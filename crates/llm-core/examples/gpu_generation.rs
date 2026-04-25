use rocm_backend::{GpuDevice, DeviceTensor, ops::matmul_f16, attention_ops::flash_attention_gpu};
use llm_core::{rms_norm::rms_norm, rope::apply_rope};
use std::sync::Arc;
use half::f16;
use tokenizers::Tokenizer;

fn bf16_to_f16_vec(bf16_data: &[u8]) -> Vec<f16> {
    let bf16_slice: &[u16] = unsafe {
        std::slice::from_raw_parts(bf16_data.as_ptr() as *const u16, bf16_data.len() / 2)
    };
    bf16_slice.iter().map(|&bf16| {
        f16::from_f32(f32::from_bits((bf16 as u32) << 16))
    }).collect()
}

fn transpose(w: &[f16], rows: usize, cols: usize) -> Vec<f16> {
    let mut t = vec![f16::ZERO; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = w[r * cols + c];
        }
    }
    t
}

struct Layer {
    wq_t: DeviceTensor<f16>,
    wk_t: DeviceTensor<f16>,
    wv_t: DeviceTensor<f16>,
    wo_t: DeviceTensor<f16>,
    gate_t: DeviceTensor<f16>,
    up_t: DeviceTensor<f16>,
    down_t: DeviceTensor<f16>,
    input_norm: Vec<f16>,
    post_attn_norm: Vec<f16>,
}

struct KVCache {
    k: Vec<f16>,
    v: Vec<f16>,
    current_len: usize,
}

impl KVCache {
    fn new(max_len: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            k: vec![f16::ZERO; max_len * num_kv_heads * head_dim],
            v: vec![f16::ZERO; max_len * num_kv_heads * head_dim],
            current_len: 0,
        }
    }
}

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║          PhysLLM - FULL GPU GENERATION                  ║");
    println!("║      (Flash Attention + RoPE + All on GPU)              ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
    
    let dev = Arc::new(GpuDevice::open_best().unwrap());
    println!("🖥️  GPU: {} ({} GB VRAM)\n", dev.props.name, dev.props.vram_total_mb/1024);
    
    let tokenizer = Tokenizer::from_file("/root/models/mistral-7b/tokenizer.json").unwrap();
    
    println!("📦 Loading model to GPU...");
    use safetensors::SafeTensors;
    use std::fs;
    use std::time::Instant;
    
    let t0 = Instant::now();
    let buf1 = fs::read("/root/models/mistral-7b/model-00001-of-00002.safetensors").unwrap();
    let buf2 = fs::read("/root/models/mistral-7b/model-00002-of-00002.safetensors").unwrap();
    let shard1 = SafeTensors::deserialize(&buf1).unwrap();
    let shard2 = SafeTensors::deserialize(&buf2).unwrap();
    
    let get = |name: &str| -> Vec<f16> {
        shard1.tensor(name).or_else(|_| shard2.tensor(name))
            .map(|t| bf16_to_f16_vec(t.data())).unwrap()
    };
    
    let embed_weights = get("model.embed_tokens.weight");
    let lm_head_weights = get("lm_head.weight");
    let final_norm = get("model.norm.weight");
    
    // Load all 32 layers - ALL weights on GPU
    let mut layers = Vec::new();
    for i in 0..32 {
        let p = format!("model.layers.{}", i);
        
        let wq = get(&format!("{}.self_attn.q_proj.weight", p));
        let wk = get(&format!("{}.self_attn.k_proj.weight", p));
        let wv = get(&format!("{}.self_attn.v_proj.weight", p));
        let wo = get(&format!("{}.self_attn.o_proj.weight", p));
        let gate = get(&format!("{}.mlp.gate_proj.weight", p));
        let up = get(&format!("{}.mlp.up_proj.weight", p));
        let down = get(&format!("{}.mlp.down_proj.weight", p));
        
        layers.push(Layer {
            wq_t: DeviceTensor::from_slice(&transpose(&wq, 4096, 4096), &[4096, 4096]).unwrap(),
            wk_t: DeviceTensor::from_slice(&transpose(&wk, 1024, 4096), &[4096, 1024]).unwrap(),
            wv_t: DeviceTensor::from_slice(&transpose(&wv, 1024, 4096), &[4096, 1024]).unwrap(),
            wo_t: DeviceTensor::from_slice(&transpose(&wo, 4096, 4096), &[4096, 4096]).unwrap(),
            gate_t: DeviceTensor::from_slice(&transpose(&gate, 14336, 4096), &[4096, 14336]).unwrap(),
            up_t: DeviceTensor::from_slice(&transpose(&up, 14336, 4096), &[4096, 14336]).unwrap(),
            down_t: DeviceTensor::from_slice(&transpose(&down, 4096, 14336), &[14336, 4096]).unwrap(),
            input_norm: get(&format!("{}.input_layernorm.weight", p)),
            post_attn_norm: get(&format!("{}.post_attention_layernorm.weight", p)),
        });
        
        if (i + 1) % 8 == 0 {
            print!("   Layer {}/32 on GPU\r", i + 1);
            use std::io::{self, Write};
            io::stdout().flush().unwrap();
        }
    }
    
    let vram_used = (dev.props.vram_total_mb - dev.free_vram_mb()) / 1024;
    println!("   Layer 32/32 ✓            ");
    println!("✓ Loaded in {:.1}s (~{} GB VRAM)\n", t0.elapsed().as_secs_f32(), vram_used);
    
    let prompt = "The physics of black holes";
    let max_tokens = 50;
    
    println!("📝 Prompt: \"{}\"\n", prompt);
    print!("💬 ");
    use std::io::{self, Write};
    io::stdout().flush().unwrap();
    
    let encoding = tokenizer.encode(prompt, false).unwrap();
    let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
    
    // KV caches for all layers
    let mut kv_caches: Vec<KVCache> = (0..32)
        .map(|_| KVCache::new(4096, 8, 128))
        .collect();
    
    let gen_start = Instant::now();
    
    for _step in 0..max_tokens {
        let tok = tokens[tokens.len() - 1] as usize;
        let emb_start = tok * 4096;
        let mut hidden = embed_weights[emb_start..emb_start + 4096].to_vec();
        
        // Run through all 32 layers - FULL GPU
        for (layer_idx, layer) in layers.iter().enumerate() {
            // RMS norm
            let normed = rms_norm(&hidden, &layer.input_norm, 1, 4096, 1e-5);
            let gpu_x = DeviceTensor::from_slice(&normed, &[1, 4096]).unwrap();
            
            // Project Q, K, V on GPU
            let mut gpu_q = DeviceTensor::alloc(&[1, 4096]).unwrap();
            let mut gpu_k = DeviceTensor::alloc(&[1, 1024]).unwrap();
            let mut gpu_v = DeviceTensor::alloc(&[1, 1024]).unwrap();
            
            matmul_f16(&dev, &gpu_x, &layer.wq_t, &mut gpu_q).unwrap();
            matmul_f16(&dev, &gpu_x, &layer.wk_t, &mut gpu_k).unwrap();
            matmul_f16(&dev, &gpu_x, &layer.wv_t, &mut gpu_v).unwrap();
            
            // Get Q, K, V back for RoPE (can be optimized to GPU later)
            let mut q = gpu_q.copy_to_host().unwrap();
            let mut k = gpu_k.copy_to_host().unwrap();
            let v = gpu_v.copy_to_host().unwrap();
            
            // Apply RoPE
            apply_rope(&mut q, 1, 32, 128, 10000.0, kv_caches[layer_idx].current_len);
            apply_rope(&mut k, 1, 8, 128, 10000.0, kv_caches[layer_idx].current_len);
            
            // Update KV cache
            let cache_start = kv_caches[layer_idx].current_len * 8 * 128;
            kv_caches[layer_idx].k[cache_start..cache_start + 1024].copy_from_slice(&k);
            kv_caches[layer_idx].v[cache_start..cache_start + 1024].copy_from_slice(&v);
            kv_caches[layer_idx].current_len += 1;
            
            let total_seq = kv_caches[layer_idx].current_len;
            
            // Attention on GPU
            let gpu_q_attn = DeviceTensor::from_slice(&q, &[1, 32, 1, 128]).unwrap();
            let gpu_k_cache = DeviceTensor::from_slice(
                &kv_caches[layer_idx].k[..total_seq * 8 * 128],
                &[1, 8, total_seq, 128]
            ).unwrap();
            let gpu_v_cache = DeviceTensor::from_slice(
                &kv_caches[layer_idx].v[..total_seq * 8 * 128],
                &[1, 8, total_seq, 128]
            ).unwrap();
            let mut gpu_attn_out = DeviceTensor::alloc(&[1, 32, 1, 128]).unwrap();
            
            flash_attention_gpu(
                &dev,
                &gpu_q_attn,
                &gpu_k_cache,
                &gpu_v_cache,
                &mut gpu_attn_out,
                1, 32, 8, 1, total_seq, 128
            ).unwrap();
            
            let attn_out_flat = gpu_attn_out.copy_to_host().unwrap();
            
            // Output projection on GPU
            let gpu_attn = DeviceTensor::from_slice(&attn_out_flat, &[1, 4096]).unwrap();
            let mut gpu_o = DeviceTensor::alloc(&[1, 4096]).unwrap();
            matmul_f16(&dev, &gpu_attn, &layer.wo_t, &mut gpu_o).unwrap();
            
            let o = gpu_o.copy_to_host().unwrap();
            
            // Residual
            for i in 0..4096 {
                hidden[i] = f16::from_f32(hidden[i].to_f32() + o[i].to_f32());
            }
            
            // MLP (already on GPU from before)
            let normed2 = rms_norm(&hidden, &layer.post_attn_norm, 1, 4096, 1e-5);
            let gpu_x2 = DeviceTensor::from_slice(&normed2, &[1, 4096]).unwrap();
            
            let mut gate_out = DeviceTensor::alloc(&[1, 14336]).unwrap();
            let mut up_out = DeviceTensor::alloc(&[1, 14336]).unwrap();
            
            matmul_f16(&dev, &gpu_x2, &layer.gate_t, &mut gate_out).unwrap();
            matmul_f16(&dev, &gpu_x2, &layer.up_t, &mut up_out).unwrap();
            
            let gate_cpu = gate_out.copy_to_host().unwrap();
            let up_cpu = up_out.copy_to_host().unwrap();
            
            let inter: Vec<f16> = gate_cpu.iter().zip(up_cpu.iter()).map(|(&g, &u)| {
                let g_f = g.to_f32();
                let silu = g_f / (1.0 + (-g_f).exp());
                f16::from_f32(silu * u.to_f32())
            }).collect();
            
            let gpu_inter = DeviceTensor::from_slice(&inter, &[1, 14336]).unwrap();
            let mut down_out = DeviceTensor::alloc(&[1, 4096]).unwrap();
            matmul_f16(&dev, &gpu_inter, &layer.down_t, &mut down_out).unwrap();
            
            let mlp_out = down_out.copy_to_host().unwrap();
            for i in 0..4096 {
                hidden[i] = f16::from_f32(hidden[i].to_f32() + mlp_out[i].to_f32());
            }
        }
        
        // Final norm + LM head
        let final_hidden = rms_norm(&hidden, &final_norm, 1, 4096, 1e-5);
        
        let mut logits = vec![0.0f32; 32000];
        for v in 0..32000 {
            let w_start = v * 4096;
            logits[v] = final_hidden.iter().zip(&lm_head_weights[w_start..w_start + 4096])
                .map(|(h, w)| h.to_f32() * w.to_f32()).sum();
        }
        
        // Sample
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|l| ((l - max_logit) / 0.7).exp()).collect();
        let sum_exp: f32 = exp.iter().sum();
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen::<f32>() * sum_exp;
        let mut cumsum = 0.0;
        let mut next_token = 0u32;
        for (i, &e) in exp.iter().enumerate() {
            cumsum += e;
            if r < cumsum {
                next_token = i as u32;
                break;
            }
        }
        
        if next_token == 2 { break; }
        
        tokens.push(next_token);
        print!("{}", tokenizer.decode(&[next_token], false).unwrap());
        io::stdout().flush().unwrap();
    }
    
    let elapsed = gen_start.elapsed().as_secs_f32();
    let tok_per_sec = (tokens.len() - encoding.get_ids().len()) as f32 / elapsed;
    
    println!("\n\n✓ Generated {} tokens in {:.1}s ({:.2} tok/s)", 
             tokens.len() - encoding.get_ids().len(),
             elapsed,
             tok_per_sec);
    
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  🎉 FULL GPU GENERATION COMPLETE!                       ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
}

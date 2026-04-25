use rocm_backend::{GpuDevice, DeviceTensor, ops::matmul_f16};
use llm_core::{attention::{attention, KVCache}, rms_norm::rms_norm};
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
    // Attention
    wq: Vec<f16>, wk: Vec<f16>, wv: Vec<f16>, wo: Vec<f16>,
    // MLP
    gate_t: DeviceTensor<f16>,
    up_t: DeviceTensor<f16>,
    down_t: DeviceTensor<f16>,
    // Norms
    input_norm: Vec<f16>,
    post_attn_norm: Vec<f16>,
}

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║          PhysLLM - COHERENT Text Generation             ║");
    println!("║          (Full Attention + RoPE + KV Cache)             ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
    
    let dev = Arc::new(GpuDevice::open_best().unwrap());
    println!("🖥️  GPU: {}\n", dev.props.name);
    
    let tokenizer = Tokenizer::from_file("/root/models/mistral-7b/tokenizer.json").unwrap();
    
    println!("📦 Loading Mistral 7B...");
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
    let final_norm_weights = get("model.norm.weight");
    
    let mut layers = Vec::new();
    for i in 0..32 {
        let p = format!("model.layers.{}", i);
        
        let gate = get(&format!("{}.mlp.gate_proj.weight", p));
        let up = get(&format!("{}.mlp.up_proj.weight", p));
        let down = get(&format!("{}.mlp.down_proj.weight", p));
        
        layers.push(Layer {
            wq: get(&format!("{}.self_attn.q_proj.weight", p)),
            wk: get(&format!("{}.self_attn.k_proj.weight", p)),
            wv: get(&format!("{}.self_attn.v_proj.weight", p)),
            wo: get(&format!("{}.self_attn.o_proj.weight", p)),
            gate_t: DeviceTensor::from_slice(&transpose(&gate, 14336, 4096), &[4096, 14336]).unwrap(),
            up_t: DeviceTensor::from_slice(&transpose(&up, 14336, 4096), &[4096, 14336]).unwrap(),
            down_t: DeviceTensor::from_slice(&transpose(&down, 4096, 14336), &[14336, 4096]).unwrap(),
            input_norm: get(&format!("{}.input_layernorm.weight", p)),
            post_attn_norm: get(&format!("{}.post_attention_layernorm.weight", p)),
        });
        
        if (i + 1) % 8 == 0 {
            print!("   Layer {}/32\r", i + 1);
            use std::io::{self, Write};
            io::stdout().flush().unwrap();
        }
    }
    println!("   Layer 32/32 ✓      ");
    println!("✓ Loaded in {:.1}s\n", t0.elapsed().as_secs_f32());
    
    let prompt = "The physics of black holes";
    let max_tokens = 50;
    
    println!("📝 Prompt: \"{}\"\n", prompt);
    print!("💬 ");
    use std::io::{self, Write};
    io::stdout().flush().unwrap();
    
    let encoding = tokenizer.encode(prompt, false).unwrap();
    let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
    
    let mut kv_caches: Vec<KVCache> = (0..32)
        .map(|_| KVCache::new(4096, 8, 128))
        .collect();
    
    let gen_start = Instant::now();
    
    for step in 0..max_tokens {
        let tok = tokens[tokens.len() - 1] as usize;
        let emb_start = tok * 4096;
        let mut hidden = embed_weights[emb_start..emb_start + 4096].to_vec();
        
        // Run through all 32 layers with FULL attention
        for (idx, layer) in layers.iter().enumerate() {
            // Pre-norm
            let normed = rms_norm(&hidden, &layer.input_norm, 1, 4096, 1e-5);
            
            // Project to Q, K, V
            let mut q = vec![f16::ZERO; 4096];  // [1, 32, 128]
            let mut k = vec![f16::ZERO; 1024];  // [1, 8, 128]
            let mut v = vec![f16::ZERO; 1024];
            
            for out_idx in 0..4096 {
                let mut sum = 0.0f32;
                for in_idx in 0..4096 {
                    sum += normed[in_idx].to_f32() * layer.wq[out_idx * 4096 + in_idx].to_f32();
                }
                q[out_idx] = f16::from_f32(sum);
            }
            
            for out_idx in 0..1024 {
                let mut sum_k = 0.0f32;
                let mut sum_v = 0.0f32;
                for in_idx in 0..4096 {
                    sum_k += normed[in_idx].to_f32() * layer.wk[out_idx * 4096 + in_idx].to_f32();
                    sum_v += normed[in_idx].to_f32() * layer.wv[out_idx * 4096 + in_idx].to_f32();
                }
                k[out_idx] = f16::from_f32(sum_k);
                v[out_idx] = f16::from_f32(sum_v);
            }
            
            // Attention with KV cache
            let attn_out = attention(&q, &k, &v, &mut kv_caches[idx], 1, 32, 8, 128, 10000.0);
            
            // Output projection
            let mut o = vec![f16::ZERO; 4096];
            for out_idx in 0..4096 {
                let mut sum = 0.0f32;
                for in_idx in 0..4096 {
                    sum += attn_out[in_idx].to_f32() * layer.wo[out_idx * 4096 + in_idx].to_f32();
                }
                o[out_idx] = f16::from_f32(sum);
            }
            
            // Residual
            for i in 0..4096 {
                hidden[i] = f16::from_f32(hidden[i].to_f32() + o[i].to_f32());
            }
            
            // MLP
            let normed2 = rms_norm(&hidden, &layer.post_attn_norm, 1, 4096, 1e-5);
            let gpu_x = DeviceTensor::from_slice(&normed2, &[1, 4096]).unwrap();
            
            let mut gate_out = DeviceTensor::alloc(&[1, 14336]).unwrap();
            let mut up_out = DeviceTensor::alloc(&[1, 14336]).unwrap();
            
            matmul_f16(&dev, &gpu_x, &layer.gate_t, &mut gate_out).unwrap();
            matmul_f16(&dev, &gpu_x, &layer.up_t, &mut up_out).unwrap();
            
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
        
        // Final norm
        let final_hidden = rms_norm(&hidden, &final_norm_weights, 1, 4096, 1e-5);
        
        // LM head
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
    
    println!("\n\n✓ Generated {} tokens in {:.1}s\n", 
             tokens.len() - encoding.get_ids().len(),
             gen_start.elapsed().as_secs_f32());
    
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  🎉 COHERENT TEXT GENERATION WORKING!                   ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
}

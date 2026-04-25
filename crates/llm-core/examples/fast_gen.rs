use rocm_backend::{GpuDevice, DeviceTensor};
use llm_core::fast_inference::{fast_layer_forward, sample_token};
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

fn transpose_weight(weight: &[f16], rows: usize, cols: usize) -> Vec<f16> {
    let mut transposed = vec![f16::ZERO; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            transposed[c * rows + r] = weight[r * cols + c];
        }
    }
    transposed
}

fn main() {
    println!("\n🚀 PhysLLM - Fast Generation (Optimized)\n");
    
    let dev = Arc::new(GpuDevice::open_best().unwrap());
    println!("GPU: {} ({} GB VRAM)\n", dev.props.name, dev.props.vram_total_mb / 1024);
    
    let tokenizer = Tokenizer::from_file("/root/models/mistral-7b/tokenizer.json").unwrap();
    
    println!("Loading model (this takes ~20 seconds)...");
    use safetensors::SafeTensors;
    use std::fs;
    use std::time::Instant;
    
    let t0 = Instant::now();
    let buf1 = fs::read("/root/models/mistral-7b/model-00001-of-00002.safetensors").unwrap();
    let buf2 = fs::read("/root/models/mistral-7b/model-00002-of-00002.safetensors").unwrap();
    let shard1 = SafeTensors::deserialize(&buf1).unwrap();
    let shard2 = SafeTensors::deserialize(&buf2).unwrap();
    
    let get_tensor = |name: &str| -> Vec<f16> {
        shard1.tensor(name).or_else(|_| shard2.tensor(name))
            .map(|t| bf16_to_f16_vec(t.data()))
            .expect(&format!("Missing: {}", name))
    };
    
    let embed_weights = get_tensor("model.embed_tokens.weight");
    let lm_head_weights = get_tensor("lm_head.weight");
    
    // Pre-transpose all weights for faster matmul
    println!("  Transposing weights for GPU...");
    let mut layers_gate = Vec::new();
    let mut layers_up = Vec::new();
    let mut layers_down = Vec::new();
    
    for i in 0..32 {
        let prefix = format!("model.layers.{}.mlp", i);
        
        let gate = get_tensor(&format!("{}.gate_proj.weight", prefix));
        let up = get_tensor(&format!("{}.up_proj.weight", prefix));
        let down = get_tensor(&format!("{}.down_proj.weight", prefix));
        
        layers_gate.push(DeviceTensor::from_slice(
            &transpose_weight(&gate, 14336, 4096),
            &[4096, 14336]
        ).unwrap());
        
        layers_up.push(DeviceTensor::from_slice(
            &transpose_weight(&up, 14336, 4096),
            &[4096, 14336]
        ).unwrap());
        
        layers_down.push(DeviceTensor::from_slice(
            &transpose_weight(&down, 4096, 14336),
            &[14336, 4096]
        ).unwrap());
        
        if (i + 1) % 8 == 0 {
            print!("  Layer {}/32\r", i + 1);
            use std::io::{self, Write};
            io::stdout().flush().unwrap();
        }
    }
    println!("  Layer 32/32 ✓      ");
    
    println!("✓ Loaded in {:.1}s ({} GB VRAM)\n", 
             t0.elapsed().as_secs_f32(),
             (dev.props.vram_total_mb - dev.free_vram_mb()) / 1024);
    
    let prompt = "The physics of black holes";
    let max_tokens = 50;
    
    println!("Prompt: \"{}\"\n", prompt);
    print!("Output: {}", prompt);
    use std::io::{self, Write};
    io::stdout().flush().unwrap();
    
    let encoding = tokenizer.encode(prompt, false).unwrap();
    let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
    
    let gen_start = Instant::now();
    
    for step in 0..max_tokens {
        let last_token = tokens[tokens.len() - 1] as usize;
        let emb_start = last_token * 4096;
        let hidden = &embed_weights[emb_start..emb_start + 4096];
        
        let mut gpu_hidden = DeviceTensor::from_slice(hidden, &[1, 4096]).unwrap();
        
        // Run through all 32 layers
        for layer_idx in 0..32 {
            gpu_hidden = fast_layer_forward(
                &dev,
                &gpu_hidden,
                &layers_gate[layer_idx],
                &layers_up[layer_idx],
                &layers_down[layer_idx],
            ).unwrap();
        }
        
        // LM head
        let final_hidden = gpu_hidden.copy_to_host().unwrap();
        let mut logits = vec![0.0f32; 32000];
        for v in 0..32000 {
            let w_start = v * 4096;
            logits[v] = final_hidden.iter().zip(&lm_head_weights[w_start..w_start + 4096])
                .map(|(h, w)| h.to_f32() * w.to_f32()).sum();
        }
        
        let next_token = sample_token(&logits, 0.7);
        if next_token == 2 { break; }
        
        tokens.push(next_token);
        print!("{}", tokenizer.decode(&[next_token], false).unwrap());
        io::stdout().flush().unwrap();
    }
    
    let elapsed = gen_start.elapsed().as_secs_f32();
    println!("\n\n✓ Generated {} tokens in {:.1}s ({:.2} tok/s)", 
             tokens.len() - encoding.get_ids().len(),
             elapsed,
             (tokens.len() - encoding.get_ids().len()) as f32 / elapsed);
}

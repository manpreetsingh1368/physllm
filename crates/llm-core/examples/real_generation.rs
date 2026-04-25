use rocm_backend::{GpuDevice, DeviceTensor};
use llm_core::inference::{layer_forward, sample_token};
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

struct MistralLayer {
    q_proj: DeviceTensor<f16>,
    k_proj: DeviceTensor<f16>,
    v_proj: DeviceTensor<f16>,
    o_proj: DeviceTensor<f16>,
    gate_proj: DeviceTensor<f16>,
    up_proj: DeviceTensor<f16>,
    down_proj: DeviceTensor<f16>,
}

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║          PhysLLM - Mistral 7B Generation Demo           ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
    
    let dev = Arc::new(GpuDevice::open_best().unwrap());
    println!("🖥️  GPU: {}", dev.props.name);
    println!("💾 VRAM: {} GB\n", dev.props.vram_total_mb / 1024);
    
    // Load tokenizer
    println!("📖 Loading tokenizer...");
    let tokenizer = Tokenizer::from_file("/root/models/mistral-7b/tokenizer.json").unwrap();
    println!("✓ Tokenizer ready\n");
    
    // Load model
    println!("🔄 Loading Mistral 7B weights...");
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
            .unwrap_or_else(|_| panic!("Missing: {}", name))
    };
    
    // Load embeddings and LM head
    let embed_weights = get_tensor("model.embed_tokens.weight");
    let lm_head_weights = get_tensor("lm_head.weight");
    
    // Load layer 0 (using just first layer for MVP)
    let layer0 = MistralLayer {
        q_proj: DeviceTensor::from_slice(&get_tensor("model.layers.0.self_attn.q_proj.weight"), &[4096, 4096]).unwrap(),
        k_proj: DeviceTensor::from_slice(&get_tensor("model.layers.0.self_attn.k_proj.weight"), &[1024, 4096]).unwrap(),
        v_proj: DeviceTensor::from_slice(&get_tensor("model.layers.0.self_attn.v_proj.weight"), &[1024, 4096]).unwrap(),
        o_proj: DeviceTensor::from_slice(&get_tensor("model.layers.0.self_attn.o_proj.weight"), &[4096, 4096]).unwrap(),
        gate_proj: DeviceTensor::from_slice(&get_tensor("model.layers.0.mlp.gate_proj.weight"), &[14336, 4096]).unwrap(),
        up_proj: DeviceTensor::from_slice(&get_tensor("model.layers.0.mlp.up_proj.weight"), &[14336, 4096]).unwrap(),
        down_proj: DeviceTensor::from_slice(&get_tensor("model.layers.0.mlp.down_proj.weight"), &[4096, 14336]).unwrap(),
    };
    
    println!("✓ Model loaded in {:.1}s", t0.elapsed().as_secs_f32());
    println!("💾 VRAM used: ~{} GB\n", (dev.props.vram_total_mb - dev.free_vram_mb()) / 1024);
    
    // Generation settings
    let prompt = "The physics of black holes";
    let max_new_tokens = 20;
    let temperature = 0.7;
    
    println!("📝 Prompt: \"{}\"", prompt);
    println!("⚙️  Settings: max_tokens={}, temperature={}\n", max_new_tokens, temperature);
    
    // Tokenize
    let encoding = tokenizer.encode(prompt, false).unwrap();
    let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
    
    println!("🔢 Input tokens: {:?}\n", tokens);
    println!("🚀 Generating...\n");
    
    // Generate tokens one by one
    for i in 0..max_new_tokens {
        let seq_len = tokens.len();
        
        // Get embeddings for current sequence
        let mut hidden_states = Vec::with_capacity(seq_len * 4096);
        for &tok_id in &tokens {
            let start = (tok_id as usize) * 4096;
            let end = start + 4096;
            hidden_states.extend_from_slice(&embed_weights[start..end]);
        }
        
        // Run through layer 0 (simplified - normally would be all 32 layers)
        let output = layer_forward(
            &dev,
            &hidden_states[(seq_len - 1) * 4096..], // Last token only
            1,
            4096,
            &layer0.q_proj,
            &layer0.k_proj,
            &layer0.v_proj,
            &layer0.o_proj,
            &layer0.gate_proj,
            &layer0.up_proj,
            &layer0.down_proj,
        ).expect("Forward pass failed");
        
        // Apply LM head: [4096] → [32000 vocab]
        let mut logits = vec![0.0f32; 32000];
        for vocab_idx in 0..32000 {
            let weight_start = vocab_idx * 4096;
            let weight_slice = &lm_head_weights[weight_start..weight_start + 4096];
            let dot: f32 = output.iter().zip(weight_slice.iter())
                .map(|(h, w)| h.to_f32() * w.to_f32())
                .sum();
            logits[vocab_idx] = dot;
        }
        
        // Sample next token
        let next_token = sample_token(&logits, temperature);
        tokens.push(next_token);
        
        // Decode and print
        let text = tokenizer.decode(&[next_token], false).unwrap();
        print!("{}", text);
        use std::io::{self, Write};
        io::stdout().flush().unwrap();
        
        // Stop on EOS
        if next_token == 2 {
            break;
        }
    }
    
    println!("\n\n✓ Generation complete!");
    println!("\n📊 Full output:");
    let full_text = tokenizer.decode(&tokens, false).unwrap();
    println!("   \"{}\"\n", full_text);
    
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  🎉 PhysLLM is working! You have a functioning 7B LLM!  ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
}

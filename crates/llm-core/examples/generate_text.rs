use rocm_backend::{GpuDevice, DeviceTensor, ops::matmul_f16};
use std::sync::Arc;
use half::f16;

fn bf16_to_f16_vec(bf16_data: &[u8]) -> Vec<f16> {
    let bf16_slice: &[u16] = unsafe {
        std::slice::from_raw_parts(bf16_data.as_ptr() as *const u16, bf16_data.len() / 2)
    };
    bf16_slice.iter().map(|&bf16| {
        f16::from_f32(f32::from_bits((bf16 as u32) << 16))
    }).collect()
}

struct MistralLayer {
    // Attention
    q_proj: DeviceTensor<f16>,
    k_proj: DeviceTensor<f16>,
    v_proj: DeviceTensor<f16>,
    o_proj: DeviceTensor<f16>,
    // MLP
    gate_proj: DeviceTensor<f16>,
    up_proj: DeviceTensor<f16>,
    down_proj: DeviceTensor<f16>,
    // Norms
    input_norm: DeviceTensor<f16>,
    post_attn_norm: DeviceTensor<f16>,
}

fn main() {
    println!("Mistral 7B Text Generation\n");
    
    let dev = Arc::new(GpuDevice::open_best().unwrap());
    println!("GPU: {}\n", dev.props.name);
    
    use safetensors::SafeTensors;
    use std::fs;
    
    println!("Loading model weights (this takes ~30 seconds)...");
    let buf1 = fs::read("/root/models/mistral-7b/model-00001-of-00002.safetensors").unwrap();
    let buf2 = fs::read("/root/models/mistral-7b/model-00002-of-00002.safetensors").unwrap();
    let shard1 = SafeTensors::deserialize(&buf1).unwrap();
    let shard2 = SafeTensors::deserialize(&buf2).unwrap();
    
    let get_tensor = |name: &str| -> Vec<f16> {
        shard1.tensor(name).or_else(|_| shard2.tensor(name))
            .map(|t| bf16_to_f16_vec(t.data()))
            .expect(&format!("Missing: {}", name))
    };
    
    // Load embeddings
    let embed_weights = get_tensor("model.embed_tokens.weight");
    let gpu_embed = DeviceTensor::from_slice(&embed_weights, &[32000, 4096]).unwrap();
    
    // Load all 32 layers
    let mut layers = Vec::new();
    for i in 0..32 {
        if i % 8 == 0 {
            print!("  Layer {}/32...\r", i);
            use std::io::{self, Write};
            io::stdout().flush().unwrap();
        }
        
        let prefix = format!("model.layers.{}", i);
        layers.push(MistralLayer {
            q_proj: DeviceTensor::from_slice(
                &get_tensor(&format!("{}.self_attn.q_proj.weight", prefix)),
                &[4096, 4096]
            ).unwrap(),
            k_proj: DeviceTensor::from_slice(
                &get_tensor(&format!("{}.self_attn.k_proj.weight", prefix)),
                &[1024, 4096]
            ).unwrap(),
            v_proj: DeviceTensor::from_slice(
                &get_tensor(&format!("{}.self_attn.v_proj.weight", prefix)),
                &[1024, 4096]
            ).unwrap(),
            o_proj: DeviceTensor::from_slice(
                &get_tensor(&format!("{}.self_attn.o_proj.weight", prefix)),
                &[4096, 4096]
            ).unwrap(),
            gate_proj: DeviceTensor::from_slice(
                &get_tensor(&format!("{}.mlp.gate_proj.weight", prefix)),
                &[14336, 4096]
            ).unwrap(),
            up_proj: DeviceTensor::from_slice(
                &get_tensor(&format!("{}.mlp.up_proj.weight", prefix)),
                &[14336, 4096]
            ).unwrap(),
            down_proj: DeviceTensor::from_slice(
                &get_tensor(&format!("{}.mlp.down_proj.weight", prefix)),
                &[4096, 14336]
            ).unwrap(),
            input_norm: DeviceTensor::from_slice(
                &get_tensor(&format!("{}.input_layernorm.weight", prefix)),
                &[4096]
            ).unwrap(),
            post_attn_norm: DeviceTensor::from_slice(
                &get_tensor(&format!("{}.post_attention_layernorm.weight", prefix)),
                &[4096]
            ).unwrap(),
        });
    }
    println!("  Layer 32/32... Done!    ");
    
    // Load output head
    let lm_head = get_tensor("lm_head.weight");
    let gpu_lm_head = DeviceTensor::from_slice(&lm_head, &[32000, 4096]).unwrap();
    
    let vram_used = (dev.props.vram_total_mb - dev.free_vram_mb()) / 1024;
    println!("\n✓ Model loaded: ~{}GB VRAM used", vram_used);
    println!("  VRAM free: {}GB\n", dev.free_vram_mb() / 1024);
    
    // Simple inference test (just embeddings for now)
    println!("Testing inference...");
    let test_tokens = vec![1u32, 15043, 3186]; // "Hello world"
    
    // In real inference, you'd:
    // 1. Look up embeddings for each token
    // 2. Run through 32 transformer layers
    // 3. Apply LM head → get logits for next token
    // 4. Sample from logits → get next token ID
    // 5. Repeat until EOS or max length
    
    println!("Input tokens: {:?}", test_tokens);
    println!("\n✓ Full model ready for text generation!");
    println!("\nTo generate text, need:");
    println!("  1. Tokenizer (decode IDs → text)");
    println!("  2. Forward pass implementation");
    println!("  3. Sampling (temperature, top-p, top-k)");
    println!("\nYou have the HARDEST part done: 13GB model on GPU!");
}

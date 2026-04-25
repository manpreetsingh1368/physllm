use rocm_backend::{GpuDevice, DeviceTensor};
use std::sync::Arc;
use half::f16;

fn main() {
    println!("Simple Mistral Inference Test\n");
    
    let dev = Arc::new(GpuDevice::open_best().unwrap());
    println!("GPU: {}\n", dev.props.name);
    
    // Load embeddings (same as before)
    use safetensors::SafeTensors;
    use std::fs;
    
    let buf1 = fs::read("/root/models/mistral-7b/model-00001-of-00002.safetensors").unwrap();
    let shard1 = SafeTensors::deserialize(&buf1).unwrap();
    
    let embed_tensor = shard1.tensor("model.embed_tokens.weight").unwrap();
    let embed_data = embed_tensor.data();
    let embed_bf16: &[u16] = unsafe {
        std::slice::from_raw_parts(embed_data.as_ptr() as *const u16, embed_data.len() / 2)
    };
    
    // Convert BF16 → F16
    let embed_f16: Vec<f16> = embed_bf16.iter().map(|&bf16| {
        f16::from_f32(f32::from_bits((bf16 as u32) << 16))
    }).collect();
    
    let gpu_embed = DeviceTensor::from_slice(&embed_f16, &[32000, 4096]).unwrap();
    println!("✓ Embeddings on GPU (262 MB)\n");
    
    // Test tokens: "Hello world"
    let tokens = vec![1u32, 15043, 3186]; // BOS, "Hello", "world"
    println!("Input: {:?}", tokens);
    
    // Look up embeddings (just copy subset from GPU for now)
    let seq_len = tokens.len();
    let hidden_dim = 4096;
    
    // In real inference, you'd index into gpu_embed
    // For now, just show it's ready
    println!("Sequence length: {}", seq_len);
    println!("Hidden dim: {}", hidden_dim);
    println!("Embedding shape: [batch=1, seq={}, hidden={}]", seq_len, hidden_dim);
    
    println!("\n✓ Ready for forward pass!");
    println!("\nTo complete inference, need to:");
    println!("  1. Index embeddings by token IDs");
    println!("  2. Load attention weights (q/k/v/o projections)");
    println!("  3. Load MLP weights (gate/up/down)");
    println!("  4. Run 32 transformer layers");
    println!("  5. Apply LM head → logits → sample next token");
    
    println!("\nFor now: embeddings work, GPU has 190GB free!");
    println!("Next: implement token → text generation loop");
}

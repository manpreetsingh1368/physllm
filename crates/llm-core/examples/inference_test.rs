use llm_core::config::ModelConfig;
use rocm_backend::{GpuDevice, DeviceTensor};
use std::sync::Arc;
use std::path::Path;
use std::collections::HashMap;
use half::f16;

fn main() {
    println!("Mistral 7B Inference Test\n");
    
    let dev = Arc::new(GpuDevice::open_best().expect("GPU"));
    println!("✓ GPU: {}\n", dev.props.name);
    
    println!("Loading weights from safetensors...");
    let model_dir = Path::new("/root/models/mistral-7b");
    
    use safetensors::SafeTensors;
    use std::fs;
    
    let buf1 = fs::read(model_dir.join("model-00001-of-00002.safetensors")).unwrap();
    let buf2 = fs::read(model_dir.join("model-00002-of-00002.safetensors")).unwrap();
    let shard1 = SafeTensors::deserialize(&buf1).unwrap();
    let shard2 = SafeTensors::deserialize(&buf2).unwrap();
    
    // Find embedding layer
    let embed_name = "model.embed_tokens.weight";
    let embed_tensor = shard1.tensor(embed_name)
        .or_else(|_| shard2.tensor(embed_name))
        .expect("embed_tokens not found");
    
    println!("✓ Found embeddings: {:?} {:?}", embed_tensor.dtype(), embed_tensor.shape());
    
    // Convert BF16 → F16
    let embed_data = embed_tensor.data();
    let embed_bf16: &[u16] = unsafe {
        std::slice::from_raw_parts(
            embed_data.as_ptr() as *const u16,
            embed_data.len() / 2
        )
    };
    
    // BF16 → F32 → F16 conversion
    let embed_f16: Vec<f16> = embed_bf16.iter().map(|&bf16_bits| {
        // BF16 is just F32 with lower 16 bits truncated
        let f32_bits = (bf16_bits as u32) << 16;
        let f32_val = f32::from_bits(f32_bits);
        f16::from_f32(f32_val)
    }).collect();
    
    println!("✓ Converted {} BF16 weights → F16", embed_f16.len());
    
    // Upload to GPU
    let shape = embed_tensor.shape();
    let gpu_embed = DeviceTensor::from_slice(&embed_f16, shape)
        .expect("Upload to GPU");
    
    println!("✓ Uploaded embeddings to GPU");
    println!("  Shape: {:?}", shape);
    println!("  VRAM used: ~{} MB\n", embed_f16.len() * 2 / 1_000_000);
    
    // Test: look up token embeddings
    println!("Testing token embedding lookup...");
    let test_tokens = vec![1u32, 15043, 310, 4628, 26467]; // "The physics of black holes"
    
    println!("✓ Token IDs: {:?}", test_tokens);
    println!("  (Would extract embeddings and run forward pass)");
    println!("\n✓ Weight loading works! Next: implement full forward pass");
}

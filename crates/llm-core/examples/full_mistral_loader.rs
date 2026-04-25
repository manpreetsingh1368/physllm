use rocm_backend::{GpuDevice, DeviceTensor};
use std::sync::Arc;
use std::collections::HashMap;
use half::f16;

fn bf16_to_f16_vec(bf16_data: &[u8]) -> Vec<f16> {
    let bf16_slice: &[u16] = unsafe {
        std::slice::from_raw_parts(bf16_data.as_ptr() as *const u16, bf16_data.len() / 2)
    };
    bf16_slice.iter().map(|&bf16| {
        f16::from_f32(f32::from_bits((bf16 as u32) << 16))
    }).collect()
}

fn main() {
    println!("Loading Full Mistral 7B Model\n");
    
    let dev = Arc::new(GpuDevice::open_best().unwrap());
    println!("GPU: {} ({}GB VRAM)\n", dev.props.name, dev.props.vram_total_mb/1024);
    
    use safetensors::SafeTensors;
    use std::fs;
    
    println!("Reading safetensors...");
    let buf1 = fs::read("/root/models/mistral-7b/model-00001-of-00002.safetensors").unwrap();
    let buf2 = fs::read("/root/models/mistral-7b/model-00002-of-00002.safetensors").unwrap();
    let shard1 = SafeTensors::deserialize(&buf1).unwrap();
    let shard2 = SafeTensors::deserialize(&buf2).unwrap();
    
    // Helper to get tensor from either shard
    let get_tensor = |name: &str| -> Vec<f16> {
        let tensor = shard1.tensor(name).or_else(|_| shard2.tensor(name))
            .expect(&format!("Tensor {} not found", name));
        bf16_to_f16_vec(tensor.data())
    };
    
    println!("Loading embeddings...");
    let embed = get_tensor("model.embed_tokens.weight");
    let gpu_embed = DeviceTensor::from_slice(&embed, &[32000, 4096]).unwrap();
    println!("✓ Embeddings: 262 MB");
    
    println!("\nLoading layer 0 (test)...");
    let q0 = get_tensor("model.layers.0.self_attn.q_proj.weight");
    let k0 = get_tensor("model.layers.0.self_attn.k_proj.weight");
    let v0 = get_tensor("model.layers.0.self_attn.v_proj.weight");
    let o0 = get_tensor("model.layers.0.self_attn.o_proj.weight");
    
    let gpu_q0 = DeviceTensor::from_slice(&q0, &[4096, 4096]).unwrap();
    let gpu_k0 = DeviceTensor::from_slice(&k0, &[1024, 4096]).unwrap();
    let gpu_v0 = DeviceTensor::from_slice(&v0, &[1024, 4096]).unwrap();
    let gpu_o0 = DeviceTensor::from_slice(&o0, &[4096, 4096]).unwrap();
    
    println!("✓ Layer 0 attention: {} MB", (q0.len() + k0.len() + v0.len() + o0.len()) * 2 / 1_000_000);
    
    let gate0 = get_tensor("model.layers.0.mlp.gate_proj.weight");
    let up0 = get_tensor("model.layers.0.mlp.up_proj.weight");
    let down0 = get_tensor("model.layers.0.mlp.down_proj.weight");
    
    let gpu_gate0 = DeviceTensor::from_slice(&gate0, &[14336, 4096]).unwrap();
    let gpu_up0 = DeviceTensor::from_slice(&up0, &[14336, 4096]).unwrap();
    let gpu_down0 = DeviceTensor::from_slice(&down0, &[4096, 14336]).unwrap();
    
    println!("✓ Layer 0 MLP: {} MB", (gate0.len() + up0.len() + down0.len()) * 2 / 1_000_000);
    
    let vram_free = dev.free_vram_mb();
    println!("\nVRAM remaining: {} GB", vram_free / 1024);
    
    println!("\n✓ Single layer loaded successfully!");
    println!("\nNext: Load all 32 layers (~13GB total)");
    println!("Then: Implement forward pass + text generation");
}

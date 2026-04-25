use llm_core::config::ModelConfig;
use rocm_backend::GpuDevice;
use std::sync::Arc;
use std::path::Path;

fn main() {
    println!("Loading Mistral 7B from safetensors...\n");
    
    let dev = Arc::new(GpuDevice::open_best().expect("GPU init"));
    println!("✓ GPU: {} ({}GB VRAM)\n", dev.props.name, dev.props.vram_total_mb/1024);
    
    println!("Reading safetensors files...");
    let model_dir = Path::new("/root/models/mistral-7b");
    
    // Load using safetensors crate
    use safetensors::SafeTensors;
    use std::fs;
    
    let buf1 = fs::read(model_dir.join("model-00001-of-00002.safetensors")).unwrap();
    let shard1 = SafeTensors::deserialize(&buf1).unwrap();
    
    let buf2 = fs::read(model_dir.join("model-00002-of-00002.safetensors")).unwrap();
    let shard2 = SafeTensors::deserialize(&buf2).unwrap();
    
    println!("✓ Loaded safetensors files");
    println!("  Shard 1: {} tensors", shard1.names().len());
    println!("  Shard 2: {} tensors", shard2.names().len());
    
    // Show what's inside
    println!("\nSample tensor names from shard 1:");
    for name in shard1.names().iter().take(10) {
        let tensor = shard1.tensor(name).unwrap();
        println!("  {}: {:?} shape={:?}", name, tensor.dtype(), tensor.shape());
    }
    
    println!("\nSample tensor names from shard 2:");
    for name in shard2.names().iter().take(10) {
        let tensor = shard2.tensor(name).unwrap();
        println!("  {}: {:?} shape={:?}", name, tensor.dtype(), tensor.shape());
    }
    
    println!("\n✓ Model weights loaded from disk!");
    println!("\nNext steps:");
    println!("  1. Map tensor names to ModelWeights fields");
    println!("  2. Convert BF16/F32 → F16 if needed");
    println!("  3. Upload to GPU");
    println!("  4. Run inference!");
}

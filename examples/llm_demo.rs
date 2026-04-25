use llm_core::{ModelWeights, config::ModelConfig};
use rocm_backend::GpuDevice;

fn main() {
    println!("PhysLLM GPU Inference Demo");
    println!("===========================\n");
    
    let dev = GpuDevice::open_best().expect("GPU init failed");
    println!("✓ GPU: {}", dev.props.name);
    println!("  VRAM: {} GB total, {} GB free\n", 
             dev.props.vram_total_mb / 1024,
             dev.props.vram_free_mb / 1024);
    
    // Small test model (fits in 2GB)
    let cfg = ModelConfig {
        vocab_size: 32000,
        hidden_dim: 512,
        num_heads: 8,
        num_kv_heads: 8,
        head_dim: 64,
        num_layers: 4,
        intermediate_dim: 1024,
        max_seq_len: 512,
        rope_theta: 10000.0,
    };
    
    println!("Initializing model...");
    println!("  Config: {} layers, {}D hidden, {} heads", 
             cfg.num_layers, cfg.hidden_dim, cfg.num_heads);
    
    let model = ModelWeights::random_init(&cfg, &dev)
        .expect("Failed to init model");
    
    let params = (cfg.vocab_size * cfg.hidden_dim * 2 +
                  cfg.num_layers * cfg.hidden_dim * cfg.hidden_dim * 12) / 1_000_000;
    
    println!("\n✓ Model loaded on GPU!");
    println!("  Parameters: ~{} million", params);
    println!("  Estimated VRAM: ~{} MB", params * 2); // f16 = 2 bytes/param
    
    let free_after = dev.free_vram_mb();
    println!("  VRAM after load: {} GB free\n", free_after / 1024);
    
    println!("✓ GPU-accelerated LLM backend is working!");
    println!("\nNext steps:");
    println!("  1. Load real weights (Llama, Mistral, etc.)");
    println!("  2. Implement tokenizer");
    println!("  3. Run inference loop");
    println!("  4. Build API server");
}

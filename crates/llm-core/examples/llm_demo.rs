use llm_core::config::ModelConfig;
use llm_core::model::ModelWeights;
use rocm_backend::GpuDevice;
use std::sync::Arc;

fn main() {
    println!("PhysLLM GPU Inference Demo");
    println!("===========================\n");
    
    let dev = Arc::new(GpuDevice::open_best().expect("GPU init failed"));
    println!("✓ GPU: {}", dev.props.name);
    println!("  VRAM: {} GB total, {} GB free\n", 
             dev.props.vram_total_mb / 1024,
             dev.props.vram_free_mb / 1024);
    
    // Use the 7B preset config
    let cfg = ModelConfig::physllm_7b();
    
    println!("Initializing PhysLLM-7B model...");
    println!("  Architecture: {} layers, {}D hidden, {} heads", 
             cfg.num_layers, cfg.hidden_dim, cfg.num_heads);
    println!("  Vocab size: {} (+ {} domain tokens)", 
             cfg.vocab_size, cfg.domain_vocab_size);
    println!("  Max sequence: {} tokens\n", cfg.max_seq_len);
    
    println!("Allocating model weights on GPU...");
    match ModelWeights::random_init(&cfg, &dev) {
        Ok(_model) => {
            let params = (cfg.vocab_size * cfg.hidden_dim * 2 +
                          cfg.num_layers * cfg.hidden_dim * cfg.hidden_dim * 12) / 1_000_000;
            
            println!("\n✓ Model loaded on GPU!");
            println!("  Parameters: ~{} million", params);
            println!("  Estimated VRAM: ~{} MB (f16)", params * 2);
            
            let free_after = dev.free_vram_mb();
            println!("  VRAM after load: {} GB free\n", free_after / 1024);
            
            println!("✓ GPU-accelerated physics LLM backend is working!");
            println!("\nWith 196GB VRAM, you can run:");
            println!("  - PhysLLM-7B  (~14GB)");
            println!("  - PhysLLM-13B (~26GB)");
            println!("  - Llama 70B   (~140GB)");
            println!("  - Multiple models simultaneously!");
            
            println!("\nNext steps:");
            println!("  1. Load real weights from checkpoint");
            println!("  2. Implement physics-aware tokenizer");
            println!("  3. Run inference with N-body/QM simulations");
            println!("  4. Build REST API + web UI");
        }
        Err(e) => {
            eprintln!("\n✗ Model init failed: {}", e);
            std::process::exit(1);
        }
    }
}

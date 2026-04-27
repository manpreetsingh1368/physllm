use rocm_backend::GpuDevice;
use llm_core::config::ModelConfig;
use std::sync::Arc;

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║          PhysLLM — GPU Inference Demo                    ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let dev = Arc::new(GpuDevice::open_best().expect("GPU init"));
    println!("🖥️  GPU: {} ({} GB VRAM)\n", dev.props.name, dev.props.vram_total_mb / 1024);

    // Show available configs
    let cfg_7b = ModelConfig::physllm_7b();
    let cfg_20b = ModelConfig::physllm_20b();
    let cfg_oss = ModelConfig::gpt_oss_20b();

    println!("Available model configs:");
    println!("  PhysLLM 7B:    {} layers, {}D hidden, {} heads, ~6B params",
             cfg_7b.num_layers, cfg_7b.hidden_dim, cfg_7b.num_heads);
    println!("  PhysLLM 20B:   {} layers, {}D hidden, {} heads, ~20B params",
             cfg_20b.num_layers, cfg_20b.hidden_dim, cfg_20b.num_heads);
    println!("  GPT-OSS 20B:   {} layers, {}D hidden, {} experts, MoE",
             cfg_oss.num_layers, cfg_oss.hidden_dim, cfg_oss.num_experts);

    println!("\n✓ PhysLLM ready for inference!");
    println!("\nNext steps:");
    println!("  1. Download model weights:");
    println!("     ./scripts/download_model.sh");
    println!("  2. Load weights and run generation");
    println!("  3. Or use GPT-OSS-20B:");
    println!("     huggingface-cli download openai/gpt-oss-20b");
}

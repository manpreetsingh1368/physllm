use rocm_backend::GpuDevice;

fn main() {
    println!("\nPhysLLM GPU Test");
    println!("════════════════\n");
    
    match GpuDevice::open_best() {
        Ok(dev) => {
            println!("✓ GPU detected!\n");
            println!("  Device:  {}", dev.props.name);
            println!("  VRAM:    {} MB total", dev.props.vram_total_mb);
            println!("  VRAM:    {} MB free", dev.props.vram_free_mb);
            println!("  CUs:     {}", dev.props.compute_units);
            println!("\n✓ ROCm backend working!");
        }
        Err(e) => {
            eprintln!("✗ GPU failed: {}", e);
            std::process::exit(1);
        }
    }
}

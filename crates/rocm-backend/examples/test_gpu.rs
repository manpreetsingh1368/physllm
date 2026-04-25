use rocm_backend::GpuDevice;

fn main() {
    println!("PhysLLM GPU Test");
    println!("================\n");
    
    match GpuDevice::open_best() {
        Ok(dev) => {
            println!("✓ GPU detected successfully!\n");
            println!("Device:       {}", dev.props.name);
            println!("Architecture: {}", dev.props.gfx_arch);
            println!("Device ID:    {}", dev.props.device_id);
            println!("VRAM Total:   {} MB", dev.props.vram_total_mb);
            println!("VRAM Free:    {} MB", dev.props.vram_free_mb);
            println!("Compute Units: {}", dev.props.compute_units);
            println!("Wavefront Size: {}", dev.props.wavefront_size);
            println!("Clock:        {} MHz", dev.props.clock_mhz);
            println!("Memory BW:    {:.1} GB/s", dev.props.memory_bw_gbps);
            println!("\n✓ ROCm backend is working!");
        }
        Err(e) => {
            eprintln!("✗ Failed to open GPU: {}", e);
            eprintln!("\nTroubleshooting:");
            eprintln!("  1. Check ROCm is installed: ls /opt/rocm");
            eprintln!("  2. Check GPU is visible: rocminfo | grep gfx");
            eprintln!("  3. Check permissions: groups | grep render");
            std::process::exit(1);
        }
    }
}

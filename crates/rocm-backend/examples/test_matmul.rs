use rocm_backend::{GpuDevice, DeviceTensor, matmul_f16};
use half::f16;

fn main() {
    println!("\nPhysLLM GPU Matmul Test");
    println!("══════════════════════\n");

    let dev = GpuDevice::open_best().expect("GPU init");
    println!("Using: {}\n", dev.props.name);

    // [2x3] @ [3x2] = [2x2]
    let a: Vec<f16> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].iter().map(|&x| f16::from_f32(x)).collect();
    let b: Vec<f16> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].iter().map(|&x| f16::from_f32(x)).collect();

    let ga = DeviceTensor::from_slice(&a, &[2, 3]).expect("alloc a");
    let gb = DeviceTensor::from_slice(&b, &[3, 2]).expect("alloc b");
    let mut gc = DeviceTensor::alloc(&[2, 2]).expect("alloc c");

    matmul_f16(&dev, &ga, &gb, &mut gc).expect("matmul");

    let result = gc.copy_to_host().expect("copy");
    println!("Result:");
    println!("  {:.1}  {:.1}", result[0].to_f32(), result[1].to_f32());
    println!("  {:.1}  {:.1}", result[2].to_f32(), result[3].to_f32());
    println!("\nExpected:");
    println!("  22.0  28.0");
    println!("  49.0  64.0");
    println!("\n✓ GPU matmul working!");
}

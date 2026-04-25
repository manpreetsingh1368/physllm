use rocm_backend::{GpuDevice, DeviceTensor, ops::matmul_f16};
use half::f16;

fn main() {
    println!("Testing GPU matrix multiply...\n");
    
    let dev = GpuDevice::open_best().expect("GPU init");
    println!("Using: {}\n", dev.props.name);
    
    // Small matmul: [2x3] · [3x2] = [2x2]
    let a_host = vec![1.0, 2.0, 3.0,  4.0, 5.0, 6.0];
    let b_host = vec![1.0, 2.0,  3.0, 4.0,  5.0, 6.0];
    
    let a_f16: Vec<f16> = a_host.iter().map(|&x| f16::from_f32(x)).collect();
    let b_f16: Vec<f16> = b_host.iter().map(|&x| f16::from_f32(x)).collect();
    
    let a = DeviceTensor::from_slice(&a_f16, &[2, 3]).expect("alloc a");
    let b = DeviceTensor::from_slice(&b_f16, &[3, 2]).expect("alloc b");
    let mut c = DeviceTensor::alloc(&[2, 2]).expect("alloc c");
    
    println!("Running matmul on GPU...");
    matmul_f16(&dev, &a, &b, &mut c).expect("matmul");
    
    let result = c.copy_to_host().expect("copy result");
    println!("Result:");
    for i in 0..2 {
        for j in 0..2 {
            print!("{:6.1} ", result[i*2 + j].to_f32());
        }
        println!();
    }
    
    println!("\n✓ GPU matmul working!");
}

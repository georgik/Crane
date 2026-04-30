use candle_core::Device;

fn main() {
    let metal_device = Device::new_metal(0);
    println!("Metal device created: {:?}", metal_device);
    
    let cpu_device = Device::Cpu;
    println!("CPU device: {:?}", cpu_device);
}

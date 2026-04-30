//! Diagnostic tool to check Metal device creation

use crane_core::models::{Device, DType, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Metal Device Diagnostic ===\n");

    // Test Metal device creation
    println!("Attempting to create Metal device...");
    let metal_device = match Device::new_metal(0) {
        Ok(device) => {
            println!("✓ Metal device created: {:?}", device);
            device
        }
        Err(e) => {
            println!("✗ Metal device creation failed: {}", e);
            println!("Falling back to CPU");
            Device::Cpu
        }
    };

    // Test tensor creation
    println!("\nCreating tensor on device...");
    let t = Tensor::zeros((2, 3), DType::F32, &metal_device)?;
    println!("✓ Tensor created");
    println!("  dtype: {:?}", t.dtype());
    println!("  device: {:?}", t.device());
    println!("  shape: {:?}", t.shape());

    println!("\n=== Diagnostic Complete ===");
    println!("Metal device: {:?}", metal_device);

    Ok(())
}

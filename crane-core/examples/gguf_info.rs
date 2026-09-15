//! Dump GGUF architecture/size metadata without any accelerator backend.
//! Run: cargo run -p crane-core --example gguf_info <path.gguf>
use anyhow::Context;
use candle_core::quantized::gguf_file::Content;

fn main() -> Result<(), anyhow::Error> {
    let path = std::env::args().nth(1).context("usage: gguf_info <path.gguf>")?;
    let mut file = std::fs::File::open(&path)?;
    let ct = Content::read(&mut file).context("read gguf")?;

    let arch = ct
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .unwrap_or_else(|| "?".into());
    let key = |k: &str| format!("{arch}.{k}");
    let u32_or = |k: &str, d: u32| ct.metadata.get(&key(k)).and_then(|v| v.to_u32().ok()).unwrap_or(d);

    println!("path:      {}", path);
    println!("architecture: {}", arch);
    println!("tensors:   {}", ct.tensor_infos.len());
    println!(
        "block_count (layers): {}",
        u32_or("block_count", 0)
    );
    println!(
        "attention.head_count_kv: {}",
        u32_or("attention.head_count_kv", 0)
    );
    println!("num_experts:            {}", u32_or("expert_count", 0));
    println!(
        "expert_used_count:       {}",
        u32_or("expert_used_count", 0)
    );
    println!(
        "total_params (metadata): {}",
        ct.metadata.get("general.total_parameters").and_then(|v| v.to_u64())
    );
    // Estimate params from tensor shapes for the embedding + blocks.
    let emb = ct
        .tensor_infos
        .get("token_embd.weight")
        .map(|i| i.shape.dims()[0]);
    if let Some(v) = emb {
        println!("token_embd vocab: {}", v);
    }
    Ok(())
}

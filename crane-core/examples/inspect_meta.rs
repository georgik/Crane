use candle_core::quantized::gguf_file::{self, Value};
use std::fs::File;

fn main() -> Result<(), candle_core::Error> {
    let path = "/home/georgik/.lmstudio/models/ornith-ai/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf";
    let file = &mut File::open(path)?;
    let content = gguf_file::Content::read(file)?;
    for (k, v) in &content.metadata {
        let kl = k.to_lowercase();
        if kl.contains("expert") || kl.contains("top_k") || kl.contains("hidden_size")
            || kl.contains("intermediate") || kl.contains("num_attention_heads")
            || kl.contains("max_position") || kl.contains("act")
        {
            let s = match v {
                Value::U8(x) => x.to_string(),
                Value::I8(x) => x.to_string(),
                Value::U16(x) => x.to_string(),
                Value::I16(x) => x.to_string(),
                Value::U32(x) => x.to_string(),
                Value::I32(x) => x.to_string(),
                Value::U64(x) => x.to_string(),
                Value::I64(x) => x.to_string(),
                Value::F32(x) => x.to_string(),
                Value::F64(x) => x.to_string(),
                Value::Bool(x) => x.to_string(),
                Value::String(x) => x.clone(),
                _ => "array".into(),
            };
            println!("{} = {}", k, s);
        }
    }
    Ok(())
}

use candle_core::quantized::gguf_file;
use std::fs::File;

fn main() -> Result<(), candle_core::Error> {
    let path = "/home/georgik/.lmstudio/models/ornith-ai/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf";
    let file = &mut File::open(path)?;
    let content = gguf_file::Content::read(file)?;
    for (name, info) in content.tensor_infos.iter() {
        let n = name.to_lowercase();
        if n.contains("gate") && (n.contains("inp") || n.contains("exp")) {
            println!("{} :: dims={:?} :: dtype={:?}", name, info.shape.dims(), info.ggml_dtype);
        }
    }
    Ok(())
}

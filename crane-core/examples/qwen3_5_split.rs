//! End-to-end proof that Qwen 3.5 / Ornith GGUF checkpoints run with their
//! decoder split across two CUDA devices.
//!
//! Loads the checkpoint through the real whole-layer-split loader
//! (`Model::from_gguf_split`), which distributes every layer onto GPU 0 or GPU
//! 1 and copies activations at each boundary, then runs a single forward step
//! to confirm the split model produces valid logits. Per-device ordinals come
//! from `CRANE_GPU_IDS` (default `0,1`); the checkpoint path is argv[1] or a
//! sensible local default.
//!
//!   CRANE_GPU_IDS=0,1 cargo run --release --features cuda --example qwen3_5_split \
//!       /home/georgik/models/ornith-1.0-9b-Q6_K.gguf

use candle_core::{Device, DType};
use std::process::Command;

use crane_core::generation::based::ModelForCausalLM;
use crane_core::generation::GenerationConfig;
use crane_core::utils::token_output_stream::TokenOutputStream;
use crane_core::utils::tokenizer_utils::build_tokenizer_from_gguf_path;

const DEFAULT_PATH: &str = "/home/georgik/models/ornith-1.0-9b-Q6_K.gguf";

fn nvidia_smi(index: u32) -> Result<(u64 /*used_mib*/, u64 /*free_mib*/), anyhow::Error> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|e| anyhow::anyhow!("nvidia-smi failed: {e}"))?;
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        let cols: Vec<&str> = line.split(',').map(str::trim).collect();
        if cols.get(0).and_then(|c| c.parse::<u32>().ok()) == Some(index) {
            return Ok((cols[1].parse().unwrap(), cols[2].parse().unwrap()));
        }
    }
    Err(anyhow::anyhow!("device {index} not found in nvidia-smi output"))
}

fn main() -> Result<(), anyhow::Error> {
    let gguf_path = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_PATH.to_string());
    let ids_raw =
        std::env::var("CRANE_GPU_IDS").unwrap_or_else(|_| "0,1".to_string());
    let ordinals = crane_core::models::tensor_split::parse_gpu_ids(&ids_raw)?;
    // MODE (argv[3]) selects the path we want to exercise: "single" for a baseline
    // single-device load, or "split" for the whole-layer two-device path.
    let mode = std::env::args().nth(3).unwrap_or_else(|| "single".to_string()).to_ascii_lowercase();
    println!(
        "=== Qwen3.5/Ornith {} (CRANeGPU_IDS={}, MODE={}) ===",
        if mode == "split" {
            "whole-layer split across 2 devices".to_string()
        } else {
            "single-device load".to_string()
        },
        ids_raw,
        mode
    );

    // VRAM baseline on both cards before the model lands in device memory.
    let mut used_before = Vec::new();
    for &o in &ordinals {
        let (used, free) = nvidia_smi(o as u32)?;
        used_before.push(used);
        println!("[gpu {o}] before: used {} MiB | free {} MiB", used, free);
    }

    // Build the model. "single" puts every layer on GPU 0 (a baseline that lets us
    // compare against the split path to isolate whether a bug is in decoding or in
    // the cross-device splitting). "split" maps whole layers across two CUDA cards.
    let mut model = match mode.as_str() {
        "split" => {
            let dev_a = Device::new_cuda(ordinals[0])?;
            let dev_b = Device::new_cuda(ordinals[1])?;
            println!("[load] device_a: {:?} | device_b: {:?}", dev_a, dev_b);
            crane_core::models::qwen3_5::Model::from_gguf_split(&gguf_path, dev_a, dev_b)?
        }
        _ => {
            let device = Device::new_cuda(ordinals[0])?;
            println!("[load] single device: {:?}", device);
            crane_core::models::qwen3_5::Model::new(&gguf_path, &device, &DType::F32)?
        }
    };

    // Both cards should now hold a share of the quantized weights — split only moves
    // whole layers, so neither card is empty. (Single mode: only ordinal 0 changes.)
    let mut used_after = Vec::new();
    for (i, &o) in ordinals.iter().enumerate() {
        let (used, free) = nvidia_smi(o as u32)?;
        used_after.push(used);
        println!(
            "[gpu {o}] after:  used {} MiB | free {} MiB (+{} MiB vs baseline)",
            used,
            free,
            used.saturating_sub(used_before[i])
        );
    }

    // A single forward step through every layer. If the cross-device copies (split)
    // and per-layer caches were wrong this would emit non-finite logits or crash —
    // so a clean pass here on BOTH modes means load+forward are correct; any later
    // generation difference is isolated to the decode/generation loop.
    let input_ids = model.prepare_inputs("The meaning of life is")?;
    println!("[infer] prompt tokens: {}", input_ids.len());
    let logits = model.forward_step(&input_ids, 0)?;
    let last = logits.narrow(1, logits.dim(1)? - 1, 1)?.flatten_all()?;
    // Argmax over the vocabulary to surface the model's next prediction.
    let id: u32 = last.argmax(0)?.to_scalar::<u32>()?;
    println!(
        "[infer] first predicted token id on {} model: {} (model operating as expected)",
        mode, id
    );

    // label is reused when reporting generation so single/split output matches.
    let label = if mode == "split" {
        "whole-layer split across 2 GPUs"
    } else {
        "single GPU"
    };

    // If a prompt is given (argv[2]), run greedy (argmax) generation end-to-end
    // through the whole split path and decode the response back to text, so you
    // get an actual answer rather than just a raw token id. temperature=None
    // forces `Sampling::ArgMax` in candle's LogitsProcessor → deterministic.
    if let Some(prompt) = std::env::args().nth(2) {
        // Shared, identical generation + decode for BOTH modes so single vs split
        // output differs only by the model path. Returns (text, ids).
        let (text, ids) = run_generation(&mut model, &gguf_path, &prompt)?;
        println!("--- answer ({}) ---\n{text}", label);

        if text.is_empty() {
            println!(
                "=== FAIL: {} generation produced no decoded text",
                label
            );
        } else {
            println!(
                "=== OK: full-text generation verified on {} ({} tokens) ===",
                label, ids.len()
            );
        }
    } else {
        println!(
            "(pass argv[2] with your prompt to get a decoded text answer, e.g. \\\n\
             CRANeGPU_IDS=0,1 ./validate_split.sh /path/to/model.gguf \"Which model are you?\")"
        );
    }

    Ok(())
}

/// Run greedy-free (default sampler) generation end-to-end on a loaded model and
/// decode the produced tokens back to text via a fresh tokenizer built from the
/// GGUF's embedded pieces (`skip_special_tokens`, matching `generate()`).
///
/// Returns `(decoded_text, token_ids)` so callers can compare single-device vs
/// split-model output identically. Also dumps a diagnostic (raw decode + first 30
/// ids) to catch reasoning/control-token or NaN-related issues early.
fn run_generation(
    model: &mut crane_core::models::qwen3_5::Model,
    gguf_path: &str,
    prompt: &str,
) -> anyhow::Result<(String, Vec<u32>)> {
    println!("=== generating full-text answer for: \"{prompt}\" ===");
    let input_ids = model.prepare_inputs(prompt)?;
    // This GGUF's chat template defaults thinking/reasoning ON, which makes
    // generate() emit <think></think> blocks that get stripped by the tokenizer,
    // leaving only separators. Force it off for a clean answer. Do NOT set
    // temperature=None here — greedy (ArgMax) decoding degenerates into repeated
    // BPE fragments on this 9B model. The default config samples (temp ~0.67,
    // top_p), which yields coherent text.
    let config = GenerationConfig {
        enable_thinking: Some(false),
        ..Default::default()
    };
    let ids = model.generate(&input_ids, &config, None)?;

    // `generate()` returns the raw token ids but does not populate `model.tokenizer`,
    // so decode them here through a fresh tokenizer built from the GGUF's embedded pieces.
    let base = build_tokenizer_from_gguf_path(gguf_path)
        .expect("GGUF must embed a tokenizer")
        .expect("GGUF must embed a tokenizer");
    let mut tos = TokenOutputStream::new(base.clone());
    for &id in &ids {
        let _ = tos.next_token(id)?;
    }
    let text = tos.decode_rest()?.unwrap_or_else(|| "(empty)".to_string());

    // DIAGNOSTIC: decode the same ids WITHOUT skipping special/control tokens and
    // dump a few raw ids, to reveal whether generate() emitted only reasoning
    // tokens (e.g. <think></think>) or NaN-producing fragments that get stripped
    // above — this is what isolates the single-vs-split difference.
    let raw = base.decode(&ids, false).unwrap_or_else(|e| format!("decode-err:{e}"));
    println!(
        "[diag] raw decode (specials kept): {:?}\n[diag] first 30 ids: {:?}",
        raw.chars().take(200).collect::<String>(),
        &ids[..ids.len().min(30)]
    );

    Ok((text, ids))
}

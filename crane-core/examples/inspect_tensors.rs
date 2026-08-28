//! Categorize every tensor in a GGUF checkpoint by its semantic group.
//!
//! Prints how many tensors fall into each group and, per decoder block, which
//! kinds of tensors it owns so a reader can see at a glance whether a layer is a
//! plain dense MLP (gate/up/down with no expert suffix) or an routed-MoE block
//! (stacked `_exps`, shared `_shexp`, routing `inp`). Default target is the
//! Ornith 35B-A3B checkpoint; pass another path as argv[1] to inspect it.
use candle_core::quantized::gguf_file::{self, Value};
use std::collections::HashMap;
use std::fs::File;

fn meta_str(v: &Value) -> String {
    match v {
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
    }
}

/// Semantic group a tensor belongs to.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Embedding,
    LmHead,
    Norm,
    Attention,
    DenseFfn,
    ExpertRouted,
    ExpertShared,
    RoutingGate,
    GdnState,
    Misc,
}

impl Kind {
    fn label(self) -> &'static str {
        match self {
            Kind::Embedding => "embedding",
            Kind::LmHead => "lm_head",
            Kind::Norm => "norm",
            Kind::Attention => "attn",
            Kind::DenseFfn => "dense_ffn",
            Kind::ExpertRouted => "expert_routed",
            Kind::ExpertShared => "expert_shared",
            Kind::RoutingGate => "routing_gate",
            Kind::GdnState => "gdn_state",
            Kind::Misc => "misc",
        }
    }

    fn classify(name: &str) -> Kind {
        let n = name.to_lowercase();
        if n.contains("token_embd") || (n.contains("embedding") && n.contains(".weight")) {
            return Kind::Embedding;
        }
        if n.contains("lm_head") || (n == "output.weight" && !n.contains("norm")) {
            return Kind::LmHead;
        }
        if n.contains("norm") {
            return Kind::Norm;
        }
        if n.contains("ssm") || n.contains("gdn") || n.contains("delta") || n.contains("state_size")
            || n.contains("time_step") || n.contains("conv_kernel") || n.contains("group_count")
        {
            return Kind::GdnState;
        }
        if n.starts_with("blk.") && (n.contains(".attn_") || n.contains("proj_q") || n.contains("proj_kv"))
        {
            return Kind::Attention;
        }
        if n.starts_with("blk.") && n.contains("ffn") {
            if n.contains("_exps") {
                return Kind::ExpertRouted;
            }
            if n.contains("_shexp") {
                return Kind::ExpertShared;
            }
            if n.contains("gate_inp") || n.contains("inp_shexp") {
                return Kind::RoutingGate;
            }
            return Kind::DenseFfn;
        }
        Kind::Misc
    }
}

fn main() -> Result<(), candle_core::Error> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/georgik/.lmstudio/models/ornith-ai/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf".into()
    });
    let file = &mut File::open(&path)?;
    let content = gguf_file::Content::read(file)?;

    if let Some(v) = content.metadata.get("general.architecture") {
        println!("architecture: {}", meta_str(v));
    }
    if let Some(v) = content.metadata.get("block_count") {
        if let Value::U64(n) = v {
            println!("num_layers: {}", n);
        }
    }

    let mut tally: HashMap<String, u32> = HashMap::new();
    let mut block_kinds: HashMap<usize, Vec<Kind>> = HashMap::new();
    for name in content.tensor_infos.keys() {
        let kind = Kind::classify(name);
        *tally.entry(kind.label().to_string()).or_insert(0) += 1;

        if let Some(rest) = name.strip_prefix("blk.") {
            if let Ok(idx) = rest.split('.').next().unwrap_or("0").parse::<usize>() {
                block_kinds.entry(idx).or_default().push(kind);
            }
        }
    }

    println!("=== tensor categories for {} ===", path);
    let mut labels: Vec<_> = tally.keys().cloned().collect();
    labels.sort_by(|a, b| tally[b].cmp(&tally[a]).then(a.cmp(b)));
    println!("{:<14} count", "group");
    for l in &labels {
        println!("{:<14} {}", l, tally[l]);
    }

    let mut blocks: Vec<_> = block_kinds.keys().copied().collect();
    blocks.sort_unstable();
    println!("=== per-block tensor kinds ===");
    println!("{:>3}  groups", "blk");
    for i in &blocks {
        let mut present: Vec<&str> = block_kinds[i].iter().map(|k| k.label()).collect();
        present.sort_unstable();
        present.dedup();
        println!("{:>3}  {}", i, present.join(","));
    }

    Ok(())
}

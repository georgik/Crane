//! Mixture-of-experts (MoE) expert block for Ornith / Qwen3.5-MoE GGUF
//! checkpoints (`general.architecture = qwen35moe`).
//!
//! Mirrors HF's `Qwen3NextSparseMoeBlock`:
//!
//! ```text
//! final = Σ_t w_t · expert_{e_t}(x) + sigmoid(shared_expert_gate(x)) · shared(x)
//! ```
//!
//! The top-k expert set comes from a softmax over the router logits; the top-k
//! weights are renormalized to sum to one (`norm_topk_prob`, matching vLLM's
//! `renormalize` default).
//!
//! # GGUF layout (llama.cpp `qwen35moe`, as shipped for Ornith-1.5-35B-A3B)
//!
//! | tensor | candle shape | content |
//! |---|---|---|
//! | `blk.N.ffn_gate_inp.weight` | `[E, H]` (F32) | router |
//! | `blk.N.ffn_gate_exps.weight` | `[E, ff, H]` | stacked gate halves, quantized |
//! | `blk.N.ffn_up_exps.weight` | `[E, ff, H]` | stacked up halves, quantized |
//! | `blk.N.ffn_down_exps.weight` | `[E, H, ff]` | stacked down, quantized |
//! | `blk.N.ffn_gate_shexp.weight` | `[ff, H]` | shared-expert gate |
//! | `blk.N.ffn_up_shexp.weight` | `[ff, H]` | shared-expert up |
//! | `blk.N.ffn_down_shexp.weight` | `[H, ff]` | shared-expert down |
//! | `blk.N.ffn_gate_inp_shexp.weight` | `[1, H]` (F32) | shared-expert sigmoid gate |
//!
//! The stacked experts stay quantized in GPU memory: dequantizing every expert
//! of a 35B-A3B `Q4_K_M` checkpoint would need ~66 GiB in BF16 against ~19 GiB
//! quantized. The per-token expert math runs through candle's CUDA
//! `QTensor::indexed_moe_forward` kernel, which computes only the selected
//! experts' rows in a single launch (the input rows are quantized to q8_1
//! inside the kernel and the output is f32).

use std::io::{Read, Seek};
use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Linear, ops};

use crate::models::hunyuan_dense::modeling::Gguf;
use crate::models::qwen3_5::config::TextConfig;
use crate::ops::linear::LinearLayer;

/// GGUF linear that keeps quantized weights as a `QMatMul` and dequantizes
/// plain f32/f16/bf16 tensors to `dense_dtype`. candle's CUDA quantized-matmul
/// path has no kernel for unquantized weights stored as `QTensor`s, so the F32
/// router / shared-expert-gate tensors land as standard `Linear`s instead.
fn linear_of<R: Read + Seek>(
    gg: &mut Gguf<R>,
    name: &str,
    dev: &Device,
    dense_dtype: DType,
) -> Result<LinearLayer> {
    let qt = gg.tensor_on(name, dev)?;
    match qt.dtype() {
        GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => {
            // `ffn_gate_inp_shexp` is stored 1-D `[H]`; `Linear` needs a
            // 2-D `[out, in]` weight, so promote row vectors.
            let mut weight = qt.dequantize(dev)?.to_dtype(dense_dtype)?;
            if weight.rank() == 1 {
                weight = weight.unsqueeze(0)?;
            }
            Ok(LinearLayer::Standard(Linear::new(weight, None)))
        },
        _ => Ok(LinearLayer::Quantized(QMatMul::from_arc(Arc::new(qt))?)),
    }
}

/// Shared SwiGLU expert (the dense MoE backbone), gated by a tiny sigmoid.
struct SharedExpert {
    gate: LinearLayer, // [H -> shared_intermediate]
    up: LinearLayer,   // [H -> shared_intermediate]
    down: LinearLayer, // [shared_intermediate -> H]
}

impl SharedExpert {
    fn from_gguf<R: Read + Seek>(
        gg: &mut Gguf<R>,
        layer_idx: usize,
        dev: &Device,
        dense_dtype: DType,
    ) -> Result<Self> {
        let p = format!("blk.{layer_idx}");
        Ok(Self {
            gate: linear_of(
                gg,
                &format!("{p}.ffn_gate_shexp.weight"),
                dev,
                dense_dtype,
            )?,
            up: linear_of(gg, &format!("{p}.ffn_up_shexp.weight"), dev, dense_dtype)?,
            down: linear_of(
                gg,
                &format!("{p}.ffn_down_shexp.weight"),
                dev,
                dense_dtype,
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let act = ops::silu(&self.gate.forward(x)?)?;
        let h = act.broadcast_mul(&self.up.forward(x)?)?;
        self.down.forward(&h)
    }
}

/// MoE expert block for GGUF checkpoints.
pub struct MoExperts {
    /// Router (`ffn_gate_inp`), `[E, H]`, evaluated in f32.
    gate: LinearLayer,
    /// Stacked gate experts `[E, ff, H]`, quantized on the layer device.
    gate_exps: Arc<QTensor>,
    /// Stacked up experts `[E, ff, H]`, quantized on the layer device.
    up_exps: Arc<QTensor>,
    /// Stacked down experts `[E, H, ff]`, quantized on the layer device.
    down_exps: Arc<QTensor>,
    /// Shared SwiGLU expert, gated by `sigmoid(shared_gate(x))`.
    shared: SharedExpert,
    /// Shared-expert sigmoid gate (`ffn_gate_inp_shexp`), `[1, H]`, in f32.
    shared_gate: LinearLayer,
    num_experts: usize,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
}

impl MoExperts {
    /// Build the MoE block from GGUF quantized weights, landing every tensor
    /// on `dev` (the layer's device in the multi-GPU split).
    ///
    /// # Errors
    ///
    /// Returns an error if a required weight tensor is missing or has a shape
    /// inconsistent with the config.
    pub fn from_gguf_on<R: Read + Seek>(
        gg: &mut Gguf<R>,
        cfg: &TextConfig,
        layer_idx: usize,
        dev: &Device,
        dense_dtype: DType,
    ) -> Result<Self> {
        let p = format!("blk.{layer_idx}");
        let num_experts = cfg.num_experts.max(1);
        let moe_i = cfg.moe_intermediate_size.max(1);
        let h = cfg.hidden_size;

        let gate_exps = gg.tensor_on(&format!("{p}.ffn_gate_exps.weight"), dev)?;
        let up_exps = gg.tensor_on(&format!("{p}.ffn_up_exps.weight"), dev)?;
        let down_exps = gg.tensor_on(&format!("{p}.ffn_down_exps.weight"), dev)?;
        for (name, t, want) in [
            (
                "ffn_gate_exps",
                &gate_exps,
                [num_experts, moe_i, h],
            ),
            ("ffn_up_exps", &up_exps, [num_experts, moe_i, h]),
            ("ffn_down_exps", &down_exps, [num_experts, h, moe_i]),
        ] {
            if t.shape().dims() != want {
                candle_core::bail!(
                    "[qwen3_5] blk.{layer_idx}.{name} shape {:?} does not match config {want:?}",
                    t.shape().dims()
                );
            }
        }

        Ok(Self {
            gate: linear_of(gg, &format!("{p}.ffn_gate_inp.weight"), dev, DType::F32)?,
            gate_exps: Arc::new(gate_exps),
            up_exps: Arc::new(up_exps),
            down_exps: Arc::new(down_exps),
            shared: SharedExpert::from_gguf(gg, layer_idx, dev, dense_dtype)?,
            shared_gate: linear_of(
                gg,
                &format!("{p}.ffn_gate_inp_shexp.weight"),
                dev,
                DType::F32,
            )?,
            num_experts,
            num_experts_per_tok: cfg.num_experts_per_tok.max(1),
            norm_topk_prob: cfg.norm_topk_prob,
        })
    }

    /// # Errors
    ///
    /// Returns an error if any tensor operation fails.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dims = x.dims().to_vec();
        let (batch, hidden) = match x_dims.as_slice() {
            [b, h] => (*b, *h),
            [b, s, h] => (b * s, *h),
            d => candle_core::bail!("[qwen3_5] MoE input must be 2-D/3-D, got {d:?}"),
        };
        let flat = x.reshape((batch, hidden))?;

        // Dense shared expert over every token, gated by a sigmoid.
        let shared_out = self.shared.forward(&flat)?;
        let shared_sig = ops::sigmoid(&self.shared_gate.forward(&flat.to_dtype(DType::F32)?)?)?;
        let shared_gated =
            shared_out.broadcast_mul(&shared_sig.to_dtype(shared_out.dtype())?)?;

        // Router: softmax, then top-k selection on the host (E = 256, so the
        // round-trip is small next to the expert matmuls and keeps the
        // selection exact and dtype-independent).
        let logits = self.gate.forward(&flat.to_dtype(DType::F32)?)?; // [N, E] f32
        let probs = ops::softmax_last_dim(&logits)?;
        let k = self.num_experts_per_tok.min(self.num_experts);
        let (ids, weights) = topk_ids_weights(&probs, k, self.norm_topk_prob)?;
        let ids = Tensor::from_vec(ids, (batch, k), &Device::Cpu)?.to_device(x.device())?;
        let route_w =
            Tensor::from_vec(weights, (batch, k, 1), &Device::Cpu)?.to_device(x.device())?;

        // All top-k expert rows per token in one launch per stacked tensor:
        // with `input_dim1 == 1` the kernel broadcasts the token row across
        // its k expert slots. `[N, 1, H]` + ids `[N, k]` -> `[N, k, ff]`.
        let x_f32 = flat.to_dtype(DType::F32)?.reshape((batch, 1, hidden))?;
        let g = self.gate_exps.indexed_moe_forward(&x_f32, &ids)?; // [N, k, ff]
        let u = self.up_exps.indexed_moe_forward(&x_f32, &ids)?; // [N, k, ff]
        let h = ops::silu(&g)?.broadcast_mul(&u)?; // [N, k, ff]
        // Down projection consumes the per-slot rows (`input_dim1 == k`).
        let expert_out = self.down_exps.indexed_moe_forward(&h, &ids)?; // [N, k, H]

        // Weighted sum over the k slots (dim 1, NOT the hidden dim), plus the
        // gated shared expert.
        let routed = expert_out.broadcast_mul(&route_w)?.sum(candle_core::D::Minus(2))?; // [N, H] f32
        let out = routed.add(&shared_gated.to_dtype(DType::F32)?)?;
        out.to_dtype(x.dtype())?.reshape(x_dims)
    }
}

/// Host-side softmax top-k: per row the k largest probs as
/// `(expert index, weight)` pairs, optionally renormalizing the weights to sum
/// to one (`norm_topk_prob`, matching vLLM's `renormalize`).
///
/// # Errors
///
/// Returns an error if the probs cannot be read back to the host.
fn topk_ids_weights(
    probs: &Tensor,
    k: usize,
    norm_topk_prob: bool,
) -> Result<(Vec<u32>, Vec<f32>)> {
    let rows = probs.to_vec2::<f32>()?;
    let mut ids = Vec::with_capacity(rows.len() * k);
    let mut weights = Vec::with_capacity(rows.len() * k);
    for row in &rows {
        let mut sel: Vec<(u32, f32)> =
            row.iter().enumerate().map(|(i, &p)| (i as u32, p)).collect();
        sel.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sel.truncate(k);
        // The top-1 prob is >= 1/E, so the denominator is strictly positive.
        let denom: f32 = sel.iter().map(|(_, p)| p).sum();
        for (expert, p) in sel {
            ids.push(expert);
            weights.push(if norm_topk_prob { p / denom } else { p });
        }
    }
    Ok((ids, weights))
}

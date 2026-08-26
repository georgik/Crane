//! Multi-GPU layer splitting ("tensor parallelism").
//!
//! Crane loads every transformer layer into a single `Device`. For models that
//! are too large to fit on one GPU, layers can be partitioned across two CUDA
//! devices so the combined VRAM is used. Weights are *not* sharded inside a
//! layer (only whole layers move), which keeps inference bit-exact versus the
//! single-device path and requires no new kernels — activations are simply
//! copied to a layer's device at each boundary, exactly as llama.cpp's
//! `--tensor-split` moves intermediate results across devices.
//!
//! Configuration is read from environment variables so operators don't have to
//! repeat it on every CLI invocation (matching the existing `CRANE_ISQ`,
//! `CRANE_PROF` … convention):
//!
//! * `CRANE_GPU_IDS="0,1"` — CUDA ordinals for GPU 0 and GPU 1. Defaults to
//!   `[0, 1]`. Any two distinct valid ordinals are accepted.
//! * `CRANE_GPU_SPLIT="g0:g1"` — fraction of layers kept on the first device
//!   (`g0`), the rest going on the second. Example `0.6:0.4`. Defaults to a
//!   balanced split.
//!
//! Both flags are optional; either may be overridden by the matching CLI flag
//! where one exists (e.g. `--tensor-split`). The feature only activates when a
//! CUDA backend is compiled in and at least two devices are reported present.

use anyhow::{bail, Result};
use candle_core::Device;

/// Whether two devices point at the same hardware. candle's `Device` does not
/// implement `PartialEq`, so compare each backend's device id within matching
/// variants; CPU is always equal to CPU. This matches how the rest of the codebase
/// decides whether a cross-device copy is needed.
#[must_use]
pub fn devices_equal(a: &Device, b: &Device) -> bool {
    match (a, b) {
        (Device::Cuda(x), Device::Cuda(y)) => x.id() == y.id(),
        _ => a.is_cpu() && b.is_cpu(),
    }
}

/// Default per-device split fraction when `CRANE_GPU_SPLIT` is unset: half the
/// layers on each GPU.
pub const DEFAULT_SPLIT_RATIO: f32 = 0.5;

/// Parse a `g0:g1` device-id pair into CUDA ordinals, validating that exactly
/// two *distinct* GPUs are named.
pub fn parse_gpu_ids(raw: &str) -> Result<[usize; 2]> {
    let parts: Vec<&str> = raw.split(',').map(str::trim).collect();
    if parts.len() != 2 {
        bail!(
            "CRANE_GPU_IDS must name exactly two GPUs (e.g. \"0,1\"), got \"{raw}\""
        );
    }
    let a: usize = parts[0]
        .parse()
        .map_err(|_| anyhow::anyhow!("GPU id \"{}\" is not a number", parts[0]))?;
    let b: usize = parts[1]
        .parse()
        .map_err(|_| anyhow::anyhow!("GPU id \"{}\" is not a number", parts[1]))?;
    if a == b {
        bail!("CRANE_GPU_IDS names the same GPU twice (\"{raw}\")");
    }
    Ok([a, b])
}

/// Parse a `g0:g1` split ratio and return the fraction of layers to keep on the
/// first device. The two numbers must be positive and sum to 1.0 (within a
/// small tolerance).
pub fn parse_split_ratio(raw: &str) -> Result<f32> {
    let parts: Vec<&str> = raw.split(':').map(str::trim).collect();
    if parts.len() != 2 {
        bail!(
            "CRANE_GPU_SPLIT must be \"g0:g1\" (e.g. \"0.6:0.4\"), got \"{raw}\""
        );
    }
    let g0: f32 = parts[0]
        .parse()
        .map_err(|_| anyhow::anyhow!("split value \"{}\" is not a number", parts[0]))?;
    let g1: f32 = parts[1]
        .parse()
        .map_err(|_| anyhow::anyhow!("split value \"{}\" is not a number", parts[1]))?;
    if !(g0 > 0.0 && g1 > 0.0) {
        bail!(
            "CRANE_GPU_SPLIT values must both be positive (got g0={g0}, g1={g1})"
        );
    }
    // Normalize by the sum so either fractional ("0.6:0.4") or relative-weight
    // ("1:1") notation yields a ratio in the open interval (0, 1).
    let sum = g0 + g1;
    if !(sum > 0.0) {
        bail!("CRANE_GPU_SPLIT values must be positive (got g0={g0}, g1={g1})");
    }
    Ok(g0 / sum)
}

/// Given the layer count and the fraction of layers kept on the first device,
/// return the cut index: layers `[0, cut)` run on GPU 0 and `[cut, num_layers)`
/// on GPU 1. The result is clamped to a strict interior split so at least one
/// layer lives on each GPU (a degenerate all-on-one-GPU split is pointless).
#[must_use]
pub fn split_point(num_layers: usize, gpu0_frac: f32) -> usize {
    if num_layers <= 2 {
        return 1;
    }
    let raw = (num_layers as f32 * gpu0_frac).round();
    // Keep at least one layer on each side.
    let mut cut = raw as usize;
    if cut < 1 {
        cut = 1;
    }
    if cut >= num_layers {
        cut = num_layers - 1;
    }
    cut
}

/// Build the per-layer device assignment for a split model: one `Device` per
/// transformer layer, indexed by layer number. `device_a` holds layers
/// `[0, cut)` and `device_b` holds `[cut, num_layers)`.
#[must_use]
pub fn per_layer_devices(device_a: &Device, device_b: &Device, num_layers: usize, cut: usize) -> Vec<Device> {
    (0..num_layers)
        .map(|i| if i < cut { device_a.clone() } else { device_b.clone() })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_gpu_ids_requires_two_distinct() {
        assert_eq!(parse_gpu_ids("0,1").unwrap(), [0, 1]);
        assert_eq!(parse_gpu_ids("2 , 5").unwrap(), [2, 5]);
        assert!(parse_gpu_ids("0").is_err());
        assert!(parse_gpu_ids("0,0").is_err());
        assert!(parse_gpu_ids("a,b").is_err());
    }

    #[test]
    fn parse_split_ratio_is_balanced_or_custom() {
        assert_eq!(split_point(2, 0.6), 1); // both GPUs keep at least one layer
        assert!((parse_split_ratio("0.6:0.4").unwrap() - 0.6).abs() < 1e-6);
        assert!((parse_split_ratio("1:1").unwrap() - 0.5).abs() < 1e-6); // equal weights normalize to 0.5
        assert!(parse_split_ratio("0.7:0.3").unwrap().abs() > 0.0);
        assert!(parse_split_ratio("0.5:0.5").is_ok());
        assert!((parse_split_ratio("0.6:0.5").unwrap() - (0.6 / 1.1)).abs() < 1e-6); // normalized, not rejected
        assert!(parse_split_ratio("1:0").is_err()); // non-positive
        assert!(parse_split_ratio("0.5").is_err()); // missing ':'
    }

    #[test]
    fn split_point_keeps_layers_on_both_sides() {
        assert_eq!(split_point(8, 0.5), 4);
        assert_eq!(split_point(8, 0.75), 6);
        assert_eq!(split_point(3, 0.5), 2);
        // Extreme ratios are clamped so nothing collapses onto one GPU.
        assert_eq!(split_point(10, 0.95), 9);
        assert_eq!(split_point(10, 0.02), 1);
    }
}

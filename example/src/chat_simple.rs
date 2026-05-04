//! Simple Chat Example
//!
//! This example shows how to create a basic chat application using the Crane SDK.

use crane::common::config::{CommonConfig, DataType, DeviceConfig};
use crane::llm::{GenerationConfig, LlmModelType};
use crane::prelude::*;
use clap::Parser;

/// Simple chat example with Qwen3 model
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the model directory
    #[arg(long, default_value = "checkpoints/Qwen3-1.7B")]
    model_path: String,

    /// Model type to use
    #[arg(long, default_value = "qwen3", value_parser = parse_model_type)]
    model_type: LlmModelType,

    /// Device to run inference on
    #[arg(long, default_value = "cpu", value_parser = parse_device)]
    device: DeviceConfig,

    /// Data type for computation
    #[arg(long, default_value = "f16", value_parser = parse_dtype)]
    dtype: DataType,

    /// Maximum number of tokens to generate
    #[arg(long, default_value_t = 100)]
    max_tokens: usize,

    /// Temperature for sampling (0.0 to 1.0)
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Number of conversation history turns to keep
    #[arg(long, default_value_t = 4)]
    history: usize,

    /// Enable streaming output
    #[arg(long, default_value_t = true)]
    streaming: bool,

    /// Message to send to the model
    #[arg(short, long, default_value = "Hello, introduce yourself briefly.")]
    message: String,
}

fn parse_model_type(s: &str) -> Result<LlmModelType, String> {
    match s.to_lowercase().as_str() {
        "qwen25" => Ok(LlmModelType::Qwen25),
        "qwen3" => Ok(LlmModelType::Qwen3),
        "qwen3vl" => Ok(LlmModelType::Qwen3VL),
        "deepseek" => Ok(LlmModelType::DeepSeek),
        "hunyuan" => Ok(LlmModelType::HunyuanDense),
        _ => Err(format!("Unknown model type: {}. Available: qwen25, qwen3, qwen3vl, deepseek, hunyuan", s)),
    }
}

fn parse_device(s: &str) -> Result<DeviceConfig, String> {
    match s.to_lowercase().as_str() {
        "cpu" => Ok(DeviceConfig::Cpu),
        "metal" => Ok(DeviceConfig::Metal),
        s if s.starts_with("cuda") => {
            let gpu_id = s.strip_prefix("cuda")
                .unwrap_or("0")
                .trim()
                .parse::<u32>()
                .map_err(|_| format!("Invalid CUDA device: {}. Use format: cuda:0", s))?;
            Ok(DeviceConfig::Cuda(gpu_id))
        }
        _ => Err(format!("Unknown device: {}. Available: cpu, metal, cuda:0", s)),
    }
}

fn parse_dtype(s: &str) -> Result<DataType, String> {
    match s.to_lowercase().as_str() {
        "f16" => Ok(DataType::F16),
        "f32" => Ok(DataType::F32),
        "bf16" => Ok(DataType::BF16),
        _ => Err(format!("Unknown dtype: {}. Available: f16, f32, bf16", s)),
    }
}

fn main() -> CraneResult<()> {
    let args = Args::parse();
    let model_path = args.model_path.clone();
    let model_type = args.model_type.clone();
    let device = args.device.clone();
    let dtype = args.dtype.clone();
    let message = args.message.clone();

    // Create a simple chat configuration from CLI arguments
    let config = ChatConfig {
        common: CommonConfig {
            model_path,
            model_type,
            device,
            dtype,
            max_memory: None,
        },
        generation: GenerationConfig {
            max_new_tokens: args.max_tokens,
            temperature: Some(args.temperature),
            ..Default::default()
        },
        max_history_turns: args.history,
        enable_streaming: args.streaming,
    };

    println!("🤖 Starting chat with {:?} model from: {}",
             args.model_type, args.model_path);
    println!("📝 Message: {}\n", args.message);

    // Create a new chat client
    let mut chat_client = ChatClient::new(config)?;

    // Send a simple message and get a response
    let response = chat_client.send_message(&message)?;
    println!("🤖 AI Response: {}\n", response);

    Ok(())
}
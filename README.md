# Crane 🦩

> Crane focusing on accelerate LLM inference speed with the power of kernels in candle framework, while reducing development overhead, make it portable and fast run model on both CPU and GPU.

![](data/aa.gif)


**Crane** - **C**andle-based **R**ust **A**ccelerated **N**eural **E**ngine
A high-performance inference framework leveraging Rust's Candle for maximum speed on CPU/GPU.

**Supported Models**:

- [x] Qwen3 (0.6B ~ 30B+) with production tool calling (Qwen3-1.7B validated)
- [x] Qwen 2.5 (0.5B ~ 72B) with function calling
- [x] Hunyuan Dense
- [x] Qwen3 VL (2B, 4B)
- [x] PaddleOCR VL 0.9B / 1.5
- [x] Moonshine ASR
- [x] Silero VAD
- [x] Qwen3-TTS (12Hz, 24kHz, 16-codebook RVQGAN + native Candle decoder, voice cloning)
- [ ] TTS: [Spark-TTS](https://github.com/SparkAudio/Spark-TTS) | [Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS) (WIP)

**Tool Calling Support**: Qwen3-1.7B validated for reliable tool calling. Models with 7B+ parameters generally provide more consistent tool recognition. Smaller models (0.5B-3B) may have inconsistent tool format adherence despite flexible parsing.


submit your models make other users use it easier!


**You can run Qwen3-VL 2B with fast speed in local, 50x faster than native PyTorch on M1/M2/M3.**

**Key Advantages**:

- **Fast Inference**: Outperforms native PyTorch with Candle's optimized kernels
- **Rust-Powered**: Eliminate C++ complexity while maintaining native performance
- **Apple Silicon Optimized**: GPU acceleration via Metal on macOS devices
- **Hardware Agnostic**: Unified codebase for CPU/CUDA/Metal execution
- **OpenAI Compatible API**: Supports OpenAI and SGLang interfaces including function calling
- **Tool Calling**: Multi-format detection, whitespace-aware parsing
- **Memory Safety**: Pre-flight memory checks prevent OOM crashes with device-aware resource management


**Crane maybe the fastest (both speed and develop speed) framework you can use to build your AI applications!**

Crane using candle as the only dependencies, inference with **fastest** speed cross CPUs and GPUs, while your code can be compiled into binary same as llama.cpp does but much more clean and simpler.

**Most important!!!**
*Crane is not a low-level SDK, you can call AI abilities out-of-box with ease*.

We include:
- Basic LLM chat
- VLM chat
- OCR with VLM
- Tool calling and function execution
- VLA (on the way)
- TTS
- ASR
- VAD
- .... (Any AI ability you want power with AI)


## Updates

- **2026.04.29**: Tool calling implementation - complete OpenAI-compatible function calling with multi-format token detection, whitespace handling, flexible argument parsing (string/object), test coverage (69 tests), Qwen3-1.7B validated
- **2026.04.29**: Memory safety system - pre-flight memory checks, device-aware concurrent scaling (Metal: 1-4, CPU: 1-4, CUDA: 4-16), F16 dtype on Metal by default (50% memory savings), sysinfo integration for accurate memory detection
- **2026.02.23**: Qwen3-TTS support added - full Talker + Code Predictor transformer in Candle, native speech-tokenizer decoder (ONNX fallback), voice cloning (Base model ICL), OpenAI `/v1/audio/speech` endpoint in crane-oai
- **2026.02.18**: Qwen3 & Hunyuan Dense inference optimization: pre-allocated KV cache, GQA 4D matmul, fused RoPE with cache pre-growth, GGUF quantization, batched decode, smart sampling fallback for large vocabularies
- **2026.01.30**: PaddleOCR-VL-1.5 supported now! model: https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5/
- **2025.03.21**: Qwen2.5 a more transformers liked Rust interface were supported, you now use Crane just like in your python
- **2025.03.19**: project initialized



## AI Abilities Use out-of-box

**1. OCR**

![](assets/image.webp)


**2. more to come**




## Why Choose Crane?

While traditional approaches face limitations:

- PyTorch's suboptimal inference performance
- llama.cpp's complex C++ codebase and model integration

Crane bridges the gap through:

1. **Candle Framework**: Combines Rust's efficiency with PyTorch-like ergonomics
2. **Cross-Platform Acceleration**: Metal GPU support achieves 3-5x speedup over CPU-only
3. **Simplified Deployment**: Add new models with <100 LOC in most cases

**Pro Tip**: For macOS developers, Crane delivers comparable performance to llama.cpp with significantly lower maintenance overhead. You can use it out of box directly without any GGUF conversion or something like install llama.cpp etc.

Speed up your LLM inference speed on M series Apple Silicon devices to 6x with almost simillar code in your python (No quantization needed!):

```rust

use clap::Parser;
use crane_core::{
    Msg,
    autotokenizer::AutoTokenizer,
    chat::Role,
    generation::{GenerationConfig, based::ModelForCausalLM, streamer::TextStreamer},
    models::{DType, Device, qwen25::Model as Qwen25Model},
};

#[derive(Parser, Debug)]
#[clap(about, version, author)]
struct Args {
    #[clap(short('m'), long, default_value = "checkpoints/Qwen2.5-0.5B-Instruct")]
    model_path: String,
}

fn main() {
    crane_core::utils::utils::print_candle_build_info();

    let args = Args::parse();
    let dtype = DType::F16;
    let device = Device::Cpu;

    let tokenizer = AutoTokenizer::from_pretrained(&args.model_path, None).unwrap();
    let mut model = Qwen25Model::new(&args.model_path, &device, &dtype).unwrap();

    let gen_config = GenerationConfig {
        max_new_tokens: 235,
        temperature: Some(0.67),
        top_p: Some(1.0),
        repetition_penalty: 1.1,
        repeat_last_n: 1,
        do_sample: false,
        pad_token_id: tokenizer.get_token("<|end_of_text|>"),
        eos_token_id: tokenizer.get_token("<|im_end|>"),
        report_speed: true,
    };

    let chats = [
        Msg!(Role::User, "hello"),
        Msg!(Role::Assistant, "Hi, how are you?"),
        Msg!(Role::User, "I am OK, tell me some truth about Yoga."),
    ];
    let prompt = tokenizer.apply_chat_template(&chats, true).unwrap();
    println!("prompt templated: {:?}\n", prompt);

    let input_ids = model.prepare_inputs(&prompt).unwrap();
    let _ = model.warmup();

    let mut streamer = TextStreamer {
        tokenizer: tokenizer.clone(),
        buffer: String::new(),
    };
    let output_ids = model
        .generate(&input_ids, &gen_config, Some(&mut streamer))
        .map_err(|e| format!("Generation failed: {}", e))
        .unwrap();

    let res = tokenizer.decode(&output_ids, false).unwrap();
    println!("Output: {}", res);
}

```

Above is all the codes you need to run end2end chat in Qwen2.5 in pure Rust, nothing overhead compare with llama.cpp.

Then, your LLM inference is 6X faster on mac without Quantization! Enabling Quantization could be even faster!

For cli chat, run:

```
# download models of Qwen2.5
mkdir -p checkpoints/
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir checkpoints/Qwen2.5-0.5B-Instruct
cargo run --bin qwenchat --release
```



## Usage

To use `crane`, here are some notes:

- `crane-core`: All models comes into core, this is a lib;
- `crane`: All Apps (runnable AI pipelines, such as Qwen2-Chat, Spark-TTS, Qwen2.5-VL etc), you can build your apps inside it, each app is a binary for demonstration purpose;
- `crane-oai`: OpenAI & SGLang compatible API server with continuous batching, see [crane-oai/README.md](crane-oai/README.md) for full documentation;

1. Make sure latest Rust were installed;
2. Build (choose based on your hardware):

   ```bash
   # CPU
   cargo build --release

   # CUDA (GPU)
   cargo build --release --features cuda
   ```

That's it!

### Testing and Development Tools

Crane includes a testing infrastructure (`xtask`) for validation and development:

```bash
# Build testing tools
cargo build -p xtask --release

# Test tool calling pipeline
cargo xtask test-tools [--model MODEL_NAME] [--timeout SECONDS]

# Test basic chat functionality
cargo xtask test-chat [--model MODEL_NAME] [--timeout SECONDS]

# Run unit tests for tool extraction
cargo test -p crane-oai --bin crane-oai 'test_parse_qwen'

# Run full test suite
cargo test --workspace
```

**Testing Features:**

- **Auto-detection**: Automatically discovers available models from `/v1/models` endpoint
- **Configurable timeouts**: CLI arguments and environment variable (`XTASK_TIMEOUT`) support
- **Structured logging**: Detailed request/response logging with timestamps
- **Tool validation**: Comprehensive tool calling format detection and parsing validation
- **Performance monitoring**: Request timing and throughput measurements

The testing framework ensures reliable tool calling behavior across different model families and formats.

### OpenAI API Server

Start a server compatible with OpenAI SDK and SGLang client:

```bash
# Build
# CPU
cargo build -p crane-oai --release
# CUDA
cargo build -p crane-oai --release --features cuda

# Start (auto-detect model type and device)
./target/release/crane-oai --model-path /path/to/Qwen2.5-7B-Instruct

# Or run directly
cargo run -p crane-oai --release -- --model-path /path/to/model --port 8000
```

Then use it with any OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

**Function Calling**

Crane supports OpenAI-compatible function calling with tool definition handling, format detection, and parsing:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="Qwen3-1.7B",
    messages=[{"role": "user", "content": "What's the current time?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current time",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }]
)

# Model returns tool_calls instead of content
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        print(f"Function: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
        # Execute function and send result back to model
```

**Tool Format Support**

Crane detects and parses tool calling formats with whitespace handling:

- **Qwen Special Tokens**: `<|tool_call|>...<|end_tool_call|>` (alternative format)
- **Qwen Tool Start**: `<|tool_start|>...<|tool_end|>` (primary format)
  - Handles whitespace variations: `<| tool_start|>`, `<|tool_start |>`, `<|tool_start\n|>`
  - Regex detection for malformed tokens from model output
- **Flexible Arguments**: Accepts both `"arguments": "{}"` (string) and `"arguments": {}` (object)
- **Conversational Context**: Extracts tool calls from model explanations and reasoning text

The parser handles whitespace in model-generated tokens that may include newlines or spaces within special token markers.

**Model Compatibility**

Tool calling requires models specifically trained for function use. Validated models:

- **Qwen3-1.7B**: Production validated with reliable tool calling
- **Qwen2.5-7B+**: Enhanced tool recognition and argument generation
- **Qwen3-4B+**: Modern tool calling format support

Models with 7B+ parameters generally provide more consistent tool recognition. Smaller models (0.5B-3B) may have inconsistent tool format adherence despite flexible parsing capabilities.

**Testing Infrastructure**

Crane includes testing infrastructure for tool calling validation:

```bash
# Test tool calling with auto-detected model
cargo xtask test-tools

# Test with specific model and timeout
cargo xtask test-tools Qwen2.5-7B-Instruct --timeout=180

# Test basic chat functionality
cargo xtask test-chat
```

The test suite includes 52+ unit tests covering tool format detection, JSON parsing, edge cases, and integration scenarios.

**Multi-turn Tool Conversations**

Crane supports multi-turn conversations with tool execution:

1. User provides tool definitions and query
2. Model generates structured tool_calls response
3. Client executes tools and returns results as tool messages
4. Model processes tool results and provides final answer
5. Continues normal conversation flow

**Logging and Debugging**

Comprehensive logging tracks the entire tool calling pipeline:

```bash
# Enable detailed logging
RUST_LOG=debug ./target/release/crane-oai --model-path /path/to/model

# Logs show: tool detection, format parsing, JSON extraction, API response structure
```

Supported endpoints:

| Family | Endpoint | Description |
|--------|----------|-------------|
| OpenAI | `POST /v1/completions` | Text completions |
| OpenAI | `POST /v1/chat/completions` | Chat completions with function calling (streaming & non-streaming) |
| OpenAI | `POST /v1/audio/speech` | Text-to-speech (Qwen3-TTS) |
| OpenAI | `GET /v1/models` | List models |
| OpenAI | `POST /v1/tokenize` | Tokenize text |
| OpenAI | `POST /v1/detokenize` | Detokenize tokens |
| SGLang | `POST /generate` | Native text generation |
| SGLang | `GET /model_info` | Model metadata |
| SGLang | `GET /server_info` | Server stats |
| SGLang | `GET /health_generate` | Deep health check |
| Mgmt   | `GET /health` | Health check |
| Mgmt   | `GET /v1/stats` | Engine statistics |

**Text-to-Speech (Qwen3-TTS)**: For TTS models, the server adds a `/v1/audio/speech` endpoint (OpenAI-compatible). Both **CustomVoice** (predefined speakers) and **Base** (voice cloning via reference audio) models are supported. `response_format` currently supports `wav` and `pcm` (other formats return `400`). See [crane-oai/README.md](crane-oai/README.md) for full TTS API documentation.

### TTS Examples

```bash
# CustomVoice — predefined speakers
cargo run --bin tts_custom_voice --release -- vendor/Qwen3-TTS-12Hz-0.6B-CustomVoice

# Voice Clone — clone speech from reference audio (Base model)
cargo run --bin tts_voice_clone --release -- vendor/Qwen3-TTS-12Hz-0.6B-Base

# Auto-detect model type
cargo run --bin tts_simple --release -- vendor/Qwen3-TTS-12Hz-0.6B-Base
```

All TTS examples save generated audio files to `data/audio/output`.

### TTS Audio Samples

- Base (voice clone): [vc1_base.wav](data/audio/output/vc1_base.wav), [vc2_base.wav](data/audio/output/vc2_base.wav)
- CustomVoice: [custom_voice_zh.wav](data/audio/output/custom_voice_zh.wav), [custom_voice_en.wav](data/audio/output/custom_voice_en.wav), [custom_voice_ja.wav](data/audio/output/custom_voice_ja.wav)

**Multimodal & Vision support**: For models like PaddleOCR-VL, the endpoints accept OpenAI's structured `messages.[]content.[{type: "image_url", image_url: {url: "..."}}]` payload or SGLang's `image_url` field. See [crane-oai/README.md](crane-oai/README.md) for full API documentation with request/response examples.

Now you can run LLM extremly fast (about 6x faster than vanilla transformers on M1)!

## Project Structure

```
Crane/
├── crane-core/          # Core library: model implementations, tokenizer, generation
│   └── src/models/      # Model architectures (Qwen 2.5, Qwen 3, Hunyuan, etc.)
├── crane/               # High-level SDK: Chat, Vision, Audio, Multimodal clients
├── crane-oai/           # OpenAI & SGLang compatible API server
│   └── src/
│       ├── engine/      # Continuous batching inference engine
│       ├── handlers/    # HTTP request handlers (OpenAI, SGLang, common)
│       ├── openai_api.rs # OpenAI request/response types with tool calling support
│       ├── sglang_api.rs # SGLang API types
│       └── main.rs      # CLI entry point & router
├── xtask/               # Testing infrastructure and development tools
│   └── src/main.rs      # Test clients for tool calling, chat, performance
├── example/             # Example binaries (chat, ASR, vision, OCR, TTS)
├── vendor/              # Vendored references (llama.cpp, sglang, vllm)
└── scripts/             # Utility scripts
```

### Tool Calling Implementation

Crane's tool calling system provides OpenAI-compatible function calling with error handling:

**Format Detection Pipeline**

The system automatically identifies the tool calling format from model-generated text:

1. **Fast Path Matching**: Direct string comparison for common formats
2. **Regex Fallback**: Handles whitespace variations in special tokens
3. **Format Validation**: Confirms both start and end tokens present
4. **Argument Parsing**: Flexible JSON extraction with type conversion

**Parsing Features**

- **Token Normalization**: Converts malformed tokens (`<|tool_start\n|>` to canonical form)
- **Flexible Arguments**: Handles both `arguments: "{}"` (string) and `arguments: {}` (object)
- **Conversational Extraction**: Finds tool calls within explanatory text
- **Multi-call Support**: Extracts multiple sequential tool calls from single response
- **Type Safety**: Strongly-typed Rust structures prevent API response errors

**Testing**

The 69 unit tests cover:

- Regex detection patterns for various whitespace combinations
- Format detection for all supported tool calling formats
- Real-world model output patterns (conversational text + tool calls)
- Edge cases (empty arguments, escaped characters, incomplete tokens)
- Multiple sequential tool calls
- Complex nested arguments with proper JSON stringification
- Integration testing with complete request/response cycle

This validates tool calling behavior across different model families and generations.

## Contribution

PR are welcomed right now! Since we need support a brand range of new models, but both Crane and HuggingFace's Candle is very limited model scope, so please join and help!

1. How to add a new model?

Generally speaking, you can reference to: `crane-core/src/models/siglip2.rs` for support new model, and all new added models should placed into `crane-core/src/models` and add `pub mod` in `crane-core/src/models/mod.rs` .

For me, the easiest way is to using Claude 3.7 to help write Rust conversion from pytorch code into Rust Candle code, and then manually fixing issues, once the float values of output are matched, the model can be ready to go.

2. How to support a new arch?

As all we know, a TTS model or any model based on LLM, it might consist of different modules, for example, in Spark-TTS, we will have a BiCodec Model before LLM, these module can be made into a separated module, and for Spark-TTS itself, we can gathering all module to inference it correctly.

One can reference to `crane-core/src/models/namo2.rs` for new arch add, which uses `Siglip2`, `mm_projector`, `Qwen2.5` to support a VL model.

3. How to add tool calling support for new models?

When adding tool calling support for new model families:

- **Identify Format**: Determine the special token format the model uses for tool calls
- **Add Detection**: Update `detect_tool_call_format()` in `crane-oai/src/openai_api.rs`
- **Implement Parser**: Add parsing function following existing patterns in `parse_qwen_tool_calls()` or `parse_qwen_tool_start_calls()`
- **Add Tests**: Include unit tests in the `openai_api::tests` module
- **Validate**: Use `cargo xtask test-tools` to validate with actual model inference

Key considerations:
- Models may insert whitespace/newlines in special tokens - handle variations
- Test with edge cases: malformed JSON, incomplete tokens, multiple sequential calls
- Ensure API response structure matches OpenAI specification
- Add logging for debugging tool detection and parsing failures


## Configuration

### Memory Management

Crane includes automatic memory safety checks to prevent out-of-memory crashes:

**Device-Aware Defaults:**
- **Metal** (Apple Silicon): F16 dtype by default (50% memory savings vs F32)
- **max_concurrent**: Scales automatically based on available system memory
  - Metal: 1 concurrent per 4GB available (capped at 4)
  - CPU: 1 concurrent per 8GB available (capped at 4)
  - CUDA: 1 concurrent per 2GB available (capped at 16)

**Memory Estimation:**
- Pre-flight check estimates model memory requirements from config.json
- Falls back to directory size if config unavailable
- Includes 25% overhead for runtime memory (activations + KV cache)

**Override Protection:**
```bash
# Auto-detect safe values (recommended)
./target/release/crane-oai --model-path /path/to/model

# Manual override (use with caution)
./target/release/crane-oai --model-path /path/to/model --max-concurrent 8

# Bypass memory checks (dangerous: may cause OOM)
./target/release/crane-oai --model-path /path/to/model --ignore-memory-limit
```

**Example Output:**
```
Device-aware max_concurrent: 2 (based on available memory)
Memory check: available=8.2G, estimated_need=7.0G
Model loaded successfully (type: Qwen25, format: Safetensors)
Device: Metal(MetalDevice(DeviceId(1))) | dtype: F16
```

### Inference Optimizations

Crane implements production-grade inference optimizations for both **Qwen3** and **Hunyuan Dense**.

Environment variables for tuning:

| Variable | Default | Description |
|----------|---------|-------------|
| `CRANE_FORCE_GPU_TOPK` | `0` | Force GPU topk sampling even for large vocabularies |
| `CRANE_TOPP_FALLBACK_TOPK` | `64` | Top-k size when top_p is active and GPU path is used |
| `CRANE_TOPK_SAMPLE_ON_CPU` | `0` | Force CPU sampling after GPU topk |
| `CRANE_SAMPLE_TRACE` | `0` | Enable detailed sampling timing logs |

### Tool Calling Implementation

Crane provides OpenAI-compatible function calling with support for Qwen models. The implementation handles model output variations including whitespace, newlines, and different argument formats.

**Supported Formats:**

Models can generate tool calls using Qwen special tokens:

```
<|tool_start|>
{"name": "function_name", "arguments": "{}"}
<|tool_end|>
```

**Key Features:**

- Multi-format token detection: Handles model-generated variations in special token formatting
- Whitespace-aware parsing: Regex-based fallback for malformed tokens
- Flexible argument handling: Accepts both `arguments: "{}"` (string) and `arguments: {}` (object) formats
- OpenAI-compatible responses: Returns structured `tool_calls` array with unique call IDs
- Test coverage: 69 unit tests covering detection, parsing, edge cases, and integration scenarios

**Request Format:**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-1.7B",
    "messages": [
      {"role": "user", "content": "What time is it?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_time",
          "description": "Get current time",
          "parameters": {
            "type": "object",
            "properties": {}
          }
        }
      }
    ]
  }'
```

**Response Format:**

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "tool_calls": [
          {
            "id": "call_95eabd13-0431-45e7-95f4-4e8ed4a8e798",
            "type": "function",
            "function": {
              "name": "get_time",
              "arguments": "{}"
            }
          }
        ]
      }
    }
  ]
}
```

**Implementation Details:**

The tool calling pipeline consists of three stages:

1. **Detection**: Identifies tool call format using fast string matching with regex fallback
2. **Normalization**: Handles whitespace variations in special tokens (`<|tool_start|>`, `<|tool_end|>`)
3. **Extraction**: Parses JSON arguments with flexible type conversion

**Testing Infrastructure:**

Comprehensive test suite validating:
- Token format detection (regex patterns, whitespace handling)
- JSON parsing (string/object arguments, escaped characters)
- Integration scenarios (conversational text, multiple calls, edge cases)
- Real-world model output patterns

**Model Compatibility:**

- Qwen3-1.7B: Validated with reliable tool calling
- Models 7B+: Generally provide more consistent tool recognition
- Smaller models (0.5B-3B): May have inconsistent tool format adherence

## Speed

Here are some speedup compare between **Crane** can other framework.

f32:

| Model/Platform | mac M1 metal | mac M1 cpu | mac M4 metal | v100 GPU | pytorch |
| -------------- | ------------- | ---------- | ------------ | -------- | ------- |
| Qwen2.5-500M   | 17.5 t/s      | 14 t/s     | /            |          | 6.9 t/s |
| Qwen2.5-VL-3B  | /             | /          | /            |          |         |

f16:

| Model/Platform | mac M1 metal | mac M1 metal 16  | mac M4 metal 16 | pytorch |
| -------------- | ------------- | ---------------- | --------------- | ------- |
| Qwen2.5-500M   | 17.5 t/s      | **35 t/s** | /               | 6.9 t/s |
| Qwen2.5-VL-3B  | /             | /                | /               |         |

- *Crane* is blazing fast on macOS with metal, useful for you to run local models;
- int8 quantization still on the way, it's even faster!


## Citation

If you use Crane in your research or projects, please cite using BibTeX:

```bibtex
@misc{Crane,
  author       = {lucasjinreal},
  title        = {{Crane: Candle-based Rust Accelerated Neural Engine}},
  howpublished = {\url{https://github.com/lucasjinreal/Crane}},
  year         = {2025}
}
```

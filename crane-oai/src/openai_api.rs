//! OpenAI-compatible request / response types.
//!
//! Covers:
//! * `/v1/chat/completions`  (chat)
//! * `/v1/completions`       (text completion)
//! * `/v1/audio/speech`      (text-to-speech)
//! * `/v1/models`            (model listing + retrieval)
//! * `/v1/tokenize`          (tokenization)
//! * `/v1/detokenize`        (detokenization)
//!
//! Wire format follows the
//! [OpenAI API reference](https://platform.openai.com/docs/api-reference).

use serde::{Deserialize, Serialize};

// ═════════════════════════════════════════════════════════════
//  Shared helpers
// ═════════════════════════════════════════════════════════════

fn default_max_tokens() -> usize {
    512
}

// ═════════════════════════════════════════════════════════════
//  Chat Completion  (/v1/chat/completions)
// ═════════════════════════════════════════════════════════════

// ── Request ──

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    pub stop: Option<Vec<String>>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub n: Option<usize>,
    /// Response format constraint (e.g., `{"type": "json_object"}`).
    pub response_format: Option<ResponseFormat>,
    /// Tools/functions that the model may call.
    pub tools: Option<Vec<Tool>>,
    /// Controls which tools to use.
    pub tool_choice: Option<ToolChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ChatMessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Extract the plain text content from the message.
    /// For multimodal messages, concatenates all text parts.
    pub fn text_content(&self) -> String {
        match &self.content {
            Some(content) => match content {
                ChatMessageContent::Text(s) => s.clone(),
                ChatMessageContent::Parts(parts) => parts
                    .iter()
                    .filter_map(|p| match p {
                        ContentPart::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(""),
            },
            None => String::new(),
        }
    }

    /// Extract image URLs from multimodal content.
    pub fn image_urls(&self) -> Vec<String> {
        match &self.content {
            Some(content) => match content {
                ChatMessageContent::Text(_) => vec![],
                ChatMessageContent::Parts(parts) => parts
                    .iter()
                    .filter_map(|p| match p {
                        ContentPart::ImageUrl { image_url } => Some(image_url.url.clone()),
                        _ => None,
                    })
                    .collect(),
            },
            None => vec![],
        }
    }
}

/// Chat message content — either a plain string or structured multimodal parts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ChatMessageContent {
    /// Plain text content (backward compatible).
    Text(String),
    /// Structured content with text and/or image_url parts.
    Parts(Vec<ContentPart>),
}

/// A single content part in a multimodal message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Text content.
    #[serde(rename = "text")]
    Text { text: String },
    /// Image URL content.
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
    /// Image content (alternative key used by some OpenAI clients).
    #[serde(rename = "image")]
    Image { image_url: Option<ImageUrl> },
}

/// An image URL reference.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ResponseFormat {
    pub r#type: String,
}

// ── Tool/Function Calling types ──

/// Tool definition for function calling.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Tool {
    pub r#type: String,
    pub function: FunctionDefinition,
}

/// Function definition within a tool.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<serde_json::Value>,
    #[serde(rename = "strict")]
    pub strict_mode: Option<bool>,
}

/// Tool choice strategy.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ToolChoice {
    String(String),
    Specific { r#type: String, function: FunctionName },
}

impl ToolChoice {
    pub fn is_auto(&self) -> bool {
        matches!(self, ToolChoice::String(s) if s == "auto")
    }

    pub fn is_none(&self) -> bool {
        matches!(self, ToolChoice::String(s) if s == "none")
    }

    pub fn is_required(&self) -> bool {
        matches!(self, ToolChoice::String(s) if s == "required")
    }
}

/// Function name for specific tool choice.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionName {
    pub name: String,
}

/// Tool call generated by the model.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}

/// Function call within a tool call.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

// ── Tool Call Detection & Parsing ──

/// Format of tool calls in model output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallFormat {
    /// Qwen special tokens: <|tool_call|>...<|end_tool_call|>
    QwenSpecialTokens,
    /// OpenAI-style JSON blocks
    OpenAIStyle,
    /// Markdown code blocks
    MarkdownCodeBlock,
    /// No tool calls detected
    None,
}

/// Detect the format of tool calls in the model output.
pub fn detect_tool_call_format(text: &str) -> ToolCallFormat {
    if text.contains("<|tool_call|>") {
        ToolCallFormat::QwenSpecialTokens
    } else if text.contains("```json") && text.contains("\"type\": \"function\"") {
        ToolCallFormat::OpenAIStyle
    } else if text.contains("```") && (text.contains("function_call") || text.contains("\"name\":")) {
        ToolCallFormat::MarkdownCodeBlock
    } else {
        ToolCallFormat::None
    }
}

/// Extract tool calls from model-generated text.
pub fn extract_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let format = detect_tool_call_format(text);

    match format {
        ToolCallFormat::QwenSpecialTokens => parse_qwen_tool_calls(text),
        ToolCallFormat::OpenAIStyle => parse_openai_style_tool_calls(text),
        ToolCallFormat::MarkdownCodeBlock => parse_markdown_tool_calls(text),
        ToolCallFormat::None => None,
    }
}

/// Parse Qwen-style tool calls: <|tool_call|>{"name": "...", "arguments": "..."}<|end_tool_call|>
fn parse_qwen_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let mut calls = Vec::new();

    let mut start = 0;
    while let Some(call_start) = text[start..].find("<|tool_call|>") {
        let absolute_start = start + call_start + "<|tool_call|>".len();
        let call_text = &text[absolute_start..];

        if let Some(call_end) = call_text.find("<|end_tool_call|>") {
            let json_str = &call_text[..call_end].trim();

            // Try to parse as JSON
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(json_str) {
                if let (Some(name), Some(args)) = (
                    value.get("name").and_then(|v| v.as_str()),
                    value.get("arguments").and_then(|v| v.as_str()),
                ) {
                    calls.push(ToolCall {
                        id: format!("call_{}", uuid::Uuid::new_v4()),
                        r#type: "function".to_string(),
                        function: FunctionCall {
                            name: name.to_string(),
                            arguments: args.to_string(),
                        },
                    });
                }
            }

            start = absolute_start + call_end + "<|end_tool_call|>".len();
        } else {
            break;
        }
    }

    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
}

/// Parse OpenAI-style JSON blocks.
fn parse_openai_style_tool_calls(_text: &str) -> Option<Vec<ToolCall>> {
    // TODO: Implement OpenAI-style parsing
    // This would look for JSON blocks with "type": "function"
    None
}

/// Parse tool calls from markdown code blocks.
fn parse_markdown_tool_calls(_text: &str) -> Option<Vec<ToolCall>> {
    // TODO: Implement markdown parsing
    // This would look for ```json blocks with function calls
    None
}

// ── Response ──

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

// ── Streaming ──

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// ═════════════════════════════════════════════════════════════
//  Text Completion  (/v1/completions)
// ═════════════════════════════════════════════════════════════

// ── Request ──

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: StringOrArray,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    pub stop: Option<Vec<String>>,
    pub suffix: Option<String>,
    pub echo: Option<bool>,
    pub seed: Option<u64>,
    pub n: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StringOrArray {
    Single(String),
    Array(Vec<String>),
}

impl StringOrArray {
    pub fn as_string(&self) -> String {
        match self {
            Self::Single(s) => s.clone(),
            Self::Array(arr) => arr.join(""),
        }
    }
}

// ── Response ──

#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: Option<String>,
}

// ── Streaming ──

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChunkChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: Option<String>,
}

// ═════════════════════════════════════════════════════════════
//  Tokenize / Detokenize  (/v1/tokenize, /v1/detokenize)
// ═════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizeRequest {
    /// Text to tokenize. Can be a single string or a list of chat messages.
    pub text: Option<String>,
    /// If provided, treat as chat messages and apply the chat template first.
    pub messages: Option<Vec<ChatMessage>>,
    /// Whether to add special tokens (default: true).
    #[serde(default = "default_true")]
    pub add_special_tokens: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenizeResponse {
    pub tokens: Vec<u32>,
    pub count: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DetokenizeRequest {
    pub tokens: Vec<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DetokenizeResponse {
    pub text: String,
}

// ═════════════════════════════════════════════════════════════
//  Audio Speech  (/v1/audio/speech)
// ═════════════════════════════════════════════════════════════

/// Audio response format.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum AudioResponseFormat {
    #[default]
    Wav,
    Pcm,
    Mp3,
    Opus,
    Aac,
    Flac,
}

fn default_speed() -> f64 {
    1.0
}

fn default_audio_max_tokens() -> usize {
    8192
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct SpeechRequest {
    /// Model ID (e.g. "qwen3-tts").
    pub model: String,
    /// The text to synthesize.
    pub input: String,
    /// Voice name (speaker ID or preset for CustomVoice models).
    #[serde(default)]
    pub voice: Option<String>,
    /// Language hint (e.g. "chinese", "english", "japanese", "auto").
    #[serde(default)]
    pub language: Option<String>,
    /// Optional system-level instructions.
    #[serde(default)]
    pub instructions: Option<String>,
    /// Response audio format.
    #[serde(default)]
    pub response_format: AudioResponseFormat,
    /// Speaking speed multiplier (currently unused, reserved).
    #[serde(default = "default_speed")]
    pub speed: f64,
    /// Generation temperature.
    pub temperature: Option<f64>,
    /// Nucleus sampling top-p.
    pub top_p: Option<f64>,
    /// Repetition penalty.
    pub repetition_penalty: Option<f32>,
    /// Max codec tokens to generate (controls max duration).
    #[serde(default = "default_audio_max_tokens")]
    pub max_tokens: usize,

    // ── Voice-clone fields (Base model only) ──────────────────

    /// URL or local path to reference audio for voice cloning.
    /// When set, triggers voice-clone mode (requires Base model).
    #[serde(default)]
    pub reference_audio: Option<String>,
    /// Transcript of the reference audio (required for voice cloning).
    #[serde(default)]
    pub reference_text: Option<String>,
}

// ═════════════════════════════════════════════════════════════
//  Common types
// ═════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

// ── Model listing ──

#[derive(Debug, Clone, Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_model_len: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub permission: Option<Vec<serde_json::Value>>,
}

// ── Error ──

#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub code: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── StringOrArray ──

    #[test]
    fn string_or_array_single() {
        let s = StringOrArray::Single("hello".into());
        assert_eq!(s.as_string(), "hello");
    }

    #[test]
    fn string_or_array_array() {
        let s = StringOrArray::Array(vec!["a".into(), "b".into(), "c".into()]);
        assert_eq!(s.as_string(), "abc");
    }

    #[test]
    fn string_or_array_empty_array() {
        let s = StringOrArray::Array(vec![]);
        assert_eq!(s.as_string(), "");
    }

    // ── Deserialization: StringOrArray ──

    #[test]
    fn string_or_array_deserialize_single() {
        let v: StringOrArray = serde_json::from_str(r#""hello world""#).unwrap();
        assert_eq!(v.as_string(), "hello world");
    }

    #[test]
    fn string_or_array_deserialize_array() {
        let v: StringOrArray = serde_json::from_str(r#"["foo","bar"]"#).unwrap();
        assert_eq!(v.as_string(), "foobar");
    }

    // ── Tool/Function Calling types ──

    #[test]
    fn test_tool_deserialization() {
        let json = r#"{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        }"#;
        let tool: Tool = serde_json::from_str(json).unwrap();
        assert_eq!(tool.r#type, "function");
        assert_eq!(tool.function.name, "get_weather");
        assert_eq!(tool.function.description, Some("Get current weather".to_string()));
    }

    #[test]
    fn test_tool_choice_auto() {
        let json = r#""auto""#;
        let choice: ToolChoice = serde_json::from_str(json).unwrap();
        assert!(choice.is_auto());
    }

    #[test]
    fn test_tool_choice_required() {
        let json = r#""required""#;
        let choice: ToolChoice = serde_json::from_str(json).unwrap();
        assert!(choice.is_required());
    }

    #[test]
    fn test_qwen_tool_call_format_detection() {
        let text = "<|tool_call|>{\"name\": \"test\", \"arguments\": \"{}\"}<|end_tool_call|>";
        let format = detect_tool_call_format(text);
        assert_eq!(format, ToolCallFormat::QwenSpecialTokens);
    }

    #[test]
    fn test_no_tool_call_format() {
        let text = "This is just regular text without any tool calls.";
        let format = detect_tool_call_format(text);
        assert_eq!(format, ToolCallFormat::None);
    }

    #[test]
    fn test_extract_qwen_tool_calls() {
        let text = "<|tool_call|>{\"name\": \"get_weather\", \"arguments\": \"{\\\"location\\\": \\\"London\\\"}\"}<|end_tool_call|>";
        let calls = extract_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].r#type, "function");
        assert!(calls[0].id.starts_with("call_"));
    }

    #[test]
    fn test_extract_multiple_qwen_tool_calls() {
        let text = "<|tool_call|>{\"name\": \"func1\", \"arguments\": \"{}\"}<|end_tool_call|>Some text<|tool_call|>{\"name\": \"func2\", \"arguments\": \"{\\\"x\\\": 1}\"}<|end_tool_call|>";
        let calls = extract_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "func1");
        assert_eq!(calls[1].function.name, "func2");
    }

    #[test]
    fn test_chat_message_with_tool_calls() {
        let msg = ChatMessage {
            role: "assistant".into(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_123".into(),
                r#type: "function".into(),
                function: FunctionCall {
                    name: "test_func".into(),
                    arguments: "{}".into(),
                },
            }]),
            tool_call_id: None,
        };

        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ChatMessage = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.role, "assistant");
        assert!(parsed.content.is_none());
        assert!(parsed.tool_calls.is_some());
        assert_eq!(parsed.tool_calls.unwrap().len(), 1);
    }

    #[test]
    fn test_chat_message_with_tool_call_id() {
        let msg = ChatMessage {
            role: "tool".into(),
            content: Some(ChatMessageContent::Text("Result from tool".into())),
            tool_calls: None,
            tool_call_id: Some("call_123".into()),
        };

        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ChatMessage = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.role, "tool");
        assert_eq!(parsed.tool_call_id, Some("call_123".into()));
    }

    // ── ChatMessage round-trip ──

    #[test]
    fn chat_message_serde_roundtrip() {
        let msg = ChatMessage {
            role: "user".into(),
            content: Some(ChatMessageContent::Text("Hello!".into())),
            tool_calls: None,
            tool_call_id: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.role, "user");
        assert_eq!(parsed.content, Some(ChatMessageContent::Text("Hello!".into())));
    }

    // ── ChatCompletionRequest deserialization ──

    #[test]
    fn chat_completion_request_minimal() {
        let json = r#"{
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "test-model");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.max_tokens, 512); // default
        assert!(!req.stream);
        assert!(req.temperature.is_none());
        assert!(req.top_p.is_none());
    }

    #[test]
    fn chat_completion_request_full() {
        let json = r#"{
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hi"}
            ],
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "stream": true,
            "stream_options": {"include_usage": true},
            "seed": 42
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, 100);
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.top_p, Some(0.9));
        assert_eq!(req.top_k, Some(50));
        assert_eq!(req.repetition_penalty, Some(1.1));
        assert!(req.stream);
        assert!(req.stream_options.as_ref().unwrap().include_usage);
        assert_eq!(req.seed, Some(42));
    }

    // ── CompletionRequest deserialization ──

    #[test]
    fn completion_request_single_prompt() {
        let json = r#"{
            "model": "m",
            "prompt": "Once upon a time"
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt.as_string(), "Once upon a time");
    }

    #[test]
    fn completion_request_array_prompt() {
        let json = r#"{
            "model": "m",
            "prompt": ["Hello", " world"]
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt.as_string(), "Hello world");
    }

    // ── TokenizeRequest ──

    #[test]
    fn tokenize_request_with_text() {
        let json = r#"{"text": "hello"}"#;
        let req: TokenizeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.text, Some("hello".into()));
        assert!(req.messages.is_none());
        assert!(req.add_special_tokens); // default true
    }

    #[test]
    fn tokenize_request_with_messages() {
        let json = r#"{"messages": [{"role": "user", "content": "Hi"}]}"#;
        let req: TokenizeRequest = serde_json::from_str(json).unwrap();
        assert!(req.text.is_none());
        assert_eq!(req.messages.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn tokenize_request_special_tokens_false() {
        let json = r#"{"text": "hello", "add_special_tokens": false}"#;
        let req: TokenizeRequest = serde_json::from_str(json).unwrap();
        assert!(!req.add_special_tokens);
    }

    // ── DetokenizeRequest ──

    #[test]
    fn detokenize_request() {
        let json = r#"{"tokens": [1, 2, 3]}"#;
        let req: DetokenizeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.tokens, vec![1, 2, 3]);
    }

    // ── Response serialization ──

    #[test]
    fn chat_completion_response_serializes() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-123".into(),
            object: "chat.completion".into(),
            created: 1234567890,
            model: "test".into(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: Some(ChatMessageContent::Text("Hello!".into())),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some("stop".into()),
            }],
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 1,
                total_tokens: 6,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"id\":\"chatcmpl-123\""));
        assert!(json.contains("\"total_tokens\":6"));
    }

    #[test]
    fn usage_serializes() {
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        };
        let json = serde_json::to_string(&usage).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["prompt_tokens"], 10);
        assert_eq!(parsed["completion_tokens"], 20);
        assert_eq!(parsed["total_tokens"], 30);
    }

    #[test]
    fn model_info_serializes_without_optional_fields() {
        let info = ModelInfo {
            id: "my-model".into(),
            object: "model".into(),
            created: 1000,
            owned_by: "crane".into(),
            max_model_len: None,
            permission: None,
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(!json.contains("max_model_len")); // skip_serializing_if
        assert!(!json.contains("permission"));
    }

    #[test]
    fn model_info_serializes_with_optional_fields() {
        let info = ModelInfo {
            id: "my-model".into(),
            object: "model".into(),
            created: 1000,
            owned_by: "crane".into(),
            max_model_len: Some(4096),
            permission: None,
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("\"max_model_len\":4096"));
    }

    #[test]
    fn model_list_serializes() {
        let list = ModelList {
            object: "list".into(),
            data: vec![],
        };
        let json = serde_json::to_string(&list).unwrap();
        assert!(json.contains("\"object\":\"list\""));
        assert!(json.contains("\"data\":[]"));
    }

    #[test]
    fn error_response_serializes() {
        let resp = ErrorResponse {
            error: ErrorDetail {
                message: "bad request".into(),
                r#type: "invalid_request_error".into(),
                code: None,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"message\":\"bad request\""));
        assert!(json.contains("\"type\":\"invalid_request_error\""));
    }

    // ── StreamOptions ──

    #[test]
    fn stream_options_default_include_usage() {
        let json = r#"{}"#;
        let opts: StreamOptions = serde_json::from_str(json).unwrap();
        assert!(!opts.include_usage);
    }

    // ── ChatCompletionChunk ──

    #[test]
    fn chat_chunk_usage_skipped_when_none() {
        let chunk = ChatCompletionChunk {
            id: "c1".into(),
            object: "chat.completion.chunk".into(),
            created: 0,
            model: "m".into(),
            choices: vec![],
            usage: None,
        };
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(!json.contains("usage"));
    }

    #[test]
    fn completion_chunk_serializes() {
        let chunk = CompletionChunk {
            id: "cmpl-1".into(),
            object: "text_completion".into(),
            created: 0,
            model: "m".into(),
            choices: vec![CompletionChunkChoice {
                index: 0,
                text: "hello".into(),
                finish_reason: None,
            }],
            usage: None,
        };
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("\"text\":\"hello\""));
    }

    // ── TokenizeResponse / DetokenizeResponse ──

    #[test]
    fn tokenize_response_serializes() {
        let resp = TokenizeResponse {
            tokens: vec![1, 2, 3],
            count: 3,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"count\":3"));
    }

    #[test]
    fn detokenize_response_serializes() {
        let resp = DetokenizeResponse {
            text: "hello world".into(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"text\":\"hello world\""));
    }
}

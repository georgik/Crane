//! Tool call parsing for Qwen models with <|tool_start|> format.
//!
//! Handles streaming tool call parsing with state management to avoid
//! leaking special tokens to the client.

use crate::openai_api::Tool;
use tracing::debug;

/// Parser state for streaming tool calls
enum ParserState {
    /// Normal text mode - no tool call detected
    Normal,
    /// Inside tool call - buffering JSON
    InToolCall,
    /// Tool call complete - waiting for end marker
    ToolCallComplete,
}

/// Tool call parser for Qwen models
pub struct ToolCallParser {
    /// Parser state machine
    state: ParserState,
    /// Buffer for accumulating content
    buffer: String,
    /// Extracted tool calls (when complete)
    tool_calls: Vec<crate::openai_api::ToolCall>,
    /// Normal text to return to client
    normal_text: String,
}

impl ToolCallParser {
    /// Create a new tool call parser
    pub fn new() -> Self {
        Self {
            state: ParserState::Normal,
            buffer: String::new(),
            tool_calls: Vec::new(),
            normal_text: String::new(),
        }
    }

    /// Parse streaming chunk and extract tool calls
    ///
    /// Returns (normal_text, tool_calls) where:
    /// - normal_text: Text content to display to user (special tokens filtered)
    /// - tool_calls: Vector of OpenAI-formatted tool calls (populated when complete)
    pub fn parse_incremental(
        &mut self,
        chunk: &str,
        _tools: &[Tool],
    ) -> Result<(String, Vec<crate::openai_api::ToolCall>), String> {
        self.buffer.push_str(chunk);
        self.normal_text.clear();

        // Check for tool call markers
        let has_start = self.buffer.contains("<|tool_start|>");
        let has_end = self.buffer.contains("<|tool_end|>");

        match self.state {
            ParserState::Normal => {
                if has_start {
                    self.state = ParserState::InToolCall;
                    // Extract text before tool_start
                    if let Some(pos) = self.buffer.find("<|tool_start|>") {
                        self.normal_text = self.buffer[..pos].to_string();
                    }
                    Ok((self.normal_text.clone(), vec![]))
                } else {
                    // Check for partial start marker
                    let has_partial = self.buffer.contains("<|tool")
                        || self.buffer.contains("tool_start")
                        || self.buffer.contains("<|");

                    if has_partial {
                        // Might be starting, but don't send yet
                        Ok((String::new(), vec![]))
                    } else {
                        // Safe to send all
                        self.normal_text = self.buffer.clone();
                        self.buffer.clear();
                        Ok((self.normal_text.clone(), vec![]))
                    }
                }
            }
            ParserState::InToolCall => {
                if has_end {
                    self.state = ParserState::ToolCallComplete;
                    // Parse tool call from buffer
                    if let Some(parsed) = self.parse_tool_call() {
                        self.tool_calls.push(parsed);
                    }
                    Ok((String::new(), self.tool_calls.clone()))
                } else {
                    // Still buffering tool call
                    Ok((String::new(), vec![]))
                }
            }
            ParserState::ToolCallComplete => {
                // Tool calls done, check for trailing content
                if let Some(pos) = self.buffer.find("<|tool_end|>") {
                    let after = &self.buffer[pos + "<|tool_end|>".len()..];
                    if !after.is_empty() {
                        self.normal_text = after.to_string();
                        self.buffer.clear();
                    }
                }
                Ok((self.normal_text.clone(), self.tool_calls.clone()))
            }
        }
    }

    /// Parse tool call JSON from buffer
    fn parse_tool_call(&self) -> Option<crate::openai_api::ToolCall> {
        let text = &self.buffer;

        // Find content between markers
        let start = text.find("<|tool_start|>")? + "<|tool_start|>".len();
        let end = text.find("<|tool_end|>")?;

        let json_str = text[start..end].trim();
        debug!("Parsing tool call JSON: {}", json_str);

        // Parse JSON
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(json_str) {
            let name = value.get("name")?.as_str()?.to_string();

            let arguments = if let Some(str_val) = value.get("arguments").and_then(|v| v.as_str()) {
                str_val.to_string()
            } else if let Some(obj) = value.get("arguments") {
                // Arguments is an object, stringify it
                serde_json::to_string(obj).unwrap_or_else(|_| "{}".to_string())
            } else {
                "{}".to_string()
            };

            Some(crate::openai_api::ToolCall {
                id: format!("call_{}", uuid::Uuid::new_v4()),
                index: Some(0),
                r#type: "function".to_string(),
                function: crate::openai_api::FunctionCall {
                    name,
                    arguments,
                },
            })
        } else {
            debug!("Failed to parse tool call JSON");
            None
        }
    }

    /// Check if text contains tool call markers
    pub fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("<|tool_start|>") || text.contains("<|tool_end|>")
    }

    /// Reset parser state for new request
    pub fn reset(&mut self) {
        self.state = ParserState::Normal;
        self.buffer.clear();
        self.tool_calls.clear();
        self.normal_text.clear();
    }

    /// Get parsed tool calls (available after parsing complete)
    pub fn get_tool_calls(&self) -> &[crate::openai_api::ToolCall] {
        &self.tool_calls
    }
}

impl Default for ToolCallParser {
    fn default() -> Self {
        Self::new()
    }
}

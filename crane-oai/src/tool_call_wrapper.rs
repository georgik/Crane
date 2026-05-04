//! Tool call parsing for Qwen models with <|tool_start|> format.
//!
//! Adapted from SMG's robust streaming tool call parsing approach.
//! SMG Source: /Users/georgik/projects/smg/crates/tool_parser

use crate::openai_api::Tool;
use serde::de::{Deserialize, IgnoredAny};
use serde_json::{Map, Value};
use tracing::debug;

/// Partial JSON parser for incomplete JSON during streaming
/// From SMG tool-parser: crates/tool_parser/src/partial_json.rs
struct PartialJson {
    max_depth: usize,
    allow_incomplete: bool,
}

impl PartialJson {
    fn new(max_depth: usize, allow_incomplete: bool) -> Self {
        Self {
            max_depth,
            allow_incomplete,
        }
    }

    /// Parse potentially incomplete JSON, returning parsed value and consumed bytes
    fn parse_value(&self, input: &str, allow_partial_strings: bool) -> Result<(Value, usize), String> {
        let mut parser = Parser::new(input, self.max_depth, self.allow_incomplete, allow_partial_strings);
        let value = parser.parse_value(0)?;
        Ok((value, parser.position))
    }
}

impl Default for PartialJson {
    fn default() -> Self {
        Self::new(32, true)
    }
}

/// Internal parser state for PartialJson
struct Parser<'a> {
    chars: std::iter::Peekable<std::str::Chars<'a>>,
    position: usize,
    max_depth: usize,
    allow_incomplete: bool,
    allow_partial_strings: bool,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str, max_depth: usize, allow_incomplete: bool, allow_partial_strings: bool) -> Self {
        Self {
            chars: input.chars().peekable(),
            position: 0,
            max_depth,
            allow_incomplete,
            allow_partial_strings,
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.chars.peek().copied()
    }

    fn advance(&mut self) {
        if self.chars.next().is_some() {
            self.position += 1;
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn parse_value(&mut self, depth: usize) -> Result<Value, String> {
        if depth > self.max_depth {
            return Err("Max depth exceeded".to_string());
        }

        self.skip_whitespace();

        match self.peek() {
            Some('{') => self.parse_object(depth + 1),
            Some('[') => self.parse_array(depth + 1),
            Some('"') => self.parse_string(),
            Some('t') | Some('f') => self.parse_bool(),
            Some('n') => self.parse_null(),
            Some(c) if c == '-' || c.is_ascii_digit() => self.parse_number(),
            _ => {
                if self.allow_incomplete {
                    Ok(Value::Null)
                } else {
                    Err("Unexpected character".to_string())
                }
            }
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<Value, String> {
        if depth > self.max_depth {
            return Err("Max depth exceeded".to_string());
        }

        let mut object = Map::new();
        self.advance(); // Consume '{'
        self.skip_whitespace();

        if self.peek() == Some('}') {
            self.advance();
            return Ok(Value::Object(object));
        }

        loop {
            let key = match self.parse_string() {
                Ok(Value::String(s)) => s,
                Err(_) if self.allow_incomplete => return Ok(Value::Object(object)),
                Err(e) => return Err(e),
                _ => return Err("Expected string key".to_string()),
            };

            self.skip_whitespace();

            if self.peek() != Some(':') {
                if self.allow_incomplete {
                    object.insert(key, Value::Null);
                    return Ok(Value::Object(object));
                }
                return Err("Expected ':'".to_string());
            }
            self.advance();
            self.skip_whitespace();

            let value = match self.parse_value(depth) {
                Ok(v) => v,
                Err(_) if self.allow_incomplete => {
                    if self.allow_partial_strings {
                        object.insert(key, Value::Null);
                    }
                    return Ok(Value::Object(object));
                }
                Err(e) => return Err(e),
            };

            object.insert(key, value);
            self.skip_whitespace();

            match self.peek() {
                Some(',') => {
                    self.advance();
                    self.skip_whitespace();
                    if self.peek() == Some('}') {
                        self.advance();
                        return Ok(Value::Object(object));
                    }
                }
                Some('}') => {
                    self.advance();
                    return Ok(Value::Object(object));
                }
                None if self.allow_incomplete => return Ok(Value::Object(object)),
                _ => {
                    if self.allow_incomplete {
                        return Ok(Value::Object(object));
                    }
                    return Err("Expected ',' or '}'".to_string());
                }
            }
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<Value, String> {
        if depth > self.max_depth {
            return Err("Max depth exceeded".to_string());
        }

        let mut array = Vec::new();
        self.advance(); // Consume '['
        self.skip_whitespace();

        if self.peek() == Some(']') {
            self.advance();
            return Ok(Value::Array(array));
        }

        loop {
            let value = match self.parse_value(depth) {
                Ok(v) => v,
                Err(_) if self.allow_incomplete => return Ok(Value::Array(array)),
                Err(e) => return Err(e),
            };

            array.push(value);
            self.skip_whitespace();

            match self.peek() {
                Some(',') => {
                    self.advance();
                    self.skip_whitespace();
                    if self.peek() == Some(']') {
                        self.advance();
                        return Ok(Value::Array(array));
                    }
                }
                Some(']') => {
                    self.advance();
                    return Ok(Value::Array(array));
                }
                None if self.allow_incomplete => return Ok(Value::Array(array)),
                _ => {
                    if self.allow_incomplete {
                        return Ok(Value::Array(array));
                    }
                    return Err("Expected ',' or ']'".to_string());
                }
            }
        }
    }

    fn parse_string(&mut self) -> Result<Value, String> {
        if self.peek() != Some('"') {
            return Err("Expected '\"'".to_string());
        }

        self.advance(); // Consume opening quote

        let mut string = String::new();
        let mut escaped = false;

        while let Some(ch) = self.peek() {
            if escaped {
                let escaped_char = match ch {
                    '"' | '\\' | '/' => ch,
                    'b' => '\u{0008}',
                    'f' => '\u{000C}',
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    'u' => {
                        self.advance();
                        let hex = self.parse_unicode_escape()?;
                        string.push(hex);
                        escaped = false;
                        continue;
                    }
                    _ => ch,
                };
                string.push(escaped_char);
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                self.advance();
                return Ok(Value::String(string));
            } else {
                string.push(ch);
            }
            self.advance();
        }

        if self.allow_incomplete && self.allow_partial_strings {
            Ok(Value::String(string))
        } else {
            Err("Unterminated string".to_string())
        }
    }

    fn parse_unicode_escape(&mut self) -> Result<char, String> {
        let mut hex = String::new();
        for _ in 0..4 {
            if let Some(ch) = self.peek() {
                if ch.is_ascii_hexdigit() {
                    hex.push(ch);
                    self.advance();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if hex.len() == 4 {
            u32::from_str_radix(&hex, 16)
                .ok()
                .and_then(char::from_u32)
                .ok_or_else(|| "Invalid unicode escape".to_string())
        } else if self.allow_incomplete {
            Ok('\u{FFFD}')
        } else {
            Err("Incomplete unicode escape".to_string())
        }
    }

    fn parse_number(&mut self) -> Result<Value, String> {
        let mut number = String::new();

        if self.peek() == Some('-') {
            number.push('-');
            self.advance();
        }

        if self.peek() == Some('0') {
            number.push('0');
            self.advance();
        } else {
            while let Some(ch) = self.peek() {
                if ch.is_ascii_digit() {
                    number.push(ch);
                    self.advance();
                } else {
                    break;
                }
            }
        }

        if self.peek() == Some('.') {
            number.push('.');
            self.advance();
            while let Some(ch) = self.peek() {
                if ch.is_ascii_digit() {
                    number.push(ch);
                    self.advance();
                } else {
                    break;
                }
            }
        }

        if let Some(ch) = self.peek() {
            if ch == 'e' || ch == 'E' {
                number.push(ch);
                self.advance();
                if let Some(sign) = self.peek() {
                    if sign == '+' || sign == '-' {
                        number.push(sign);
                        self.advance();
                    }
                }
                while let Some(ch) = self.peek() {
                    if ch.is_ascii_digit() {
                        number.push(ch);
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
        }

        if let Ok(n) = number.parse::<i64>() {
            Ok(Value::Number(serde_json::Number::from(n)))
        } else if let Ok(n) = number.parse::<f64>() {
            Ok(Value::Number(
                serde_json::Number::from_f64(n).unwrap_or_else(|| serde_json::Number::from(0)),
            ))
        } else if self.allow_incomplete {
            Ok(Value::Number(serde_json::Number::from(0)))
        } else {
            Err("Invalid number".to_string())
        }
    }

    fn parse_bool(&mut self) -> Result<Value, String> {
        let mut word = String::new();
        let mut temp_chars = self.chars.clone();
        while let Some(&ch) = temp_chars.peek() {
            if ch.is_alphabetic() && word.len() < 5 {
                word.push(ch);
                temp_chars.next();
            } else {
                break;
            }
        }

        let is_valid = word == "true"
            || word == "false"
            || (self.allow_incomplete && ("true".starts_with(&word) || "false".starts_with(&word)));

        if !is_valid {
            return Err("Invalid boolean".to_string());
        }

        word.clear();
        while let Some(ch) = self.peek() {
            if ch.is_alphabetic() {
                word.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        match word.as_str() {
            "true" => Ok(Value::Bool(true)),
            "false" => Ok(Value::Bool(false)),
            partial if self.allow_incomplete => {
                if "true".starts_with(partial) {
                    Ok(Value::Bool(true))
                } else if "false".starts_with(partial) {
                    Ok(Value::Bool(false))
                } else {
                    Err("Invalid boolean".to_string())
                }
            }
            _ => Err("Invalid boolean".to_string()),
        }
    }

    fn parse_null(&mut self) -> Result<Value, String> {
        let mut word = String::new();
        let mut temp_chars = self.chars.clone();
        while let Some(&ch) = temp_chars.peek() {
            if ch.is_alphabetic() && word.len() < 4 {
                word.push(ch);
                temp_chars.next();
            } else {
                break;
            }
        }

        let is_valid = word == "null" || (self.allow_incomplete && "null".starts_with(&word));

        if !is_valid {
            return Err("Invalid null".to_string());
        }

        word.clear();
        while let Some(ch) = self.peek() {
            if ch.is_alphabetic() {
                word.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        if word == "null" || (self.allow_incomplete && "null".starts_with(&word)) {
            Ok(Value::Null)
        } else {
            Err("Invalid null".to_string())
        }
    }
}

/// Check if buffer ends with partial occurrence of a token
/// From SMG tool-parser: crates/tool_parser/src/parsers/helpers.rs:75
fn ends_with_partial_token(buffer: &str, token: &str) -> Option<usize> {
    if buffer.is_empty() || token.is_empty() {
        return None;
    }
    (1..token.len()).find(|&i| buffer.ends_with(&token[..i]))
}

/// Check if a string contains complete, valid JSON
/// From SMG tool-parser: crates/tool_parser/src/parsers/helpers.rs:138
fn is_complete_json(input: &str) -> bool {
    let mut de = serde_json::Deserializer::from_str(input);
    IgnoredAny::deserialize(&mut de).is_ok() && de.end().is_ok()
}

/// Normalize tool call fields (arguments/parameters)
/// From SMG tool-parser: crates/tool_parser/src/parsers/helpers.rs:188
fn normalize_tool_call_fields(mut obj: Value) -> Value {
    if obj.get("arguments").is_none() {
        if let Some(params) = obj.get("parameters").cloned() {
            if let Value::Object(ref mut map) = obj {
                map.insert("arguments".to_string(), params);
            }
        }
    }
    obj
}

/// Streaming parse result (matches SMG structure)
struct StreamingParseResult {
    normal_text: String,
    calls: Vec<ToolCallItem>,
}

impl Default for StreamingParseResult {
    fn default() -> Self {
        Self {
            normal_text: String::new(),
            calls: Vec::new(),
        }
    }
}

/// Tool call item for incremental streaming
struct ToolCallItem {
    tool_index: usize,
    name: Option<String>,
    parameters: String,
}

/// Tool call parser for Qwen models
/// Adapted from SMG's QwenParser with proven streaming approach
pub struct ToolCallParser {
    /// Partial JSON parser for incomplete JSON
    partial_json: PartialJson,
    /// Buffer for accumulating content
    buffer: String,
    /// Extracted tool calls (when complete)
    tool_calls: Vec<crate::openai_api::ToolCall>,
    /// Normal text to return to client
    normal_text: String,
    /// Special tokens to detect
    tool_start_token: &'static str,
    tool_end_token: &'static str,
    /// Current tool ID being parsed
    current_tool_id: i32,
    /// Whether current tool's name has been sent
    current_tool_name_sent: bool,
    /// Tool call array for tracking previous states
    prev_tool_call_arr: Vec<Value>,
    /// Tracks raw JSON string content streamed to client for each tool's arguments
    streamed_args_for_tool: Vec<String>,
}

impl ToolCallParser {
    /// Create a new tool call parser
    pub fn new() -> Self {
        Self {
            partial_json: PartialJson::default(),
            buffer: String::new(),
            tool_calls: Vec::new(),
            normal_text: String::new(),
            tool_start_token: "<|tool_start|>",
            tool_end_token: "<|tool_end|>",
            current_tool_id: -1,
            current_tool_name_sent: false,
            prev_tool_call_arr: Vec::new(),
            streamed_args_for_tool: Vec::new(),
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
        tools: &[Tool],
    ) -> Result<(String, Vec<crate::openai_api::ToolCall>), String> {
        self.buffer.push_str(chunk);
        self.normal_text.clear();

        let current_text = &self.buffer.clone();

        // Check if current_text has tool_call
        let has_tool_start = self.has_tool_markers(current_text);

        if !has_tool_start {
            // Only clear buffer if we're sure no tool call is starting
            if ends_with_partial_token(&self.buffer, self.tool_start_token).is_none() {
                // Safe to send all
                self.normal_text = self.buffer.clone();
                self.buffer.clear();
                Ok((self.normal_text.clone(), vec![]))
            } else {
                // Might end with partial tool_start token, keep buffering
                Ok((String::new(), vec![]))
            }
        } else {
            // Use SMG's proven handle_json_tool_streaming approach
            match self.handle_json_tool_streaming(tools) {
                Ok(result) => {
                    self.normal_text = result.normal_text;

                    // Convert SMG ToolCallItems to our ToolCall format
                    for call_item in result.calls {
                        let tool_id = call_item.tool_index;

                        if let Some(name) = &call_item.name {
                            // New tool call - create with empty arguments
                            if self.current_tool_id == -1 {
                                self.current_tool_id = 0;
                            }

                            self.tool_calls.push(crate::openai_api::ToolCall {
                                id: format!("call_{}", uuid::Uuid::new_v4()),
                                index: Some(tool_id),
                                r#type: "function".to_string(),
                                function: crate::openai_api::FunctionCall {
                                    name: name.clone(),
                                    arguments: String::new(),
                                },
                            });
                        } else {
                            // Argument delta - update existing tool call with COMPLETE accumulated arguments
                            if tool_id < self.streamed_args_for_tool.len() {
                                let complete_args = &self.streamed_args_for_tool[tool_id];
                                if let Some(tool_call) = self.tool_calls.get_mut(tool_id) {
                                    tool_call.function.arguments = complete_args.clone();
                                }
                            }
                        }
                    }

                    Ok((self.normal_text.clone(), self.tool_calls.clone()))
                }
                Err(e) => {
                    debug!("Tool streaming error: {}", e);
                    Ok((String::new(), vec![]))
                }
            }
        }
    }

    /// Handle JSON tool streaming using SMG's proven approach
    /// From SMG: crates/tool_parser/src/parsers/helpers.rs:219
    fn handle_json_tool_streaming(
        &mut self,
        tools: &[Tool],
    ) -> Result<StreamingParseResult, String> {
        use std::collections::HashMap;

        // Build tool indices for validation
        let tool_indices: HashMap<String, usize> = tools
            .iter()
            .enumerate()
            .map(|(i, tool)| (tool.function.name.clone(), i))
            .collect();

        let start_idx = if let Some(pos) = self.buffer.find(self.tool_start_token) {
            pos + self.tool_start_token.len()
        } else {
            return Err("Tool start token not found".to_string());
        };

        if start_idx >= self.buffer.len() {
            return Ok(StreamingParseResult::default());
        }

        let json_str = &self.buffer[start_idx..];

        // When current_tool_name_sent is false, don't allow partial strings
        let allow_partial_strings = self.current_tool_name_sent;

        // Parse partial JSON
        let (obj, end_idx) = match self.partial_json.parse_value(json_str, allow_partial_strings) {
            Ok(result) => result,
            Err(_) => return Ok(StreamingParseResult::default()),
        };

        // Check if JSON is complete - validate only the parsed portion
        // Ensure end_idx is on a valid UTF-8 character boundary
        let safe_end_idx = if json_str.is_char_boundary(end_idx) {
            end_idx
        } else {
            (0..end_idx)
                .rev()
                .find(|&i| json_str.is_char_boundary(i))
                .unwrap_or(0)
        };
        let is_complete = is_complete_json(&json_str[..safe_end_idx]);

        // Normalize tool call fields first
        let current_tool_call = normalize_tool_call_fields(obj);

        // Validate tool name if present
        if let Some(name) = current_tool_call.get("name").and_then(|v| v.as_str()) {
            if !tool_indices.contains_key(name) {
                debug!("Invalid tool name '{}' - skipping", name);
                self.reset_current_tool_state();
                return Ok(StreamingParseResult::default());
            }
        }

        let mut result = StreamingParseResult::default();

        // Case 1: Handle tool name streaming
        if !self.current_tool_name_sent {
            if let Some(function_name) = current_tool_call.get("name").and_then(|v| v.as_str()) {
                if tool_indices.contains_key(function_name) {
                    // Initialize if first tool
                    if self.current_tool_id == -1 {
                        self.current_tool_id = 0;
                        self.streamed_args_for_tool.push(String::new());
                    } else if (self.current_tool_id as usize) >= self.streamed_args_for_tool.len() {
                        // Ensure capacity for subsequent tools
                        self.ensure_capacity();
                    }

                    // Send tool name with empty parameters
                    self.current_tool_name_sent = true;
                    result.calls.push(ToolCallItem {
                        tool_index: self.current_tool_id as usize,
                        name: Some(function_name.to_string()),
                        parameters: String::new(),
                    });
                }
            }
        }
        // Case 2: Handle streaming arguments
        else if let Some(cur_arguments) = current_tool_call.get("arguments") {
            let tool_id = self.current_tool_id as usize;
            let sent = self.streamed_args_for_tool
                .get(tool_id)
                .map(|s| s.len())
                .unwrap_or(0);
            let cur_args_json = serde_json::to_string(cur_arguments)
                .map_err(|e| format!("JSON serialization failed: {}", e))?;

            // Get prev_arguments
            let prev_arguments = if tool_id < self.prev_tool_call_arr.len() {
                self.prev_tool_call_arr[tool_id].get("arguments")
            } else {
                None
            };

            // Calculate diff: everything after we've already sent
            let mut argument_diff = None;

            if is_complete {
                // Send all remaining arguments
                argument_diff = if sent < cur_args_json.len() {
                    Some(cur_args_json[sent..].to_string())
                } else {
                    Some(String::new())
                };
            } else if let Some(prev_args) = prev_arguments {
                let prev_args_json = serde_json::to_string(prev_args)
                    .map_err(|e| format!("JSON serialization failed: {}", e))?;

                if cur_args_json != prev_args_json {
                    let prefix = self.find_common_prefix(&prev_args_json, &cur_args_json);
                    argument_diff = if sent < prefix.len() {
                        Some(prefix[sent..].to_string())
                    } else {
                        Some(String::new())
                    };
                }
            }

            // Send diff if present
            if let Some(diff) = argument_diff {
                if !diff.is_empty() {
                    if tool_id < self.streamed_args_for_tool.len() {
                        self.streamed_args_for_tool[tool_id].push_str(&diff);
                    }
                    result.calls.push(ToolCallItem {
                        tool_index: tool_id,
                        name: None,
                        parameters: diff,
                    });
                }
            }

            // Update prev_tool_call_arr with current state
            if self.current_tool_id >= 0 {
                self.ensure_capacity();

                if tool_id < self.prev_tool_call_arr.len() {
                    self.prev_tool_call_arr[tool_id] = current_tool_call;
                }
            }

            // If complete, advance to next tool
            if is_complete {
                let mut new_buffer_start = start_idx + end_idx;

                // Also skip <|tool_end|> token if present
                let remaining = &self.buffer[new_buffer_start..];
                if let Some(tool_end_pos) = remaining.find(self.tool_end_token) {
                    // Check if it's at the start (allowing for whitespace)
                    let after_whitespace = remaining.trim_start();
                    if after_whitespace.starts_with(self.tool_end_token) {
                        new_buffer_start += tool_end_pos + self.tool_end_token.len();
                    }
                }

                self.buffer = self.buffer[new_buffer_start..].to_string();
                self.current_tool_name_sent = false;
                self.current_tool_id += 1;
            }
        }

        Ok(result)
    }

    /// Find common prefix of two strings
    /// From SMG helpers.rs:23
    fn find_common_prefix(&self, s1: &str, s2: &str) -> String {
        s1.chars()
            .zip(s2.chars())
            .take_while(|(c1, c2)| c1 == c2)
            .map(|(c1, _)| c1)
            .collect()
    }

    /// Reset state for the current tool being parsed
    fn reset_current_tool_state(&mut self) {
        self.buffer.clear();
        self.current_tool_name_sent = false;

        // Only pop if we added an entry for the current (invalid) tool
        if self.streamed_args_for_tool.len() > self.prev_tool_call_arr.len() {
            self.streamed_args_for_tool.pop();
        }
    }

    /// Ensure arrays have capacity for the given tool ID
    fn ensure_capacity(&mut self) {
        if self.current_tool_id < 0 {
            return;
        }
        let needed = (self.current_tool_id + 1) as usize;

        if self.prev_tool_call_arr.len() < needed {
            self.prev_tool_call_arr.resize_with(needed, || Value::Null);
        }
        if self.streamed_args_for_tool.len() < needed {
            self.streamed_args_for_tool.resize_with(needed, String::new);
        }
    }

    /// Parse tool call JSON from buffer
    fn parse_tool_call(&self) -> Option<crate::openai_api::ToolCall> {
        let text = &self.buffer;

        // Find content between markers
        let start = text.find(self.tool_start_token)? + self.tool_start_token.len();
        let end = text.find(self.tool_end_token)?;

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
        text.contains(self.tool_start_token)
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

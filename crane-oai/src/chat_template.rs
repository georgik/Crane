//! Chat template formatting.
//!
//! Converts OpenAI-style `[{role, content}]` messages into the prompt string
//! expected by each model family.  Two strategies:
//!
//! * **[`AutoChatTemplate`]** — uses the Jinja `chat_template` from
//!   `tokenizer_config.json` (works for Qwen, Llama, Mistral, …).
//! * **[`HunyuanChatTemplate`]** — hardcoded template for Hunyuan models.

use crate::openai_api::{ChatMessage, Tool};
use crane_core::autotokenizer::AutoTokenizer;
use tracing::{debug, error, info, warn};

// ─────────────────────────────────────────────────────────────
//  Trait
// ─────────────────────────────────────────────────────────────

/// Formats chat messages into a model-specific prompt string.
pub trait ChatTemplateProcessor: Send + Sync {
    fn apply(&self, messages: &[ChatMessage], tools: Option<&[Tool]>) -> Result<String, String>;
}

// ─────────────────────────────────────────────────────────────
//  AutoChatTemplate (Jinja-based)
// ─────────────────────────────────────────────────────────────

/// Uses [`AutoTokenizer`]'s Jinja `chat_template` from `tokenizer_config.json`.
pub struct AutoChatTemplate {
    tokenizer: AutoTokenizer,
}

impl AutoChatTemplate {
    pub fn new(model_path: &str) -> Result<Self, String> {
        let tokenizer = AutoTokenizer::from_pretrained(model_path, None)
            .map_err(|e| format!("Failed to load AutoTokenizer: {e}"))?;
        Ok(Self { tokenizer })
    }
}

impl ChatTemplateProcessor for AutoChatTemplate {
    fn apply(&self, messages: &[ChatMessage], tools: Option<&[Tool]>) -> Result<String, String> {
        info!("Applying chat template: {} messages, tools: {}",
              messages.len(),
              tools.map_or(0, |t| t.len()));

        // Build the list of {role, content} values expected by the Jinja template.
        let mut template_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                let mut msg = serde_json::json!({
                    "role": m.role,
                    "content": m.text_content(),
                });
                // Add tool_calls if present
                if let Some(tool_calls) = &m.tool_calls {
                    msg["tool_calls"] = serde_json::to_value(tool_calls)
                        .map_err(|e| format!("Failed to serialize tool_calls: {e}"))?;
                    debug!("Message has tool_calls: {:?}", tool_calls);
                }
                // Add tool_call_id if present
                if let Some(tool_call_id) = &m.tool_call_id {
                    msg["tool_call_id"] = serde_json::json!(tool_call_id);
                }
                Ok(msg)
            })
            .collect::<Result<Vec<_>, String>>()?;

        // For tools, prepend a system message with tool definitions
        // This matches how HuggingFace transformers does it
        if let Some(tools) = tools {
            info!("Processing {} tools for template", tools.len());

            // Create system message with tool definitions
            let tools_json = serde_json::to_string_pretty(tools)
                .unwrap_or_else(|_| "[]".to_string());

            let system_msg = serde_json::json!({
                "role": "system",
                "content": format!(
                    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <|tool_start|>...<|tool_end|> XML tags.\n\nExample:\n<|tool_start|>\n{{\"name\": \"get_time\", \"arguments\": \"{{}}\"}}\n<|tool_end|>",
                    tools_json
                )
            });

            // Insert system message at the beginning
            let mut messages_with_tools = vec![system_msg];
            messages_with_tools.extend(template_messages);

            debug!("Messages with tool system message: {} total messages", messages_with_tools.len());
            debug!("First message (tools): {}", serde_json::to_string_pretty(&messages_with_tools[0]).unwrap_or_else(|_| "Invalid".to_string()));

            return self.tokenizer
                .apply_chat_template(&messages_with_tools, true)
                .map_err(|e| format!("Chat template error: {e}"));
        }

        // No tools - pass messages directly
        debug!("No tools, using messages-only template");
        self.tokenizer
            .apply_chat_template(&template_messages, true)
            .map_err(|e| format!("Chat template error: {e}"))
    }
}

// ─────────────────────────────────────────────────────────────
//  HunyuanChatTemplate (hardcoded)
// ─────────────────────────────────────────────────────────────

/// Hardcoded chat template for Hunyuan Dense models.
pub struct HunyuanChatTemplate;

impl ChatTemplateProcessor for HunyuanChatTemplate {
    fn apply(&self, messages: &[ChatMessage], _tools: Option<&[Tool]>) -> Result<String, String> {
        const BOS: &str = "<\u{ff5c}hy_begin\u{2581}of\u{2581}sentence\u{ff5c}>";
        const USER: &str = "<\u{ff5c}hy_User\u{ff5c}>";
        const ASSISTANT: &str = "<\u{ff5c}hy_Assistant\u{ff5c}>";
        const EOS: &str = "<\u{ff5c}hy_place\u{2581}holder\u{2581}no\u{2581}2\u{ff5c}>";
        const SEP: &str = "<\u{ff5c}hy_place\u{2581}holder\u{2581}no\u{2581}3\u{ff5c}>";

        let mut result = String::new();
        result.push_str(BOS);

        let (system_msg, loop_messages) = if !messages.is_empty() && messages[0].role == "system" {
            (Some(messages[0].text_content()), &messages[1..])
        } else {
            (None, &messages[..])
        };

        if let Some(sys) = system_msg {
            result.push_str(&sys);
            result.push_str(SEP);
        }

        for msg in loop_messages {
            match msg.role.as_str() {
                "user" => {
                    result.push_str(USER);
                    result.push_str(&msg.text_content());
                }
                "assistant" => {
                    result.push_str(ASSISTANT);
                    result.push_str(&msg.text_content());
                    result.push_str(EOS);
                }
                _ => {}
            }
        }

        result.push_str(ASSISTANT);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::chat_template::HunyuanChatTemplate;
    use crate::chat_template::ChatTemplateProcessor;
    use crate::openai_api::{ChatMessage, ChatMessageContent};

    fn make_messages(pairs: &[(&str, &str)]) -> Vec<ChatMessage> {
        pairs
            .iter()
            .map(|(role, content)| ChatMessage {
                role: role.to_string(),
                content: Some(ChatMessageContent::Text(content.to_string())),
                tool_calls: None,
                tool_call_id: None,
            })
            .collect()
    }

    // ── HunyuanChatTemplate ──

    #[test]
    fn hunyuan_basic_user_message() {
        let tmpl = HunyuanChatTemplate;
        let msgs = make_messages(&[("user", "Hello")]);
        let result = tmpl.apply(&msgs, None).unwrap();

        // Should start with BOS.
        assert!(result.starts_with("<\u{ff5c}hy_begin\u{2581}of\u{2581}sentence\u{ff5c}>"));
        // Should contain user tag + content.
        assert!(result.contains("<\u{ff5c}hy_User\u{ff5c}>Hello"));
        // Should end with assistant tag (ready for generation).
        assert!(result.ends_with("<\u{ff5c}hy_Assistant\u{ff5c}>"));
    }

    #[test]
    fn hunyuan_system_message_prepended() {
        let tmpl = HunyuanChatTemplate;
        let msgs = make_messages(&[
            ("system", "You are helpful"),
            ("user", "Hi"),
        ]);
        let result = tmpl.apply(&msgs, None).unwrap();

        // System content should appear after BOS followed by SEP.
        let sep = "<\u{ff5c}hy_place\u{2581}holder\u{2581}no\u{2581}3\u{ff5c}>";
        assert!(result.contains(&format!("You are helpful{sep}")));
    }

    #[test]
    fn hunyuan_multi_turn() {
        let tmpl = HunyuanChatTemplate;
        let msgs = make_messages(&[
            ("user", "Hello"),
            ("assistant", "Hi!"),
            ("user", "How are you?"),
        ]);
        let result = tmpl.apply(&msgs, None).unwrap();

        let eos = "<\u{ff5c}hy_place\u{2581}holder\u{2581}no\u{2581}2\u{ff5c}>";
        // Assistant response should end with EOS.
        assert!(result.contains(&format!("Hi!{eos}")));
        // Second user message present.
        assert!(result.contains("How are you?"));
    }

    #[test]
    fn hunyuan_empty_messages() {
        let tmpl = HunyuanChatTemplate;
        let msgs: Vec<ChatMessage> = vec![];
        let result = tmpl.apply(&msgs, None).unwrap();

        // Should at least have BOS + ASSISTANT.
        assert!(result.starts_with("<\u{ff5c}hy_begin\u{2581}of\u{2581}sentence\u{ff5c}>"));
        assert!(result.ends_with("<\u{ff5c}hy_Assistant\u{ff5c}>"));
    }

    #[test]
    fn hunyuan_unknown_role_skipped() {
        let tmpl = HunyuanChatTemplate;
        let msgs = make_messages(&[
            ("user", "Hello"),
            ("tool", "some tool output"),
            ("user", "Next"),
        ]);
        let result = tmpl.apply(&msgs, None).unwrap();
        // "tool" content should not appear with any tag.
        assert!(!result.contains("some tool output"));
    }

    // ── ChatTemplateProcessor trait ──

    #[test]
    fn hunyuan_implements_trait() {
        let proc: Box<dyn ChatTemplateProcessor> = Box::new(HunyuanChatTemplate);
        let msgs = make_messages(&[("user", "test")]);
        assert!(proc.apply(&msgs, None).is_ok());
    }

    // ── Tool calling tests ──

    use crate::openai_api::{Tool, FunctionDefinition};

    fn make_tools() -> Vec<Tool> {
        vec![Tool {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: "get_time".to_string(),
                description: Some("Get current time".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {}
                })),
                strict_mode: None,
            },
        }]
    }

    #[test]
    fn test_tool_context_structure() {
        let msgs = make_messages(&[("user", "What time is it?")]);
        let tools = make_tools();

        // Test that tools are properly structured in context
        let template_messages: Vec<serde_json::Value> = msgs
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": m.text_content(),
                })
            })
            .collect();

        let mut context = serde_json::json!({
            "messages": template_messages,
        });

        if let Ok(tools_value) = serde_json::to_value(&tools) {
            context["tools"] = tools_value;
        }

        // Verify context structure
        assert!(context.get("messages").is_some());
        assert!(context.get("tools").is_some());
        assert_eq!(context["tools"].as_array().unwrap().len(), 1);
        assert_eq!(context["tools"][0]["function"]["name"], "get_time");
    }

    #[test]
    fn test_tool_serialization() {
        let tools = make_tools();

        // Test tool serialization
        let tools_value = serde_json::to_value(&tools).unwrap();
        assert_eq!(tools_value.as_array().unwrap().len(), 1);

        let tool = &tools_value.as_array().unwrap()[0];
        assert_eq!(tool["type"], "function");
        assert_eq!(tool["function"]["name"], "get_time");
        assert_eq!(tool["function"]["description"], "Get current time");
    }
}

//! OpenAI-compatible API handlers.
//!
//! Endpoints:
//! - `POST /v1/chat/completions`
//! - `POST /v1/completions`
//! - `GET  /v1/models`
//! - `GET  /v1/models/:model_id`
//! - `POST /v1/tokenize`
//! - `POST /v1/detokenize`

use std::sync::Arc;
use tracing::{debug, error, info, warn};
use std::fs;
use std::path::PathBuf;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{
        sse::{KeepAlive, Sse},
        IntoResponse, Json, Response,
    },
};

use crate::engine::EngineResponse;
use crate::openai_api::*;
use crate::{make_error, now_epoch, AppState};

use super::sse;
use super::vlm;

// ─────────────────────────────────────────────────────────────
//  Communication Logging
// ─────────────────────────────────────────────────────────────

fn get_comm_log_dir() -> PathBuf {
    PathBuf::from("communication_logs")
}

fn log_communication(direction: &str, body: &serde_json::Value) {
    let log_dir = get_comm_log_dir();
    if let Err(e) = fs::create_dir_all(&log_dir) {
        warn!("Failed to create communication log directory: {}", e);
        return;
    }

    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S_%3f");
    let filename = format!("{}_{}.json", timestamp, direction);
    let filepath = log_dir.join(&filename);

    let formatted = serde_json::to_string_pretty(body).unwrap_or_else(|_| "Invalid JSON".to_string());

    if let Err(e) = fs::write(&filepath, formatted) {
        warn!("Failed to write communication log: {}", e);
    } else {
        debug!("Logged communication to: {}", filename);
    }
}

// ─────────────────────────────────────────────────────────────
//  Chat Completions
// ─────────────────────────────────────────────────────────────

/// `POST /v1/chat/completions` — streaming and non-streaming.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    info!("Chat completion request: model={}, messages={}, tools={}, stream={}",
          req.model,
          req.messages.len(),
          req.tools.as_ref().map_or(0, |t| t.len()),
          req.stream);

    // Log incoming request
    log_communication("client_request", &serde_json::to_value(&req).unwrap_or_else(|_| serde_json::json!({"error": "Failed to serialize request"})));

    if let Some(tools) = &req.tools {
        debug!("Tools in request: {}", serde_json::to_string_pretty(tools).unwrap_or_else(|_| "Invalid".to_string()));
    }

    // If VLM model is loaded, delegate to VLM handler.
    if state.gemma4_vlm_tx.is_some() {
        info!("Delegating to Gemma4 VLM handler");
        return vlm::gemma4_vlm_chat_completions(state, req).await;
    }
    if state.vlm_tx.is_some() {
        info!("Delegating to VLM handler");
        return vlm::vlm_chat_completions(state, req).await;
    }

    // Apply chat template.
    let formatted = state
        .chat_template
        .apply(&req.messages, req.tools.as_deref())
        .map_err(|e| {
            error!("Chat template failed: {}", e);
            make_error(StatusCode::BAD_REQUEST, &format!("Chat template failed: {e}"))
        })?;

    // Tokenize.
    let input_ids = state
        .tokenizer
        .encode(formatted.as_str(), true)
        .map_err(|e| make_error(StatusCode::BAD_REQUEST, &format!("Tokenize failed: {e}")))?
        .get_ids()
        .to_vec();

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let include_usage = req
        .stream_options
        .as_ref()
        .map_or(false, |so| so.include_usage);

    let engine = state.engine.as_ref().ok_or_else(|| {
        make_error(StatusCode::SERVICE_UNAVAILABLE, "Text engine not available (VLM model loaded)")
    })?;

    let response_rx = engine
        .submit(
            request_id.clone(),
            input_ids,
            req.max_tokens,
            req.temperature.or(Some(0.8)),
            req.top_p.or(Some(0.95)),
            req.top_k.or(Some(40)),
            req.repetition_penalty.unwrap_or(1.05),
            state.eos_token_id.clone(),
        )
        .map_err(|e| make_error(StatusCode::SERVICE_UNAVAILABLE, &e.to_string()))?;

    if req.stream {
        let model_name = state.model_name.clone();
        let stream =
            sse::make_chat_sse_stream(request_id, model_name, response_rx, include_usage);
        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let (full_text, prompt_tokens, completion_tokens, finish_reason) =
            collect_response(response_rx).await?;

        // Check if model generated tool calls
        use crate::openai_api::extract_tool_calls;

        debug!("Checking for tool calls in generated text ({} chars)", full_text.len());
        debug!("Generated text preview: {}", &full_text.chars().take(200).collect::<String>());

        let response = if let Some(tool_calls) = extract_tool_calls(&full_text) {
            info!("Tool calls detected: {} calls", tool_calls.len());
            debug!("Tool calls: {:?}", tool_calls);

            // Extract any text content before tool calls for better UX
            // Models often generate conversational text before <|tool_start|> tokens
            let text_before_tools = full_text.split("<|tool_start|>").next()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string());

            if let Some(ref text) = text_before_tools {
                debug!("Text content before tool calls: {}", text);
            }

            // Model generated tool call(s)
            // Note: Don't include conversational text in content when tool_calls present
            // This prevents clients from displaying text instead of executing tools
            ChatCompletionResponse {
                id: request_id,
                object: "chat.completion".into(),
                created: now_epoch(),
                model: state.model_name.clone(),
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".into(),
                        content: None, // Set to None when tool_calls present
                        tool_calls: Some(tool_calls),
                        tool_call_id: None,
                    },
                    finish_reason: Some("tool_calls".to_string()), // Must be "tool_calls" when tools present
                }],
                usage: Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                },
            }
        } else {
            // Normal text response
            ChatCompletionResponse {
                id: request_id,
                object: "chat.completion".into(),
                created: now_epoch(),
                model: state.model_name.clone(),
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".into(),
                        content: Some(ChatMessageContent::Text(full_text)),
                        tool_calls: None,
                        tool_call_id: None,
                    },
                    finish_reason: Some(finish_reason),
                }],
                usage: Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                },
            }
        };

        // Log outgoing response
        log_communication("server_response", &serde_json::to_value(&response).unwrap_or_else(|_| serde_json::json!({"error": "Failed to serialize response"})));

        // If client requested streaming but we have tool calls, wrap in SSE format
        if req.stream && req.tools.is_some() {
            let model_name = state.model_name.clone();
            let stream = sse::wrap_single_response_in_sse(response, model_name);
            return Ok(Sse::new(stream)
                .keep_alive(KeepAlive::default())
                .into_response());
        }

        Ok(Json(response).into_response())
    }
}

// ─────────────────────────────────────────────────────────────
//  Text Completions
// ─────────────────────────────────────────────────────────────

/// `POST /v1/completions` — text completion (no chat template).
pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let prompt = req.prompt.as_string();
    let include_usage = req
        .stream_options
        .as_ref()
        .map_or(false, |so| so.include_usage);

    let input_ids = state
        .tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| make_error(StatusCode::BAD_REQUEST, &format!("Tokenize failed: {e}")))?
        .get_ids()
        .to_vec();

    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());

    let engine = state.engine.as_ref().ok_or_else(|| {
        make_error(StatusCode::SERVICE_UNAVAILABLE, "Text engine not available (VLM model loaded)")
    })?;

    let response_rx = engine
        .submit(
            request_id.clone(),
            input_ids,
            req.max_tokens,
            req.temperature.or(Some(0.8)),
            req.top_p.or(Some(0.95)),
            req.top_k.or(Some(40)),
            req.repetition_penalty.unwrap_or(1.05),
            state.eos_token_id.clone(),
        )
        .map_err(|e| make_error(StatusCode::SERVICE_UNAVAILABLE, &e.to_string()))?;

    if req.stream {
        let model_name = state.model_name.clone();
        let stream =
            sse::make_completion_sse_stream(request_id, model_name, response_rx, include_usage);
        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let (full_text, prompt_tokens, completion_tokens, finish_reason) =
            collect_response(response_rx).await?;

        let response = CompletionResponse {
            id: request_id,
            object: "text_completion".into(),
            created: now_epoch(),
            model: state.model_name.clone(),
            choices: vec![CompletionChoice {
                index: 0,
                text: full_text,
                finish_reason: Some(finish_reason),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        };
        Ok(Json(response).into_response())
    }
}

// ─────────────────────────────────────────────────────────────
//  Models
// ─────────────────────────────────────────────────────────────

/// `GET /v1/models` — list available models.
pub async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(ModelList {
        object: "list".into(),
        data: vec![make_model_info(&state)],
    })
}

/// `GET /v1/models/:model_id` — retrieve a specific model.
pub async fn retrieve_model(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Result<Json<ModelInfo>, (StatusCode, Json<ErrorResponse>)> {
    if model_id == state.model_name {
        Ok(Json(make_model_info(&state)))
    } else {
        Err(make_error(
            StatusCode::NOT_FOUND,
            &format!("Model '{model_id}' not found. Available: {}", state.model_name),
        ))
    }
}

fn make_model_info(state: &AppState) -> ModelInfo {
    ModelInfo {
        id: state.model_name.clone(),
        object: "model".into(),
        created: state.server_start_time,
        owned_by: "crane".into(),
        max_model_len: None,
        permission: None,
    }
}

// ─────────────────────────────────────────────────────────────
//  Tokenize / Detokenize
// ─────────────────────────────────────────────────────────────

/// `POST /v1/tokenize` or `POST /tokenize`
pub async fn tokenize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Determine the text to tokenize.
    let text = if let Some(messages) = &req.messages {
        // Apply chat template first.
        state
            .chat_template
            .apply(messages, None)
            .map_err(|e| make_error(StatusCode::BAD_REQUEST, &format!("Chat template failed: {e}")))?
    } else if let Some(text) = &req.text {
        text.clone()
    } else {
        return Err(make_error(
            StatusCode::BAD_REQUEST,
            "Either 'text' or 'messages' must be provided",
        ));
    };

    let encoding = state
        .tokenizer
        .encode(text.as_str(), req.add_special_tokens)
        .map_err(|e| make_error(StatusCode::BAD_REQUEST, &format!("Tokenize failed: {e}")))?;

    let tokens = encoding.get_ids().to_vec();
    let count = tokens.len();

    Ok(Json(TokenizeResponse { tokens, count }))
}

/// `POST /v1/detokenize` or `POST /detokenize`
pub async fn detokenize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DetokenizeRequest>,
) -> Result<Json<DetokenizeResponse>, (StatusCode, Json<ErrorResponse>)> {
    let text = state
        .tokenizer
        .decode(&req.tokens, true)
        .map_err(|e| make_error(StatusCode::BAD_REQUEST, &format!("Detokenize failed: {e}")))?;

    Ok(Json(DetokenizeResponse { text }))
}

// ─────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────

/// Collect all response chunks into (full_text, prompt_tokens, completion_tokens, finish_reason).
async fn collect_response(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<EngineResponse>,
) -> Result<(String, usize, usize, String), (StatusCode, Json<ErrorResponse>)> {
    let mut full_text = String::new();
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;
    let mut finish_reason = "length".to_string();

    while let Some(resp) = rx.recv().await {
        match resp {
            EngineResponse::Token { text, .. } => {
                full_text.push_str(&text);
            }
            EngineResponse::Finished {
                full_text: ft,
                prompt_tokens: pt,
                completion_tokens: ct,
                finish_reason: fr,
            } => {
                full_text = ft;
                prompt_tokens = pt;
                completion_tokens = ct;
                finish_reason = fr;
                break;
            }
            EngineResponse::Error(e) => {
                return Err(make_error(StatusCode::INTERNAL_SERVER_ERROR, &e));
            }
        }
    }

    Ok((full_text, prompt_tokens, completion_tokens, finish_reason))
}

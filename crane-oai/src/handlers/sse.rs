//! SSE stream builders for OpenAI-compatible and native streaming.

use std::convert::Infallible;

use axum::response::sse::Event;
use futures::stream::{self, Stream};
use tokio::sync::mpsc;

use crate::engine::EngineResponse;
use crate::openai_api::*;
use crate::sglang_api::*;
use crate::now_epoch;
use crate::tool_call_wrapper::ToolCallParser;
use serde::Serialize;
use std::iter::Iterator;

// ─────────────────────────────────────────────────────────────
//  Chat completions SSE
// ─────────────────────────────────────────────────────────────

pub fn make_chat_sse_stream(
    request_id: String,
    model_name: String,
    mut rx: mpsc::UnboundedReceiver<EngineResponse>,
    include_usage: bool,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let created = now_epoch();

    async_stream::stream! {
        // Role announcement chunk.
        let first_chunk = ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk".into(),
            created,
            model: model_name.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some("assistant".into()),
                    content: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };
        yield Ok(Event::default().json_data(&first_chunk).unwrap());

        let mut _prompt_tokens = 0usize;
        let mut _completion_tokens = 0usize;
        let mut tool_parser = ToolCallParser::new();
        let mut tool_calls_sent = false;

        while let Some(resp) = rx.recv().await {
            match resp {
                EngineResponse::Token { text, .. } => {
                    _completion_tokens += 1;

                    // Use tool parser to handle streaming with special token filtering
                    match tool_parser.parse_incremental(&text, &[]) {
                        Ok((normal_text, tool_calls)) => {
                            // Stream normal text (special tokens filtered)
                            if !normal_text.is_empty() {
                                let chunk = ChatCompletionChunk {
                                    id: request_id.clone(),
                                    object: "chat.completion.chunk".into(),
                                    created,
                                    model: model_name.clone(),
                                    choices: vec![ChunkChoice {
                                        index: 0,
                                        delta: ChunkDelta {
                                            role: None,
                                            content: Some(normal_text),
                                            tool_calls: None,
                                        },
                                        finish_reason: None,
                                    }],
                                    usage: None,
                                };
                                yield Ok(Event::default().json_data(&chunk).unwrap());
                            }

                            // Stream tool calls if available (but don't send finish_reason yet)
                            if !tool_calls.is_empty() {
                                tool_calls_sent = true;
                                for tool_call in tool_calls {
                                    let chunk = ChatCompletionChunk {
                                        id: request_id.clone(),
                                        object: "chat.completion.chunk".into(),
                                        created,
                                        model: model_name.clone(),
                                        choices: vec![ChunkChoice {
                                            index: 0,
                                            delta: ChunkDelta {
                                                role: None,
                                                content: None,
                                                tool_calls: Some(vec![tool_call]),
                                            },
                                            finish_reason: None,
                                        }],
                                        usage: None,
                                    };
                                    yield Ok(Event::default().json_data(&chunk).unwrap());
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!("Tool parsing error: {}", e);
                            // Continue streaming even if parsing fails
                        }
                    }
                }
                EngineResponse::Finished {
                    finish_reason,
                    prompt_tokens: pt,
                    completion_tokens: ct,
                    ..
                } => {
                    _prompt_tokens = pt;
                    _completion_tokens = ct;

                    // Check if tool parser has any remaining tool calls
                    let tool_calls = tool_parser.get_tool_calls();

                    // Determine finish reason and whether to send tool calls
                    let actual_finish_reason = if !tool_calls.is_empty() {
                        "tool_calls"
                    } else {
                        &finish_reason
                    };

                    if !tool_calls.is_empty() && !tool_calls_sent {
                        // Send tool calls in SSE format (only if not already sent during streaming)
                        for tool_call in tool_calls {
                            let chunk = ChatCompletionChunk {
                                id: request_id.clone(),
                                object: "chat.completion.chunk".into(),
                                created,
                                model: model_name.clone(),
                                choices: vec![ChunkChoice {
                                    index: 0,
                                    delta: ChunkDelta {
                                        role: None,
                                        content: None,
                                        tool_calls: Some(vec![tool_call.clone()]),
                                    },
                                    finish_reason: None,
                                }],
                                usage: None,
                            };
                            yield Ok(Event::default().json_data(&chunk).unwrap());
                        }
                    }

                    // Send final chunk with finish_reason
                    let finish_chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: None,
                                tool_calls: None,
                            },
                            finish_reason: Some(actual_finish_reason.to_string()),
                        }],
                        usage: None,
                    };
                    yield Ok(Event::default().json_data(&finish_chunk).unwrap());

                    if include_usage {
                        let usage_chunk = ChatCompletionChunk {
                            id: request_id.clone(),
                            object: "chat.completion.chunk".into(),
                            created,
                            model: model_name.clone(),
                            choices: vec![],
                            usage: Some(Usage {
                                prompt_tokens: _prompt_tokens,
                                completion_tokens: _completion_tokens,
                                total_tokens: _prompt_tokens + _completion_tokens,
                            }),
                        };
                        yield Ok(Event::default().json_data(&usage_chunk).unwrap());
                    }

                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                EngineResponse::Error(e) => {
                    yield Ok(Event::default().data(format!("error: {e}")));
                    break;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Text completions SSE
// ─────────────────────────────────────────────────────────────

pub fn make_completion_sse_stream(
    request_id: String,
    model_name: String,
    mut rx: mpsc::UnboundedReceiver<EngineResponse>,
    include_usage: bool,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let created = now_epoch();

    async_stream::stream! {
        let mut _prompt_tokens = 0usize;
        let mut _completion_tokens = 0usize;

        while let Some(resp) = rx.recv().await {
            match resp {
                EngineResponse::Token { text, .. } => {
                    _completion_tokens += 1;
                    let chunk = CompletionChunk {
                        id: request_id.clone(),
                        object: "text_completion".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![CompletionChunkChoice {
                            index: 0,
                            text,
                            finish_reason: None,
                        }],
                        usage: None,
                    };
                    yield Ok(Event::default().json_data(&chunk).unwrap());
                }
                EngineResponse::Finished {
                    finish_reason,
                    prompt_tokens: pt,
                    completion_tokens: ct,
                    ..
                } => {
                    _prompt_tokens = pt;
                    _completion_tokens = ct;

                    let chunk = CompletionChunk {
                        id: request_id.clone(),
                        object: "text_completion".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![CompletionChunkChoice {
                            index: 0,
                            text: String::new(),
                            finish_reason: Some(finish_reason),
                        }],
                        usage: None,
                    };
                    yield Ok(Event::default().json_data(&chunk).unwrap());

                    if include_usage {
                        let usage_chunk = CompletionChunk {
                            id: request_id.clone(),
                            object: "text_completion".into(),
                            created,
                            model: model_name.clone(),
                            choices: vec![],
                            usage: Some(Usage {
                                prompt_tokens: _prompt_tokens,
                                completion_tokens: _completion_tokens,
                                total_tokens: _prompt_tokens + _completion_tokens,
                            }),
                        };
                        yield Ok(Event::default().json_data(&usage_chunk).unwrap());
                    }

                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                EngineResponse::Error(e) => {
                    yield Ok(Event::default().data(format!("error: {e}")));
                    break;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Native /generate SSE
// ─────────────────────────────────────────────────────────────

pub fn make_generate_sse_stream(
    request_id: String,
    mut rx: mpsc::UnboundedReceiver<EngineResponse>,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        while let Some(resp) = rx.recv().await {
            match resp {
                EngineResponse::Token { text, .. } => {
                    let chunk = GenerateStreamChunk {
                        text,
                        meta_info: None,
                    };
                    yield Ok(Event::default().json_data(&chunk).unwrap());
                }
                EngineResponse::Finished {
                    prompt_tokens,
                    completion_tokens,
                    finish_reason,
                    ..
                } => {
                    let chunk = GenerateStreamChunk {
                        text: String::new(),
                        meta_info: Some(GenerateMetaInfo {
                            id: request_id.clone(),
                            prompt_tokens,
                            completion_tokens,
                            finish_reason,
                        }),
                    };
                    yield Ok(Event::default().json_data(&chunk).unwrap());
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                EngineResponse::Error(e) => {
                    yield Ok(Event::default().data(format!("error: {e}")));
                    break;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  SSE Wrapper for Non-Streaming Responses
// ─────────────────────────────────────────────────────────────

/// Wrap a single ChatCompletionResponse in SSE format for clients requesting streaming.
/// This is needed when we disable streaming for tool calls but the client expects SSE.
pub fn wrap_single_response_in_sse(
    response: ChatCompletionResponse,
    model_name: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    // Convert response to streaming chunk format
    let chunk = ChatCompletionChunk {
        id: response.id.clone(),
        object: "chat.completion.chunk".to_string(),
        created: response.created,
        model: model_name,
        choices: vec![ChunkChoice {
            index: 0,
            delta: ChunkDelta {
                role: Some("assistant".to_string()),
                content: response.choices[0].message.content.as_ref().and_then(|c| {
                    if let ChatMessageContent::Text(s) = c {
                        Some(s.clone())
                    } else {
                        None
                    }
                }),
                tool_calls: response.choices[0].message.tool_calls.clone(),
            },
            finish_reason: response.choices[0].finish_reason.clone(),
        }],
        usage: None, // Usage sent in separate chunk if needed
    };

    // Create the main chunk event
    let chunk_event = Event::default()
        .json_data(&chunk)
        .unwrap_or_else(|_| Event::default().data("error serializing response"));

    // Create the [DONE] event
    let done_event = Event::default().data("[DONE]");

    // Return a stream with both events
    stream::iter(vec![chunk_event, done_event].into_iter().map(Ok))
}

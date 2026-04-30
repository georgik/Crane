use log::{debug, error, info, warn};
use std::env;
use std::time::{Duration, SystemTime};

fn format_time(time: SystemTime) -> String {
    use std::time::UNIX_EPOCH;
    match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => {
            let secs = duration.as_secs();
            let datetime = chrono::DateTime::from_timestamp(secs as i64, 0)
                .unwrap_or_else(|| chrono::DateTime::from_timestamp(0, 0).unwrap());
            datetime.format("%Y-%m-%d %H:%M:%S").to_string()
        }
        Err(_) => "Invalid time".to_string(),
    }
}

fn parse_timeout(args: &[String]) -> Duration {
    const DEFAULT_TIMEOUT_SECS: u64 = 120;

    // Check CLI args first, then env var, then default
    let timeout_str = args
        .iter()
        .find(|a| a.starts_with("--timeout="))
        .map(|s| s.trim_start_matches("--timeout=").to_string())
        .or_else(|| env::var("XTASK_TIMEOUT").ok())
        .unwrap_or_else(|| DEFAULT_TIMEOUT_SECS.to_string());

    timeout_str
        .parse::<u64>()
        .map(Duration::from_secs)
        .unwrap_or_else(|_| Duration::from_secs(DEFAULT_TIMEOUT_SECS))
}

fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        println!("Usage: cargo xtask <command> [model_name] [--timeout=SECONDS]");
        println!("Commands:");
        println!("  test-tools [model]  - Test tool calling with crane-oai");
        println!("  test-shell [model]  - Test shell command tool (ls case)");
        println!("  test-goose-shell [model] - Test with Goose's developer__shell tool");
        println!("  test-chat [model]   - Test basic chat completion");
        println!("Options:");
        println!(
            "  --timeout=SECONDS   - Request timeout in seconds (default: 120, env: XTASK_TIMEOUT)"
        );
        println!("\nIf model_name is not provided, will auto-detect from server.");
        return;
    }

    let timeout = parse_timeout(&args);
    info!("Request timeout: {} seconds", timeout.as_secs());

    let model = args.get(2).map(|s| s.as_str());

    match args[1].as_str() {
        "test-tools" => test_tools(model, timeout),
        "test-shell" => test_shell(model, timeout),
        "test-goose-shell" => test_goose_shell(model, timeout),
        "test-chat" => test_chat(model, timeout),
        _ => println!("Unknown command: {}", args[1]),
    }
}

fn get_model_name() -> Option<String> {
    let client = reqwest::blocking::Client::new();
    debug!("Fetching available models from server...");
    match client.get("http://localhost:8080/v1/models").send() {
        Ok(response) => {
            if response.status().is_success() {
                if let Ok(text) = response.text() {
                    debug!("Models response: {}", text);
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                        if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
                            if let Some(first_model) = data.first() {
                                if let Some(id) = first_model.get("id").and_then(|i| i.as_str()) {
                                    info!("Auto-detected model: {}", id);
                                    return Some(id.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }
        Err(e) => {
            debug!("Failed to fetch models: {}", e);
        }
    }
    warn!("Could not auto-detect model");
    None
}

fn test_tools(model: Option<&str>, timeout: Duration) {
    info!("Testing tool calling...");

    let start = std::time::Instant::now();

    let model_name = model.map(String::from).unwrap_or_else(|| {
        info!("Auto-detecting model...");
        match get_model_name() {
            Some(name) => name,
            None => {
                warn!("Could not auto-detect model, using default");
                "unknown".to_string()
            }
        }
    });

    let client = reqwest::blocking::Client::builder()
        .timeout(timeout)
        .build()
        .unwrap();
    let url = "http://localhost:8080/v1/chat/completions";

    let payload = serde_json::json!({
        "model": model_name,
        "messages": [
            {"role": "user", "content": "What is the current time in UTC? Use the get_time tool."}
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
    });

    info!("Sending request to: {}", url);
    info!("Timeout: {} seconds", timeout.as_secs());
    debug!(
        "Payload: {}",
        serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "Invalid JSON".to_string())
    );
    info!(
        "Request started at: {}",
        format_time(std::time::SystemTime::now())
    );

    match client.post(url).json(&payload).send() {
        Ok(response) => {
            let status = response.status();
            let duration = start.elapsed();

            info!(
                "Response received in {:.2}s - Status: {}",
                duration.as_secs_f64(),
                status
            );

            if status.is_success() {
                if let Ok(text) = response.text() {
                    debug!("Response body: {}", text);
                    info!(
                        "Response: {}",
                        serde_json::to_string_pretty(
                            &serde_json::from_str::<serde_json::Value>(&text)
                                .unwrap_or_else(|_| serde_json::json!({"raw": text}))
                        )
                        .unwrap_or_else(|_| text.clone())
                    );

                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                        // Enhanced diagnostics for Goose compatibility
                        if let Some(choice) = json["choices"].get(0) {
                            if let Some(message) = choice.get("message") {
                                if let Some(tool_calls) = message.get("tool_calls") {
                                    info!(
                                        "✓ Tool calls detected! Count: {}",
                                        tool_calls.as_array().map(|v| v.len()).unwrap_or(0)
                                    );
                                    info!(
                                        "Tool calls: {}",
                                        serde_json::to_string_pretty(tool_calls).unwrap()
                                    );

                                    // Validate Goose compatibility
                                    if let Some(calls) = tool_calls.as_array() {
                                        for (i, call) in calls.iter().enumerate() {
                                            info!("Tool call #{}:", i);
                                            if let Some(fn_obj) = call.get("function") {
                                                let name = fn_obj
                                                    .get("name")
                                                    .and_then(|n| n.as_str())
                                                    .unwrap_or("MISSING");
                                                let args = fn_obj
                                                    .get("arguments")
                                                    .unwrap_or(&serde_json::json!(null));
                                                info!("  name: {}", name);
                                                info!(
                                                    "  arguments type: {}",
                                                    match args {
                                                        serde_json::Value::String(_) => "string ✓",
                                                        serde_json::Value::Object(_) =>
                                                            "object (should be string)",
                                                        serde_json::Value::Null => "null",
                                                        _ => "other",
                                                    }
                                                );

                                                // Convert to owned String to avoid temporary value issue
                                                let args_display = args
                                                    .as_str()
                                                    .map(|s| s.to_string())
                                                    .unwrap_or_else(|| {
                                                        serde_json::to_string(args)
                                                            .unwrap_or_else(|_| "null".to_string())
                                                    });
                                                info!("  arguments value: {}", args_display);

                                                // Check if arguments are properly stringified (Goose requirement)
                                                if let Some(args_str) = args.as_str() {
                                                    // Verify it's valid JSON
                                                    match serde_json::from_str::<serde_json::Value>(
                                                        args_str,
                                                    ) {
                                                        Ok(_) => {
                                                            info!("  ✓ Arguments valid JSON string")
                                                        }
                                                        Err(e) => warn!(
                                                            "  ✗ Arguments invalid JSON: {}",
                                                            e
                                                        ),
                                                    }
                                                } else if args.is_object() {
                                                    warn!("  ⚠ Arguments as object - Goose expects string!");
                                                }
                                            }
                                            info!(
                                                "  type: {}",
                                                call.get("type")
                                                    .and_then(|t| t.as_str())
                                                    .unwrap_or("MISSING")
                                            );
                                            info!(
                                                "  id: {}",
                                                call.get("id")
                                                    .and_then(|i| i.as_str())
                                                    .unwrap_or("MISSING")
                                            );
                                        }
                                    }
                                } else if let Some(content) = message.get("content") {
                                    warn!("✗ No tool calls - model generated text instead");
                                    info!(
                                        "Content preview: {}",
                                        &content
                                            .as_str()
                                            .unwrap_or("")
                                            .chars()
                                            .take(200)
                                            .collect::<String>()
                                    );

                                    // Check if model tried to use tool tokens
                                    let content_str = content.as_str().unwrap_or("");
                                    if content_str.contains("<|tool_start|>")
                                        || content_str.contains("<|tool_call|>")
                                    {
                                        info!("⚠ Model output contains tool tokens but wasn't parsed!");
                                    }
                                }

                                // Check for finish_reason
                                if let Some(finish_reason) =
                                    choice.get("finish_reason").and_then(|f| f.as_str())
                                {
                                    info!("Finish reason: {}", finish_reason);
                                }
                            }
                        }

                        // Show usage if available
                        if let Some(usage) = json.get("usage") {
                            info!(
                                "Usage: prompt_tokens={}, completion_tokens={}, total_tokens={}",
                                usage
                                    .get("prompt_tokens")
                                    .and_then(|v| v.as_i64())
                                    .unwrap_or(0),
                                usage
                                    .get("completion_tokens")
                                    .and_then(|v| v.as_i64())
                                    .unwrap_or(0),
                                usage
                                    .get("total_tokens")
                                    .and_then(|v| v.as_i64())
                                    .unwrap_or(0)
                            );
                        }
                    }
                }
            } else {
                error!("Request failed with status: {}", status);
                if let Ok(text) = response.text() {
                    error!("Error response: {}", text);
                }
            }
        }
        Err(e) => {
            error!(
                "Request error after {:.2}s (timeout was {}s): {:?}",
                start.elapsed().as_secs_f64(),
                timeout.as_secs(),
                e
            );
            error!("Make sure crane-oai is running on {}", url);
            info!("Check server health: curl http://localhost:8080/health");
        }
    }
}

fn test_shell(model: Option<&str>, timeout: Duration) {
    info!("Testing shell command tool (ls case)...");

    let start = std::time::Instant::now();

    let model_name = model.map(String::from).unwrap_or_else(|| {
        info!("Auto-detecting model...");
        match get_model_name() {
            Some(name) => name,
            None => {
                warn!("Could not auto-detect model, using default");
                "unknown".to_string()
            }
        }
    });

    let client = reqwest::blocking::Client::builder()
        .timeout(timeout)
        .build()
        .unwrap();
    let url = "http://localhost:8080/v1/chat/completions";

    let payload = serde_json::json!({
        "model": model_name,
        "messages": [
            {"role": "user", "content": "List files in local directory calling ls command"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "shell",
                    "description": "Execute shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Shell command to execute"
                            }
                        },
                        "required": ["command"]
                    }
                }
            }
        ]
    });

    info!("Sending request to: {}", url);
    info!("Timeout: {} seconds", timeout.as_secs());
    debug!(
        "Payload: {}",
        serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "Invalid JSON".to_string())
    );
    info!(
        "Request started at: {}",
        format_time(std::time::SystemTime::now())
    );

    match client.post(url).json(&payload).send() {
        Ok(response) => {
            let status = response.status();
            let duration = start.elapsed();

            info!(
                "Response received in {:.2}s - Status: {}",
                duration.as_secs_f64(),
                status
            );

            if status.is_success() {
                if let Ok(text) = response.text() {
                    debug!("Response body: {}", text);
                    info!(
                        "Response: {}",
                        serde_json::to_string_pretty(
                            &serde_json::from_str::<serde_json::Value>(&text)
                                .unwrap_or_else(|_| serde_json::json!({"raw": text}))
                        )
                        .unwrap_or_else(|_| text.clone())
                    );

                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                        // Same enhanced diagnostics as test-tools
                        if let Some(choice) = json["choices"].get(0) {
                            if let Some(message) = choice.get("message") {
                                if let Some(tool_calls) = message.get("tool_calls") {
                                    info!("✓ Shell tool call detected!");
                                    info!(
                                        "Tool calls: {}",
                                        serde_json::to_string_pretty(tool_calls).unwrap()
                                    );

                                    // Validate the shell command structure
                                    if let Some(calls) = tool_calls.as_array() {
                                        for call in calls.iter() {
                                            if let Some(fn_obj) = call.get("function") {
                                                let name = fn_obj
                                                    .get("name")
                                                    .and_then(|n| n.as_str())
                                                    .unwrap_or("MISSING");
                                                if name == "shell" {
                                                    info!("✓ Correct tool name: shell");
                                                    if let Some(args) = fn_obj
                                                        .get("arguments")
                                                        .and_then(|a| a.as_str())
                                                    {
                                                        match serde_json::from_str::<
                                                            serde_json::Value,
                                                        >(
                                                            args
                                                        ) {
                                                            Ok(parsed_args) => {
                                                                if let Some(cmd) = parsed_args
                                                                    .get("command")
                                                                    .and_then(|c| c.as_str())
                                                                {
                                                                    info!(
                                                                        "✓ Shell command: {}",
                                                                        cmd
                                                                    );
                                                                    if cmd.contains("ls") {
                                                                        info!("✓ Command contains 'ls' as expected");
                                                                    }
                                                                } else {
                                                                    warn!("✗ Missing 'command' in arguments");
                                                                }
                                                            }
                                                            Err(e) => {
                                                                warn!("✗ Failed to parse arguments: {}", e);
                                                            }
                                                        }
                                                    }
                                                } else {
                                                    warn!(
                                                        "✗ Wrong tool name: {}, expected 'shell'",
                                                        name
                                                    );
                                                }
                                            }
                                        }
                                    }
                                } else if let Some(content) = message.get("content") {
                                    warn!("✗ No tool calls - model generated text instead");
                                    info!("Content: {}", content.as_str().unwrap_or(""));
                                }
                            }
                        }
                    }
                }
            } else {
                error!("Request failed with status: {}", status);
                if let Ok(text) = response.text() {
                    error!("Error response: {}", text);
                }
            }
        }
        Err(e) => {
            error!(
                "Request error after {:.2}s (timeout was {}s): {:?}",
                start.elapsed().as_secs_f64(),
                timeout.as_secs(),
                e
            );
            error!("Make sure crane-oai is running on {}", url);
        }
    }
}

fn test_goose_shell(model: Option<&str>, timeout: Duration) {
    info!("Testing with Goose's developer__shell tool format...");

    let start = std::time::Instant::now();

    let model_name = model.map(String::from).unwrap_or_else(|| {
        info!("Auto-detecting model...");
        match get_model_name() {
            Some(name) => name,
            None => {
                warn!("Could not auto-detect model, using default");
                "unknown".to_string()
            }
        }
    });

    let client = reqwest::blocking::Client::builder()
        .timeout(timeout)
        .build()
        .unwrap();
    let url = "http://localhost:8080/v1/chat/completions";

    // Use Goose's actual tool name and format
    let payload = serde_json::json!({
        "model": model_name,
        "messages": [
            {"role": "user", "content": "List files in the current directory"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "developer__shell",
                    "description": "Execute shell commands in the developer's working directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The shell command to execute"
                            }
                        },
                        "required": ["command"]
                    }
                }
            }
        ]
    });

    info!("Sending request to: {}", url);
    info!("Timeout: {} seconds", timeout.as_secs());
    debug!(
        "Payload: {}",
        serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "Invalid JSON".to_string())
    );
    info!(
        "Request started at: {}",
        format_time(std::time::SystemTime::now())
    );

    match client.post(url).json(&payload).send() {
        Ok(response) => {
            let status = response.status();
            let duration = start.elapsed();

            info!(
                "Response received in {:.2}s - Status: {}",
                duration.as_secs_f64(),
                status
            );

            if status.is_success() {
                if let Ok(text) = response.text() {
                    debug!("Response body: {}", text);

                    // Parse and analyze response
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                        info!("Response structure analysis:");

                        // Check content field
                        if let Some(choice) = json.get("choices").and_then(|c| c.get(0)) {
                            if let Some(message) = choice.get("message") {
                                // Check content field
                                if let Some(content) = message.get("content") {
                                    let content_str = content.as_str().unwrap_or("");
                                    info!(
                                        "Content field: {}",
                                        if content_str.is_empty() {
                                            "empty (null)".to_string()
                                        } else {
                                            format!("{} chars", content_str.len())
                                        }
                                    );
                                    if !content_str.is_empty() {
                                        info!(
                                            "Content preview: {}",
                                            &content_str.chars().take(200).collect::<String>()
                                        );
                                    }
                                } else {
                                    info!("Content field: null");
                                }

                                // Check tool_calls
                                if let Some(tool_calls) = message.get("tool_calls") {
                                    info!("✓ tool_calls field present");
                                    if let Some(calls) = tool_calls.as_array() {
                                        info!("Tool call count: {}", calls.len());
                                        for (i, call) in calls.iter().enumerate() {
                                            info!("Tool call #{}:", i);
                                            if let Some(fn_obj) = call.get("function") {
                                                let name = fn_obj
                                                    .get("name")
                                                    .and_then(|n| n.as_str())
                                                    .unwrap_or("MISSING");
                                                info!("  Function name: {}", name);
                                                if name == "developer__shell" {
                                                    info!("  ✓ Correct Goose tool name!");
                                                } else if name == "shell" {
                                                    info!("  ⚠ Wrong tool name: 'shell' instead of 'developer__shell'");
                                                    info!("  This is why Goose won't execute it!");
                                                } else {
                                                    info!("  ✗ Unknown tool name");
                                                }

                                                let args = fn_obj
                                                    .get("arguments")
                                                    .unwrap_or(&serde_json::json!(null));
                                                info!("  Arguments: {}", args);
                                            }
                                            info!(
                                                "  Type: {}",
                                                call.get("type")
                                                    .and_then(|t| t.as_str())
                                                    .unwrap_or("MISSING")
                                            );
                                            info!(
                                                "  ID: {}",
                                                call.get("id")
                                                    .and_then(|i| i.as_str())
                                                    .unwrap_or("MISSING")
                                            );
                                        }
                                    } else {
                                        warn!("✗ tool_calls is not an array");
                                    }
                                } else {
                                    warn!("✗ No tool_calls field in response");
                                }

                                // Check finish_reason
                                if let Some(finish_reason) = choice.get("finish_reason") {
                                    info!("Finish reason: {}", finish_reason);
                                }
                            }
                        }

                        // Show usage if available
                        if let Some(usage) = json.get("usage") {
                            info!(
                                "Usage: prompt_tokens={}, completion_tokens={}, total_tokens={}",
                                usage
                                    .get("prompt_tokens")
                                    .and_then(|v| v.as_i64())
                                    .unwrap_or(0),
                                usage
                                    .get("completion_tokens")
                                    .and_then(|v| v.as_i64())
                                    .unwrap_or(0),
                                usage
                                    .get("total_tokens")
                                    .and_then(|v| v.as_i64())
                                    .unwrap_or(0)
                            );
                        }
                    }
                }
            } else {
                error!("Request failed with status: {}", status);
                if let Ok(text) = response.text() {
                    error!("Error response: {}", text);
                }
            }
        }
        Err(e) => {
            error!(
                "Request error after {:.2}s (timeout was {}s): {:?}",
                start.elapsed().as_secs_f64(),
                timeout.as_secs(),
                e
            );
            error!("Make sure crane-oai is running on {}", url);
        }
    }
}

fn test_chat(model: Option<&str>, timeout: Duration) {
    info!("Testing basic chat...");

    let start = std::time::Instant::now();

    let model_name = model.map(String::from).unwrap_or_else(|| {
        info!("Auto-detecting model...");
        match get_model_name() {
            Some(name) => name,
            None => {
                warn!("Could not auto-detect model, using default");
                "unknown".to_string()
            }
        }
    });

    let client = reqwest::blocking::Client::builder()
        .timeout(timeout)
        .build()
        .unwrap();
    let url = "http://localhost:8080/v1/chat/completions";

    let payload = serde_json::json!({
        "model": model_name,
        "messages": [
            {"role": "user", "content": "Hello! How are you?"}
        ]
    });

    info!("Sending request to: {}", url);
    info!("Timeout: {} seconds", timeout.as_secs());
    debug!(
        "Payload: {}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    info!(
        "Request started at: {}",
        format_time(std::time::SystemTime::now())
    );

    match client.post(url).json(&payload).send() {
        Ok(response) => {
            let status = response.status();
            let duration = start.elapsed();

            info!(
                "Response received in {:.2}s - Status: {}",
                duration.as_secs_f64(),
                status
            );

            if status.is_success() {
                if let Ok(text) = response.text() {
                    info!("Response body: {}", text);
                }
            } else {
                error!("Request failed with status: {}", status);
                if let Ok(text) = response.text() {
                    error!("Error response: {}", text);
                }
            }
        }
        Err(e) => {
            error!(
                "Request error after {:.2}s (timeout was {}s): {:?}",
                start.elapsed().as_secs_f64(),
                timeout.as_secs(),
                e
            );
            error!("Make sure crane-oai is running on {}", url);
            info!("Check server health: curl http://localhost:8080/health");
        }
    }
}

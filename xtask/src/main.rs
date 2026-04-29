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
                        if let Some(choice) = json["choices"].get(0) {
                            if let Some(message) = choice.get("message") {
                                if let Some(tool_calls) = message.get("tool_calls") {
                                    info!("Tool calls detected!");
                                    info!(
                                        "Tool calls: {}",
                                        serde_json::to_string_pretty(tool_calls).unwrap()
                                    );
                                } else if let Some(content) = message.get("content") {
                                    warn!("No tool calls - model generated text instead");
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
            info!("Check server health: curl http://localhost:8080/health");
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

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        println!("Usage: cargo xtask <command>");
        println!("Commands:");
        println!("  test-tools    - Test tool calling with crane-oai");
        println!("  test-chat     - Test basic chat completion");
        return;
    }

    match args[1].as_str() {
        "test-tools" => test_tools(),
        "test-chat" => test_chat(),
        _ => println!("Unknown command: {}", args[1]),
    }
}

fn test_tools() {
    println!("Testing tool calling...");

    let client = reqwest::blocking::Client::new();
    let url = "http://localhost:8080/v1/chat/completions";

    let payload = serde_json::json!({
        "model": "qwen25",
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

    match client.post(url).json(&payload).send() {
        Ok(response) => {
            if response.status().is_success() {
                if let Ok(text) = response.text() {
                    println!("Response:\n{}", &text);

                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                        if let Some(choice) = json["choices"].get(0) {
                            if let Some(message) = choice.get("message") {
                                if let Some(tool_calls) = message.get("tool_calls") {
                                    println!("\n✅ Tool calls detected!");
                                    println!(
                                        "Tool calls: {}",
                                        serde_json::to_string_pretty(tool_calls).unwrap()
                                    );
                                } else if let Some(content) = message.get("content") {
                                    println!("\n❌ No tool calls - model generated text instead");
                                    println!("Content: {}", content.as_str().unwrap_or(""));
                                }
                            }
                        }
                    }
                }
            } else {
                println!("❌ Request failed: {}", response.status());
            }
        }
        Err(e) => {
            println!("❌ Request error: {}", e);
        }
    }
}

fn test_chat() {
    println!("Testing basic chat...");

    let client = reqwest::blocking::Client::new();
    let url = "http://localhost:8080/v1/chat/completions";

    let payload = serde_json::json!({
        "model": "qwen25",
        "messages": [
            {"role": "user", "content": "Hello! How are you?"}
        ]
    });

    match client.post(url).json(&payload).send() {
        Ok(response) => {
            if response.status().is_success() {
                if let Ok(text) = response.text() {
                    println!("Response:\n{}", &text);
                }
            } else {
                println!("❌ Request failed: {}", response.status());
            }
        }
        Err(e) => {
            println!("❌ Request error: {}", e);
        }
    }
}

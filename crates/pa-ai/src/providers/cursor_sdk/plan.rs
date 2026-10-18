//! Decide whether a Cursor agent can continue or must be reseeded.
//!
//! The sidecar keeps its own transcript. A pure append of tool results
//! resolves parked `execute` calls. A new user message is sent on its own.
//! Anything else (compaction, a tool-set change, a shrunk transcript) replays
//! the harness context into a fresh agent.

use std::collections::HashMap;

use serde_json::Value;

use crate::types::{
    AssistantContent, AssistantMessage, Context, Message, ToolResultMessage, UserOrToolContent,
};

#[derive(Clone, Debug, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolResultSpec {
    pub id: String,
    pub text: String,
    pub is_error: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PendingTool {
    pub id: String,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Memory {
    pub fingerprint: String,
    pub tool_fingerprint: String,
    pub cwd: String,
    pub synced_messages: usize,
    pub pending: Vec<PendingTool>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Plan {
    Reseed {
        prompt: String,
        tools: Vec<ToolSpec>,
    },
    Send {
        prompt: String,
        tools: Vec<ToolSpec>,
    },
    Resolve {
        results: Vec<ToolResultSpec>,
    },
}

pub fn decide(memory: Option<&Memory>, context: &Context, cwd: &str) -> Plan {
    let tools = tool_specs(context);
    let reseed = || Plan::Reseed {
        prompt: render(context),
        tools: tools.clone(),
    };
    let Some(memory) = memory else {
        return reseed();
    };
    if memory.fingerprint != fingerprint(context)
        || memory.tool_fingerprint != tool_fingerprint(context)
        || memory.cwd != cwd
        || context.messages.len() < memory.synced_messages
    {
        return reseed();
    }
    if !memory.pending.is_empty() {
        if exact_resolution(&context.messages, memory.synced_messages, &memory.pending) {
            return Plan::Resolve {
                results: result_specs(&context.messages, &memory.pending),
            };
        }
        return reseed();
    }
    match new_user_text(&context.messages, memory.synced_messages) {
        Some(prompt) => Plan::Send { prompt, tools },
        None => reseed(),
    }
}

pub fn remember(context: &Context, cwd: &str, pending: Vec<PendingTool>) -> Memory {
    Memory {
        fingerprint: fingerprint(context),
        tool_fingerprint: tool_fingerprint(context),
        cwd: cwd.to_string(),
        synced_messages: context.messages.len(),
        pending,
    }
}

fn tool_specs(context: &Context) -> Vec<ToolSpec> {
    context
        .tools
        .iter()
        .flatten()
        .map(|tool| ToolSpec {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
        })
        .collect()
}

fn fingerprint(context: &Context) -> String {
    let system = context.system_prompt.as_deref().unwrap_or("");
    let first = context
        .messages
        .iter()
        .find_map(|message| match message {
            Message::User(user) => Some(user.content.text()),
            Message::Assistant(_) | Message::ToolResult(_) => None,
        })
        .unwrap_or_default();
    format!("{system}\n{first}")
}

fn tool_fingerprint(context: &Context) -> String {
    let specs: Vec<Value> = tool_specs(context)
        .iter()
        .map(|tool| {
            serde_json::json!({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            })
        })
        .collect();
    serde_json::to_string(&specs).unwrap_or_default()
}

fn render(context: &Context) -> String {
    let mut parts = Vec::new();
    if let Some(system) = context.system_prompt.as_deref() {
        if !system.is_empty() {
            parts.push(system.to_string());
        }
    }
    let mut saw_result = false;
    for message in &context.messages {
        match message {
            Message::User(user) => parts.push(user.content.text()),
            Message::Assistant(assistant) => parts.push(render_assistant(assistant)),
            Message::ToolResult(result) => {
                saw_result = true;
                parts.push(render_result(result));
            }
        }
    }
    if saw_result {
        parts.push("The tool results above are already complete. Continue from them.".to_string());
    }
    let prompt = parts
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    if prompt.is_empty() {
        "Continue.".to_string()
    } else {
        prompt
    }
}

fn render_assistant(message: &AssistantMessage) -> String {
    let mut parts = Vec::new();
    for block in &message.content {
        match block {
            AssistantContent::Text(text) if !text.text.is_empty() => parts.push(text.text.clone()),
            AssistantContent::Text(_) | AssistantContent::Thinking(_) => {}
            AssistantContent::ToolCall(call) => {
                let args = Value::Object(call.arguments.clone());
                parts.push(format!("[tool call {} id={}]\n{args}", call.name, call.id));
            }
        }
    }
    parts.join("\n")
}

fn render_result(result: &ToolResultMessage) -> String {
    let body = block_text(&result.content);
    let status = if result.is_error { "error" } else { "ok" };
    format!(
        "[tool result {} id={} {status}]\n{body}",
        result.tool_name, result.tool_call_id
    )
}

fn block_text(blocks: &[UserOrToolContent]) -> String {
    let text = blocks
        .iter()
        .filter_map(UserOrToolContent::text)
        .collect::<Vec<_>>()
        .join("\n");
    if text.is_empty() {
        "[no text]".to_string()
    } else {
        text
    }
}

fn new_user_text(messages: &[Message], synced: usize) -> Option<String> {
    let new = messages.get(synced..)?;
    if new.is_empty() {
        return None;
    }
    let mut parts = Vec::new();
    for message in new {
        match message {
            Message::User(user) => parts.push(user.content.text()),
            Message::Assistant(_) | Message::ToolResult(_) => return None,
        }
    }
    let prompt = parts.join("\n\n");
    if prompt.is_empty() {
        None
    } else {
        Some(prompt)
    }
}

fn exact_resolution(messages: &[Message], synced: usize, pending: &[PendingTool]) -> bool {
    let Some(new) = messages.get(synced..) else {
        return false;
    };
    let mut results = Vec::new();
    for message in new {
        match message {
            Message::Assistant(_) if results.is_empty() => {}
            Message::ToolResult(result) => results.push(result.tool_call_id.clone()),
            Message::User(_) | Message::Assistant(_) => return false,
        }
    }
    let mut expected: Vec<String> = pending.iter().map(|tool| tool.id.clone()).collect();
    expected.sort();
    results.sort();
    !expected.is_empty() && results == expected
}

fn result_specs(messages: &[Message], pending: &[PendingTool]) -> Vec<ToolResultSpec> {
    let mut found = HashMap::new();
    for message in messages {
        if let Message::ToolResult(result) = message {
            found.insert(result.tool_call_id.clone(), result);
        }
    }
    pending
        .iter()
        .filter_map(|tool| {
            let result = found.get(&tool.id)?;
            Some(ToolResultSpec {
                id: tool.id.clone(),
                text: block_text(&result.content),
                is_error: result.is_error,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Map;

    use crate::types::{TextContent, Tool, ToolCall, ToolResultMessage, UserMessage};

    fn user(text: &str) -> Message {
        Message::User(UserMessage {
            content: crate::types::UserMessageContent::Text(text.to_string()),
            timestamp: 0,
            rest: Map::new(),
        })
    }

    fn context(messages: Vec<Message>) -> Context {
        Context {
            system_prompt: Some("system".to_string()),
            messages,
            tools: Some(vec![Tool {
                name: "bash".to_string(),
                description: "run".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            }]),
        }
    }

    fn tool_result(id: &str, text: &str, is_error: bool) -> Message {
        Message::ToolResult(ToolResultMessage {
            tool_call_id: id.to_string(),
            tool_name: "bash".to_string(),
            content: vec![UserOrToolContent::Text(TextContent {
                text: text.to_string(),
                text_signature: None,
                rest: Map::new(),
            })],
            details: None,
            is_error,
            timestamp: 0,
            rest: Map::new(),
        })
    }

    fn assistant_call(id: &str) -> Message {
        Message::Assistant(AssistantMessage {
            content: vec![AssistantContent::ToolCall(ToolCall {
                id: id.to_string(),
                name: "bash".to_string(),
                arguments: Map::new(),
                thought_signature: None,
                rest: Map::new(),
            })],
            api: "cursor-sdk".to_string(),
            provider: "cursor".to_string(),
            model: "composer-2.5".to_string(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: crate::types::Usage::default(),
            stop_reason: crate::types::StopReason::ToolUse,
            stop_reason_raw: None,
            error_message: None,
            timestamp: 0,
            rest: Map::new(),
        })
    }

    #[test]
    fn the_first_turn_reseeds_with_the_system_prompt_and_tools() {
        let context = context(vec![user("build it")]);
        let plan = decide(None, &context, "/tmp/work");
        match plan {
            Plan::Reseed { prompt, tools } => {
                assert!(prompt.contains("system"));
                assert!(prompt.contains("build it"));
                assert_eq!(tools.len(), 1);
                assert_eq!(tools[0].name, "bash");
            }
            Plan::Send { .. } | Plan::Resolve { .. } => panic!("expected reseed"),
        }
    }

    #[test]
    fn a_matching_tool_result_resolves_the_parked_call() {
        let first = context(vec![user("build it")]);
        let memory = remember(
            &first,
            "/tmp/work",
            vec![PendingTool {
                id: "t1".to_string(),
                name: "bash".to_string(),
            }],
        );
        let next = context(vec![
            user("build it"),
            assistant_call("t1"),
            tool_result("t1", "hi\n", false),
        ]);
        match decide(Some(&memory), &next, "/tmp/work") {
            Plan::Resolve { results } => {
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].id, "t1");
                assert_eq!(results[0].text, "hi\n");
                assert!(!results[0].is_error);
            }
            Plan::Reseed { .. } | Plan::Send { .. } => panic!("expected resolve"),
        }
    }

    #[test]
    fn a_later_user_message_is_sent_without_the_system_prompt() {
        let first = context(vec![user("build it")]);
        let memory = remember(&first, "/tmp/work", Vec::new());
        let next = context(vec![user("build it"), user("also add tests")]);
        match decide(Some(&memory), &next, "/tmp/work") {
            Plan::Send { prompt, tools } => {
                assert_eq!(prompt, "also add tests");
                assert_eq!(tools.len(), 1);
            }
            Plan::Reseed { .. } | Plan::Resolve { .. } => panic!("expected send"),
        }
    }

    #[test]
    fn a_changed_head_or_directory_reseeds() {
        let first = context(vec![user("build it")]);
        let memory = remember(&first, "/tmp/work", Vec::new());
        let rewritten = context(vec![user("summarized")]);
        assert!(matches!(
            decide(Some(&memory), &rewritten, "/tmp/work"),
            Plan::Reseed { .. }
        ));
        assert!(matches!(
            decide(Some(&memory), &first, "/tmp/elsewhere"),
            Plan::Reseed { .. }
        ));
        let shrunk = context(vec![]);
        assert!(matches!(
            decide(Some(&memory), &shrunk, "/tmp/work"),
            Plan::Reseed { .. }
        ));
    }

    #[test]
    fn an_error_result_keeps_its_flag() {
        let first = context(vec![user("build it")]);
        let memory = remember(
            &first,
            "/tmp/work",
            vec![PendingTool {
                id: "t1".to_string(),
                name: "bash".to_string(),
            }],
        );
        let next = context(vec![
            user("build it"),
            assistant_call("t1"),
            tool_result("t1", "boom", true),
        ]);
        match decide(Some(&memory), &next, "/tmp/work") {
            Plan::Resolve { results } => {
                assert!(results[0].is_error);
                assert_eq!(results[0].text, "boom");
            }
            Plan::Reseed { .. } | Plan::Send { .. } => panic!("expected resolve"),
        }
    }
}

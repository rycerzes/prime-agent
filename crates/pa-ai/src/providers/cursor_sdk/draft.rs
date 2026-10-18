//! Assemble one assistant message from sidecar events.

use serde_json::{Map, Value};

use crate::event_stream::{initial_assistant_message, AssistantMessageEventWriter};
use crate::types::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, DoneStopReason, StopReason,
    TextContent, ThinkingContent, ToolCall, Usage,
};

pub struct Draft {
    message: AssistantMessage,
    text_open: bool,
    thinking_open: bool,
}

impl Draft {
    pub fn new(api: &str, provider: &str, model_id: &str) -> Self {
        Self {
            message: initial_assistant_message(api, provider, model_id),
            text_open: false,
            thinking_open: false,
        }
    }

    pub fn set_stop(&mut self, stop: StopReason) {
        self.message.stop_reason = stop;
    }

    pub fn set_usage(&mut self, usage: Usage) {
        self.message.usage = usage;
    }

    pub fn begin(&self, writer: &AssistantMessageEventWriter) {
        writer.push(AssistantMessageEvent::Start {
            partial: self.message.clone(),
        });
    }

    pub fn text(&mut self, writer: &AssistantMessageEventWriter, delta: &str) {
        if delta.is_empty() {
            return;
        }
        self.close_thinking(writer);
        if !self.text_open {
            self.message
                .content
                .push(AssistantContent::Text(TextContent {
                    text: String::new(),
                    text_signature: None,
                    rest: Map::new(),
                }));
            writer.push(AssistantMessageEvent::TextStart {
                content_index: last_index(&self.message),
                partial: self.message.clone(),
            });
            self.text_open = true;
        }
        if let Some(AssistantContent::Text(text)) = self.message.content.last_mut() {
            text.text.push_str(delta);
        }
        writer.push(AssistantMessageEvent::TextDelta {
            content_index: last_index(&self.message),
            delta: delta.to_string(),
            partial: self.message.clone(),
        });
    }

    pub fn thinking(&mut self, writer: &AssistantMessageEventWriter, delta: &str) {
        if delta.is_empty() {
            return;
        }
        self.close_text(writer);
        if !self.thinking_open {
            self.message
                .content
                .push(AssistantContent::Thinking(ThinkingContent {
                    thinking: String::new(),
                    thinking_signature: None,
                    redacted: None,
                    rest: Map::new(),
                }));
            writer.push(AssistantMessageEvent::ThinkingStart {
                content_index: last_index(&self.message),
                partial: self.message.clone(),
            });
            self.thinking_open = true;
        }
        if let Some(AssistantContent::Thinking(thinking)) = self.message.content.last_mut() {
            thinking.thinking.push_str(delta);
        }
        writer.push(AssistantMessageEvent::ThinkingDelta {
            content_index: last_index(&self.message),
            delta: delta.to_string(),
            partial: self.message.clone(),
        });
    }

    pub fn tool(
        &mut self,
        writer: &AssistantMessageEventWriter,
        id: String,
        name: String,
        arguments: Map<String, Value>,
    ) {
        self.close_text(writer);
        self.close_thinking(writer);
        self.message
            .content
            .push(AssistantContent::ToolCall(ToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: Map::new(),
                thought_signature: None,
                rest: Map::new(),
            }));
        let index = last_index(&self.message);
        writer.push(AssistantMessageEvent::ToolcallStart {
            content_index: index,
            partial: self.message.clone(),
        });
        let arguments_text = Value::Object(arguments.clone()).to_string();
        writer.push(AssistantMessageEvent::ToolcallDelta {
            content_index: index,
            delta: arguments_text,
            partial: self.message.clone(),
        });
        if let Some(AssistantContent::ToolCall(call)) = self.message.content.last_mut() {
            call.arguments.clone_from(&arguments);
        }
        writer.push(AssistantMessageEvent::ToolcallEnd {
            content_index: index,
            tool_call: ToolCall {
                id,
                name,
                arguments,
                thought_signature: None,
                rest: Map::new(),
            },
            partial: self.message.clone(),
        });
    }

    pub fn finish(mut self, writer: &AssistantMessageEventWriter) {
        self.close_text(writer);
        self.close_thinking(writer);
        let reason = match self.message.stop_reason {
            StopReason::Length => DoneStopReason::Length,
            StopReason::ToolUse => DoneStopReason::ToolUse,
            StopReason::Stop | StopReason::Error | StopReason::Aborted => DoneStopReason::Stop,
        };
        writer.push(AssistantMessageEvent::Done {
            reason,
            message: self.message.clone(),
        });
        writer.end(Some(self.message));
    }

    fn close_text(&mut self, writer: &AssistantMessageEventWriter) {
        if !self.text_open {
            return;
        }
        self.text_open = false;
        let content = match self.message.content.last() {
            Some(AssistantContent::Text(text)) => text.text.clone(),
            _ => String::new(),
        };
        writer.push(AssistantMessageEvent::TextEnd {
            content_index: last_index(&self.message),
            content,
            partial: self.message.clone(),
        });
    }

    fn close_thinking(&mut self, writer: &AssistantMessageEventWriter) {
        if !self.thinking_open {
            return;
        }
        self.thinking_open = false;
        let content = match self.message.content.last() {
            Some(AssistantContent::Thinking(thinking)) => thinking.thinking.clone(),
            _ => String::new(),
        };
        writer.push(AssistantMessageEvent::ThinkingEnd {
            content_index: last_index(&self.message),
            content,
            partial: self.message.clone(),
        });
    }
}

fn last_index(message: &AssistantMessage) -> u64 {
    u64::try_from(message.content.len().saturating_sub(1)).unwrap_or(0)
}

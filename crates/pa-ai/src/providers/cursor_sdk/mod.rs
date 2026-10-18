//! `cursor-sdk` provider.
//!
//! Composer is not a chat-completions endpoint. A long-lived Python sidecar
//! holds one local Cursor agent per Prime session. The agent's custom tools
//! are Prime's tools, allowlisted through `mcp`. `execute` parks until the
//! next provider turn supplies the harness result, so `pa-agent` keeps the
//! tool loop.

mod draft;
mod plan;

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use serde_json::{Map, Value};

use crate::event_stream::{
    initial_assistant_message, AssistantMessageEventStream, AssistantMessageEventWriter,
};
use crate::registry::Provider;
use crate::types::{
    AssistantMessageEvent, Context, ErrorStopReason, Model, SimpleStreamOptions, StopReason,
    StreamOptions, Usage, UsageCost,
};

use draft::Draft;
use plan::{decide, remember, Memory, PendingTool, Plan, ToolResultSpec, ToolSpec};

const API: &str = "cursor-sdk";
const DEFAULT_GRACE_MS: u64 = 400;
const DEFAULT_SILENCE_MS: u64 = 480_000;

pub struct CursorSdkProvider;

impl Provider for CursorSdkProvider {
    fn api(&self) -> &str {
        API
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        start_stream(model, context, options)
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        start_stream(model, context, options.map(|options| &options.base))
    }
}

fn start_stream(
    model: &Model,
    context: &Context,
    options: Option<&StreamOptions>,
) -> AssistantMessageEventStream {
    let (writer, stream) = AssistantMessageEventStream::new();
    let owned_model = model.clone();
    let context = context.clone();
    let options = options.cloned();
    let writer_for_thread = writer.clone();
    let model_for_thread = owned_model.clone();
    let spawned = std::thread::Builder::new()
        .name("cursor-sdk-turn".to_string())
        .spawn(move || {
            turn_thread(
                &writer_for_thread,
                &model_for_thread,
                &context,
                options.as_ref(),
            );
        });
    if let Err(error) = spawned {
        emit_failure(&writer, &owned_model, &error.to_string(), false);
    }
    stream
}

fn turn_thread(
    writer: &AssistantMessageEventWriter,
    model: &Model,
    context: &Context,
    options: Option<&StreamOptions>,
) {
    let session_id = session_id_of(options);
    let sidecar = match sidecar_for(&session_id) {
        Ok(sidecar) => sidecar,
        Err(error) => {
            emit_failure(writer, model, &error, false);
            return;
        }
    };
    let _guard = sidecar
        .turn
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Err(error) = run_turn(writer, &sidecar, model, context, options) {
        {
            let mut memory = sidecar
                .memory
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *memory = None;
        }
        if error.drop_process {
            drop_session(&session_id);
        } else {
            let _ = write_line(&sidecar.stdin, &serde_json::json!({"op": "abort"}));
            drain(events_lock(&sidecar));
        }
        emit_failure(writer, model, &error.message, error.aborted);
    }
}

struct TurnError {
    message: String,
    aborted: bool,
    drop_process: bool,
}

impl TurnError {
    fn failed(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            aborted: false,
            drop_process: false,
        }
    }

    fn broken(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            aborted: false,
            drop_process: true,
        }
    }

    fn aborted() -> Self {
        Self {
            message: "Request aborted".to_string(),
            aborted: true,
            drop_process: false,
        }
    }
}

fn run_turn(
    writer: &crate::event_stream::AssistantMessageEventWriter,
    sidecar: &Sidecar,
    model: &Model,
    context: &Context,
    options: Option<&StreamOptions>,
) -> Result<(), TurnError> {
    let api_key = api_key_of(options).ok_or_else(|| {
        TurnError::failed(
            "CURSOR_API_KEY is not set. Put the env var name in models.json and export the key.",
        )
    })?;
    let cwd = std::env::current_dir()
        .map_err(|error| TurnError::failed(format!("current directory: {error}")))?
        .display()
        .to_string();
    let plan = {
        let memory = sidecar
            .memory
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        decide(memory.as_ref(), context, &cwd)
    };
    if matches!(plan, Plan::Reseed { .. }) && sidecar.spoken.load(Ordering::SeqCst) {
        write_line(&sidecar.stdin, &serde_json::json!({"op": "abort"}))
            .map_err(TurnError::broken)?;
        drain(events_lock(sidecar));
        let mut memory = sidecar
            .memory
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *memory = None;
    }
    let turn_id = next_turn_id();
    tracing::info!(session = %session_id_of(options), %turn_id, plan = plan_name(&plan), "cursor-sdk turn");
    let payload = command_json(&plan, &turn_id, &model.id, &api_key, &cwd);
    write_line(&sidecar.stdin, &payload).map_err(TurnError::broken)?;
    sidecar.spoken.store(true, Ordering::SeqCst);
    let signal = options.and_then(|options| options.signal.clone());
    let step = read_step(events_lock(sidecar), &turn_id, signal.as_ref())?;
    let pending = pending_from(&step);
    {
        let mut memory = sidecar
            .memory
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *memory = Some(remember(context, &cwd, pending));
    }
    emit_step(writer, model, step);
    Ok(())
}

fn session_id_of(options: Option<&StreamOptions>) -> String {
    options
        .and_then(|options| options.session_id.clone())
        .filter(|id| !id.is_empty())
        .unwrap_or_else(|| "default".to_string())
}

fn api_key_of(options: Option<&StreamOptions>) -> Option<String> {
    options
        .and_then(|options| options.api_key.clone())
        .filter(|key| !key.is_empty())
}

fn plan_name(plan: &Plan) -> &'static str {
    match plan {
        Plan::Reseed { .. } => "reseed",
        Plan::Send { .. } => "send",
        Plan::Resolve { .. } => "resolve",
    }
}

fn command_json(plan: &Plan, turn_id: &str, model_id: &str, api_key: &str, cwd: &str) -> Value {
    match plan {
        Plan::Resolve { results } => serde_json::json!({
            "op": "tools",
            "id": turn_id,
            "results": results.iter().map(result_json).collect::<Vec<_>>(),
        }),
        Plan::Reseed { prompt, tools } | Plan::Send { prompt, tools } => serde_json::json!({
            "op": "turn",
            "id": turn_id,
            "model": model_id,
            "apiKey": api_key,
            "prompt": prompt,
            "cwd": cwd,
            "tools": tools.iter().map(tool_json).collect::<Vec<_>>(),
        }),
    }
}

fn tool_json(tool: &ToolSpec) -> Value {
    serde_json::json!({
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    })
}

fn result_json(result: &ToolResultSpec) -> Value {
    serde_json::json!({
        "id": result.id,
        "text": result.text,
        "isError": result.is_error,
    })
}

fn next_turn_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    format!("turn-{id}")
}

struct Sidecar {
    stdin: Mutex<ChildStdin>,
    events: Mutex<Receiver<BridgeEvent>>,
    child: Mutex<Child>,
    turn: Mutex<()>,
    memory: Mutex<Option<Memory>>,
    spoken: AtomicBool,
}

fn sessions() -> &'static Mutex<HashMap<String, Arc<Sidecar>>> {
    static SESSIONS: OnceLock<Mutex<HashMap<String, Arc<Sidecar>>>> = OnceLock::new();
    SESSIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn bridge_override() -> &'static Mutex<Option<PathBuf>> {
    static OVERRIDE: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
    OVERRIDE.get_or_init(|| Mutex::new(None))
}

fn sidecar_for(session_id: &str) -> Result<Arc<Sidecar>, String> {
    let mut map = sessions()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(existing) = map.get(session_id) {
        return Ok(Arc::clone(existing));
    }
    let sidecar = spawn_sidecar()?;
    map.insert(session_id.to_string(), Arc::clone(&sidecar));
    Ok(sidecar)
}

fn drop_session(session_id: &str) {
    let removed = {
        let mut map = sessions()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        map.remove(session_id)
    };
    if let Some(sidecar) = removed {
        let mut child = sidecar
            .child
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _ = child.kill();
        let _ = child.wait();
    }
}

fn spawn_sidecar() -> Result<Arc<Sidecar>, String> {
    let script = bridge_script()?;
    let python = std::env::var("PRIME_CURSOR_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let mut command = Command::new(python);
    command
        .arg(&script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .env("PYTHONUNBUFFERED", "1");
    let mut child = command
        .spawn()
        .map_err(|error| format!("spawn cursor bridge: {error}"))?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| "cursor bridge stdin missing".to_string())?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "cursor bridge stdout missing".to_string())?;
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("cursor-sdk-bridge".to_string())
        .spawn(move || read_stdout(stdout, &tx))
        .map_err(|error| format!("spawn cursor bridge reader: {error}"))?;
    Ok(Arc::new(Sidecar {
        stdin: Mutex::new(stdin),
        events: Mutex::new(rx),
        child: Mutex::new(child),
        turn: Mutex::new(()),
        memory: Mutex::new(None),
        spoken: AtomicBool::new(false),
    }))
}

fn bridge_script() -> Result<PathBuf, String> {
    if let Some(path) = bridge_override()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
    {
        return Ok(path);
    }
    if let Some(path) = std::env::var_os("PRIME_CURSOR_BRIDGE") {
        return Ok(PathBuf::from(path));
    }
    let dir = std::env::temp_dir().join("prime-cursor-bridge");
    std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
    let path = dir.join("bridge.py");
    std::fs::write(&path, include_str!("bridge.py")).map_err(|error| error.to_string())?;
    Ok(path)
}

fn read_stdout(stdout: ChildStdout, tx: &Sender<BridgeEvent>) {
    let reader = BufReader::new(stdout);
    for line in reader.lines() {
        match line {
            Ok(line) if line.trim().is_empty() => {}
            Ok(line) => {
                if tx.send(parse_event(&line)).is_err() {
                    break;
                }
            }
            Err(error) => {
                let _ = tx.send(BridgeEvent::Error {
                    turn: None,
                    message: format!("cursor bridge read: {error}"),
                });
                break;
            }
        }
    }
    let _ = tx.send(BridgeEvent::Eof);
}

enum BridgeEvent {
    Text {
        turn: Option<String>,
        delta: String,
    },
    Thinking {
        turn: Option<String>,
        delta: String,
    },
    Tool {
        turn: Option<String>,
        id: String,
        name: String,
        arguments: Map<String, Value>,
    },
    Done {
        turn: Option<String>,
        stop: String,
        usage: Usage,
    },
    Error {
        turn: Option<String>,
        message: String,
    },
    Eof,
}

fn parse_event(line: &str) -> BridgeEvent {
    let value: Value = match serde_json::from_str(line) {
        Ok(value) => value,
        Err(_) => {
            return BridgeEvent::Error {
                turn: None,
                message: "bridge sent malformed json".to_string(),
            }
        }
    };
    let turn = value
        .get("turn")
        .and_then(Value::as_str)
        .map(ToString::to_string);
    let op = value.get("op").and_then(Value::as_str).unwrap_or("");
    match op {
        "text" => BridgeEvent::Text {
            turn,
            delta: string_field(&value, "delta"),
        },
        "thinking" => BridgeEvent::Thinking {
            turn,
            delta: string_field(&value, "delta"),
        },
        "tool" => BridgeEvent::Tool {
            turn,
            id: string_field(&value, "id"),
            name: string_field(&value, "name"),
            arguments: arguments_map(
                value
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| Value::Object(Map::new())),
            ),
        },
        "done" => BridgeEvent::Done {
            turn,
            stop: string_field(&value, "stop"),
            usage: parse_usage(&value),
        },
        "error" => BridgeEvent::Error {
            turn,
            message: redact(&string_field(&value, "message")),
        },
        _ => BridgeEvent::Error {
            turn,
            message: format!("unknown bridge op {op}"),
        },
    }
}

fn string_field(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn arguments_map(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        other => {
            let mut map = Map::new();
            map.insert("value".to_string(), other);
            map
        }
    }
}

fn parse_usage(value: &Value) -> Usage {
    let usage = value.get("usage");
    let number = |key: &str| {
        usage
            .and_then(|usage| usage.get(key))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    let input = number("input");
    let output = number("output");
    Usage {
        input,
        output,
        cache_read: number("cacheRead"),
        cache_write: number("cacheWrite"),
        total_tokens: input.saturating_add(output),
        cost: UsageCost::default(),
    }
}

fn event_turn(event: &BridgeEvent) -> Option<&str> {
    match event {
        BridgeEvent::Text { turn, .. }
        | BridgeEvent::Thinking { turn, .. }
        | BridgeEvent::Tool { turn, .. }
        | BridgeEvent::Done { turn, .. }
        | BridgeEvent::Error { turn, .. } => turn.as_deref(),
        BridgeEvent::Eof => None,
    }
}

struct Step {
    events: Vec<Incoming>,
    stop: StopReason,
    usage: Usage,
}

enum Incoming {
    Text(String),
    Thinking(String),
    Tool {
        id: String,
        name: String,
        arguments: Map<String, Value>,
    },
}

fn read_step(
    events: &Mutex<Receiver<BridgeEvent>>,
    turn_id: &str,
    signal: Option<&tokio_util::sync::CancellationToken>,
) -> Result<Step, TurnError> {
    let rx = events
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut incoming = Vec::new();
    let mut saw_tool = false;
    let mut usage = Usage::default();
    let mut tool_seq = 0_u64;
    loop {
        if signal.is_some_and(tokio_util::sync::CancellationToken::is_cancelled) {
            return Err(TurnError::aborted());
        }
        let timeout = if saw_tool { grace() } else { silence() };
        let event = match rx.recv_timeout(timeout) {
            Ok(event) => event,
            Err(RecvTimeoutError::Timeout) if saw_tool => {
                if !sweep_quiet(
                    &rx,
                    turn_id,
                    &mut incoming,
                    &mut saw_tool,
                    &mut usage,
                    &mut tool_seq,
                )? {
                    return Ok(Step {
                        events: incoming,
                        stop: StopReason::ToolUse,
                        usage,
                    });
                }
                continue;
            }
            Err(RecvTimeoutError::Timeout) => {
                return Err(TurnError::failed(
                    "cursor bridge timed out waiting for the model",
                ));
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(TurnError::broken("cursor bridge closed"));
            }
        };
        if !accept_event(turn_id, &event) {
            continue;
        }
        match apply_event(
            event,
            &mut incoming,
            &mut saw_tool,
            &mut usage,
            &mut tool_seq,
        )? {
            Control::Continue => {}
            Control::Finish(stop) => {
                return Ok(Step {
                    events: incoming,
                    stop,
                    usage,
                });
            }
        }
    }
}

fn sweep_quiet(
    rx: &Receiver<BridgeEvent>,
    turn_id: &str,
    incoming: &mut Vec<Incoming>,
    saw_tool: &mut bool,
    usage: &mut Usage,
    tool_seq: &mut u64,
) -> Result<bool, TurnError> {
    let mut extended = false;
    loop {
        let event = match rx.try_recv() {
            Ok(event) => event,
            Err(mpsc::TryRecvError::Empty) => return Ok(extended),
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(TurnError::broken("cursor bridge closed"));
            }
        };
        if !accept_event(turn_id, &event) {
            continue;
        }
        extended = true;
        if let Control::Finish(_) = apply_event(event, incoming, saw_tool, usage, tool_seq)? {
            return Ok(false);
        }
    }
}

enum Control {
    Continue,
    Finish(StopReason),
}

fn apply_event(
    event: BridgeEvent,
    incoming: &mut Vec<Incoming>,
    saw_tool: &mut bool,
    usage: &mut Usage,
    tool_seq: &mut u64,
) -> Result<Control, TurnError> {
    match event {
        BridgeEvent::Text { delta, .. } => incoming.push(Incoming::Text(delta)),
        BridgeEvent::Thinking { delta, .. } => incoming.push(Incoming::Thinking(delta)),
        BridgeEvent::Tool {
            id,
            name,
            arguments,
            ..
        } => {
            *saw_tool = true;
            *tool_seq = tool_seq.saturating_add(1);
            let id = if id.is_empty() {
                format!("cursor-tool-{tool_seq}")
            } else {
                id
            };
            let name = if name.is_empty() {
                "tool".to_string()
            } else {
                name
            };
            incoming.push(Incoming::Tool {
                id,
                name,
                arguments,
            });
        }
        BridgeEvent::Done {
            stop,
            usage: reported,
            ..
        } => {
            *usage = reported;
            let reason = if *saw_tool {
                StopReason::ToolUse
            } else {
                map_stop(&stop)
            };
            return Ok(Control::Finish(reason));
        }
        BridgeEvent::Error { message, .. } => return Err(TurnError::failed(message)),
        BridgeEvent::Eof => return Err(TurnError::broken("cursor bridge closed")),
    }
    Ok(Control::Continue)
}

fn accept_event(turn_id: &str, event: &BridgeEvent) -> bool {
    match event {
        BridgeEvent::Eof => true,
        _ => match event_turn(event) {
            Some(turn) => turn == turn_id,
            None => true,
        },
    }
}

fn map_stop(stop: &str) -> StopReason {
    match stop {
        "toolUse" | "tool_use" => StopReason::ToolUse,
        "length" => StopReason::Length,
        _ => StopReason::Stop,
    }
}

fn pending_from(step: &Step) -> Vec<PendingTool> {
    if step.stop != StopReason::ToolUse {
        return Vec::new();
    }
    step.events
        .iter()
        .filter_map(|event| match event {
            Incoming::Tool { id, name, .. } => Some(PendingTool {
                id: id.clone(),
                name: name.clone(),
            }),
            Incoming::Text(_) | Incoming::Thinking(_) => None,
        })
        .collect()
}

fn emit_step(writer: &crate::event_stream::AssistantMessageEventWriter, model: &Model, step: Step) {
    let mut draft = Draft::new(&model.api, &model.provider, &model.id);
    draft.set_stop(step.stop);
    draft.set_usage(step.usage);
    draft.begin(writer);
    for event in step.events {
        match event {
            Incoming::Text(delta) => draft.text(writer, &delta),
            Incoming::Thinking(delta) => draft.thinking(writer, &delta),
            Incoming::Tool {
                id,
                name,
                arguments,
            } => draft.tool(writer, id, name, arguments),
        }
    }
    draft.finish(writer);
}

fn events_lock(sidecar: &Sidecar) -> &Mutex<Receiver<BridgeEvent>> {
    &sidecar.events
}

fn write_line(stdin: &Mutex<ChildStdin>, value: &Value) -> Result<(), String> {
    let mut line = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    line.push(b'\n');
    let mut stdin = stdin
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    stdin
        .write_all(&line)
        .map_err(|error| format!("cursor bridge write: {error}"))?;
    stdin
        .flush()
        .map_err(|error| format!("cursor bridge flush: {error}"))
}

fn drain(events: &Mutex<Receiver<BridgeEvent>>) {
    let rx = events
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    while rx.try_recv().is_ok() {}
}

fn grace() -> Duration {
    Duration::from_millis(env_ms("PRIME_CURSOR_TOOL_GRACE_MS", DEFAULT_GRACE_MS))
}

fn silence() -> Duration {
    Duration::from_millis(env_ms("PRIME_CURSOR_SILENCE_MS", DEFAULT_SILENCE_MS))
}

fn env_ms(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn redact(text: &str) -> String {
    let mut rest = text;
    let mut out = String::new();
    while let Some(index) = rest.find("crsr_") {
        out.push_str(&rest[..index]);
        rest = &rest[index + 5..];
        let end = rest
            .find(|ch: char| !ch.is_ascii_alphanumeric())
            .unwrap_or(rest.len());
        out.push_str("crsr_[redacted]");
        rest = &rest[end..];
    }
    out.push_str(rest);
    out
}

fn emit_failure(
    writer: &crate::event_stream::AssistantMessageEventWriter,
    model: &Model,
    message: &str,
    aborted: bool,
) {
    let mut error = initial_assistant_message(&model.api, &model.provider, &model.id);
    error.stop_reason = if aborted {
        StopReason::Aborted
    } else {
        StopReason::Error
    };
    error.error_message = Some(redact(message));
    let reason = if aborted {
        ErrorStopReason::Aborted
    } else {
        ErrorStopReason::Error
    };
    writer.push(AssistantMessageEvent::Error {
        reason,
        error: error.clone(),
    });
    writer.end(Some(error));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        AssistantMessage, TextContent, Tool, ToolCall, ToolResultMessage, UserMessage,
        UserOrToolContent,
    };
    const FAKE_BRIDGE: &str = r#"
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    cmd = json.loads(line)
    op = cmd.get("op")
    if op == "turn":
        sys.stdout.write(json.dumps({"op": "text", "delta": "working"}) + "\n")
        sys.stdout.write(json.dumps({"op": "tool", "id": "t1", "name": "bash", "arguments": {"command": "echo hi"}}) + "\n")
        sys.stdout.flush()
    elif op == "tools":
        text = cmd["results"][0]["text"].strip()
        sys.stdout.write(json.dumps({"op": "text", "delta": "RESULT=" + text}) + "\n")
        sys.stdout.write(json.dumps({"op": "done", "stop": "stop", "usage": {"input": 3, "output": 4}}) + "\n")
        sys.stdout.flush()
    elif op in ("abort", "reset", "cancel"):
        pass
"#;

    fn set_override(path: Option<PathBuf>) {
        *bridge_override()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = path;
    }

    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            set_override(None);
        }
    }

    fn test_model() -> Model {
        serde_json::from_value(serde_json::json!({
            "id": "composer-2.5",
            "name": "Composer 2.5",
            "api": "cursor-sdk",
            "provider": "cursor",
            "baseUrl": "cursor-sdk://local",
            "reasoning": true,
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 200_000,
            "maxTokens": 32000
        }))
        .expect("model")
    }

    fn options(session: &str) -> SimpleStreamOptions {
        SimpleStreamOptions {
            base: StreamOptions {
                api_key: Some("test-key".to_string()),
                session_id: Some(session.to_string()),
                ..StreamOptions::default()
            },
            ..SimpleStreamOptions::default()
        }
    }

    fn user(text: &str) -> crate::types::Message {
        crate::types::Message::User(UserMessage {
            content: crate::types::UserMessageContent::Text(text.to_string()),
            timestamp: 0,
            rest: Map::new(),
        })
    }

    fn first_context() -> Context {
        Context {
            system_prompt: Some("system".to_string()),
            messages: vec![user("build it")],
            tools: Some(vec![Tool {
                name: "bash".to_string(),
                description: "run".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            }]),
        }
    }

    fn second_context() -> Context {
        let mut context = first_context();
        context
            .messages
            .push(crate::types::Message::Assistant(AssistantMessage {
                content: vec![crate::types::AssistantContent::ToolCall(ToolCall {
                    id: "t1".to_string(),
                    name: "bash".to_string(),
                    arguments: Map::from_iter([(
                        "command".to_string(),
                        Value::String("echo hi".to_string()),
                    )]),
                    thought_signature: None,
                    rest: Map::new(),
                })],
                api: API.to_string(),
                provider: "cursor".to_string(),
                model: "composer-2.5".to_string(),
                response_model: None,
                response_id: None,
                diagnostics: None,
                usage: Usage::default(),
                stop_reason: StopReason::ToolUse,
                stop_reason_raw: None,
                error_message: None,
                timestamp: 0,
                rest: Map::new(),
            }));
        context
            .messages
            .push(crate::types::Message::ToolResult(ToolResultMessage {
                tool_call_id: "t1".to_string(),
                tool_name: "bash".to_string(),
                content: vec![UserOrToolContent::Text(TextContent {
                    text: "hi\n".to_string(),
                    text_signature: None,
                    rest: Map::new(),
                })],
                details: None,
                is_error: false,
                timestamp: 0,
                rest: Map::new(),
            }));
        context
    }

    fn collect(stream: AssistantMessageEventStream) -> Vec<AssistantMessageEvent> {
        futures::executor::block_on(stream.collect())
    }

    fn done_message(events: &[AssistantMessageEvent]) -> AssistantMessage {
        events
            .iter()
            .rev()
            .find_map(|event| match event {
                AssistantMessageEvent::Done { message, .. } => Some(message.clone()),
                _ => None,
            })
            .expect("done event")
    }

    #[test]
    fn a_parked_tool_is_resolved_on_the_following_turn() {
        let _guard = Guard;
        let dir = std::env::temp_dir().join(format!("prime-cursor-fake-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let script = dir.join("fake_bridge.py");
        std::fs::write(&script, FAKE_BRIDGE).expect("fake bridge");
        set_override(Some(script));
        let session = format!("cursor-sdk-test-{}", std::process::id());
        drop_session(&session);
        let model = test_model();
        let provider = CursorSdkProvider;
        let options = options(&session);
        let first = collect(provider.stream_simple(&model, &first_context(), Some(&options)));
        let parked = done_message(&first);
        assert_eq!(parked.stop_reason, StopReason::ToolUse);
        let call = parked.content.iter().find_map(|block| match block {
            crate::types::AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        });
        let call = call.expect("tool call");
        assert_eq!(call.id, "t1");
        assert_eq!(call.name, "bash");
        assert!(parked.content.iter().any(|block| matches!(
            block,
            crate::types::AssistantContent::Text(text) if text.text.contains("working")
        )));

        let second = collect(provider.stream_simple(&model, &second_context(), Some(&options)));
        let finished = done_message(&second);
        assert_eq!(finished.stop_reason, StopReason::Stop);
        assert!(finished.content.iter().any(|block| matches!(
            block,
            crate::types::AssistantContent::Text(text) if text.text.contains("RESULT=hi")
        )));
        assert_eq!(finished.usage.input, 3);
        assert_eq!(finished.usage.output, 4);
        drop_session(&session);
    }

    #[test]
    fn redact_strips_cursor_keys() {
        assert_eq!(
            redact("failed crsr_abc123DEF for user"),
            "failed crsr_[redacted] for user"
        );
    }
}

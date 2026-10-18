"""Prime Agent sidecar for a local Cursor agent.

The harness owns tool execution. This process owns one Cursor agent whose
custom tools park inside ``execute`` until the harness writes a result.
Protocol is NDJSON on stdin and a duplicated stdout. Diagnostics go to
stderr and never include the API key.
"""

from __future__ import annotations

import faulthandler
import json
import os
import queue
import re
import sys
import tempfile
import threading
import traceback
import uuid
from collections.abc import Mapping
from typing import Any

_SECRET = re.compile(r"crsr_[A-Za-z0-9]+")
_PROTO: Any = None
_STDIN: Any = None
_OUT_LOCK = threading.Lock()


def scrub(text: str) -> str:
    return _SECRET.sub("crsr_[redacted]", text)


def prepare_environment() -> None:
    origin = os.environ.get("HOME", "")
    local_bin = os.path.join(origin, ".local", "bin")
    path = os.environ.get("PATH", "")
    if os.path.isdir(local_bin) and local_bin not in path.split(os.pathsep):
        os.environ["PATH"] = local_bin + os.pathsep + path
    clean = os.environ.get("PRIME_CURSOR_HOME")
    if not clean:
        clean = tempfile.mkdtemp(prefix="prime-cursor-home-")
    os.makedirs(clean, exist_ok=True)
    os.environ["HOME"] = clean
    os.environ.setdefault("XDG_CONFIG_HOME", os.path.join(clean, ".config"))
    os.environ.setdefault("XDG_DATA_HOME", os.path.join(clean, ".local", "share"))


def prepare_stdio() -> None:
    """Detach the protocol pipes from fds 0 and 1.

    The Cursor bridge subprocess inherits the process stdio. Leaving the
    command pipe on fd 0 makes that child close it when a run starts, which
    ends this process. The command stream moves to a private dup and fd 0
    becomes ``/dev/null``.
    """
    global _PROTO, _STDIN
    proto_in = os.dup(0)
    null_fd = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null_fd, 0)
    os.close(null_fd)
    _STDIN = os.fdopen(proto_in, "r", encoding="utf-8")
    proto_out = os.dup(1)
    _PROTO = os.fdopen(proto_out, "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr


def emit(payload: dict[str, Any]) -> None:
    if _PROTO is None:
        return
    line = json.dumps(payload, ensure_ascii=False, default=str)
    with _OUT_LOCK:
        _PROTO.write(line + "\n")
        _PROTO.flush()


def excepthook(etype, value, tb) -> None:  # noqa: ANN001
    sys.stderr.write(scrub("".join(traceback.format_exception(etype, value, tb))))
    sys.stderr.flush()


prepare_environment()
prepare_stdio()
faulthandler.enable(file=sys.stderr, all_threads=True)
sys.excepthook = excepthook

try:
    from cursor_sdk import Agent, AgentOptions, CustomTool, LocalAgentOptions
except Exception as exc:  # noqa: BLE001
    emit({"op": "error", "message": scrub(f"cursor_sdk import failed: {exc}")})
    raise SystemExit(1) from exc


def assistant_text(message: Any) -> str:
    body = getattr(message, "message", None)
    content = body.get("content") if isinstance(body, dict) else getattr(body, "content", None)
    if content is None:
        content = getattr(message, "text", None)
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    try:
        blocks = list(content)
    except TypeError:
        return ""
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
            continue
        kind = block.get("type") if isinstance(block, dict) else getattr(block, "type", "")
        if kind not in ("text", "", None):
            continue
        text = block.get("text") if isinstance(block, dict) else getattr(block, "text", "")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    return {"value": value if isinstance(value, (str, int, float, bool)) or value is None else str(value)}


class Bridge:
    def __init__(self) -> None:
        self.agent: Any = None
        self.agent_key = ""
        self.turn_id = ""
        self.generation = 0
        self.closing = False
        self.pending: dict[str, queue.Queue[dict[str, Any]]] = {}
        self.pending_lock = threading.Lock()
        self.current_run: Any = None
        self.run_thread: threading.Thread | None = None
        self.assistant_sent = ""
        self.thinking_sent = ""

    def serve(self) -> None:
        sys.stderr.write("cursor bridge: serving\n")
        sys.stderr.flush()
        assert _STDIN is not None
        for line in _STDIN:
            text = line.strip()
            if not text:
                continue
            try:
                command = json.loads(text)
            except json.JSONDecodeError as exc:
                emit({"op": "error", "message": f"bad command: {exc}"})
                continue
            if not isinstance(command, dict):
                emit({"op": "error", "message": "command must be an object"})
                continue
            try:
                self.dispatch(command)
            except BaseException as exc:  # noqa: BLE001
                sys.stderr.write(
                    f"cursor bridge: command failed {type(exc).__name__}: {scrub(str(exc))}\n"
                )
                sys.stderr.flush()
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                emit(
                    {
                        "op": "error",
                        "turn": self.turn_id,
                        "message": scrub(str(exc)),
                    }
                )
        sys.stderr.write("cursor bridge: stdin closed\n")
        sys.stderr.flush()
        self.handle_abort()

    def dispatch(self, command: dict[str, Any]) -> None:
        op = command.get("op")
        if op == "turn":
            self.handle_turn(command)
        elif op == "tools":
            self.handle_tools(command)
        elif op in ("abort", "reset", "cancel"):
            self.handle_abort()
        else:
            emit({"op": "error", "message": f"unknown op {op}"})

    def handle_turn(self, command: dict[str, Any]) -> None:
        self.turn_id = str(command.get("id") or "")
        if self.run_thread is not None and self.run_thread.is_alive():
            self.handle_abort()
            self.turn_id = str(command.get("id") or "")
        model = str(command.get("model") or "composer-2.5")
        api_key = str(command.get("apiKey") or "")
        cwd = str(command.get("cwd") or os.getcwd())
        os.makedirs(cwd, exist_ok=True)
        tools = command.get("tools") or []
        if not isinstance(tools, list):
            tools = []
        key = json.dumps({"cwd": cwd, "model": model, "tools": tools}, sort_keys=True, default=str)
        if self.agent is None or self.agent_key != key:
            self.close_agent()
            if not api_key:
                emit({"op": "error", "turn": self.turn_id, "message": "CURSOR_API_KEY is not set"})
                return
            self.agent = self.build_agent(model, api_key, cwd, tools)
            self.agent_key = key
        prompt = str(command.get("prompt") or "") or "Continue."
        self.start_run(prompt)

    def build_agent(self, model: str, api_key: str, cwd: str, tools: list[Any]) -> Any:
        custom: dict[str, Any] = {}
        for tool in tools:
            if not isinstance(tool, dict) or not tool.get("name"):
                continue
            name = str(tool["name"])
            parameters = tool.get("parameters")
            if not isinstance(parameters, dict):
                parameters = {"type": "object", "properties": {}}
            description = str(tool.get("description") or name)
            custom[name] = CustomTool(
                description=description,
                input_schema=parameters,
                execute=self.make_execute(name),
            )
        sys.stderr.write(f"cursor bridge: create agent model={model} tools={list(custom)}\n")
        sys.stderr.flush()
        try:
            agent = Agent.create(
                AgentOptions(
                    model=model,
                    api_key=api_key,
                    tools=["mcp"],
                    local=LocalAgentOptions(cwd=cwd, custom_tools=custom),
                )
            )
        except BaseException as exc:
            sys.stderr.write(f"cursor bridge: create failed {type(exc).__name__}: {scrub(str(exc))}\n")
            sys.stderr.flush()
            raise
        sys.stderr.write("cursor bridge: create returned\n")
        sys.stderr.flush()
        return agent

    def make_execute(self, name: str):
        def execute(args: Any, context: Any) -> str:
            call_id = getattr(context, "tool_call_id", None) or str(uuid.uuid4())
            if not isinstance(call_id, str) or not call_id:
                call_id = str(uuid.uuid4())
            box: queue.Queue[dict[str, Any]] = queue.Queue()
            with self.pending_lock:
                if self.closing:
                    return "Error: aborted"
                self.pending[call_id] = box
            sys.stderr.write(f"cursor bridge: parked {name} id={call_id}\n")
            sys.stderr.flush()
            self.emit_turn({"op": "tool", "id": call_id, "name": name, "arguments": as_dict(args)})
            result = box.get()
            with self.pending_lock:
                self.pending.pop(call_id, None)
            text = result.get("text") or ""
            if not isinstance(text, str):
                text = json.dumps(text)
            return text

        return execute

    def start_run(self, prompt: str) -> None:
        if self.agent is None:
            emit({"op": "error", "turn": self.turn_id, "message": "no cursor agent"})
            return
        sys.stderr.write("cursor bridge: send\n")
        sys.stderr.flush()
        self.generation += 1
        generation = self.generation
        self.assistant_sent = ""
        self.thinking_sent = ""
        try:
            run = self.agent.send(prompt)
        except BaseException as exc:
            sys.stderr.write(
                f"cursor bridge: send failed {type(exc).__name__}: {scrub(str(exc))}\n"
            )
            sys.stderr.flush()
            raise
        sys.stderr.write("cursor bridge: send returned\n")
        sys.stderr.flush()
        self.current_run = run

        def consume() -> None:
            sys.stderr.write("cursor bridge: consuming\n")
            sys.stderr.flush()
            try:
                for message in run.messages():
                    if generation != self.generation:
                        return
                    self.handle_message(message)
                if generation != self.generation:
                    return
                self.emit_terminal(run.wait())
            except Exception as exc:  # noqa: BLE001
                if generation != self.generation:
                    return
                emit({"op": "error", "turn": self.turn_id, "message": scrub(str(exc))})

        self.run_thread = threading.Thread(target=consume, name="cursor-run", daemon=True)
        self.run_thread.start()
        sys.stderr.write("cursor bridge: run thread started\n")
        sys.stderr.flush()

    def handle_message(self, message: Any) -> None:
        kind = getattr(message, "type", "")
        if kind == "assistant":
            self.emit_delta("text", assistant_text(message), "assistant_sent")
        elif kind == "thinking":
            text = getattr(message, "text", "")
            if isinstance(text, str):
                self.emit_delta("thinking", text, "thinking_sent")

    def emit_delta(self, op: str, text: str, slot: str) -> None:
        if not text:
            return
        sent = getattr(self, slot)
        if text.startswith(sent):
            delta = text[len(sent) :]
            updated = text
        elif sent.startswith(text):
            return
        else:
            delta = text
            updated = sent + text
        setattr(self, slot, updated)
        if delta:
            emit({"op": op, "turn": self.turn_id, "delta": delta})

    def emit_terminal(self, result: Any) -> None:
        extra = result_text(result)
        if extra:
            self.emit_delta("text", extra, "assistant_sent")
        status = str(getattr(result, "status", "") or "").lower()
        if status in {"error", "failed"}:
            detail = getattr(result, "error", None) or status or "run failed"
            emit({"op": "error", "turn": self.turn_id, "message": scrub(str(detail))})
            return
        if status in {"cancelled", "canceled", "aborted"}:
            return
        payload: dict[str, Any] = {"op": "done", "turn": self.turn_id, "stop": "stop"}
        usage = usage_of(result)
        if usage is not None:
            payload["usage"] = usage
        emit(payload)

    def handle_tools(self, command: dict[str, Any]) -> None:
        self.turn_id = str(command.get("id") or self.turn_id)
        results = command.get("results") or []
        if not isinstance(results, list):
            emit({"op": "error", "turn": self.turn_id, "message": "tools results must be a list"})
            return
        with self.pending_lock:
            parked = dict(self.pending)
        for result in results:
            if not isinstance(result, dict):
                continue
            call_id = str(result.get("id") or "")
            box = parked.get(call_id)
            if box is None:
                emit(
                    {
                        "op": "error",
                        "turn": self.turn_id,
                        "message": f"no parked tool {call_id}",
                    }
                )
                continue
            text = result.get("text")
            if text is None:
                text = ""
            if not isinstance(text, str):
                text = json.dumps(text)
            if result.get("isError") and not text.startswith("Error"):
                text = f"Error: {text}"
            box.put({"text": text})

    def handle_abort(self) -> None:
        self.generation += 1
        with self.pending_lock:
            self.closing = True
            boxes = list(self.pending.values())
            self.pending.clear()
        for box in boxes:
            box.put({"abort": True, "text": "Error: aborted"})
        run = self.current_run
        if run is not None:
            try:
                run.cancel()
            except Exception:  # noqa: BLE001
                pass
        thread = self.run_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=8)
        self.close_agent()
        with self.pending_lock:
            self.closing = False
        self.current_run = None
        self.run_thread = None

    def close_agent(self) -> None:
        current = self.agent
        self.agent = None
        self.agent_key = ""
        if current is not None:
            try:
                current.close()
            except Exception:  # noqa: BLE001
                pass

    def emit_turn(self, payload: dict[str, Any]) -> None:
        payload["turn"] = self.turn_id
        emit(payload)


def result_text(result: Any) -> str:
    text = getattr(result, "text", None)
    if callable(text):
        try:
            text = text()
        except Exception:  # noqa: BLE001
            return ""
    return text if isinstance(text, str) else ""


def usage_of(result: Any) -> dict[str, int] | None:
    usage = getattr(result, "usage", None)
    if usage is None:
        return None

    def num(name: str) -> int:
        value = getattr(usage, name, 0) or 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    return {
        "input": num("input_tokens"),
        "output": num("output_tokens"),
        "cacheRead": num("cache_read_tokens"),
        "cacheWrite": num("cache_write_tokens"),
    }


if __name__ == "__main__":
    Bridge().serve()

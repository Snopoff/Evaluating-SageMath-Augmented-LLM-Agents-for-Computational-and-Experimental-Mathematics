from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from src.tools.registry import ToolRegistry
from src.utils.logging import progress


@dataclass(frozen=True)
class ControllerConfig:
    """Runtime knobs for the tool-using chat loop.

    Args:
        max_turns: Maximum number of model turns before stopping.
        temperature: Sampling temperature passed to the chat backend.
        progress_logs: Whether to emit controller progress messages.
        max_tool_calls: Maximum number of tool dispatches allowed per solve.
    """

    max_turns: int = 6
    temperature: float = 0.0
    progress_logs: bool = False
    max_tool_calls: int = 4

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> ControllerConfig:
        cfg_dict = dict(cfg or {})
        return cls(
            max_turns=int(cfg_dict.get("max_turns", 6)),
            temperature=float(cfg_dict.get("temperature", 0.0)),
            progress_logs=bool(cfg_dict.get("progress_logs", False)),
            max_tool_calls=int(cfg_dict.get("max_tool_calls", 4)),
        )


@dataclass(frozen=True)
class SolveResult:
    """Final output and execution trace returned by the controller.

    Args:
        final_answer: Final answer returned to the caller.
        tool_traces: Per-tool execution records collected during the solve loop.
        turn_count: Number of model turns consumed.
        stop_reason: Terminal reason such as ``finalized`` or ``max_turns_reached``.
    """

    final_answer: str
    tool_traces: list[dict[str, Any]]
    turn_count: int
    stop_reason: str


@dataclass(frozen=True)
class ParsedTurn:
    """Structured representation of one model turn.

    Args:
        answer: Assistant answer extracted from the model payload.
        tool_call: Optional tool call payload with ``name`` and ``arguments``.
    """

    answer: str
    tool_call: dict[str, Any] | None


class AgentController:
    """Runs the iterative chat loop and dispatches tool calls.

    Args:
        client: Chat-completions-compatible provider client.
        model_name: Model identifier passed to the provider.
        tool_registry: Registry used to expose and dispatch tools.
        config: Optional controller configuration. Defaults to ``ControllerConfig()``.
    """

    def __init__(
        self,
        client: Any,
        model_name: str,
        tool_registry: ToolRegistry,
        config: ControllerConfig | None = None,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.tool_registry = tool_registry
        self.config = config or ControllerConfig()

    def _progress(self, message: str) -> None:
        if self.config.progress_logs:
            progress(f"[bold orange1]\[controller][/bold orange1] {message}")  # type: ignore because rich markup

    @staticmethod
    def _preview_text(text: str, max_chars: int = 240) -> str:
        compact = " ".join(text.strip().split())
        if len(compact) <= max_chars:
            return compact
        return f"{compact[: max_chars - 3]}..."

    def _log_model_reply(self, raw: str) -> None:
        self._progress(f"model reply: {self._preview_text(raw)}")

    def _log_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        self._progress(f"tool call: {tool_name} args={self._preview_text(json.dumps(arguments, ensure_ascii=True), max_chars=320)}")

    def _log_tool_result(self, trace: Mapping[str, Any]) -> None:
        metadata = trace.get("metadata", {})
        status = metadata.get("status", "unknown") if isinstance(metadata, Mapping) else "unknown"
        self._progress(
            f"tool result: {trace.get('name', '?')} ok={trace.get('ok', False)} status={status} "
            f"content={self._preview_text(str(trace.get('content', '')))}"
        )

    def solve(self, question: str) -> SolveResult:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": question},
        ]
        tool_traces: list[dict[str, Any]] = []
        last_answer = ""

        for turn_index in range(self.config.max_turns):
            self._progress(f"turn {turn_index + 1}/{self.config.max_turns}")
            raw = self._chat_completion(messages)
            self._log_model_reply(raw)
            parsed = self._parse_turn(raw)
            if parsed is None:
                final = raw.strip() or last_answer
                self._progress("model reply did not contain a valid JSON payload")
                return SolveResult(
                    final_answer=final,
                    tool_traces=tool_traces,
                    turn_count=turn_index + 1,
                    stop_reason="invalid_model_output",
                )

            if parsed.answer.strip():
                last_answer = parsed.answer.strip()

            if parsed.tool_call is None:
                return SolveResult(
                    final_answer=parsed.answer.strip() or last_answer,
                    tool_traces=tool_traces,
                    turn_count=turn_index + 1,
                    stop_reason="finalized",
                )

            if len(tool_traces) >= self.config.max_tool_calls:
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": "Tool call limit reached. Return JSON with tool_call=null.",
                    }
                )
                continue

            tool_name = parsed.tool_call.get("name")
            tool_args = parsed.tool_call.get("arguments", {})
            if not isinstance(tool_name, str) or not tool_name.strip() or not isinstance(tool_args, Mapping):
                self._progress("model emitted invalid tool_call shape; requesting corrected JSON")
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": "Invalid tool_call format. Use {name, arguments} or null.",
                    }
                )
                continue

            tool_name_str = tool_name.strip()
            tool_args_dict = dict(tool_args)
            self._log_tool_call(tool_name_str, tool_args_dict)
            tool_result = self.tool_registry.execute(tool_name_str, tool_args_dict)
            trace = {
                "turn": turn_index + 1,
                "name": tool_name_str,
                "arguments": tool_args_dict,
                "ok": tool_result.ok,
                "content": tool_result.content,
                "metadata": dict(tool_result.metadata),
            }
            tool_traces.append(trace)
            self._log_tool_result(trace)

            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (f"Tool result JSON:\n{json.dumps(trace, ensure_ascii=True)}\nIf you can finalize, return tool_call=null."),
                }
            )

        return SolveResult(
            final_answer=last_answer,
            tool_traces=tool_traces,
            turn_count=self.config.max_turns,
            stop_reason="max_turns_reached",
        )

    def _chat_completion(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.config.temperature,
        )
        content = response.choices[0].message.content

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        parts.append(text_value)
            return "\n".join(parts)
        return str(content)

    def _system_prompt(self) -> str:
        tool_lines: list[str] = []
        for spec in self.tool_registry.list_tools():
            tool_lines.append(f"- {spec.name}: {spec.description}; schema={json.dumps(spec.input_schema, ensure_ascii=True)}")

        if not tool_lines:
            tool_lines = ["- (no tools registered)"]

        return (
            "You are a math research assistant. "
            "Reply with exactly one JSON object and no extra prose. "
            'Schema: {"answer": string, "tool_call": null | {"name": string, "arguments": object}}. '
            "Use tools only when needed for computation or verification.\n"
            "Available tools:\n"
            f"{'\n'.join(tool_lines)}"
        )

    @classmethod
    def _parse_turn(cls, text: str) -> ParsedTurn | None:
        payload = cls._extract_json_payload(text)
        if payload is None:
            return None

        answer = payload.get("answer", "")
        if not isinstance(answer, str):
            return None

        tool_call = payload.get("tool_call")
        if tool_call is not None and not isinstance(tool_call, dict):
            return None

        return ParsedTurn(answer=answer, tool_call=tool_call)

    @staticmethod
    def _extract_json_payload(text: str) -> dict[str, Any] | None:
        stripped = text.strip()
        if not stripped:
            return None

        decoder = json.JSONDecoder()
        search_index = 0
        while True:
            brace_index = stripped.find("{", search_index)
            if brace_index < 0:
                break
            try:
                parsed, consumed = decoder.raw_decode(stripped[brace_index:])
            except json.JSONDecodeError:
                search_index = brace_index + 1
                continue
            if isinstance(parsed, dict):
                return parsed
            search_index = brace_index + max(consumed, 1)

        return None

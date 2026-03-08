from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from llmxm2.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ControllerConfig:
    max_turns: int = 6
    temperature: float = 0.0
    progress_logs: bool = False
    max_tool_calls: int = 4

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> "ControllerConfig":
        cfg_dict = dict(cfg or {})
        return cls(
            max_turns=int(cfg_dict.get("max_turns", 6)),
            temperature=float(cfg_dict.get("temperature", 0.0)),
            progress_logs=bool(cfg_dict.get("progress_logs", False)),
            max_tool_calls=int(cfg_dict.get("max_tool_calls", 4)),
        )


@dataclass(frozen=True)
class SolveResult:
    final_answer: str
    tool_traces: list[dict[str, Any]]
    turn_count: int
    stop_reason: str


@dataclass(frozen=True)
class ParsedTurn:
    answer: str
    tool_call: dict[str, Any] | None


class AgentController:
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
            print(f"[progress][controller] {message}", flush=True)

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
            parsed = self._parse_turn(raw)
            if parsed is None:
                final = raw.strip() or last_answer
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
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": "Invalid tool_call format. Use {name, arguments} or null.",
                    }
                )
                continue

            tool_result = self.tool_registry.execute(tool_name.strip(), dict(tool_args))
            trace = {
                "turn": turn_index + 1,
                "name": tool_name.strip(),
                "arguments": dict(tool_args),
                "ok": tool_result.ok,
                "content": tool_result.content,
                "metadata": dict(tool_result.metadata),
            }
            tool_traces.append(trace)

            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Tool result JSON:\n"
                        f"{json.dumps(trace, ensure_ascii=True)}\n"
                        "If you can finalize, return tool_call=null."
                    ),
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
            tool_lines.append(
                f"- {spec.name}: {spec.description}; schema={json.dumps(spec.input_schema, ensure_ascii=True)}"
            )

        if not tool_lines:
            tool_lines = ["- (no tools registered)"]

        return (
            "You are a math research assistant. "
            "Reply with exactly one JSON object and no extra prose. "
            "Schema: {\"answer\": string, \"tool_call\": null | {\"name\": string, \"arguments\": object}}. "
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

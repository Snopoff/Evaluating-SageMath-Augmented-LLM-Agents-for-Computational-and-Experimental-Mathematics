from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from src.tools.registry import ToolRegistry
from src.utils.console_logging import ConsoleLogger


@dataclass(frozen=True)
class ControllerConfig:
    """Runtime knobs for the tool-using chat loop.

    Args:
        max_steps: Maximum number of model steps before stopping.
        temperature: Sampling temperature passed to the chat backend.
        progress_logs: Whether to emit controller progress messages.
        max_tool_calls: Maximum number of tool dispatches allowed per solve.
        require_successful_tool_call_for_final: Whether a successful ``sage_exec``
            call is required before the controller accepts finalization.
        require_verification_for_final: Whether finalization requires an
            explicitly verified successful ``sage_exec`` result.
    """

    max_steps: int = 6
    temperature: float = 0.0
    progress_logs: bool = False
    max_tool_calls: int = 4
    require_successful_tool_call_for_final: bool = False
    require_verification_for_final: bool = False

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> ControllerConfig:
        cfg_dict = dict(cfg or {})
        return cls(
            max_steps=int(cfg_dict.get("max_steps", 6)),
            temperature=float(cfg_dict.get("temperature", 0.0)),
            progress_logs=bool(cfg_dict.get("progress_logs", False)),
            max_tool_calls=int(cfg_dict.get("max_tool_calls", 4)),
            require_successful_tool_call_for_final=bool(cfg_dict.get("require_successful_tool_call_for_final", False)),
            require_verification_for_final=bool(cfg_dict.get("require_verification_for_final", False)),
        )


@dataclass(frozen=True)
class SolveResult:
    """Final output and execution trace returned by the controller.

    Args:
        final_answer: Final answer returned to the caller.
        tool_traces: Per-tool execution records collected during the solve loop.
        turn_count: Number of model turns consumed.
        stop_reason: Terminal reason such as ``finalized`` or ``max_steps_reached``.
        verified_sage_code: Exact successful ``sage_exec`` snippet used to verify
            the finalized answer.
    """

    final_answer: str
    tool_traces: list[dict[str, Any]]
    turn_count: int
    stop_reason: str
    verified_sage_code: str = ""


@dataclass(frozen=True)
class ParsedTurn:
    """Structured representation of one model turn.

    Args:
        answer: Assistant answer extracted from the model payload.
        tool_call: Optional tool call payload with ``name`` and ``arguments``.
    """

    answer: str
    tool_call: dict[str, Any] | None


@dataclass(frozen=True)
class ModelCallResult:
    """Raw model response plus best-effort token usage metadata."""

    raw_response: str
    token_usage: dict[str, int | None]


class AgentController:
    """Runs the iterative chat loop and dispatches tool calls.

    Args:
        client: Chat-completions-compatible provider client.
        model_name: Model identifier passed to the provider.
        tool_registry: Registry used to expose and dispatch tools.
        config: Optional controller configuration. Defaults to ``ControllerConfig()``.
        logger: Optional experiment logger for recording progress and results.
        agent_id: Optional identifier for the agent instance, used in logging.
    """

    def __init__(
        self,
        client: Any,
        model_name: str,
        tool_registry: ToolRegistry,
        config: ControllerConfig | None = None,
        logger: ConsoleLogger | None = None,
        agent_id: str = "single_agent",
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.tool_registry = tool_registry
        self.config = config or ControllerConfig()
        self.logger = logger or ConsoleLogger()
        self.agent_id = agent_id.strip() or "single_agent"

    def _progress(self, message: str) -> None:
        if self.config.progress_logs:
            self.logger.progress(f"[bold orange1]\\[controller][/bold orange1] {message}")  # type: ignore because rich markup

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

    @staticmethod
    def _sage_trace_is_verified(trace: Mapping[str, Any] | None) -> bool:
        if not isinstance(trace, Mapping):
            return False
        metadata = trace.get("metadata")
        if not isinstance(metadata, Mapping):
            return False
        result_data = metadata.get("result_data")
        if not isinstance(result_data, Mapping):
            return False
        return result_data.get("verified") is True

    def solve(self, question: str) -> SolveResult:
        system_prompt = self._system_prompt()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        self.logger.start_run(
            metadata={
                "agent_id": self.agent_id,
                "question": question,
                "system_prompt": system_prompt,
                "model_name": self.model_name,
                "controller_config": asdict(self.config),
                "tool_specs": [asdict(spec) for spec in self.tool_registry.list_tools()],
            }
        )
        tool_traces: list[dict[str, Any]] = []
        last_answer = ""
        last_successful_sage_code = ""
        last_successful_sage_trace: dict[str, Any] | None = None

        for turn_index in range(self.config.max_steps):
            self._progress(f"turn {turn_index + 1}/{self.config.max_steps}")
            model_call = self._chat_completion(messages)
            raw = model_call.raw_response
            self._log_model_reply(raw)
            parsed = self._parse_turn(raw)
            self.logger.log_model_call(
                agent_id=self.agent_id,
                turn=turn_index + 1,
                model_name=self.model_name,
                messages=messages,
                raw_response=raw,
                parsed_payload=None if parsed is None else {"answer": parsed.answer, "tool_call": parsed.tool_call},
                token_usage=model_call.token_usage,
            )
            if parsed is None:
                final = raw.strip() or last_answer
                self._progress("model reply did not contain a valid JSON payload")
                return self._record_solve_result(
                    SolveResult(
                        final_answer=final,
                        tool_traces=tool_traces,
                        turn_count=turn_index + 1,
                        stop_reason="invalid_model_output",
                    )
                )

            if parsed.answer.strip():
                last_answer = parsed.answer.strip()

            if parsed.tool_call is None:
                if self.config.require_successful_tool_call_for_final and last_successful_sage_trace is None:
                    self._progress("rejecting finalization without a successful sage_exec call")
                    messages.append({"role": "assistant", "content": raw})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Do not finalize yet. Execute sage_exec again and only finalize after a successful verification "
                                'result with RESULT["verified"] = True.'
                            ),
                        }
                    )
                    continue
                if self.config.require_verification_for_final and not self._sage_trace_is_verified(last_successful_sage_trace):
                    self._progress("rejecting finalization without an explicitly verified sage_exec result")
                    messages.append({"role": "assistant", "content": raw})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Do not finalize yet. Execute sage_exec again and only finalize after a successful verification "
                                'result with RESULT["verified"] = True.'
                            ),
                        }
                    )
                    continue
                return self._record_solve_result(
                    SolveResult(
                        final_answer=parsed.answer.strip() or last_answer,
                        tool_traces=tool_traces,
                        turn_count=turn_index + 1,
                        stop_reason="finalized",
                        verified_sage_code=last_successful_sage_code,
                    )
                )

            if len(tool_traces) >= self.config.max_tool_calls:
                self._progress("tool call limit reached")
                return self._record_solve_result(
                    SolveResult(
                        final_answer=last_answer,
                        tool_traces=tool_traces,
                        turn_count=turn_index + 1,
                        stop_reason="max_tool_calls_reached",
                    )
                )

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
            self.logger.log_tool_call(
                agent_id=self.agent_id,
                turn=turn_index + 1,
                tool_name=tool_name_str,
                arguments=tool_args_dict,
            )
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
            self.logger.log_tool_result(
                agent_id=self.agent_id,
                turn=turn_index + 1,
                tool_name=tool_name_str,
                ok=tool_result.ok,
                content=tool_result.content,
                metadata=tool_result.metadata,
            )
            if tool_name_str == "sage_exec" and tool_result.ok:
                last_successful_sage_trace = trace
                code_value = tool_args_dict.get("code")
                if isinstance(code_value, str) and code_value.strip():
                    last_successful_sage_code = code_value

            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (f"Tool result JSON:\n{json.dumps(trace, ensure_ascii=True)}\nIf you can finalize, return tool_call=null."),
                }
            )

        return self._record_solve_result(
            SolveResult(
                final_answer=last_answer,
                tool_traces=tool_traces,
                turn_count=self.config.max_steps,
                stop_reason="max_steps_reached",
            )
        )

    def _record_solve_result(self, result: SolveResult) -> SolveResult:
        self.logger.log_solve_result(
            agent_id=self.agent_id,
            final_answer=result.final_answer,
            turn_count=result.turn_count,
            stop_reason=result.stop_reason,
            tool_traces=result.tool_traces,
            verified_sage_code=result.verified_sage_code,
        )
        return result

    def _chat_completion(self, messages: list[dict[str, str]]) -> ModelCallResult:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.config.temperature,
        )
        content = response.choices[0].message.content
        token_usage = self._extract_token_usage(response)

        if isinstance(content, str):
            return ModelCallResult(raw_response=content, token_usage=token_usage)
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        parts.append(text_value)
            return ModelCallResult(raw_response="\n".join(parts), token_usage=token_usage)
        return ModelCallResult(raw_response=str(content), token_usage=token_usage)

    @classmethod
    def _extract_token_usage(cls, response: Any) -> dict[str, int | None]:
        usage = cls._read_value(response, "usage")
        input_tokens = cls._read_int(usage, "input_tokens")
        if input_tokens is None:
            input_tokens = cls._read_int(usage, "prompt_tokens")

        output_tokens = cls._read_int(usage, "output_tokens")
        if output_tokens is None:
            output_tokens = cls._read_int(usage, "completion_tokens")

        total_tokens = cls._read_int(usage, "total_tokens")
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _read_value(value: Any, key: str) -> Any:
        if isinstance(value, Mapping):
            return value.get(key)
        return getattr(value, key, None)

    @classmethod
    def _read_int(cls, value: Any, key: str) -> int | None:
        candidate = cls._read_value(value, key)
        if candidate is None:
            return None
        if isinstance(candidate, bool):
            return int(candidate)
        if isinstance(candidate, int):
            return candidate
        if isinstance(candidate, float):
            return int(candidate)
        return None

    def _system_prompt(self) -> str:
        tool_lines: list[str] = []
        for spec in self.tool_registry.list_tools():
            tool_lines.append(f"- {spec.name}: {spec.description}; schema={json.dumps(spec.input_schema, ensure_ascii=True)}")

        if not tool_lines:
            tool_lines = ["- (no tools registered)"]

        verification_guardrail = ""
        if self.config.require_verification_for_final:
            verification_guardrail = (
                "Do not return tool_call=null until you have a successful sage_exec call that verifies the answer.\n"
                'When using Sage for verification, set RESULT["verified"] = True only if the candidate answer is actually verified; '
                "otherwise set it to False.\n"
            )
        elif self.config.require_successful_tool_call_for_final:
            verification_guardrail = "Do not return tool_call=null until you have a successful sage_exec call.\n"

        return (
            "You are a math research assistant. "
            "Reply with exactly one JSON object and no extra prose. "
            'Schema: {"answer": string, "tool_call": null | {"name": string, "arguments": object}}. '
            "Use tools only when needed for computation or verification.\n"
            f"{verification_guardrail}"
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

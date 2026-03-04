from __future__ import annotations

import ast
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from llmxm2.mcp.client import SageToolClient

_JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_TOOL_USE_MODES = {"auto", "required", "disabled"}


@dataclass(frozen=True)
class ToolBudget:
    """Limits for total tool calls and cumulative tool runtime."""

    max_calls: int = 8
    max_cumulative_cpu_seconds: float = 45.0
    max_cumulative_wall_seconds: float = 45.0


@dataclass(frozen=True)
class ControllerConfig:
    """Controller behavior settings for turns, tool policy, and budgets."""

    max_turns: int = 3
    temperature: float = 0.0
    progress_logs: bool = False
    tool_use_mode: str = "auto"
    min_required_tool_calls: int = 1
    tool_budget: ToolBudget = field(default_factory=ToolBudget)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> "ControllerConfig":
        cfg = dict(cfg or {})
        budget_cfg = dict(cfg.get("tool_budget", {}))
        tool_use_mode = str(cfg.get("tool_use_mode", "auto"))
        if tool_use_mode not in _TOOL_USE_MODES:
            raise ValueError(f"Invalid tool_use_mode={tool_use_mode!r}. Expected one of {_TOOL_USE_MODES}.")

        min_required_tool_calls = int(cfg.get("min_required_tool_calls", 1))
        if min_required_tool_calls < 1:
            min_required_tool_calls = 1

        return cls(
            max_turns=int(cfg.get("max_turns", 3)),
            temperature=float(cfg.get("temperature", 0.0)),
            progress_logs=bool(cfg.get("progress_logs", False)),
            tool_use_mode=tool_use_mode,
            min_required_tool_calls=min_required_tool_calls,
            tool_budget=ToolBudget(
                max_calls=int(budget_cfg.get("max_calls", 8)),
                max_cumulative_cpu_seconds=float(budget_cfg.get("max_cumulative_cpu_seconds", 45.0)),
                max_cumulative_wall_seconds=float(budget_cfg.get("max_cumulative_wall_seconds", 45.0)),
            ),
        )


@dataclass(frozen=True)
class ModelTurn:
    """Parsed model output for one turn in the controller loop."""

    needs_tool: bool
    draft_answer: str
    tool_request: dict[str, Any] | None


@dataclass(frozen=True)
class SolveResult:
    """Final answer plus execution trace metadata for one solve request."""

    final_answer: str
    tool_traces: list[dict[str, Any]]
    turn_count: int
    stop_reason: str


class AgentController:
    """Runs a guarded model-tool loop and returns a structured solve result."""

    def __init__(
        self,
        client: Any,
        model_name: str,
        tool_client: SageToolClient,
        config: ControllerConfig | None = None,
    ):
        self.client = client
        self.model_name = model_name
        self.tool_client = tool_client
        self.config = config or ControllerConfig()

    def _progress(self, message: str) -> None:
        if self.config.progress_logs:
            print(f"[progress][controller] {message}", flush=True)

    @staticmethod
    def _truncate(text: str, max_chars: int = 500) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    def solve(self, question: str) -> SolveResult:
        self._progress(f"solve started (question_chars={len(question)})")
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._question_prompt(question)},
        ]

        tool_traces: list[dict[str, Any]] = []
        tool_calls = 0
        cumulative_wall_ms = 0
        cumulative_cpu_ms = 0
        last_draft = ""

        for turn_index in range(self.config.max_turns):
            self._progress(f"turn {turn_index + 1}/{self.config.max_turns}: requesting model completion")
            try:
                raw_text = self._chat_completion(messages)
            except Exception as exc:
                self._progress(f"model API error: {self._truncate(str(exc), max_chars=300)}")
                return SolveResult(
                    final_answer=f"Model API error: {exc}",
                    tool_traces=tool_traces,
                    turn_count=turn_index + 1,
                    stop_reason="model_api_error",
                )

            parsed = self._parse_model_turn(raw_text)
            if parsed is None:
                if turn_index + 1 < self.config.max_turns:
                    self._progress("model response invalid JSON schema; requesting strict JSON retry")
                    messages.append({"role": "assistant", "content": self._history_assistant_content(raw_text)})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response was not parseable. "
                                "Return exactly one JSON object with keys "
                                "needs_tool (bool), draft_answer (string), tool_request (object|null). "
                                "Do not output any extra text."
                            ),
                        }
                    )
                    continue

                self._progress("model response was not valid JSON schema; returning raw text")
                return SolveResult(
                    final_answer=raw_text.strip(),
                    tool_traces=tool_traces,
                    turn_count=turn_index + 1,
                    stop_reason="non_json_model_output",
                )

            if parsed.draft_answer:
                last_draft = parsed.draft_answer
                self._progress(f"turn {turn_index + 1}: draft_answer={self._truncate(parsed.draft_answer)!r}")

            if parsed.needs_tool:
                if self.config.tool_use_mode == "disabled":
                    self._progress("tool usage disabled by config; requesting final answer without tool")
                    messages.append({"role": "assistant", "content": self._history_assistant_content(raw_text)})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Tool use is disabled for this run. "
                                "Return JSON with needs_tool=false and your best final answer."
                            ),
                        }
                    )
                    continue

                if not isinstance(parsed.tool_request, dict):
                    self._progress("tool_request missing/invalid; asking model to finalize without tool")
                    messages.append({"role": "assistant", "content": self._history_assistant_content(raw_text)})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your response requested a tool but did not provide a valid tool_request object. "
                                "Return valid JSON with needs_tool=false and your best final answer."
                            ),
                        }
                    )
                    continue

                normalized_tool_request = self._normalize_tool_request(parsed.tool_request)
                if normalized_tool_request != parsed.tool_request:
                    self._progress("normalized tool_request aliases for schema compatibility")

                local_request_error = self._validate_normalized_tool_request(normalized_tool_request)
                if local_request_error:
                    self._progress("tool_request failed local validation; requesting corrected tool request")
                    if turn_index + 1 >= self.config.max_turns:
                        final_answer = parsed.draft_answer.strip() or last_draft.strip()
                        if not final_answer:
                            final_answer = "Invalid tool request at final turn."
                        return SolveResult(
                            final_answer=final_answer,
                            tool_traces=tool_traces,
                            turn_count=turn_index + 1,
                            stop_reason="invalid_tool_request_final_turn",
                        )

                    messages.append({"role": "assistant", "content": self._history_assistant_content(raw_text)})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your tool_request was invalid: {local_request_error}. "
                                "Return JSON again with a corrected tool_request. "
                                "If no tool is needed, set needs_tool=false."
                            ),
                        }
                    )
                    continue

                if tool_calls >= self.config.tool_budget.max_calls:
                    self._progress("tool call budget exhausted; requesting final answer")
                    messages.append({"role": "assistant", "content": self._history_assistant_content(raw_text)})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Tool call budget exhausted. "
                                "Do not request additional tools. Return final JSON with needs_tool=false."
                            ),
                        }
                    )
                    continue

                if self._is_cumulative_budget_exhausted(cumulative_wall_ms, cumulative_cpu_ms):
                    self._progress("cumulative tool budget exhausted; requesting final answer")
                    messages.append({"role": "assistant", "content": self._history_assistant_content(raw_text)})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Cumulative tool time budget reached. "
                                "Do not request additional tools. Return final JSON with needs_tool=false."
                            ),
                        }
                    )
                    continue

                operation = normalized_tool_request.get("operation", "unknown")
                self._progress(
                    "tool_request payload="
                    f"{self._truncate(json.dumps(normalized_tool_request, ensure_ascii=True), max_chars=900)}"
                )
                self._progress(f"executing tool call #{tool_calls + 1} (operation={operation})")

                tool_response = self.tool_client.sage_eval(normalized_tool_request)
                tool_calls += 1
                runtime_ms = int(tool_response.get("runtime_ms", 0) or 0)
                cumulative_wall_ms += runtime_ms
                cumulative_cpu_ms += runtime_ms

                self._progress(
                    "tool response received "
                    f"(status={tool_response.get('status')}, runtime_ms={runtime_ms}, "
                    f"cumulative_ms={cumulative_wall_ms})"
                )

                tool_traces.append(
                    {
                        "turn": turn_index + 1,
                        "request": normalized_tool_request,
                        "response": tool_response,
                    }
                )

                messages.append({"role": "assistant", "content": self._history_assistant_content(raw_text)})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Tool response summary (use it only to verify/correct your draft):\n"
                            f"{json.dumps(self._compact_tool_response(tool_response), ensure_ascii=True)}\n"
                            "Now return JSON again. If enough evidence exists, set needs_tool=false."
                        ),
                    }
                )

                if self._is_cumulative_budget_exhausted(cumulative_wall_ms, cumulative_cpu_ms):
                    self._progress("cumulative tool budget reached; requesting final answer")
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Cumulative tool time budget reached. "
                                "Do not request additional tools. Return final JSON with needs_tool=false."
                            ),
                        }
                    )
                continue

            if self.config.tool_use_mode == "required" and tool_calls < self.config.min_required_tool_calls:
                if turn_index + 1 >= self.config.max_turns:
                    self._progress("required tool-call minimum unmet at final turn; returning latest draft")
                    final_answer = parsed.draft_answer.strip() or last_draft.strip()
                    if not final_answer:
                        final_answer = "Required tool-call minimum unmet at final turn."
                    return SolveResult(
                        final_answer=final_answer,
                        tool_traces=tool_traces,
                        turn_count=turn_index + 1,
                        stop_reason="required_tool_calls_unmet",
                    )

                self._progress(
                    "tool_use_mode=required and minimum tool calls not reached; "
                    f"requesting more tool use ({tool_calls}/{self.config.min_required_tool_calls})"
                )
                messages.append({"role": "assistant", "content": self._history_assistant_content(raw_text)})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Before finalizing, you must call sage_eval again for targeted verification. "
                            "Return JSON with needs_tool=true and a valid tool_request. "
                            f"Minimum required tool calls: {self.config.min_required_tool_calls}."
                        ),
                    }
                )
                continue

            self._progress(f"turn {turn_index + 1}: finalizing without additional tool call")
            final_answer = parsed.draft_answer.strip()
            if not final_answer:
                final_answer = raw_text.strip()
            return SolveResult(
                final_answer=final_answer,
                tool_traces=tool_traces,
                turn_count=turn_index + 1,
                stop_reason="finalized",
            )

        self._progress("max turns reached; returning latest draft")
        return SolveResult(
            final_answer=last_draft.strip() or "",
            tool_traces=tool_traces,
            turn_count=self.config.max_turns,
            stop_reason="max_turns_reached",
        )

    def _is_cumulative_budget_exhausted(self, cumulative_wall_ms: int, cumulative_cpu_ms: int) -> bool:
        wall_budget_ms = int(self.config.tool_budget.max_cumulative_wall_seconds * 1000)
        cpu_budget_ms = int(self.config.tool_budget.max_cumulative_cpu_seconds * 1000)
        return cumulative_wall_ms >= wall_budget_ms or cumulative_cpu_ms >= cpu_budget_ms

    def _chat_completion(self, messages: list[dict[str, str]]) -> str:
        self._progress("waiting for model API response")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.config.temperature,
        )
        self._progress("model API response received")
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

    def _history_assistant_content(self, raw_text: str) -> str:
        payload = self._extract_json_payload(raw_text)
        if not isinstance(payload, dict):
            return self._truncate(raw_text, max_chars=2000)

        compact: dict[str, Any] = {}
        needs_tool = payload.get("needs_tool")
        if isinstance(needs_tool, bool):
            compact["needs_tool"] = needs_tool

        draft = payload.get("draft_answer")
        if isinstance(draft, str):
            compact["draft_answer"] = self._truncate(draft, max_chars=600)

        tool_request = payload.get("tool_request")
        if isinstance(tool_request, dict):
            req_compact: dict[str, Any] = {}
            operation = tool_request.get("operation")
            if isinstance(operation, str):
                req_compact["operation"] = operation
            request_id = tool_request.get("request_id")
            if isinstance(request_id, str):
                req_compact["request_id"] = request_id
            budget_profile = tool_request.get("budget_profile")
            if isinstance(budget_profile, str):
                req_compact["budget_profile"] = budget_profile
            compact["tool_request"] = req_compact
        else:
            compact["tool_request"] = None

        return self._truncate(json.dumps(compact, ensure_ascii=True), max_chars=2000)

    def _compact_tool_response(self, tool_response: Mapping[str, Any]) -> dict[str, Any]:
        compact: dict[str, Any] = {
            "status": tool_response.get("status"),
            "error_code": tool_response.get("error_code"),
            "runtime_ms": tool_response.get("runtime_ms"),
        }
        plain = tool_response.get("result_plain")
        if isinstance(plain, str) and plain.strip():
            compact["result_plain"] = self._truncate(plain.strip(), max_chars=1200)

        complexity = tool_response.get("complexity_report")
        if isinstance(complexity, dict):
            compact["complexity_report"] = {
                "policy_decision": complexity.get("policy_decision"),
                "reason": self._truncate(str(complexity.get("reason", "")), max_chars=200),
            }
        return compact

    def _system_prompt(self) -> str:
        mode_instruction = (
            "Use sage_eval only when needed for computation or verification."
            if self.config.tool_use_mode == "auto"
            else (
                f"You must call sage_eval at least {self.config.min_required_tool_calls} time(s) before finalizing."
                if self.config.tool_use_mode == "required"
                else "Do not request sage_eval."
            )
        )

        return (
            "You are a computational commutative-algebra assistant with access to a constrained SageMath tool. "
            "Always answer with exactly one JSON object and no surrounding prose. "
            'Schema: {"needs_tool": bool, "draft_answer": str, "tool_request": object|null}. '
            "If you need computation or verification, set needs_tool=true and provide tool_request with keys "
            "{operation, args, assumptions, request_id, budget_profile}. "
            "Prefer sage_snippet for arbitrary Sage workflows or multi-step checks. "
            "For sage_snippet, provide args.code as full Python/Sage code and optional args.result_var. "
            "For generic callable execution, set operation to a Sage callable name and pass "
            "args.positional_args (list), optional args.keyword_args (object), and optional "
            "args.coerce_symbolic_strings (bool). "
            f"{mode_instruction} "
            "If you can finalize, set needs_tool=false and tool_request=null."
        )

    @staticmethod
    def _question_prompt(question: str) -> str:
        return (
            "Stage A (draft): reason about the question and draft an answer. "
            "Call the tool only for targeted compute/check steps.\n\n"
            "Question:\n"
            f"{question}"
        )

    @classmethod
    def _parse_model_turn(cls, text: str) -> ModelTurn | None:
        payload = cls._extract_json_payload(text)
        if payload is None:
            return None

        needs_tool = payload.get("needs_tool")
        if not isinstance(needs_tool, bool):
            return None

        draft_answer = payload.get("draft_answer", "")
        if draft_answer is None:
            draft_answer = ""
        if not isinstance(draft_answer, str):
            return None

        tool_request = payload.get("tool_request")
        if tool_request is not None and not isinstance(tool_request, dict):
            return None

        return ModelTurn(needs_tool=needs_tool, draft_answer=draft_answer, tool_request=tool_request)

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
                parsed_stream, consumed = decoder.raw_decode(stripped[brace_index:])
            except json.JSONDecodeError:
                search_index = brace_index + 1
                continue
            if isinstance(parsed_stream, dict):
                return parsed_stream
            search_index = brace_index + max(consumed, 1)

        match = _JSON_BLOCK_PATTERN.search(stripped)
        candidates: list[str] = []
        if match:
            candidates.append(match.group(1))

        if stripped.startswith("{") and stripped.endswith("}"):
            candidates.append(stripped)

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(stripped[start : end + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

        return None

    @staticmethod
    def _normalize_tool_request(tool_request: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(tool_request)

        args = normalized.get("args")
        if isinstance(args, dict):
            args_norm = dict(args)

            if "code" not in args_norm and isinstance(args_norm.get("script"), str):
                args_norm["code"] = args_norm["script"]
            if "result_var" not in args_norm and isinstance(args_norm.get("result"), str):
                args_norm["result_var"] = args_norm["result"]

            if "positional_args" not in args_norm and isinstance(args_norm.get("args"), list):
                args_norm["positional_args"] = args_norm["args"]
            if "keyword_args" not in args_norm and isinstance(args_norm.get("kwargs"), dict):
                args_norm["keyword_args"] = args_norm["kwargs"]

            normalized["args"] = args_norm

        assumptions = normalized.get("assumptions")
        if not isinstance(assumptions, dict):
            normalized["assumptions"] = {}

        request_id = normalized.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip():
            normalized["request_id"] = f"auto_{uuid.uuid4().hex[:12]}"

        budget_profile = normalized.get("budget_profile")
        if not isinstance(budget_profile, str) or not budget_profile.strip():
            normalized["budget_profile"] = "conservative"

        return normalized

    @staticmethod
    def _validate_normalized_tool_request(tool_request: Mapping[str, Any]) -> str | None:
        operation = tool_request.get("operation")
        if not isinstance(operation, str) or not operation.strip():
            return "tool_request.operation must be a non-empty string"

        args = tool_request.get("args")
        if not isinstance(args, Mapping):
            return "tool_request.args must be an object"

        if operation == "sage_snippet":
            code = args.get("code")
            if not isinstance(code, str) or not code.strip():
                return "sage_snippet requires args.code as a non-empty string"
            if "<<omitted" in code:
                return "sage_snippet args.code contains placeholder text"
            try:
                ast.parse(code)
            except SyntaxError as exc:
                return f"sage_snippet args.code has invalid Python syntax: {exc.msg}"
            return None

        positional_args = args.get("positional_args", [])
        if not isinstance(positional_args, list):
            return "generic operations require args.positional_args as a list"

        keyword_args = args.get("keyword_args", {})
        if not isinstance(keyword_args, Mapping):
            return "generic operations require args.keyword_args as an object"

        coerce_symbolic_strings = args.get("coerce_symbolic_strings")
        if coerce_symbolic_strings is not None and not isinstance(coerce_symbolic_strings, bool):
            return "generic operations require args.coerce_symbolic_strings as boolean when provided"

        return None

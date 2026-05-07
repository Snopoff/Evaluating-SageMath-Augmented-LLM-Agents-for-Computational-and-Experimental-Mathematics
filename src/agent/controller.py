import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from src.agent.schemas import FinalAnswerArgs, SageFinalAnswerArgs
from src.agent.verification import verification_passes
from src.agent.controller_utils import (
    answer_has_explicit_failure_language,
    extract_token_usage,
    forced_finalization_message,
    message_payload,
    messages_for_logging,
    preview_text,
    structured_final_request,
    trace_from_tool_message,
    trace_verification,
)
from src.tools.catalog import FINAL_ANSWER_TOOL_NAME, SAGE_EXEC_TOOL_NAME, make_submit_final_answer_tool
from src.utils.console_logging import ConsoleLogger


@dataclass(frozen=True)
class ControllerConfig:
    """Runtime knobs for one agent run."""

    max_steps: int = 6
    progress_logs: bool = False
    require_verification_for_final: bool = False


@dataclass(frozen=True)
class SolveResult:
    final_answer: str
    tool_traces: list[dict[str, Any]]
    turn_count: int
    stop_reason: str
    verified_sage_code: str = ""
    explanation: str = ""
    confidence: int = 0
    verified_claims: list[str] = field(default_factory=list)
    final_payload: dict[str, Any] = field(default_factory=dict)


class AgentController:
    """Runs either a plain structured LLM call or a tool loop."""

    def __init__(
        self,
        model: Any,
        tools: Sequence[BaseTool],
        config: ControllerConfig | None = None,
        logger: ConsoleLogger | None = None,
        agent_id: str = "single_agent",
        model_name: str = "",
        system_prompt: str = "",
    ) -> None:
        self.config = config or ControllerConfig()
        self.logger = logger or ConsoleLogger()
        self.agent_id = agent_id.strip() or "single_agent"
        self.model_name = model_name
        self.system_prompt = system_prompt.strip()

        runtime_tools = list(tools)
        self.uses_react = bool(runtime_tools)
        if self.uses_react and not any(tool.name == SAGE_EXEC_TOOL_NAME for tool in runtime_tools):
            raise ValueError("Tool-enabled AgentController currently requires the sage_exec tool.")

        self.tools = [*runtime_tools, make_submit_final_answer_tool(SageFinalAnswerArgs)] if self.uses_react else []
        self.tool_by_name = {tool.name: tool for tool in self.tools}
        self.model = (
            model.bind_tools(self.tools, parallel_tool_calls=False)
            if self.uses_react
            else model.with_structured_output(FinalAnswerArgs, include_raw=True)
        )

    def solve(self, question: str) -> SolveResult:
        messages: list[BaseMessage] = [SystemMessage(content=self.system_prompt)] if self.system_prompt else []
        messages.append(HumanMessage(content=question))

        self._start_run(question)
        if self.uses_react:
            return self._solve_with_tools(messages)
        return self._solve_plain(messages)

    def _solve_plain(self, messages: list[BaseMessage]) -> SolveResult:
        self._log("plain structured model call", level="turn", color="yellow")
        payload = self._invoke_structured_model(messages, turn=1)
        result = self._result_from_final_payload(
            payload=payload,
            tool_traces=[],
            turn_count=1,
            stop_reason="finalized",
            verified_sage_code="",
        )
        self._log_solve_result(result)
        return result

    def _solve_with_tools(self, messages: list[BaseMessage]) -> SolveResult:
        tool_traces: list[dict[str, Any]] = []
        last_successful_sage_code = ""
        last_successful_sage_trace: dict[str, Any] | None = None
        last_successful_verification_code = ""
        last_successful_verification_trace: dict[str, Any] | None = None

        for turn_index in range(self.config.max_steps):
            turn = turn_index + 1
            self._log(f"{turn}/{self.config.max_steps}", level="turn", color="yellow")
            ai_message = self._invoke_and_log_tool_model(messages, turn=turn)
            tool_calls = list(ai_message.tool_calls or [])

            if not tool_calls:
                messages.extend([ai_message, HumanMessage(content=structured_final_request())])
                continue

            if len(tool_calls) > 1:
                messages.append(ai_message)
                for i, call in enumerate(tool_calls, start=1):
                    messages.append(
                        ToolMessage(
                            content="Rejected tool call. Call at most one tool at a time.",
                            name=str(call.get("name", "")),
                            tool_call_id=str(call.get("id") or f"call_{turn}_{i}"),
                            artifact={"ok": False, "status": "parallel_tool_calls_rejected"},
                            status="error",
                        )
                    )
                continue

            messages.append(ai_message)
            tool_call = tool_calls[0]
            tool_name = str(tool_call.get("name", ""))
            tool_args = dict(tool_call.get("args", {}))
            tool_call_id = str(tool_call.get("id") or f"call_{turn}_{len(tool_traces) + 1}")

            if tool_name == FINAL_ANSWER_TOOL_NAME:
                final_payload, rejection = self._read_final_answer(
                    tool_args,
                    last_successful_sage_trace=last_successful_sage_trace,
                    last_successful_verification_trace=last_successful_verification_trace,
                    forced=False,
                )
                if rejection is not None:
                    messages.append(ToolMessage(content=rejection, name=tool_name, tool_call_id=tool_call_id, status="error"))
                    continue
                result = self._result_from_final_payload(
                    payload=final_payload,
                    tool_traces=tool_traces,
                    turn_count=turn,
                    stop_reason="finalized",
                    verified_sage_code=last_successful_verification_code or last_successful_sage_code,
                )
                self._log_solve_result(result)
                return result

            self._log_tool_call(tool_name, tool_args)
            self.logger.log_tool_call(agent_id=self.agent_id, turn=turn, tool_name=tool_name, arguments=tool_args)
            tool_message = self._execute_tool(tool_name, tool_args, tool_call_id)
            trace = trace_from_tool_message(turn, tool_name, tool_args, tool_message)
            tool_traces.append(trace)
            self._log_tool_result(trace)
            self.logger.log_tool_result(
                agent_id=self.agent_id,
                turn=turn,
                tool_name=tool_name,
                ok=bool(trace["ok"]),
                content=str(trace["content"]),
                metadata=trace["metadata"],
            )

            if tool_name == SAGE_EXEC_TOOL_NAME and trace["ok"]:
                last_successful_sage_trace = trace
                code_value = tool_args.get("code")
                if isinstance(code_value, str) and code_value.strip():
                    last_successful_sage_code = code_value
                if trace_verification(trace) is not None:
                    last_successful_verification_trace = trace
                    last_successful_verification_code = code_value if isinstance(code_value, str) and code_value.strip() else ""

            messages.append(tool_message)

        return self._force_finalization(
            messages=messages,
            tool_traces=tool_traces,
            last_successful_sage_trace=last_successful_sage_trace,
            last_successful_verification_trace=last_successful_verification_trace,
            verified_sage_code=last_successful_verification_code or last_successful_sage_code,
        )

    def _force_finalization(
        self,
        *,
        messages: list[BaseMessage],
        tool_traces: list[dict[str, Any]],
        last_successful_sage_trace: Mapping[str, Any] | None,
        last_successful_verification_trace: Mapping[str, Any] | None,
        verified_sage_code: str,
    ) -> SolveResult:
        self._log("step limit reached; requesting forced final answer")
        messages.append(HumanMessage(content=forced_finalization_message(last_successful_sage_trace)))
        turn = self.config.max_steps + 1
        ai_message = self._invoke_and_log_tool_model(messages, turn=turn)

        stop_reason = "forced_finalized"
        if last_successful_sage_trace is None:
            stop_reason = "forced_finalized_without_successful_sage"
        if self.uses_react and self.config.require_verification_for_final:
            verification_ok, _ = verification_passes(trace_verification(last_successful_verification_trace))
            if not verification_ok:
                stop_reason = "forced_finalized_without_verification"

        for tool_call in list(ai_message.tool_calls or []):
            if str(tool_call.get("name", "")) != FINAL_ANSWER_TOOL_NAME:
                continue
            final_payload, rejection = self._read_final_answer(
                dict(tool_call.get("args", {})),
                last_successful_sage_trace=last_successful_sage_trace,
                last_successful_verification_trace=last_successful_verification_trace,
                forced=True,
            )
            if rejection is None:
                result = self._result_from_final_payload(
                    payload=final_payload,
                    tool_traces=tool_traces,
                    turn_count=turn,
                    stop_reason=stop_reason,
                    verified_sage_code=verified_sage_code,
                )
                self._log_solve_result(result)
                return result

        result = SolveResult(
            final_answer="",
            explanation="Forced finalization failed: the model did not call submit_final_answer with valid arguments.",
            confidence=1,
            final_payload={"final_answer": "", "explanation": "Forced finalization failed.", "confidence": 1},
            tool_traces=tool_traces,
            turn_count=turn,
            stop_reason="forced_finalization_failed",
            verified_sage_code=verified_sage_code,
        )
        self._log_solve_result(result)
        return result

    def _invoke_structured_model(self, messages: list[BaseMessage], *, turn: int) -> FinalAnswerArgs:
        response = self.model.invoke(messages)
        raw = response.get("raw")
        raw_message = raw if isinstance(raw, AIMessage) else None
        parsing_error: Any = response.get("parsing_error")
        parsed: Any = response.get("parsed")
        if parsing_error is not None:
            raise ValueError(f"Structured output parsing failed: {parsing_error}") from parsing_error

        self.logger.log_model_call(
            agent_id=self.agent_id,
            turn=turn,
            model_name=self.model_name,
            messages=messages_for_logging(messages),
            raw_response=json.dumps(message_payload(raw_message), ensure_ascii=True),
            parsed_payload=parsed.model_dump(),
            token_usage=extract_token_usage(raw_message),
        )
        return parsed

    def _invoke_and_log_tool_model(self, messages: list[BaseMessage], *, turn: int) -> AIMessage:
        response = self.model.invoke(messages)
        self._log_model_reply(response)

        self.logger.log_model_call(
            agent_id=self.agent_id,
            turn=turn,
            model_name=self.model_name,
            messages=messages_for_logging(messages),
            raw_response=json.dumps(message_payload(response), ensure_ascii=True),
            parsed_payload=message_payload(response),
            token_usage=extract_token_usage(response),
        )
        return response

    def _execute_tool(self, tool_name: str, tool_args: dict[str, Any], tool_call_id: str) -> ToolMessage:
        selected_tool = self.tool_by_name.get(tool_name)
        if selected_tool is None:
            return ToolMessage(
                content=f"Unknown tool: {tool_name}",
                name=tool_name,
                tool_call_id=tool_call_id,
                artifact={"ok": False, "status": "unknown_tool"},
                status="error",
            )
        try:
            return selected_tool.invoke({"type": "tool_call", "id": tool_call_id, "name": tool_name, "args": tool_args})
        except Exception as exc:  # noqa: BLE001 - tool errors should be returned to the model
            return ToolMessage(
                content=f"Tool error: {exc}",
                name=tool_name,
                tool_call_id=tool_call_id,
                artifact={"ok": False, "status": "tool_error", "error": str(exc)},
                status="error",
            )

    def _read_final_answer(
        self,
        tool_args: dict[str, Any],
        *,
        last_successful_sage_trace: Mapping[str, Any] | None,
        last_successful_verification_trace: Mapping[str, Any] | None,
        forced: bool,
    ) -> tuple[FinalAnswerArgs | SageFinalAnswerArgs | None, str | None]:
        try:
            payload = (SageFinalAnswerArgs if self.uses_react else FinalAnswerArgs).model_validate(tool_args)
        except ValidationError as exc:
            return None, f"Rejected final answer. Invalid {FINAL_ANSWER_TOOL_NAME} arguments: {exc}"

        if forced:
            return payload, None

        rejection = self._finalization_rejection(
            final_answer=payload.final_answer,
            last_successful_sage_trace=last_successful_sage_trace,
            last_successful_verification_trace=last_successful_verification_trace,
        )
        return payload, rejection

    def _finalization_rejection(
        self,
        *,
        final_answer: str,
        last_successful_sage_trace: Mapping[str, Any] | None,
        last_successful_verification_trace: Mapping[str, Any] | None,
    ) -> str | None:
        if not final_answer.strip():
            return "Rejected final answer. The final answer must be non-empty."
        if self.uses_react and last_successful_sage_trace is None:
            return f"Rejected final answer. Execute {SAGE_EXEC_TOOL_NAME} successfully before finalizing."
        if self.uses_react and self.config.require_verification_for_final:
            verification_ok, failures = verification_passes(trace_verification(last_successful_verification_trace))
            if not verification_ok:
                details = "; ".join(failures[:4]) if failures else "verification is incomplete"
                return f"Rejected final answer. The latest verification is insufficient: {details}."
        if answer_has_explicit_failure_language(final_answer):
            return "Rejected final answer. Do not finalize with language that admits failed or unresolved constraints."
        return None

    def _result_from_final_payload(
        self,
        *,
        payload: FinalAnswerArgs | SageFinalAnswerArgs,
        tool_traces: list[dict[str, Any]],
        turn_count: int,
        stop_reason: str,
        verified_sage_code: str,
    ) -> SolveResult:
        final_payload = payload.model_dump()
        verified_claims = list(getattr(payload, "verified_claims", []) or [])
        if self.uses_react:
            final_payload["verified_claims"] = verified_claims
            final_payload["sage_code"] = verified_sage_code

        return SolveResult(
            final_answer=payload.final_answer.strip(),
            explanation=payload.explanation.strip(),
            confidence=payload.confidence,
            final_payload=final_payload,
            verified_claims=verified_claims,
            tool_traces=tool_traces,
            turn_count=turn_count,
            stop_reason=stop_reason,
            verified_sage_code=verified_sage_code,
        )

    def _start_run(self, question: str) -> None:
        self.logger.start_run(
            metadata={
                "agent_id": self.agent_id,
                "agent_mode": "react" if self.uses_react else "plain",
                "question": question,
                "system_prompt": self.system_prompt,
                "model_name": self.model_name,
                "controller_config": asdict(self.config),
                "structured_output_schema": (SageFinalAnswerArgs if self.uses_react else FinalAnswerArgs).model_json_schema(),
                "tool_specs": [{"name": tool.name} for tool in self.tools],
            }
        )

    def _log_solve_result(self, result: SolveResult) -> None:
        self.logger.log_solve_result(
            agent_id=self.agent_id,
            final_answer=result.final_answer,
            turn_count=result.turn_count,
            stop_reason=result.stop_reason,
            tool_traces=result.tool_traces,
            verified_sage_code=result.verified_sage_code,
            explanation=result.explanation,
            confidence=result.confidence,
            verified_claims=result.verified_claims,
            final_payload=result.final_payload,
        )

    def _log(self, *args, **kwargs) -> None:
        if self.config.progress_logs:
            self.logger.log(*args, **kwargs)

    def _log_model_reply(self, message: AIMessage) -> None:
        self._log(preview_text(json.dumps(message_payload(message), ensure_ascii=True)), level="model_reply", color="green")

    def _log_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        self._log(
            f"{tool_name} args={preview_text(json.dumps(arguments, ensure_ascii=True), max_chars=320)}",
            level="tool_call",
            color="blue",
        )

    def _log_tool_result(self, trace: Mapping[str, Any]) -> None:
        metadata = trace.get("metadata", {})
        status = metadata.get("status", "unknown") if isinstance(metadata, Mapping) else "unknown"
        self._log(
            f"{trace.get('name', '?')} ok={trace.get('ok', False)} status={status} content={preview_text(str(trace.get('content', '')))}",
            level="tool_result",
            color="magenta",
        )

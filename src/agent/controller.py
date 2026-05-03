import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from src.agent.schemas import FinalAnswerArgs
from src.agent.verification import verification_passes
from src.tools.catalog import FINAL_ANSWER_TOOL_NAME, SAGE_EXEC_TOOL_NAME, make_submit_final_answer_tool
from src.utils.console_logging import ConsoleLogger


@dataclass(frozen=True)
class ControllerConfig:
    """Runtime knobs for the lightweight LangChain tool loop."""

    max_steps: int = 6
    progress_logs: bool = False
    require_verification_for_final: bool = False
    require_structured_final: bool = True


@dataclass(frozen=True)
class SolveResult:
    final_answer: str
    tool_traces: list[dict[str, Any]]
    turn_count: int
    stop_reason: str
    verified_sage_code: str = ""
    explanation: str = ""
    verified_claims: list[str] = field(default_factory=list)
    final_payload: dict[str, Any] = field(default_factory=dict)


class AgentController:
    """Runs a small model-agnostic LangChain tool-calling loop."""

    def __init__(
        self,
        model: BaseChatModel,
        tools: Sequence[BaseTool],
        config: ControllerConfig | None = None,
        logger: ConsoleLogger | None = None,
        agent_id: str = "single_agent",
        model_name: str | None = None,
        system_prompt: str = "",
    ) -> None:
        self.model = model
        self.config = config or ControllerConfig()
        self.logger = logger or ConsoleLogger()
        self.agent_id = agent_id.strip() or "single_agent"
        self.model_name = model_name or self._infer_model_name(model)
        self.system_prompt = system_prompt.strip()
        self.submit_final_answer_tool = make_submit_final_answer_tool()
        self.tools = [*tools, self.submit_final_answer_tool]
        self.tool_by_name = {tool.name: tool for tool in self.tools}
        self.bound_model = self._bind_model_tools(model, self.tools)

    def _bind_model_tools(self, model: BaseChatModel, tools: Sequence[BaseTool]) -> Runnable[LanguageModelInput, AIMessage]:
        if not hasattr(model, "bind_tools"):
            raise TypeError("Configured model must provide bind_tools(...).")
        try:
            return model.bind_tools(tools, parallel_tool_calls=False)
        except TypeError:
            return model.bind_tools(tools)

    @staticmethod
    def _infer_model_name(model: BaseChatModel) -> str:
        for attr in ("model_name", "model", "model_id"):
            value = getattr(model, attr, None)
            if isinstance(value, str) and value.strip():
                return value
        return model.__class__.__name__

    def _log(self, *args, **kwargs) -> None:
        if self.config.progress_logs:
            self.logger.log(*args, **kwargs)

    @staticmethod
    def _preview_text(text: str, max_chars: int = 240) -> str:
        compact = " ".join(text.strip().split())
        if len(compact) <= max_chars:
            return compact
        return f"{compact[: max_chars - 3]}..."

    def _log_model_reply(self, message: AIMessage) -> None:
        self._log(self._preview_text(json.dumps(self._message_payload(message), ensure_ascii=True)), level="model_reply", color="green")

    def _log_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        self._log(
            f"{tool_name} args={self._preview_text(json.dumps(arguments, ensure_ascii=True), max_chars=320)}",
            level="tool_call",
            color="blue",
        )

    def _log_tool_result(self, trace: Mapping[str, Any]) -> None:
        metadata = trace.get("metadata", {})
        status = metadata.get("status", "unknown") if isinstance(metadata, Mapping) else "unknown"
        self._log(
            f"{trace.get('name', '?')} ok={trace.get('ok', False)} status={status} "
            f"content={self._preview_text(str(trace.get('content', '')))}",
            level="tool_result",
            color="magenta",
        )

    @staticmethod
    def _trace_verification(trace: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(trace, Mapping):
            return None
        metadata = trace.get("metadata")
        if not isinstance(metadata, Mapping):
            return None
        verification = metadata.get("verification")
        if not isinstance(verification, Mapping):
            return None
        return dict(verification)

    @staticmethod
    def _answer_has_explicit_failure_language(answer: str) -> bool:
        lowered = answer.lower()
        failure_markers = (
            "not satisfied",
            "does not satisfy",
            "constraint failed",
            "constraint remains failed",
            "failed verification",
            "not verified",
            "cannot verify",
            "could not verify",
            "unresolved",
        )
        return any(marker in lowered for marker in failure_markers)

    def solve(self, question: str) -> SolveResult:
        messages: list[BaseMessage] = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(HumanMessage(content=question))
        self.logger.start_run(
            metadata={
                "agent_id": self.agent_id,
                "question": question,
                "system_prompt": self.system_prompt,
                "model_name": self.model_name,
                "controller_config": asdict(self.config),
                "tool_specs": [self._tool_spec_for_logging(tool) for tool in self.tools],
            }
        )

        tool_traces: list[dict[str, Any]] = []
        last_successful_sage_code = ""
        last_successful_sage_trace: dict[str, Any] | None = None
        last_successful_verification_code = ""
        last_successful_verification_trace: dict[str, Any] | None = None

        for turn_index in range(self.config.max_steps):
            self._log(f"{turn_index + 1}/{self.config.max_steps}", level="turn", color="yellow")
            ai_message: AIMessage = self._invoke_and_log_model(messages, turn=turn_index + 1)

            content: str = self._message_text(ai_message).strip()
            tool_calls: list = list(ai_message.tool_calls or [])
            if not tool_calls:
                if self.config.require_structured_final:
                    messages.extend(
                        [
                            ai_message,
                            HumanMessage(content=f"Use the {FINAL_ANSWER_TOOL_NAME} tool to submit the final answer."),
                        ]
                    )
                    continue

                rejection: str | None = self._finalization_rejection(
                    final_answer=content,
                    last_successful_sage_trace=last_successful_sage_trace,
                    last_successful_verification_trace=last_successful_verification_trace,
                )
                if rejection is not None:
                    messages.extend([ai_message, HumanMessage(content=rejection)])
                    continue
                result = SolveResult(
                    final_answer=content,
                    tool_traces=tool_traces,
                    turn_count=turn_index + 1,
                    stop_reason="finalized",
                    verified_sage_code=last_successful_verification_code or last_successful_sage_code,
                )
                self._log_solve_result(result)
                return result

            if len(tool_calls) > 1:
                messages.append(ai_message)
                for index, tool_call in enumerate(tool_calls, start=1):
                    tool_name: str = str(tool_call.get("name", ""))
                    tool_call_id: str = str(tool_call.get("id") or f"call_{turn_index + 1}_{index}")
                    messages.append(
                        ToolMessage(
                            content="Rejected tool call. Call at most one tool at a time.",
                            name=tool_name,
                            tool_call_id=tool_call_id,
                            artifact={"ok": False, "status": "parallel_tool_calls_rejected"},
                            status="error",
                        )
                    )
                continue

            messages.append(ai_message)
            for tool_call in tool_calls:
                tool_name: str = str(tool_call.get("name", ""))
                tool_args: dict = dict(tool_call.get("args", {}))
                tool_call_id: str = str(tool_call.get("id") or f"call_{turn_index + 1}_{len(tool_traces) + 1}")

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
                    result = SolveResult(
                        final_answer=final_payload.final_answer.strip(),
                        tool_traces=tool_traces,
                        turn_count=turn_index + 1,
                        stop_reason="finalized",
                        verified_sage_code=last_successful_verification_code or last_successful_sage_code,
                        explanation=final_payload.explanation.strip(),
                        verified_claims=final_payload.verified_claims,
                        final_payload=final_payload.model_dump(),
                    )
                    self._log_solve_result(result)
                    return result

                self._log_tool_call(tool_name, tool_args)
                self.logger.log_tool_call(agent_id=self.agent_id, turn=turn_index + 1, tool_name=tool_name, arguments=tool_args)
                tool_message: ToolMessage = self._execute_tool(tool_name, tool_args, tool_call_id)
                trace: dict = self._trace_from_tool_message(turn_index + 1, tool_name, tool_args, tool_message)
                tool_traces.append(trace)
                self._log_tool_result(trace)
                self.logger.log_tool_result(
                    agent_id=self.agent_id,
                    turn=turn_index + 1,
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
                    if self._trace_verification(trace) is not None:
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

    def _invoke_and_log_model(self, messages: list[BaseMessage], *, turn: int) -> AIMessage:
        ai_message: AIMessage = self._invoke_model(messages)
        self._log_model_reply(ai_message)
        self.logger.log_model_call(
            agent_id=self.agent_id,
            turn=turn,
            model_name=self.model_name,
            messages=self._messages_for_logging(messages),
            raw_response=json.dumps(self._message_payload(ai_message), ensure_ascii=True),
            parsed_payload=self._message_payload(ai_message),
            token_usage=self._extract_token_usage(ai_message),
        )
        return ai_message

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
        messages.append(HumanMessage(content=self._forced_finalization_message(last_successful_sage_trace)))
        turn = self.config.max_steps + 1
        ai_message = self._invoke_and_log_model(messages, turn=turn)

        stop_reason = self._forced_stop_reason(
            last_successful_sage_trace=last_successful_sage_trace,
            last_successful_verification_trace=last_successful_verification_trace,
        )
        tool_calls = list(ai_message.tool_calls or [])
        for tool_call in tool_calls:
            tool_name = str(tool_call.get("name", ""))
            if tool_name != FINAL_ANSWER_TOOL_NAME:
                continue
            final_payload, rejection = self._read_final_answer(
                dict(tool_call.get("args", {})),
                last_successful_sage_trace=last_successful_sage_trace,
                last_successful_verification_trace=last_successful_verification_trace,
                forced=True,
            )
            if rejection is None:
                result = SolveResult(
                    final_answer=final_payload.final_answer.strip(),
                    tool_traces=tool_traces,
                    turn_count=turn,
                    stop_reason=stop_reason,
                    verified_sage_code=verified_sage_code,
                    explanation=final_payload.explanation.strip(),
                    verified_claims=final_payload.verified_claims,
                    final_payload=final_payload.model_dump(),
                )
                self._log_solve_result(result)
                return result

        result = SolveResult(
            final_answer="",
            tool_traces=tool_traces,
            turn_count=turn,
            stop_reason="forced_finalization_failed",
            verified_sage_code=verified_sage_code,
            explanation="Forced finalization failed: the model did not call submit_final_answer with valid arguments.",
        )
        self._log_solve_result(result)
        return result

    def _forced_stop_reason(
        self,
        *,
        last_successful_sage_trace: Mapping[str, Any] | None,
        last_successful_verification_trace: Mapping[str, Any] | None,
    ) -> str:
        if last_successful_sage_trace is None:
            return "forced_finalized_without_successful_sage"
        if self.config.require_verification_for_final:
            verification_ok, _ = verification_passes(self._trace_verification(last_successful_verification_trace))
            if not verification_ok:
                return "forced_finalized_without_verification"
        return "forced_finalized"

    @staticmethod
    def _forced_finalization_message(last_successful_sage_trace: Mapping[str, Any] | None) -> str:
        evidence_note = (
            "Use the successful Sage evidence already in the conversation."
            if last_successful_sage_trace is not None
            else "No successful Sage execution is available; say explicitly that the answer is not CAS-verified."
        )
        return (
            "The step limit has been reached. Do not call sage_exec again. "
            f"{evidence_note} Call submit_final_answer now with the best final answer you can justify. "
            "Put the exact checkable result in final_answer and the context in explanation. "
            "If the evidence is incomplete, state what is verified and what remains unverified in explanation."
        )

    def _invoke_model(self, messages: list[BaseMessage]) -> AIMessage:
        response = self.bound_model.invoke(messages)
        if not isinstance(response, AIMessage):
            raise TypeError(f"Bound chat model returned {type(response).__name__}, expected AIMessage.")
        return response

    def _execute_tool(self, tool_name: str, tool_args: dict[str, Any], tool_call_id: str) -> ToolMessage:
        selected_tool: BaseTool | None = self.tool_by_name.get(tool_name)
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
    ) -> tuple[FinalAnswerArgs | None, str | None]:
        try:
            payload = FinalAnswerArgs.model_validate(tool_args)
        except ValidationError as exc:
            return None, f"Rejected final answer. Invalid {FINAL_ANSWER_TOOL_NAME} arguments: {exc}"

        final_answer = payload.final_answer.strip()
        if not payload.explanation.strip():
            return None, "Rejected final answer. The explanation must be non-empty."
        if forced:
            if not final_answer:
                return None, "Rejected final answer. The final answer must be non-empty."
            return payload, None
        rejection = self._finalization_rejection(
            final_answer=final_answer,
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
        if last_successful_sage_trace is None:
            return f"Rejected final answer. Execute {SAGE_EXEC_TOOL_NAME} successfully before finalizing."
        if self.config.require_verification_for_final:
            verification_ok, failures = verification_passes(self._trace_verification(last_successful_verification_trace))
            if not verification_ok:
                details = "; ".join(failures[:4]) if failures else "verification is incomplete"
                return f"Rejected final answer. The latest verification is insufficient: {details}."
        if self._answer_has_explicit_failure_language(final_answer):
            return "Rejected final answer. Do not finalize with language that admits failed or unresolved constraints."
        return None

    @staticmethod
    def _trace_from_tool_message(turn: int, tool_name: str, tool_args: dict[str, Any], message: ToolMessage) -> dict[str, Any]:
        artifact = message.artifact if isinstance(message.artifact, Mapping) else {}
        ok = bool(artifact.get("ok", message.status != "error"))
        metadata = dict(artifact)
        return {
            "turn": turn,
            "name": tool_name,
            "arguments": tool_args,
            "ok": ok,
            "content": AgentController._message_text(message),
            "metadata": metadata,
        }

    def _log_solve_result(self, result: SolveResult) -> None:
        self.logger.log_solve_result(
            agent_id=self.agent_id,
            final_answer=result.final_answer,
            turn_count=result.turn_count,
            stop_reason=result.stop_reason,
            tool_traces=result.tool_traces,
            verified_sage_code=result.verified_sage_code,
            explanation=result.explanation,
            verified_claims=result.verified_claims,
            final_payload=result.final_payload,
        )

    @staticmethod
    def _message_text(message: BaseMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, Mapping):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
        return "" if content is None else str(content)

    @staticmethod
    def _message_payload(message: BaseMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": message.type,
            "content": AgentController._message_text(message),
        }
        if isinstance(message, AIMessage):
            payload["tool_calls"] = [
                {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "args": item.get("args", {}),
                }
                for item in (message.tool_calls or [])
            ]
        if isinstance(message, ToolMessage):
            payload["name"] = message.name
            payload["tool_call_id"] = message.tool_call_id
            payload["status"] = message.status
            if isinstance(message.artifact, Mapping):
                payload["artifact"] = dict(message.artifact)
        return payload

    @staticmethod
    def _messages_for_logging(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
        return [AgentController._message_payload(message) for message in messages]

    @staticmethod
    def _extract_token_usage(message: AIMessage) -> dict[str, int | None]:
        usage = message.usage_metadata or {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")
        return {
            "input_tokens": input_tokens if isinstance(input_tokens, int) else None,
            "output_tokens": output_tokens if isinstance(output_tokens, int) else None,
            "total_tokens": total_tokens if isinstance(total_tokens, int) else None,
        }

    @staticmethod
    def _tool_spec_for_logging(tool: BaseTool) -> dict[str, Any]:
        args_schema = getattr(tool, "args_schema", None)
        schema: dict[str, Any] = {}
        if hasattr(args_schema, "model_json_schema"):
            schema = args_schema.model_json_schema()  # type: ignore
        elif isinstance(args_schema, Mapping):
            schema = dict(args_schema)
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": schema,
        }

from typing import Any, Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from src.tools.catalog import FINAL_ANSWER_TOOL_NAME


def preview_text(text: str, max_chars: int = 240) -> str:
    compact = " ".join(text.strip().split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3]}..."


def message_text(message: BaseMessage) -> str:
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


def message_payload(message: BaseMessage | None) -> dict[str, Any]:
    if message is None:
        return {}
    payload: dict[str, Any] = {
        "role": message.type,
        "content": message_text(message),
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


def messages_for_logging(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    return [message_payload(message) for message in messages]


def extract_token_usage(message: AIMessage | None) -> dict[str, int | None]:
    usage = (message.usage_metadata or {}) if message is not None else {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")
    return {
        "input_tokens": input_tokens if isinstance(input_tokens, int) else None,
        "output_tokens": output_tokens if isinstance(output_tokens, int) else None,
        "total_tokens": total_tokens if isinstance(total_tokens, int) else None,
    }


def trace_from_tool_message(turn: int, tool_name: str, tool_args: dict[str, Any], message: ToolMessage) -> dict[str, Any]:
    artifact = message.artifact if isinstance(message.artifact, Mapping) else {}
    ok = bool(artifact.get("ok", message.status != "error"))
    return {
        "turn": turn,
        "name": tool_name,
        "arguments": tool_args,
        "ok": ok,
        "content": message_text(message),
        "metadata": dict(artifact),
    }


def trace_verification(trace: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(trace, Mapping):
        return None
    metadata = trace.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    verification = metadata.get("verification")
    if not isinstance(verification, Mapping):
        return None
    return dict(verification)


def answer_has_explicit_failure_language(answer: str) -> bool:
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


def structured_final_request() -> str:
    return (
        f"Use the {FINAL_ANSWER_TOOL_NAME} tool to submit the final answer. "
        "Provide final_answer, explanation, confidence as an integer from 1 to 5, "
        "and verified_claims supported by successful Sage output or explicit reasoning."
    )


def forced_finalization_message(last_successful_sage_trace: Mapping[str, Any] | None) -> str:
    evidence_note = (
        "Use the successful Sage evidences already in the conversation."
        if last_successful_sage_trace is not None
        else "No successful Sage execution is available; say explicitly that the answer is not CAS-verified."
    )
    return (
        "The step limit has been reached. Do not call sage_exec again. "
        f"{evidence_note} Call `submit_final_answer` now with the best final answer you can justify. "
        "Put the exact checkable result in final_answer, the context in explanation, an integer confidence from 1 to 5, "
        "and verified_claims supported by the available evidence. "
        "If the evidence is incomplete, state what is verified and what remains unverified in explanation."
    )

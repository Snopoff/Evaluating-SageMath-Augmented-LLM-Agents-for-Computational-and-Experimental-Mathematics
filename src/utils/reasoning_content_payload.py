"""Helpers for multi-turn thinking models that require reasoning_content in history."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage


def inject_reasoning_content_into_payload(
    input_messages: Sequence[BaseMessage],
    payload: dict[str, Any],
    *,
    use_responses_api: bool = False,
) -> dict[str, Any]:
    """Attach stored reasoning_content to assistant messages in an API payload."""
    if use_responses_api:
        return payload

    ai_messages = [message for message in input_messages if isinstance(message, AIMessage)]
    assistant_payloads = [
        message for message in payload.get("messages", []) if message.get("role") == "assistant"
    ]
    for ai_message, api_message in zip(ai_messages, assistant_payloads, strict=False):
        if "reasoning_content" in api_message:
            continue

        reasoning_content = ai_message.additional_kwargs.get("reasoning_content")
        if reasoning_content is not None:
            api_message["reasoning_content"] = reasoning_content
        elif api_message.get("tool_calls"):
            api_message["reasoning_content"] = ""

    return payload

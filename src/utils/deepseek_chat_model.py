"""DeepSeek chat model helpers."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_deepseek import ChatDeepSeek as _ChatDeepSeek

from src.utils.reasoning_content_payload import inject_reasoning_content_into_payload


class ChatDeepSeek(_ChatDeepSeek):
    """ChatDeepSeek that preserves reasoning_content across multi-turn tool calls."""

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        input_messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        return inject_reasoning_content_into_payload(
            input_messages,
            payload,
            use_responses_api=self._use_responses_api(payload),
        )

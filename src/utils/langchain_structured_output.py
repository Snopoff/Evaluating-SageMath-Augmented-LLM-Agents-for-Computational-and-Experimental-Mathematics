from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel


def structured_output_kwargs(model: BaseChatModel) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"include_raw": True}

    module_name = type(model).__module__
    thinking = getattr(model, "thinking", None)
    if (
        (module_name == "langchain_anthropic.chat_models" or module_name.startswith("langchain_anthropic."))
        and isinstance(thinking, dict)
        and thinking.get("type") in {"enabled", "adaptive"}
    ):
        kwargs["method"] = "json_schema"

    return kwargs

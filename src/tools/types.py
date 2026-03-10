from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    """Metadata exposed to the controller and prompt construction.

    Args:
        name: Stable tool name used for registration and dispatch.
        description: Human-readable description shown to the model.
        input_schema: JSON-schema-like shape describing expected tool arguments.
    """

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """Normalized result returned by any tool handler.

    Args:
        ok: Whether the tool call succeeded.
        content: Main textual payload returned to the controller.
        metadata: Optional structured metadata attached to the result.
    """

    ok: bool
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


ToolHandler = Callable[[dict[str, Any]], ToolResult]


@dataclass(frozen=True)
class ToolDefinition:
    """Concrete tool registration entry with metadata and handler.

    Args:
        spec: Tool metadata exposed to the controller.
        handler: Callable that executes the tool and returns ``ToolResult``.
    """

    spec: ToolSpec
    handler: ToolHandler

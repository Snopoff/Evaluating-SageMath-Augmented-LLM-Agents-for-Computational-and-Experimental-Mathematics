from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


class SageToolClient(Protocol):
    def sage_eval(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Execute a tool call and return a JSON-serializable response."""


@dataclass
class InProcessSageToolClient:
    service: Any

    def sage_eval(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self.service.sage_eval(dict(payload))

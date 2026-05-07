from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from rich.console import Console
from rich.logging import RichHandler


class ConsoleLogger:
    """Minimal in-memory logger that owns console output and trace payloads."""

    _console = Console(record=True)
    _logging_configured = False

    def __init__(self, mode: str = "") -> None:
        self.mode = mode
        self.run_metadata: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        self.artifacts: list[dict[str, Any]] = []
        self.final_result: dict[str, Any] = {}
        self.final_results: dict[str, dict[str, Any]] = {}
        self.token_usage_totals: dict[str, int] = {}
        self.finish_status: str = ""
        self._run_active = False

    @classmethod
    def setup_logging(cls) -> None:
        if cls._logging_configured:
            return
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=cls._console, rich_tracebacks=True, markup=True)],
        )
        cls._logging_configured = True

    def log(self, message: str, level: str = "info", color: str = "white", *args, **kwargs) -> None:
        self._console.print(f"[{color}]\\[{level}][/{color}] {message}", *args, **kwargs)

    def progress(self, message: str, *args, **kwargs) -> None:
        self.log(message, level="progress", color="cyan", *args, **kwargs)
        # self._console.print(f"[cyan]\\[progress][/cyan] {message}", *args, **kwargs)

    @staticmethod
    def _normalize_payload(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): ConsoleLogger._normalize_payload(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [ConsoleLogger._normalize_payload(item) for item in value]
        return value

    def start_run(self, *, metadata: Mapping[str, Any] | None = None) -> None:
        metadata_dict = dict(self._normalize_payload(dict(metadata or {})))
        if self._run_active:
            self._merge_run_metadata(metadata_dict)
            self._record_event("run_start", metadata_dict)
            return

        self.run_metadata = metadata_dict
        self._ensure_agent_ids(self.run_metadata)
        self.events = []
        self.artifacts = []
        self.final_result = {}
        self.final_results = {}
        self.token_usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self.finish_status = ""
        self._run_active = True
        self._record_event("run_start", metadata_dict)

    def log_model_call(
        self,
        *,
        agent_id: str,
        turn: int,
        model_name: str,
        messages: list[dict[str, str]],
        raw_response: str,
        parsed_payload: dict[str, Any] | None,
        token_usage: Mapping[str, int | None] | None = None,
    ) -> None:
        usage_payload = dict(self._normalize_payload(dict(token_usage or {})))
        self._record_event(
            "model_call",
            {
                "agent_id": agent_id,
                "turn": turn,
                "model_name": model_name,
                "messages": messages,
                "raw_response": raw_response,
                "parsed_payload": parsed_payload,
                "token_usage": usage_payload,
            },
        )
        self._accumulate_token_usage(usage_payload)

    def log_tool_call(
        self,
        *,
        agent_id: str,
        turn: int,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> None:
        self._record_event(
            "tool_call",
            {
                "agent_id": agent_id,
                "turn": turn,
                "tool_name": tool_name,
                "arguments": dict(arguments),
            },
        )

    def log_tool_result(
        self,
        *,
        agent_id: str,
        turn: int,
        tool_name: str,
        ok: bool,
        content: str,
        metadata: Mapping[str, Any],
    ) -> None:
        self._record_event(
            "tool_result",
            {
                "agent_id": agent_id,
                "turn": turn,
                "tool_name": tool_name,
                "ok": ok,
                "content": content,
                "metadata": dict(metadata),
            },
        )

    def log_solve_result(
        self,
        *,
        agent_id: str,
        final_answer: str,
        turn_count: int,
        stop_reason: str,
        tool_traces: list[dict[str, Any]],
        verified_sage_code: str = "",
        explanation: str = "",
        confidence: int | None = None,
        verified_claims: list[str] | None = None,
        final_payload: Mapping[str, Any] | None = None,
    ) -> None:
        self.final_result = dict(
            self._normalize_payload(
                {
                    "agent_id": agent_id,
                    "final_answer": final_answer,
                    "explanation": explanation,
                    "confidence": confidence,
                    "verified_claims": list(verified_claims or []),
                    "final_payload": dict(final_payload or {}),
                    "turn_count": turn_count,
                    "stop_reason": stop_reason,
                    "tool_traces": tool_traces,
                    "verified_sage_code": verified_sage_code,
                }
            )
        )
        self.final_results[agent_id] = dict(self.final_result)
        self._record_event("solve_result", self.final_result)

    def log_artifact(
        self,
        *,
        name: str,
        path: str | Path,
        artifact_type: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        record = {
            "name": name,
            "path": str(Path(path)),
            "artifact_type": artifact_type,
            "metadata": dict(self._normalize_payload(dict(metadata or {}))),
        }
        self.artifacts.append(record)

    def finish_run(self, *, status: str) -> None:
        self.finish_status = status
        self._run_active = False

    def build_trace_payload(self, *, status: str | None = None) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "status": status or self.finish_status,
            "run_metadata": self.run_metadata,
            "events": self.events,
            "final_result": self.final_result,
            "final_results": self.final_results,
            "token_usage_totals": self.token_usage_totals,
            "artifacts": self.artifacts,
        }

    def save_output(self, output_dir: str | Path, *, prefix: str = "logger", status: str | None = None) -> dict[str, Path]:
        """Save all console output and trace data to files.

        Args:
            output_dir: Directory to save outputs to.
            prefix: Filename prefix for saved files (default: "logger").
            status: Optional status override for the trace payload.

        Returns:
            Dictionary mapping output type to saved file paths.
        """
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files: dict[str, Path] = {}

        # Save console output as text
        console_text_path = output_dir / f"{prefix}_console.txt"
        self._console.save_text(str(console_text_path))
        saved_files["console_text"] = console_text_path

        # Save console output as HTML
        console_html_path = output_dir / f"{prefix}_console.html"
        self._console.save_html(str(console_html_path))
        saved_files["console_html"] = console_html_path

        # Save trace payload as JSON
        trace_payload = self.build_trace_payload(status=status)
        trace_path = output_dir / f"{prefix}_trace.json"
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace_payload, f, indent=2, ensure_ascii=False)
        saved_files["trace_json"] = trace_path

        return saved_files

    def _record_event(self, kind: str, payload: Mapping[str, Any]) -> None:
        self.events.append({"kind": kind, "payload": dict(self._normalize_payload(dict(payload)))})

    def _merge_run_metadata(self, metadata: Mapping[str, Any]) -> None:
        merged = dict(self.run_metadata)
        merged.update(metadata)
        self._ensure_agent_ids(merged)
        self.run_metadata = merged

    def _accumulate_token_usage(self, token_usage: Mapping[str, Any]) -> None:
        if not self.token_usage_totals:
            self.token_usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = token_usage.get(key)
            if isinstance(value, int):
                self.token_usage_totals[key] += value

    @staticmethod
    def _ensure_agent_ids(metadata: dict[str, Any]) -> None:
        agent_ids = metadata.get("agent_ids", [])
        if not isinstance(agent_ids, list):
            agent_ids = []
        normalized_agent_ids = [agent_id for agent_id in agent_ids if isinstance(agent_id, str) and agent_id.strip()]
        current_agent_id = metadata.get("agent_id")
        if isinstance(current_agent_id, str) and current_agent_id.strip() and current_agent_id not in normalized_agent_ids:
            normalized_agent_ids.append(current_agent_id)
        if normalized_agent_ids:
            metadata["agent_ids"] = normalized_agent_ids

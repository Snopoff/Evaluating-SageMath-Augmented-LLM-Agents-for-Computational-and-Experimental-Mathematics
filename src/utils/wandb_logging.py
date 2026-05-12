import importlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

from src.utils.console_logging import ConsoleLogger


class WandbWeaveLogger(ConsoleLogger):
    """Console logger with W&B run logging and Weave trace events."""

    def __init__(
        self,
        entity: str,
        project: str,
        mode: str = "",
        run_name: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        super().__init__(mode=mode)
        if not entity.strip():
            raise ValueError("W&B entity must be a non-empty string.")
        if not project.strip():
            raise ValueError("W&B project must be a non-empty string.")

        self.entity = entity
        self.project = project
        self.run_name = run_name
        self.group = group
        self.tags = list(tags or [])

        self._wandb = importlib.import_module("wandb")
        self._weave = importlib.import_module("weave")
        self._wandb_run: Any | None = None
        self._weave_active = False
        self._model_call_index = 0
        self._trace_run_start = self._weave.op(name="llmxcas_run_start")(self._trace_run_start_payload)
        self._trace_model_call = self._weave.op(name="llmxcas_model_call")(self._trace_model_call_payload)
        self._trace_tool_call = self._weave.op(name="llmxcas_tool_call")(self._trace_tool_call_payload)
        self._trace_tool_result = self._weave.op(name="llmxcas_tool_result")(self._trace_tool_result_payload)
        self._trace_solve_result = self._weave.op(name="llmxcas_solve_result")(self._trace_solve_result_payload)
        self._trace_artifact = self._weave.op(name="llmxcas_artifact")(self._trace_artifact_payload)

    def start_run(self, *, metadata: Mapping[str, Any] | None = None) -> None:
        was_active = self._run_active
        super().start_run(metadata=metadata)
        run_start_payload = self.events[-1]["payload"]
        if was_active:
            if self._wandb_run is not None:
                self._wandb_run.config.update(
                    {
                        "mode": self.mode,
                        "model_name": self.run_metadata.get("model_name", ""),
                        "problem_id": self.run_metadata.get("problem_id", ""),
                        "prediction_batch_id": self.run_metadata.get("prediction_batch_id", ""),
                        "problem_attempt": self.run_metadata.get("problem_attempt", 1),
                        "controller_config": self.run_metadata.get("controller_config", {}),
                        "tool_names": [spec.get("name", "") for spec in self.run_metadata.get("tool_specs", [])],
                        "agent_ids": self.run_metadata.get("agent_ids", []),
                    },
                    allow_val_change=True,
                )
            self._trace_run_start(dict(run_start_payload))
            return

        project_path = f"{self.entity}/{self.project}"
        problem_id = self.run_metadata.get("problem_id", "")
        prediction_batch_id = self.run_metadata.get("prediction_batch_id", "")
        problem_attempt = self.run_metadata.get("problem_attempt", 1)
        resolved_run_name = self.run_name
        if resolved_run_name is None and isinstance(problem_id, str) and problem_id.strip():
            resolved_run_name = problem_id
            if isinstance(prediction_batch_id, str) and prediction_batch_id.strip():
                resolved_run_name = f"{resolved_run_name}@{prediction_batch_id}"
            if isinstance(problem_attempt, int) and problem_attempt > 1:
                resolved_run_name = f"{resolved_run_name} (attempt {problem_attempt})"
        resolved_group = self.group
        if resolved_group is None and isinstance(prediction_batch_id, str) and prediction_batch_id.strip():
            resolved_group = prediction_batch_id
        self._weave.init(project_path, global_attributes={"mode": self.mode or "unknown"})
        self._weave_active = True
        self._wandb_run = self._wandb.init(
            entity=self.entity,
            project=self.project,
            name=resolved_run_name,
            group=resolved_group,
            tags=self.tags,
            job_type=self.mode or None,
            config={
                "mode": self.mode,
                "model_name": self.run_metadata.get("model_name", ""),
                "problem_id": problem_id,
                "prediction_batch_id": prediction_batch_id,
                "problem_attempt": problem_attempt,
                "controller_config": self.run_metadata.get("controller_config", {}),
                "tool_names": [spec.get("name", "") for spec in self.run_metadata.get("tool_specs", [])],
                "agent_ids": self.run_metadata.get("agent_ids", []),
            },
        )
        self._model_call_index = 0
        self._trace_run_start(dict(run_start_payload))

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
        super().log_model_call(
            agent_id=agent_id,
            turn=turn,
            model_name=model_name,
            messages=messages,
            raw_response=raw_response,
            parsed_payload=parsed_payload,
            token_usage=token_usage,
        )
        self._trace_model_call(dict(self.events[-1]["payload"]))
        if self._wandb_run is not None:
            self._model_call_index += 1
            history_payload: dict[str, Any] = {
                "model_call/index": self._model_call_index,
                "model_call/turn": turn,
            }
            for source_key, target_key in (
                ("input_tokens", "model_call/input_tokens"),
                ("output_tokens", "model_call/output_tokens"),
                ("total_tokens", "model_call/total_tokens"),
            ):
                value = self.events[-1]["payload"]["token_usage"].get(source_key)
                if isinstance(value, int):
                    history_payload[target_key] = value
            self._wandb_run.log(history_payload, step=self._model_call_index)

    def log_tool_call(
        self,
        *,
        agent_id: str,
        turn: int,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> None:
        super().log_tool_call(agent_id=agent_id, turn=turn, tool_name=tool_name, arguments=arguments)
        self._trace_tool_call(dict(self.events[-1]["payload"]))

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
        super().log_tool_result(
            agent_id=agent_id,
            turn=turn,
            tool_name=tool_name,
            ok=ok,
            content=content,
            metadata=metadata,
        )
        self._trace_tool_result(dict(self.events[-1]["payload"]))

    def log_solve_result(
        self,
        *,
        agent_id: str,
        final_answer: str,
        turn_count: int,
        stop_reason: str,
        tool_traces: list[dict[str, Any]],
        token_usage: Mapping[str, int] | None = None,
        verified_sage_code: str = "",
        explanation: str = "",
        confidence: int | None = None,
        verified_claims: list[str] | None = None,
        final_payload: Mapping[str, Any] | None = None,
    ) -> None:
        super().log_solve_result(
            agent_id=agent_id,
            final_answer=final_answer,
            turn_count=turn_count,
            stop_reason=stop_reason,
            token_usage=token_usage,
            tool_traces=tool_traces,
            verified_sage_code=verified_sage_code,
            explanation=explanation,
            confidence=confidence,
            verified_claims=verified_claims,
            final_payload=final_payload,
        )
        self._trace_solve_result(dict(self.final_result))

    def log_artifact(
        self,
        *,
        name: str,
        path: str | Path,
        artifact_type: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().log_artifact(name=name, path=path, artifact_type=artifact_type, metadata=metadata)
        if self._wandb_run is None:
            return

        artifact = self._wandb.Artifact(name=name, type=artifact_type, metadata=dict(self._normalize_payload(dict(metadata or {}))))
        artifact.add_file(str(Path(path)))
        self._wandb_run.log_artifact(artifact)
        self._trace_artifact(dict(self.artifacts[-1]))

    def finish_run(self, *, status: str) -> None:
        super().finish_run(status=status)
        if self._wandb_run is None:
            self._finish_weave()
            return

        wandb_run = self._wandb_run
        try:
            trace_path = self._write_trace_artifact(status=status)
            self.log_artifact(
                name=f"{self.mode or 'run'}-trace",
                path=trace_path,
                artifact_type="llmxcas-trace",
                metadata={"status": status},
            )

            tool_call_count = sum(1 for event in self.events if event.get("kind") == "tool_call")
            wandb_run.summary["event_count"] = len(self.events)
            wandb_run.summary["tool_call_count"] = tool_call_count
            wandb_run.summary["agent_count"] = len(self.final_results)
            wandb_run.summary["input_tokens"] = self.token_usage_totals.get("input_tokens", 0)
            wandb_run.summary["output_tokens"] = self.token_usage_totals.get("output_tokens", 0)
            wandb_run.summary["total_tokens"] = self.token_usage_totals.get("total_tokens", 0)
            wandb_run.summary["stop_reason"] = self.final_result.get("stop_reason", status)
            wandb_run.summary["turn_count"] = self.final_result.get("turn_count", 0)
        finally:
            wandb_run.finish()
            self._wandb_run = None
            self._finish_weave()

    def _finish_weave(self) -> None:
        if not self._weave_active:
            return
        finish = getattr(self._weave, "finish", None)
        if callable(finish):
            finish()
        self._weave_active = False

    def _write_trace_artifact(self, *, status: str) -> Path:
        payload = self.build_trace_payload(status=status)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            return Path(handle.name)

    @staticmethod
    def _trace_run_start_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    @staticmethod
    def _trace_model_call_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    @staticmethod
    def _trace_tool_call_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    @staticmethod
    def _trace_tool_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    @staticmethod
    def _trace_solve_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    @staticmethod
    def _trace_artifact_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

import json
import logging
from time import perf_counter
from typing import Any, Mapping

from .docker_executor import DockerSageExecutor, ExecutionResult
from .policy import (
    ERROR_EXEC,
    ERROR_INPUT_TOO_LARGE,
    ERROR_UNSAFE,
    PolicyEngine,
    PolicyValidationError,
    SageEvalRequest,
    SageEvalResponse,
    make_blocked_response,
)


class SageMCPService:
    """Applies policy checks and executes Sage requests in Docker."""

    def __init__(self, policy_engine: PolicyEngine, executor: DockerSageExecutor, progress_logs: bool = False):
        self.policy_engine = policy_engine
        self.executor = executor
        self.progress_logs = progress_logs
        self.audit_logger = logging.getLogger("llmxm2.audit")
        if self.progress_logs:
            cfg = self.executor.config
            image = cfg.image
            short_image = image if len(image) <= 120 else f"{image[:117]}..."
            user = cfg.user if cfg.user else "<image-default>"
            entrypoint = cfg.entrypoint if cfg.entrypoint else "<default>"
            print(
                "[progress][sage-mcp] runtime config "
                f"(image={short_image}, platform={cfg.platform or '<default>'}, "
                f"entrypoint={entrypoint}, user={user})",
                flush=True,
            )

    def _progress(self, message: str) -> None:
        if self.progress_logs:
            print(f"[progress][sage-mcp] {message}", flush=True)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "SageMCPService":
        policy_cfg = dict(cfg.get("policy", {}))
        docker_cfg = dict(cfg.get("docker", {}))
        return cls(
            policy_engine=PolicyEngine.from_config(policy_cfg),
            executor=DockerSageExecutor.from_config(docker_cfg),
            progress_logs=bool(cfg.get("progress_logs", False)),
        )

    def sage_eval(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        started = perf_counter()
        self._progress("received sage_eval payload")

        try:
            request = SageEvalRequest.from_payload(payload)
        except PolicyValidationError as exc:
            self._progress(f"payload validation blocked: {exc}")
            reason = str(exc)
            error_code = ERROR_UNSAFE
            if "maximum characters" in reason.lower() or "chars" in reason.lower():
                error_code = ERROR_INPUT_TOO_LARGE
            response = make_blocked_response(
                report=self._deny_report(reason),
                runtime_ms=0,
                error_code=error_code,
            )
            self._audit(payload, response)
            return response.to_dict()

        self._progress(f"policy assessment started (operation={request.operation}, request_id={request.request_id})")
        report = self.policy_engine.assess(request)
        if report.policy_decision != "allow":
            self._progress(f"policy denied request: {report.reason}")
            response = make_blocked_response(report=report, runtime_ms=0, error_code=ERROR_UNSAFE)
            self._audit(request, response)
            return response.to_dict()

        self._progress("policy allowed request; executing in docker")
        exec_result = self.executor.execute(request)
        response = self._response_from_execution(report=report, result=exec_result)
        total_ms = int((perf_counter() - started) * 1000)
        self._progress(f"execution finished (status={response.status}, runtime_ms={response.runtime_ms}, total_ms={total_ms})")
        self._audit(request, response)

        # Defensive enforcement in case executor returns weird values.
        if response.runtime_ms < 0:
            sanitized = SageEvalResponse(
                status="error",
                result_plain="",
                result_latex="",
                complexity_report=report,
                runtime_ms=int((perf_counter() - started) * 1000),
                error_code=ERROR_EXEC,
            )
            return sanitized.to_dict()

        return response.to_dict()

    @staticmethod
    def _deny_report(reason: str):
        from .policy import ComplexityReport

        return ComplexityReport(features={}, policy_decision="deny", reason=reason)

    @staticmethod
    def _response_from_execution(report, result: ExecutionResult) -> SageEvalResponse:
        result_plain = result.result_plain
        if result.status != "ok" and not result_plain and result.message:
            result_plain = result.message
        return SageEvalResponse(
            status=result.status,
            result_plain=result_plain,
            result_latex=result.result_latex,
            complexity_report=report,
            runtime_ms=result.runtime_ms,
            error_code=result.error_code,
        )

    def _audit(self, request_payload: Mapping[str, Any] | SageEvalRequest, response: SageEvalResponse) -> None:
        if isinstance(request_payload, SageEvalRequest):
            request_id = request_payload.request_id
            operation = request_payload.operation
        else:
            request_id = str(request_payload.get("request_id", "unknown"))
            operation = str(request_payload.get("operation", "unknown"))

        entry = {
            "request_id": request_id,
            "operation": operation,
            "status": response.status,
            "error_code": response.error_code,
            "runtime_ms": response.runtime_ms,
            "policy_decision": response.complexity_report.policy_decision,
            "policy_reason": response.complexity_report.reason,
        }
        self.audit_logger.info(json.dumps(entry, ensure_ascii=True, sort_keys=True))


def build_fastmcp_app(service: SageMCPService):
    from mcp.server.fastmcp import FastMCP

    app = FastMCP("SageMath", json_response=True)

    @app.tool()
    def sage_eval(payload: dict[str, Any]) -> dict[str, Any]:
        """Constrained SageMath execution with static policy and runtime limits."""

        return service.sage_eval(payload)

    return app


def run_mcp_server(service: SageMCPService, transport: str = "streamable-http") -> None:
    app = build_fastmcp_app(service)
    app.run(transport=transport)

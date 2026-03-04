from __future__ import annotations

import ast
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

ERROR_NONE = "NONE"
ERROR_INPUT_TOO_LARGE = "INPUT_TOO_LARGE"
ERROR_UNSAFE = "UNSAFE_OPERATION"
ERROR_BUDGET = "BUDGET_EXCEEDED"
ERROR_TIMEOUT = "TIMEOUT"
ERROR_EXEC = "EXEC_ERROR"

_ALLOWED_DOMAINS = {"QQ", "RR", "CC", "ZZ"}
_SAFE_TOKEN_PATTERN = re.compile(r"(__|import\b|open\s*\(|exec\b|eval\b|subprocess|os\.|sys\.|socket|fork|spawn|;|`)")
_SYMBOL_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_]*\b")
_EXP_PATTERN = re.compile(r"\^(\d+)")
_INT_PATTERN = re.compile(r"-?\d+")
_MONOMIAL_SPLIT_PATTERN = re.compile(r"(?<!\^)[+-]")

_KNOWN_TOKENS = {
    "sin",
    "cos",
    "tan",
    "log",
    "ln",
    "sqrt",
    "exp",
    "pi",
    "e",
    "i",
    "true",
    "false",
    "and",
    "or",
    "not",
}

_SNIPPET_BANNED_CALLS = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "input",
    "breakpoint",
    "quit",
    "exit",
}

_SNIPPET_ALLOWED_IMPORT_ROOTS = {
    "sage",
    "math",
    "random",
    "re",
    "heapq",
    "itertools",
    "collections",
    "functools",
    "statistics",
    "fractions",
}

_SNIPPET_BANNED_MODULE_PATTERN = re.compile(
    r"\b("
    r"os|sys|subprocess|socket|pathlib|shutil|tempfile|glob|importlib|ctypes|threading|multiprocessing|"
    r"pickle|marshal|requests|httpx|urllib|ftplib|telnetlib|resource|signal|pty"
    r")\b"
)


class PolicyValidationError(ValueError):
    """Raised when the tool request payload is malformed."""


@dataclass(frozen=True)
class GenericLimits:
    max_total_args: int = 64


@dataclass(frozen=True)
class SnippetLimits:
    max_code_chars: int = 6000
    max_ast_nodes: int = 3000


@dataclass(frozen=True)
class PolicyLimits:
    input_max_chars: int = 5000
    max_depth: int = 8
    max_nodes: int = 1500
    uncertainty_denies: bool = True
    generic: GenericLimits = field(default_factory=GenericLimits)
    snippet: SnippetLimits = field(default_factory=SnippetLimits)


@dataclass(frozen=True)
class SageEvalRequest:
    operation: str
    args: dict[str, Any]
    assumptions: dict[str, Any]
    request_id: str
    budget_profile: str = "conservative"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SageEvalRequest":
        if not isinstance(payload, Mapping):
            raise PolicyValidationError("Payload must be an object.")

        operation = payload.get("operation")
        if not isinstance(operation, str) or not operation.strip():
            raise PolicyValidationError("Field 'operation' must be a non-empty string.")

        args = payload.get("args")
        if not isinstance(args, Mapping):
            raise PolicyValidationError("Field 'args' must be an object.")

        assumptions = payload.get("assumptions", {})
        if assumptions is None:
            assumptions = {}
        if not isinstance(assumptions, Mapping):
            raise PolicyValidationError("Field 'assumptions' must be an object.")

        domain = assumptions.get("domain")
        if domain is not None and domain not in _ALLOWED_DOMAINS:
            raise PolicyValidationError(f"Unsupported assumptions.domain: {domain!r}")

        symbols = assumptions.get("symbols")
        if symbols is not None:
            if not isinstance(symbols, list) or not all(isinstance(item, str) for item in symbols):
                raise PolicyValidationError("assumptions.symbols must be a list of strings.")

        request_id = payload.get("request_id")
        if request_id is None:
            request_id = str(uuid.uuid4())
        if not isinstance(request_id, str) or not request_id:
            raise PolicyValidationError("Field 'request_id' must be a non-empty string.")

        budget_profile = payload.get("budget_profile", "conservative")
        if not isinstance(budget_profile, str) or not budget_profile:
            raise PolicyValidationError("Field 'budget_profile' must be a non-empty string.")

        return cls(
            operation=operation,
            args=dict(args),
            assumptions=dict(assumptions),
            request_id=request_id,
            budget_profile=budget_profile,
        )


@dataclass(frozen=True)
class ComplexityReport:
    features: dict[str, Any]
    policy_decision: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": self.features,
            "policy_decision": self.policy_decision,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SageEvalResponse:
    status: str
    result_plain: str
    result_latex: str
    complexity_report: ComplexityReport
    runtime_ms: int
    error_code: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "result_plain": self.result_plain,
            "result_latex": self.result_latex,
            "complexity_report": self.complexity_report.to_dict(),
            "runtime_ms": self.runtime_ms,
            "error_code": self.error_code,
        }


class PolicyEngine:
    """Fail-closed static checks for Sage tool requests."""

    def __init__(self, limits: PolicyLimits | None = None):
        self.limits = limits or PolicyLimits()

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> "PolicyEngine":
        cfg = dict(cfg or {})
        generic_cfg = dict(cfg.get("generic", {}))
        snippet_cfg = dict(cfg.get("snippet", {}))

        limits = PolicyLimits(
            input_max_chars=int(cfg.get("input_max_chars", 5000)),
            max_depth=int(cfg.get("max_depth", 8)),
            max_nodes=int(cfg.get("max_nodes", 1500)),
            uncertainty_denies=bool(cfg.get("uncertainty_denies", True)),
            generic=GenericLimits(
                max_total_args=int(generic_cfg.get("max_total_args", 64)),
            ),
            snippet=SnippetLimits(
                max_code_chars=int(snippet_cfg.get("max_code_chars", 6000)),
                max_ast_nodes=int(snippet_cfg.get("max_ast_nodes", 3000)),
            ),
        )
        return cls(limits=limits)

    def assess(self, request: SageEvalRequest) -> ComplexityReport:
        features: dict[str, Any] = {
            "operation": request.operation,
            "string_chars": 0,
            "node_count": 0,
            "max_depth": 0,
            "contains_unsafe_token": False,
            "symbol_count": 0,
            "max_polynomial_degree": 0,
            "total_monomials": 0,
            "max_coefficient_bits": 0,
            "ast_nodes_estimate": 0,
            "uncertainty_score": 0,
        }

        try:
            self._scan_value(request.args, depth=1, features=features)
        except Exception as exc:
            features["uncertainty_score"] = 1
            return ComplexityReport(
                features=features,
                policy_decision="deny",
                reason=f"Feature extraction failed: {exc}",
            )

        if features["string_chars"] > self.limits.input_max_chars:
            return ComplexityReport(features, "deny", "Input exceeds maximum characters.")
        if features["node_count"] > self.limits.max_nodes:
            return ComplexityReport(features, "deny", "Input exceeds maximum node count.")
        if features["max_depth"] > self.limits.max_depth:
            return ComplexityReport(features, "deny", "Input exceeds maximum nesting depth.")
        if features["contains_unsafe_token"] and request.operation != "sage_snippet":
            return ComplexityReport(features, "deny", "Unsafe tokens detected in request arguments.")

        operation_reason = self._check_operation_constraints(request, features)
        if operation_reason is not None:
            return ComplexityReport(features, "deny", operation_reason)

        if self.limits.uncertainty_denies and features.get("uncertainty_score", 0) > 0:
            return ComplexityReport(features, "deny", "Denied due to uncertain complexity analysis.")

        return ComplexityReport(features, "allow", "Request is within static policy limits.")

    def _scan_value(self, value: Any, depth: int, features: dict[str, Any]) -> None:
        features["node_count"] += 1
        features["max_depth"] = max(features["max_depth"], depth)

        if isinstance(value, str):
            features["string_chars"] += len(value)
            features["ast_nodes_estimate"] += self._estimate_ast_nodes(value)
            if _SAFE_TOKEN_PATTERN.search(value):
                features["contains_unsafe_token"] = True
            features["symbol_count"] = max(features["symbol_count"], self._count_symbols(value))
            features["max_polynomial_degree"] = max(features["max_polynomial_degree"], self._max_degree(value))
            features["total_monomials"] += self._monomial_count(value)
            features["max_coefficient_bits"] = max(features["max_coefficient_bits"], self._max_coefficient_bits(value))
            return

        if isinstance(value, bool) or value is None:
            return

        if isinstance(value, int):
            features["max_coefficient_bits"] = max(features["max_coefficient_bits"], abs(value).bit_length())
            return

        if isinstance(value, float):
            return

        if isinstance(value, list):
            for item in value:
                self._scan_value(item, depth + 1, features)
            return

        if isinstance(value, Mapping):
            for item in value.values():
                self._scan_value(item, depth + 1, features)
            return

        features["uncertainty_score"] += 1

    def _check_operation_constraints(self, request: SageEvalRequest, features: dict[str, Any]) -> str | None:
        args = request.args
        if request.operation == "sage_snippet":
            code = args.get("code")
            if not isinstance(code, str) or not code.strip():
                return "sage_snippet requires non-empty 'code' string."
            if len(code) > self.limits.snippet.max_code_chars:
                return "sage_snippet code length exceeds limit."

            result_var = args.get("result_var")
            if result_var is not None and (not isinstance(result_var, str) or not result_var.strip()):
                return "sage_snippet 'result_var' must be a non-empty string when provided."

            include_locals = args.get("include_locals")
            if include_locals is not None and not isinstance(include_locals, bool):
                return "sage_snippet 'include_locals' must be a boolean when provided."

            snippet_error = self._check_snippet_safety(code)
            if snippet_error is not None:
                return snippet_error

            return None

        # Generic operation path: operation name maps to a callable in sage.all namespace.
        positional_args = args.get("positional_args", [])
        if not isinstance(positional_args, list):
            return "generic operations require args.positional_args as a list."

        keyword_args = args.get("keyword_args", {})
        if not isinstance(keyword_args, Mapping):
            return "generic operations require args.keyword_args as an object."

        coerce_symbolic_strings = args.get("coerce_symbolic_strings")
        if coerce_symbolic_strings is not None and not isinstance(coerce_symbolic_strings, bool):
            return "generic operations require args.coerce_symbolic_strings as a boolean when provided."

        if len(positional_args) + len(keyword_args) > self.limits.generic.max_total_args:
            return "generic operation argument count exceeds limit."

        return None

    def _check_snippet_safety(self, code: str) -> str | None:
        if _SNIPPET_BANNED_MODULE_PATTERN.search(code):
            return "sage_snippet contains banned modules/APIs."

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"sage_snippet contains invalid Python syntax: {exc.msg}"

        node_count = sum(1 for _ in ast.walk(tree))
        if node_count > self.limits.snippet.max_ast_nodes:
            return "sage_snippet AST node count exceeds limit."

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root not in _SNIPPET_ALLOWED_IMPORT_ROOTS:
                        return f"sage_snippet import '{alias.name}' is not allowed."

            if isinstance(node, ast.ImportFrom):
                if node.module is None:
                    return "sage_snippet relative imports are not allowed."
                root = node.module.split(".", 1)[0]
                if root not in _SNIPPET_ALLOWED_IMPORT_ROOTS:
                    return f"sage_snippet import from '{node.module}' is not allowed."

            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                return "sage_snippet dunder attribute access is not allowed."

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    call_name = node.func.id
                    if call_name in _SNIPPET_BANNED_CALLS:
                        return f"sage_snippet call '{call_name}' is not allowed."

        return None

    @staticmethod
    def _estimate_ast_nodes(text: str) -> int:
        operators = sum(text.count(token) for token in ["+", "-", "*", "/", "^", "(", ")", "=", ","])
        symbols = len(_SYMBOL_PATTERN.findall(text))
        return operators + symbols

    @staticmethod
    def _count_symbols(text: str) -> int:
        names = {name for name in _SYMBOL_PATTERN.findall(text) if name.lower() not in _KNOWN_TOKENS}
        return len(names)

    @staticmethod
    def _max_degree(text: str) -> int:
        degrees = [int(item) for item in _EXP_PATTERN.findall(text)]
        return max(degrees) if degrees else 1

    @staticmethod
    def _monomial_count(text: str) -> int:
        stripped = text.replace(" ", "")
        if not stripped:
            return 0
        parts = [segment for segment in _MONOMIAL_SPLIT_PATTERN.split(stripped) if segment]
        return max(1, len(parts))

    @staticmethod
    def _max_coefficient_bits(text: str) -> int:
        ints = [abs(int(token)).bit_length() for token in _INT_PATTERN.findall(text)]
        return max(ints) if ints else 0


def make_blocked_response(
    report: ComplexityReport,
    runtime_ms: int = 0,
    error_code: str = ERROR_UNSAFE,
) -> SageEvalResponse:
    return SageEvalResponse(
        status="blocked",
        result_plain="",
        result_latex="",
        complexity_report=report,
        runtime_ms=runtime_ms,
        error_code=error_code,
    )

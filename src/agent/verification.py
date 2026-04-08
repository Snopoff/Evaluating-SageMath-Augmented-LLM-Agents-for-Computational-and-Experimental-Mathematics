from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ConstraintSpec:
    id: str
    text: str
    requires_cas: bool = True


@dataclass(frozen=True)
class OutputSpec:
    id: str
    text: str


@dataclass(frozen=True)
class SolveContract:
    hard_constraints: tuple[ConstraintSpec, ...]
    required_outputs: tuple[OutputSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hard_constraints": [{"id": item.id, "text": item.text, "requires_cas": item.requires_cas} for item in self.hard_constraints],
            "required_outputs": [{"id": item.id, "text": item.text} for item in self.required_outputs],
        }


def normalize_solve_contract(payload: Any) -> SolveContract | None:
    if not isinstance(payload, Mapping):
        return None

    raw_constraints = payload.get("hard_constraints", [])
    raw_outputs = payload.get("required_outputs", [])
    if not isinstance(raw_constraints, list) or not isinstance(raw_outputs, list):
        return None

    hard_constraints: list[ConstraintSpec] = []
    seen_constraint_ids: set[str] = set()
    for index, item in enumerate(raw_constraints, start=1):
        if not isinstance(item, Mapping):
            return None
        text = str(item.get("text", "")).strip()
        if not text:
            return None
        item_id = _normalize_identifier(item.get("id"), prefix="constraint", index=index)
        if item_id in seen_constraint_ids:
            return None
        seen_constraint_ids.add(item_id)
        requires_cas = bool(item.get("requires_cas", True))
        hard_constraints.append(ConstraintSpec(id=item_id, text=text, requires_cas=requires_cas))

    required_outputs: list[OutputSpec] = []
    seen_output_ids: set[str] = set()
    for index, item in enumerate(raw_outputs, start=1):
        if not isinstance(item, Mapping):
            return None
        text = str(item.get("text", "")).strip()
        if not text:
            return None
        item_id = _normalize_identifier(item.get("id"), prefix="output", index=index)
        if item_id in seen_output_ids:
            return None
        seen_output_ids.add(item_id)
        required_outputs.append(OutputSpec(id=item_id, text=text))

    if not hard_constraints and not required_outputs:
        return None

    return SolveContract(
        hard_constraints=tuple(hard_constraints),
        required_outputs=tuple(required_outputs),
    )


def normalize_verification_payload(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, Mapping) and isinstance(payload.get("verification"), Mapping):
        payload = payload.get("verification")

    if not isinstance(payload, Mapping):
        return None

    checks_value = payload.get("checks", [])
    normalized_checks: list[dict[str, str]] = []
    if isinstance(checks_value, Mapping):
        for index, (key, value) in enumerate(checks_value.items(), start=1):
            item_id = _normalize_identifier(key, prefix="constraint", index=index)
            status, evidence = _normalize_check_value(value)
            if status is None:
                return None
            normalized_checks.append({"id": item_id, "status": status, "evidence": evidence})
    elif isinstance(checks_value, list):
        for index, item in enumerate(checks_value, start=1):
            if not isinstance(item, Mapping):
                return None
            item_id = _normalize_identifier(item.get("id"), prefix="constraint", index=index)
            status = _normalize_status(item.get("status"))
            if status is None:
                return None
            evidence = str(item.get("evidence", "")).strip()
            normalized_checks.append({"id": item_id, "status": status, "evidence": evidence})
    else:
        return None

    outputs_value = payload.get("outputs", {})
    normalized_outputs: dict[str, Any] = {}
    if isinstance(outputs_value, Mapping):
        for key, value in outputs_value.items():
            normalized_outputs[str(key)] = value
    elif outputs_value not in ({}, None):
        return None

    summary = _normalize_status(payload.get("summary"))
    if summary is None:
        summary = _derive_summary(normalized_checks)

    return {
        "summary": summary,
        "checks": normalized_checks,
        "outputs": normalized_outputs,
    }


def verification_satisfies_contract(
    verification: Mapping[str, Any] | None,
    contract: SolveContract | None,
    *,
    require_full_coverage: bool,
) -> tuple[bool, list[str]]:
    if not isinstance(verification, Mapping):
        return False, ["missing verification payload"]

    summary = _normalize_status(verification.get("summary"))
    if summary != "pass":
        return False, [f"verification summary is {summary or 'missing'}"]

    if contract is None or not require_full_coverage:
        return True, []

    checks = verification.get("checks", [])
    outputs = verification.get("outputs", {})
    if not isinstance(checks, list) or not isinstance(outputs, Mapping):
        return False, ["verification payload has invalid shape"]

    check_status_by_id = {str(item.get("id", "")): _normalize_status(item.get("status")) for item in checks if isinstance(item, Mapping)}

    failures: list[str] = []
    for item in contract.hard_constraints:
        status = check_status_by_id.get(item.id)
        if status != "pass":
            failures.append(f"constraint {item.id} status is {status or 'missing'}")

    for item in contract.required_outputs:
        if item.id not in outputs:
            failures.append(f"required output {item.id} is missing")

    return not failures, failures


def _normalize_check_value(value: Any) -> tuple[str | None, str]:
    if isinstance(value, Mapping):
        status = _normalize_status(value.get("status"))
        evidence = str(value.get("evidence", "")).strip()
        return status, evidence

    status = _normalize_status(value)
    return status, ""


def _normalize_status(value: Any) -> str | None:
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if not isinstance(value, str):
        return None

    normalized = value.strip().lower()
    if normalized in {"pass", "passed", "verified", "true", "ok"}:
        return "pass"
    if normalized in {"fail", "failed", "false", "error"}:
        return "fail"
    if normalized in {"unresolved", "inconclusive", "unknown", "pending"}:
        return "unresolved"
    return None


def _derive_summary(checks: list[dict[str, str]]) -> str:
    statuses = [item["status"] for item in checks]
    if not statuses:
        return "unresolved"
    if any(status == "fail" for status in statuses):
        return "fail"
    if all(status == "pass" for status in statuses):
        return "pass"
    return "unresolved"


def _normalize_identifier(value: Any, *, prefix: str, index: int) -> str:
    if isinstance(value, str):
        candidate = value.strip()
        if candidate:
            return candidate
    return f"{prefix}_{index}"

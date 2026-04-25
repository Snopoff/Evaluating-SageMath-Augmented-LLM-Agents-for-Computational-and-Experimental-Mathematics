from typing import Any, Mapping


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


def verification_passes(verification: Mapping[str, Any] | None) -> tuple[bool, list[str]]:
    if not isinstance(verification, Mapping):
        return False, ["missing verification payload"]

    summary = _normalize_status(verification.get("summary"))
    if summary != "pass":
        return False, [f"verification summary is {summary or 'missing'}"]

    return True, []


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

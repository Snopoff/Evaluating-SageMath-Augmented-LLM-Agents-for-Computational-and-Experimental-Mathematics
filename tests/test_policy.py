from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llmxm2.mcp.policy import PolicyEngine, SageEvalRequest


class PolicyEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = PolicyEngine()

    def test_allows_generic_operation_with_argument_lists(self) -> None:
        request = SageEvalRequest.from_payload(
            {
                "operation": "factor",
                "args": {
                    "positional_args": ["x^2-1"],
                    "keyword_args": {},
                    "coerce_symbolic_strings": True,
                },
                "assumptions": {"domain": "QQ", "symbols": ["x"]},
                "request_id": "req-1",
                "budget_profile": "conservative",
            }
        )
        report = self.engine.assess(request)
        self.assertEqual(report.policy_decision, "allow")

    def test_blocks_generic_operation_with_invalid_positional_args(self) -> None:
        request = SageEvalRequest.from_payload(
            {
                "operation": "factor",
                "args": {
                    "positional_args": "not-a-list",
                    "keyword_args": {},
                },
                "assumptions": {"domain": "QQ"},
                "request_id": "req-2",
                "budget_profile": "conservative",
            }
        )
        report = self.engine.assess(request)
        self.assertEqual(report.policy_decision, "deny")
        self.assertIn("positional_args", report.reason)

    def test_blocks_generic_operation_with_invalid_keyword_args(self) -> None:
        request = SageEvalRequest.from_payload(
            {
                "operation": "factor",
                "args": {
                    "positional_args": [],
                    "keyword_args": "not-an-object",
                },
                "assumptions": {"domain": "QQ"},
                "request_id": "req-3",
                "budget_profile": "conservative",
            }
        )
        report = self.engine.assess(request)
        self.assertEqual(report.policy_decision, "deny")
        self.assertIn("keyword_args", report.reason)

    def test_blocks_generic_operation_with_too_many_args(self) -> None:
        request = SageEvalRequest.from_payload(
            {
                "operation": "factor",
                "args": {
                    "positional_args": list(range(65)),
                    "keyword_args": {},
                },
                "assumptions": {"domain": "QQ"},
                "request_id": "req-4",
                "budget_profile": "conservative",
            }
        )
        report = self.engine.assess(request)
        self.assertEqual(report.policy_decision, "deny")
        self.assertIn("argument count", report.reason)

    def test_blocks_unsafe_tokens_for_non_snippet(self) -> None:
        request = SageEvalRequest.from_payload(
            {
                "operation": "factor",
                "args": {
                    "positional_args": ["import os"],
                    "keyword_args": {},
                },
                "assumptions": {"domain": "QQ"},
                "request_id": "req-5",
                "budget_profile": "conservative",
            }
        )
        report = self.engine.assess(request)
        self.assertEqual(report.policy_decision, "deny")
        self.assertIn("Unsafe tokens", report.reason)

    def test_allows_sage_snippet_with_safe_code(self) -> None:
        request = SageEvalRequest.from_payload(
            {
                "operation": "sage_snippet",
                "args": {
                    "code": "from sage.all import *\nR = PolynomialRing(QQ, 'x')\nx = R.gen()\nRESULT = factor(x^2 - 1)",
                    "result_var": "RESULT",
                },
                "assumptions": {"domain": "ZZ"},
                "request_id": "req-6",
                "budget_profile": "conservative",
            }
        )
        report = self.engine.assess(request)
        self.assertEqual(report.policy_decision, "allow")

    def test_blocks_sage_snippet_with_os_import(self) -> None:
        request = SageEvalRequest.from_payload(
            {
                "operation": "sage_snippet",
                "args": {
                    "code": "import os\nRESULT = os.listdir('/')",
                    "result_var": "RESULT",
                },
                "assumptions": {"domain": "ZZ"},
                "request_id": "req-7",
                "budget_profile": "conservative",
            }
        )
        report = self.engine.assess(request)
        self.assertEqual(report.policy_decision, "deny")
        self.assertIn("banned modules", report.reason)


if __name__ == "__main__":
    unittest.main()

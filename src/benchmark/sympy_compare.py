from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sympy import Equality, FiniteSet, simplify, sympify
from sympy.core.containers import Tuple as SympyTuple
from sympy.matrices.matrixbase import MatrixBase
from sympy.parsing.sympy_parser import parse_expr

SympyAnswer = str | list[str]


@dataclass(frozen=True)
class ScoreResult:
    correct: bool
    match_type: str
    normalized_prediction: SympyAnswer
    normalized_reference: SympyAnswer


class SympyAnswerComparator:
    def score(self, prediction: SympyAnswer, reference: SympyAnswer) -> ScoreResult:
        norm_prediction = self.normalize(prediction)
        norm_reference = self.normalize(reference)

        if norm_prediction == norm_reference:
            return ScoreResult(
                correct=True,
                match_type="exact",
                normalized_prediction=norm_prediction,
                normalized_reference=norm_reference,
            )

        if self.payload_equal(norm_prediction, norm_reference):
            return ScoreResult(
                correct=True,
                match_type="symbolic",
                normalized_prediction=norm_prediction,
                normalized_reference=norm_reference,
            )

        return ScoreResult(
            correct=False,
            match_type="mismatch",
            normalized_prediction=norm_prediction,
            normalized_reference=norm_reference,
        )

    @staticmethod
    def coerce(value: Any) -> SympyAnswer:
        if isinstance(value, list):
            return [str(item) for item in value]
        return str(value)

    @staticmethod
    def normalize(value: SympyAnswer) -> SympyAnswer:
        if isinstance(value, list):
            return [item.strip() for item in value]
        return value.strip()

    def payload_equal(self, left: SympyAnswer, right: SympyAnswer) -> bool:
        if isinstance(left, list) or isinstance(right, list):
            if not isinstance(left, list) or not isinstance(right, list):
                return False
            if len(left) != len(right):
                return False
            return all(self.string_equal(lhs, rhs) for lhs, rhs in zip(left, right))

        return self.string_equal(left, right)

    def string_equal(self, left: str, right: str) -> bool:
        if not left or not right:
            return False
        try:
            left_expr = parse_expr(left, evaluate=False)
        except Exception:
            try:
                left_expr = sympify(left, evaluate=False)
            except Exception:
                return False

        try:
            right_expr = parse_expr(right, evaluate=False)
        except Exception:
            try:
                right_expr = sympify(right, evaluate=False)
            except Exception:
                return False

        return self.expr_equal(left_expr, right_expr)

    def expr_equal(self, left: Any, right: Any) -> bool:
        if left == right:
            return True

        if isinstance(left, SympyTuple) and isinstance(right, SympyTuple):
            if len(left) != len(right):
                return False
            return all(self.expr_equal(lhs, rhs) for lhs, rhs in zip(left, right))

        if isinstance(left, MatrixBase) and isinstance(right, MatrixBase):
            if left.shape != right.shape:
                return False
            return all(self.expr_equal(lhs, rhs) for lhs, rhs in zip(left, right))

        if isinstance(left, FiniteSet) and isinstance(right, FiniteSet):
            unmatched = list(right)
            for lhs in left:
                for index, rhs in enumerate(unmatched):
                    if self.expr_equal(lhs, rhs):
                        unmatched.pop(index)
                        break
                else:
                    return False
            return not unmatched

        if isinstance(left, Equality) and isinstance(right, Equality):
            return (
                self.expr_equal(left.lhs, right.lhs)
                and self.expr_equal(left.rhs, right.rhs)
            ) or (
                self.expr_equal(left.lhs, right.rhs)
                and self.expr_equal(left.rhs, right.lhs)
            )

        if getattr(left, "func", None) == getattr(right, "func", None) and getattr(left, "args", ()) and len(left.args) == len(right.args):
            if all(self.expr_equal(lhs, rhs) for lhs, rhs in zip(left.args, right.args)):
                return True

        try:
            if simplify(left - right) == 0:
                return True
        except Exception:
            pass

        equals_method = getattr(left, "equals", None)
        if callable(equals_method):
            try:
                equals_result = equals_method(right)
            except Exception:
                equals_result = None
            if equals_result is not None:
                return bool(equals_result)

        return False

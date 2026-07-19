from pydantic import BaseModel, ConfigDict, Field, field_validator


def _validate_answer_string(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} strings must be non-empty.")
    if any(marker in normalized for marker in ("$", "\\")):
        raise ValueError(f"{field_name} must not contain LaTeX wrappers or backslashes.")
    return value


class SageExecArgs(BaseModel):
    """Arguments for executing Sage code as part of an agent's reasoning process."""

    code: str = Field(
        min_length=1,
        description=(
            "Sage script code to execute with Sage's preparser. "
            "Sage shorthand such as R.<x> declarations and ^ exponentiation is allowed. "
            "Assign the final value to RESULT."
        ),
    )
    result_var: str = Field(default="RESULT", description="Variable name to read after execution.")


class FinalAnswerArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    explanation: str = Field(
        description="Post-hoc summary of the final answer in 3-6 sentences: the decisive identity / lemma and any sanity check used. Not a full derivation (that belongs in internal reasoning) and not a restatement of the question.",
    )

    final_answer: str = Field(
        description=(
            "The final answer to score. Put only the checkable result here, not the narrative and reasoning."
        ),
    )

    sympy_answer: str | list[str] = Field(
        description=(
            "Normalized SymPy form of the final answer for benchmarking. "
            "Use a single string for one answer or list[str] for multiple answers. "
            "No prose, no LaTeX wrappers, and no backslashes. "
            "Flatten indexed names into ASCII identifiers like M_n_minus_1, not M_{n-1}."
        ),
    )

    confidence: int = Field(
        ge=1,
        le=5,
        description="Confidence in the final answer on a 1-5 scale, where 5 is highest.",
    )

    @field_validator("sympy_answer")
    @classmethod
    def validate_sympy_answer(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, str):
            return _validate_answer_string(value, field_name="sympy_answer")
        if not value:
            raise ValueError("sympy_answer lists must be non-empty.")
        return [
            _validate_answer_string(item, field_name=f"sympy_answer[{index}]")
            for index, item in enumerate(value)
        ]


class SageFinalAnswerArgs(FinalAnswerArgs):
    verified_claims: list[str] = Field(
        description="Short list of final claims supported by successful Sage output or explicit reasoning.",
    )

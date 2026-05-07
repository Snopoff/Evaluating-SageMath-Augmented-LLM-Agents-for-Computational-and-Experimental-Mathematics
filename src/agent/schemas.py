from pydantic import BaseModel, ConfigDict, Field


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

    final_answer: str = Field(
        min_length=1,
        description=(
            "The exact, concise answer to score. Put only the checkable result here, not the derivation or verification narrative."
        ),
    )
    explanation: str = Field(
        min_length=1,
        description="Brief explanation or verification summary supporting the final answer.",
    )
    confidence: int = Field(
        ge=1,
        le=5,
        description="Confidence in the final answer on a 1-5 scale, where 5 is highest.",
    )


class SageFinalAnswerArgs(FinalAnswerArgs):
    verified_claims: list[str] = Field(
        description="Short list of final claims supported by successful Sage output or explicit reasoning.",
    )

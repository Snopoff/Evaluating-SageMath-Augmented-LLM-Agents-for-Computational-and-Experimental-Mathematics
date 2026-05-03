from pydantic import BaseModel, ConfigDict, Field


class SageExecArgs(BaseModel):
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
    verified_claims: list[str] = Field(
        default_factory=list,
        description="Optional short list of final claims supported by successful Sage output or explicit reasoning.",
    )

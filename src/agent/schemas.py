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
    
    explanation: str = Field(
        description="Post-hoc summary of the final answer in 3-6 sentences: the decisive identity / lemma and any sanity check used. Not a full derivation (that belongs in internal reasoning) and not a restatement of the question.",
    )

    final_answer: str = Field(
        description=(
            "The final answer to score. Put only the checkable result here, not the narrative and reasoning."
        ),
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

from __future__ import annotations

from pydantic import BaseModel, Field


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
    final_answer: str = Field(min_length=1, description="The final answer to return to the caller.")

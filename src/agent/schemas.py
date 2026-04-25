from __future__ import annotations

from pydantic import BaseModel, Field


class SageExecArgs(BaseModel):
    code: str = Field(min_length=1, description="Sage code to execute. Assign the final value to RESULT.")
    result_var: str = Field(default="RESULT", description="Variable name to read after execution.")
    timeout_sec: float | None = Field(default=None, description="Optional per-call timeout in seconds.")


class FinalAnswerArgs(BaseModel):
    final_answer: str = Field(min_length=1, description="The final answer to return to the caller.")

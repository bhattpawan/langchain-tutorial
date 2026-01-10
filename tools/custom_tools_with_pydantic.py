from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class MultiplyInput(BaseModel):
    a: int = Field(description="The first number to multiply")
    b: int = Field(description="The second number to multiply")


def multiply(a: int, b: int) -> int:
    """Multiplies two numbers and returns the result."""
    return a * b


multiply_tool = StructuredTool.from_function(
    func=multiply,
    name="multiply",
    description="Multiplies two numbers and returns the result.",
    args_schema=MultiplyInput,
)

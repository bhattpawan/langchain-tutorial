from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class MultiplyInput(BaseModel):
    a: int = Field(description="The first number to multiply")
    b: int = Field(description="The second number to multiply")


class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiplies two numbers"
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b


multiply_tool = MultiplyTool()

result = multiply_tool.invoke({"a": 3, "b": 2})

print(result)

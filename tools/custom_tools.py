from langchain_core.tools import tool


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers and returns the result."""
    return a * b


result = multiply.invoke({"a": 6, "b": 7})

print(result)

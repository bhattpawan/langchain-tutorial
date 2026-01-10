from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import requests
from dotenv import load_dotenv

load_dotenv()


@tool
def multiply(a: int, b: int) -> int:
    """
    Given two numbers, this tool returns their product
    """

    return a * b


tool_mapping = {"multiply": multiply}

res = multiply.invoke({"a": 3, "b": 4})

llm = ChatOpenAI(model="gpt-5-nano")

llm_with_tools = llm.bind_tools([multiply])

query = "Can you multiply 5 with 6"
human_message = HumanMessage(query)

messages = [human_message]

try:
    res = llm_with_tools.invoke(messages)
    messages.append(res)
    tool_res_list = res.tool_calls
    required_tool = None
    for tool_res in tool_res_list:
        required_tool = tool_mapping.get(tool_res["name"])
        if required_tool:
            tool_response = required_tool.invoke(tool_res)
            messages.append(tool_response)
            final_response = llm_with_tools.invoke(messages)
            print(final_response.content)
except Exception as e:
    print(e)

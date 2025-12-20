import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

chat_history = [
    SystemMessage(
        content="You are a helpful assistant. Please answer all my queries in a summarised manner."
    )
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in {"exit", "quit", "close", "stop"}:
        print("Closing chat")
        break
    result = model.invoke(chat_history).content
    chat_history.append(AIMessage(content=result))
    print("AI:", result)

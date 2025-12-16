from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv

# load_dotenv()

# if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
#     raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

chat_template = ChatPromptTemplate(
    [
        ("system", "You are a {domain} expert."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}"),
    ]
)

# Load chat history
chat_history = []
with open("chat_history.txt", "r") as f:
    chat_history.extend(f.readlines())

prompt = chat_template.invoke(
    {
        "domain": "support",
        "query": "where is my refund",
        "chat_history": chat_history,
    }
)

print(prompt)

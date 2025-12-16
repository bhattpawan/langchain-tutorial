from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

chat_template = ChatPromptTemplate(
    [
        ("system", "You are a helpful {domain} expert."),
        ("human", "Please provide a detailed explanation about {topic}."),
    ]
)

prompt = chat_template.invoke({"domain": "science", "topic": "quantum computing"})

print(prompt)

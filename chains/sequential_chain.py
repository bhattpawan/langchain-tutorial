from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

initial_prompt = PromptTemplate(
    template="Generate a detailed report on {topic}", input_variables=["topic"]
)

final_prompt = PromptTemplate(
    template="Summarise 5 interesting facts from the following {text}",
    input_variables=["text"],
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

parser = StrOutputParser()

# Langchain expression language
chain = initial_prompt | model | parser | final_prompt | model | parser

result = chain.invoke({"topic": "Lionel Messi"})

print(result)

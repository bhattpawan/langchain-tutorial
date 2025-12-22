from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

file_path = f"{BASE_DIR}/data/data.txt"

loader = TextLoader(file_path, encoding="utf-8")

docs = loader.load()

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

prompt = PromptTemplate(
    template="Write summary for the following: {text}", input_variables=["text"]
)

parser = StrOutputParser()

content = docs[0].page_content

chain = prompt | model | parser

response = chain.invoke({"text": content})

print(response)

# for doc in docs:
#     print(f"METADATA:\n{doc.metadata}")
#     print(f"PAGE CONTENT:\n{doc.page_content}")

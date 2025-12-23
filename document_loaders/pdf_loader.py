from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

file_path = f"{BASE_DIR}/data/my_data.pdf"

loader = PyPDFLoader(file_path)

docs = loader.load()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

parser = StrOutputParser()

prompt = PromptTemplate(template="Summarise the text: {text}", input_variables=["text"])

chain = prompt | model | parser

response = chain.invoke({"text": docs[0].page_content})

print(response)

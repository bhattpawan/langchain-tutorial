from pathlib import Path
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader

BASE_DIR = Path(__file__).resolve().parent
file_path = f"{BASE_DIR}/semantic_data.txt"

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

loader = TextLoader(file_path, encoding="utf-8")
text = loader.load()[0].page_content

text_splitter = SemanticChunker(
    embedding,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1,
)

docs = text_splitter.create_documents([text])

for doc in docs:
    print("=================================")
    print(f"\n{doc}\n")
    print("=================================")

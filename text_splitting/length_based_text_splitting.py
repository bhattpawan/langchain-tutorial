from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
file_path = f"{BASE_DIR}/data.txt"

loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=80, chunk_overlap=0, separator="")
split_text = splitter.split_documents(docs)

print(split_text)

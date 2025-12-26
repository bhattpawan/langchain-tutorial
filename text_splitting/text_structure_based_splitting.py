from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

BASE_DIR = Path(__file__).resolve().parent
file_path = f"{BASE_DIR}/data.txt"

loader = TextLoader(file_path, encoding="utf-8")
text = loader.load()[0].page_content

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
chunks = splitter.split_text(text)

print(chunks)

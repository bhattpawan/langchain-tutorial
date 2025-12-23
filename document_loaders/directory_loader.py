from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader

BASE_DIR = Path(__file__).resolve().parent

dir_path = f"{BASE_DIR}/data/"

loader = DirectoryLoader(path=dir_path, glob="*.txt", loader_cls=TextLoader)

docs = loader.load()

# docs = loader.lazy_load()

for doc in docs:
    print(doc.metadata)

import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

documents = [
    "Langchain is a language model framework",
    "Python is for AI and ML",
    "Google Generative AI provides embedding APIs",
]

result = embedding.embed_documents(documents)

print(str(result))

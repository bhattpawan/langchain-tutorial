import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", dimensions=32)

result = embedding.embed_query("Delhi is the capital of India.")

print(str(result))

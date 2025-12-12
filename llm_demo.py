import os
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

response = llm.invoke("Write a short poem about the beauty of nature.")

print(response)

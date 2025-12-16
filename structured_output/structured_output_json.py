import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

json_schema = {
    "title": "student",
    "description": "schema about students",
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}

structured_model = model.with_structured_output(json_schema)

response = structured_model.invoke("Tell me about a student")

print(response)

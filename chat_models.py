import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", temperature=0, max_output_tokens=1024
)
# temperature affects the output randomness; 0 means deterministic output and higher values increase randomness. [0,2]
# for low value of temperature, it gives same output for same input every time.
# this randomness increasees with increase in temperature.
#  max_completion_tokens sets the maximum length of the generated response.

response = model.invoke("What is the capital of India?")

print(response)

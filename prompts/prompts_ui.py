import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import load_prompt

load_dotenv()

if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", temperature=0, max_output_tokens=1024
)

st.header("Know your celebrity")
celebrity = st.selectbox(
    "Select a celebrity:",
    [
        "Elon Musk",
        "Virat Kohli",
        "Lionel Messi",
        "Albert Einstein",
        "MS Dhoni",
        "Stephen Hawking",
        "Cristiano Ronaldo",
        "Dr Homi Jahangir Bhabha",
        "Rajesh Dhawan",
        "Sachin Tendulkar",
    ],
)
style = st.selectbox("Explanation style:", ["Casual", "Academic"])
length = st.selectbox("Explanation length:", ["Short", "Medium", "Long"])

template = load_prompt("template.json")

if st.button("Get Info"):
    chain = template | model
    response = chain.invoke({"celebrity": celebrity, "style": style, "length": length})
    # prompt = template.invoke({"celebrity": celebrity, "style": style, "length": length})
    # response = model.invoke(prompt)
    st.write(response.content)

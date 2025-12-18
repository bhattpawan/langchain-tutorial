from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
model2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

initial_prompt = PromptTemplate(
    template="Generate a detailed report on {topic}", input_variables=["topic"]
)

notes_prompt = PromptTemplate(
    template="Generate short notes on:\n {text}", input_variables=["text"]
)

quiz_prompt = PromptTemplate(
    template="Create a quiz with 5 questions based on the following text:\n {text}",
    input_variables=["text"],
)

final_prompt = PromptTemplate(
    template="Merge the provided notes and quiz into a single document:\n\nNotes:\n{notes}\n\nQuiz:\n{quiz}",
    input_variables=["notes", "quiz"],
)

parser = StrOutputParser()

initial_chain = initial_prompt | model1 | parser

# Parallel Chains
parallel_chain = RunnableParallel(
    {
        "notes": notes_prompt | model1 | parser,
        "quiz": quiz_prompt | model2 | parser,
    }
)

# Merge Chain
merge_chain = final_prompt | model1 | parser

chain = initial_chain | parallel_chain | merge_chain

response = chain.invoke({"topic": "Lionel Messi"})

print(response)

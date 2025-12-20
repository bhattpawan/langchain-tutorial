from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

prompt = PromptTemplate(
    template="Write 20 words about {topic}", input_variables=["topic"]
)

parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser)

response = chain.invoke({"topic": "Tony Stark"})

print(response)

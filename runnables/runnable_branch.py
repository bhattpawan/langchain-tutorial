from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnableBranch,
    RunnablePassthrough,
)
from dotenv import load_dotenv

# Used to make conditonal chains. Can be considered as if-else of langchain world

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

prompt1 = PromptTemplate(
    template="Write a detailed report about {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarise the report {text}", input_variables=["text"]
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 100, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough(),
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

response = final_chain.invoke({"topic": "Barcelona vs Real Madrid"})

print(response)

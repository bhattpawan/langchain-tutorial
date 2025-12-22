from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableSequence,
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)


def word_counter(text):
    return len(text.split())


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

prompt = PromptTemplate(
    template="Write a joke about {topic}", input_variables=["topic"]
)

parser = StrOutputParser()

runnable_word_counter = RunnableLambda(word_counter)

joke_generator_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "word_count": runnable_word_counter,
    }
)

final_chain = RunnableSequence(joke_generator_chain, parallel_chain)

response = final_chain.invoke({"topic": "corporate interviews"})

print(response)

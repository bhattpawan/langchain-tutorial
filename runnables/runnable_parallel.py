from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

tweet_prompt = PromptTemplate(
    template="Write tweet about {topic}", input_variables=["topic"]
)

linkedin_prompt = PromptTemplate(
    template="Write linked in post about {topic}",
    input_variables=["topic"],
)

parser = StrOutputParser()

chain = RunnableParallel(
    {
        "tweet": RunnableSequence(tweet_prompt, model, parser),
        "linkedin": RunnableSequence(linkedin_prompt, model, parser),
    }
)

response = chain.invoke({"topic": "Tony Stark"})
tweet = response["tweet"]
linkedin = response["linkedin"]
print(tweet)
print(linkedin)

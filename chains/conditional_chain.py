from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


class SentimentAnalyzer(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(
        description="The sentiment classification of the feedback"
    )


str_parser = StrOutputParser()
pydantic_parser = PydanticOutputParser(pydantic_object=SentimentAnalyzer)

classifier_prompt = PromptTemplate(
    template="Classify the sentiment of the following feedback as positive or negative: \n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": pydantic_parser.get_format_instructions()},
)

classifier_chain = classifier_prompt | model | pydantic_parser

response = classifier_chain.invoke(
    {"feedback": "This is a terrible smartphone"}
).sentiment

positive_prompt = PromptTemplate(
    template="Write an appropriate response under 50 words to this positive feedback:\n {feedback}",
    input_variables=["feedback"],
)

negative_prompt = PromptTemplate(
    template="Write an appropriate response under 50 words to this negative feedback:\n {feedback}",
    input_variables=["feedback"],
)

positive_chain = positive_prompt | model | str_parser
negative_chain = negative_prompt | model | str_parser

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "Positive", positive_chain),
    (lambda x: x.sentiment == "Negative", negative_chain),
    RunnableLambda(lambda x: "could not find sentiment"),
)

final_chain = classifier_chain | branch_chain

response = final_chain.invoke({"feedback": "This is a terrific phone"})

print(response)

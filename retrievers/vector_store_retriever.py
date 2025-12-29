from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

documents = [
    Document(page_content="Langchain is a cool library for learning genAI"),
    Document(page_content="AI is going to be the future"),
    Document(page_content="Messi is the GOAT in football"),
    Document(page_content="India is the world's largest democracy"),
]

embedding_model = OpenAIEmbeddings()

vector_store = Chroma.from_documents(
    documents=documents, embedding=embedding_model, collection_name="my_collection"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 1})

query = "Who is the GOAT?"

results = retriever.invoke(query)

print(results)

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

embedding_model = OpenAIEmbeddings()

documents = [
    Document(page_content="Use FAISS for fast vector search"),
    Document(page_content="FAISS supports approximate nearest neighbors"),
    Document(page_content="Chroma provides persistent vector storage"),
    Document(page_content="Vector databases store embeddings"),
    Document(page_content="Embeddings convert text to vectors"),
    Document(page_content="Cosine similarity measures vector distance"),
    Document(page_content="MMR balances relevance and diversity"),
    Document(page_content="Retrievers fetch documents for LLMs"),
]


vector_store = FAISS.from_documents(documents=documents, embedding=embedding_model)

# lambda_mult => relevance-diversity balance

retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 1}
)

query = "What are Vector Databases?"

result = retriever.invoke(query)

for i in result:
    print(i.page_content)

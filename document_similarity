import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

documents = [
    "Langchain is a language model framework",
    "Python is best language for AI and ML",
    "Google Generative AI provides embedding APIs",
]

query = "Which is the best language for AI"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(documents[index])
print(f"Similarity Score: {score}")

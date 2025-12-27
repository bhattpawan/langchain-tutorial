from dotenv import load_dotenv
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.schema import Document

try:
    load_dotenv()
except:
    raise AssertionError("Required key not found")

BASE_DIR = Path(__file__).resolve().parent

doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
    metadata={"team": "Royal Challengers Bangalore"},
)
doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
    metadata={"team": "Mumbai Indians"},
)
doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
    metadata={"team": "Chennai Super Kings"},
)
doc4 = Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
    metadata={"team": "Mumbai Indians"},
)
doc5 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
    metadata={"team": "Chennai Super Kings"},
)

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    ),
    persist_directory=f"{BASE_DIR}/chroma_db",
    collection_name="sample",
)

# Add new documents to vector store
vector_store.add_documents(docs)

# View Documents
vector_store.get(include=["embeddings", "documents", "metadatas"])

# Search documents
# k controls how many responses do we need in return
response = vector_store.similarity_search("Who among these is a bowler?", k=1)

# Search with similarity score
response = vector_store.similarity_search_with_score(
    "Who among these is a bowler?", k=1
)

# Filter with metadata
response = vector_store.similarity_search_with_score(
    query="Captain", k=1, filter={"team": "Chennai Super Kings"}
)

# Update existing documents
updated_doc = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Rajasthan Royals, his quick fielding and match-winning performances make him a key player.",
    metadata={"team": "Rajasthan Royals"},
)

vector_store.update_document(
    document_id="f39f6279-9369-41ee-af49-426ca7ab5989", document=updated_doc
)

# Delete emtries

vector_store.delete(ids=["f39f6279-9369-41ee-af49-426ca7ab5989"])

from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang="en")

query = "the history of world war 1"

docs = retriever.invoke(query)

print(docs)

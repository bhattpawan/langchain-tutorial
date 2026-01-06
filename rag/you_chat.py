from dotenv import load_dotenv
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
from urllib.parse import urlparse, parse_qs

load_dotenv()

# Configure Streamlit

st.set_page_config(page_title="YouChat - Chat with Youtube Videos")

st.title("YouChat - Chat with YouTube Video")


def get_video_id(youtube_url):
    parsed_url = urlparse(youtube_url)

    if parsed_url.hostname in ("www.youtube.com", "youtube.com"):
        return parse_qs(parsed_url.query).get("v", [None])[0]

    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")

    return None


def fetch_transcript(video_id):
    transcript_fetched = YouTubeTranscriptApi().fetch(video_id).to_raw_data()
    transcript = " ".join(item["text"] for item in transcript_fetched)
    return transcript


# Streamlit UI

with st.form("youtube_form"):
    youtube_url = st.text_input("Enter youtube video URL:")
    transcript_form_submit = st.form_submit_button("Load Video Transcripts")

# Load Transcripts
if transcript_form_submit:
    video_id = get_video_id(youtube_url)

    if not video_id:
        st.error("Invalid URL")
        st.stop()

    try:
        transcript_text = fetch_transcript(video_id)
        doc = Document(page_content=transcript_text, metadata={"video_id": video_id})

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

        split_docs = splitter.split_documents([doc])

        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(split_docs, embeddings)

        st.session_state.vectorstore = vector_store
        st.session_state.video_id = video_id

        st.success("Video Transcripts loaded successfully")

    except TranscriptsDisabled:
        st.error("Transcripts are disabled for the video")
    except NoTranscriptFound:
        st.error("No Transcripts found for the video")
    except Exception as e:
        st.error(str(e))

# Ask Questions from transcripts
if "vectorstore" in st.session_state:
    st.subheader("Ask your queries below:")

    with st.form("query_form"):
        query = st.text_input("Your query here:")
        query_form_submit = st.form_submit_button("Ask")

    if query_form_submit:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})

        prompt = PromptTemplate(
            template="""
                Answer the question using ONLY the context below.
                If the answer is not present in the context, say "I don't know".

                Context:
                {context}

                Question:
                {query} """,
            input_variables=["context", "query"],
        )

        model = ChatOpenAI(model="gpt-5-nano")

        chain = {"context": retriever, "query": RunnablePassthrough()} | prompt | model

        response = chain.invoke(query)

        st.markdown("### Answer:")
        st.write(response.content)


# https://www.youtube.com/watch?v=-CQatuQdQv4

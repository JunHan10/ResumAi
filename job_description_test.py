from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from ollama import Client
import streamlit as st
import pdfplumber
import os
import io


# Custom embedding function (same as in ingest.py)
class OllamaEmbeddings(Embeddings):
    def __init__(self, model="mxbai-embed-large"):
        self.model = model
        self.client = Client()
    
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            response = self.client.embeddings(
                model=self.model,
                prompt=text
            )
            embeddings.append(response["embedding"])
        return embeddings
    
    def embed_query(self, text):
        response = self.client.embeddings(
            model=self.model,
            prompt=text
        )
        return response["embedding"]

# Load FAISS vectorstore
def load_vectorstore(path="vectorstore"):
    """Load the LangChain FAISS vectorstore"""
    embedding_function = OllamaEmbeddings()
    vectorstore = FAISS.load_local(
        path, 
        embeddings=embedding_function,
        allow_dangerous_deserialization=True
    )
    return vectorstore

# Retrieve top k chunks
def search(vectorstore, query, k=5, filter=None):
    """Search for relevant chunks using similarity search"""
    results = vectorstore.similarity_search(query, k=k, filter=filter)
    # Extract just the text content from Document objects
    chunks = [doc.page_content for doc in results]
    return chunks

def query_llm(job_descriptions, question, model="llama3.2"):
    """Query the LLM with context from the vectorstore"""
    client = Client()

    prompt = f"""{question}
use this list of jobs as a reference:
{job_descriptions}"""
    
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

st.set_page_config(page_title="Resume Editor", page_icon=":briefcase:", layout="wide")

st.title("Resume Editor (Powered by llama3.2)")

uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])
question = st.text_input("Your Question", "How can I improve my resume for a software engineering position?")

analyze = st.button("Critque Resume")

if __name__ == "__main__":
    import sys
    
    # Load vectorstore
    print("Loading vectorstore...")
    try:
        vectorstore = load_vectorstore()
        print("✅ Vectorstore loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading vectorstore: {e}")
        print("Make sure you've run ingest.py first to create the vectorstore.")
        sys.exit(1)
    
    if analyze:
        st.markdown("### 📄 Analyzing Resume...")
        st.markdown("### 🔍 Searching relevant information...")
        job_descriptions = search(vectorstore, question, k=5, filter={"source": "job_description"})
        job_description_context = "\n\n".join(job_descriptions)
        json_schema = search(vectorstore, question, k=3, filter={"source": "resume_schema"})
        json_schema_context = "\n\n".join(json_schema)
        st.markdown("### 💬 Generating answer...")
        answer = query_llm(job_description_context, question)
        st.markdown("### Answer:")
        st.markdown(answer)
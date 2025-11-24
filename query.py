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
def search(vectorstore, query, k=5):
    """Search for relevant chunks using similarity search"""
    results = vectorstore.similarity_search(query, k=k)
    # Extract just the text content from Document objects
    chunks = [doc.page_content for doc in results]
    return chunks

# Build prompt + call Ollama
def query_llm(job_descriptions, json_schema, question, file_content, model="llama3.2"):
    """Query the LLM with context from the vectorstore"""
    client = Client()

    prompt = f"""You are a resume analysis assistant.

Resume Content:
{file_content}

Context of job descriptions:
{job_descriptions}

Question:
{question}

Provide a detailed and specific answer based on the context provided."""
    
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# Interactive query function
def interactive_query(vectorstore, model="llama3.2"):
    """Allow interactive querying of the vectorstore"""
    print("\n=== Resume & Job Description Query System ===")
    print("Type 'quit' or 'exit' to stop\n")
    
    while True:
        question = input("Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        print("\n🔍 Searching relevant information...")
        top_chunks = search(vectorstore, question, k=5)
        context = "\n\n".join(top_chunks)
        
        print("💬 Generating answer...\n")
        answer = query_llm(context, question, model)
        
        print("Answer:")
        print("-" * 60)
        print(answer)
        print("-" * 60)
        print()

# --- Start of Streamlit UI ---
st.set_page_config(page_title="Resume Editor", page_icon=":briefcase:", layout="wide")

st.title("Resume Editor (Powered by llama3.2)")

uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])
question = st.text_input("Your Question", "How can I improve my resume for a software engineering position?")

analyze = st.button("Critque Resume")

def extract_text_from_pdf(pdf_file):
    pdf_reader = pdfplumber.open(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text(layout=True) + "\n"
    return text

def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
    return uploaded_file.read().decode("utf-8")

# --- MAIN ---
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
    
    if analyze and uploaded_file:
        st.markdown("### 📄 Analyzing Resume...")
        try:
            file_content = extract_text(uploaded_file)
            if not file_content.strip():
                st.error("The uploaded resume is empty.")
                st.stop()
            st.markdown("### 🔍 Searching relevant information...")
            job_descriptions = search(vectorstore, question, k=5, filter={"source": "job_description"})
            job_description_context = "\n\n".join(job_descriptions)
            json_schema = search(vectorstore, question, k=3, filter={"source": "resume_schema"})
            json_schema_context = "\n\n".join(json_schema)
            st.markdown("### 💬 Generating answer...")
            answer = query_llm(job_description_context, json_schema_context, question, file_content)
            st.markdown("### Answer:")
            st.markdown(answer)
        except Exception as e:
            st.error(f"Error analyzing resume: {e}")
            st.stop()
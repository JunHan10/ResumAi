import os
import pickle
import pdfplumber
from docx import Document
from ollama import Client
import numpy as np
import faiss
from langchain_core.documents import Document as LangChainDocument
from langchain_core.embeddings import Embeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

def load_pdf_text(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def load_docx_text(path):
    doc = Document(path)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip() != ""])
    return text

def load_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

    return chunks

# Ollama client
client = Client()

# Custom embedding function wrapper for LangChain
class OllamaEmbeddings(Embeddings):
    def __init__(self, model="mxbai-embed-large"):
        self.model = model
        self.client = Client()
    
    def embed_documents(self, texts):
        embeddings = []
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"  Embedding document {i+1}/{len(texts)}...")
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

# Embed chunks
def embed_chunks(chunks, model="mxbai-embed-large"):
    embeddings = []
    for i, chunk in enumerate(chunks):
        if i % 10 == 0:
            print(f"  Embedding chunk {i+1}/{len(chunks)}...")
        response = client.embeddings(
            model=model,
            prompt=chunk
        )
        embeddings.append(response["embedding"])
    return embeddings

# Build FAISS index with LangChain
def build_faiss_index(chunks, vectors):
    vectors_array = np.array(vectors).astype("float32")
    
    # Create FAISS index
    dimension = vectors_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors_array)
    
    # Create docstore with LangChain Documents
    docstore = InMemoryDocstore({i: LangChainDocument(page_content=chunk) for i, chunk in enumerate(chunks)})
    
    # Create index_to_docstore_id mapping
    index_to_docstore_id = {i: i for i in range(len(chunks))}
    
    # Create embedding function
    embedding_function = OllamaEmbeddings()
    
    # Initialize FAISS vectorstore properly
    vectorstore = FAISS(
        embedding_function=embedding_function,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    
    return vectorstore

# Save vector store
def save_index(vectorstore, path="vectorstore"):
    os.makedirs(path, exist_ok=True)
    vectorstore.save_local(path)


# --- MAIN ---
if __name__ == "__main__":
    resume_texts = []

    # Load resumes
    resume_folder = "data/resumes"
    if not os.path.exists(resume_folder):
        print(f"Warning: {resume_folder} does not exist!")
    else:
        for file in os.listdir(resume_folder):
            path = os.path.join(resume_folder, file)
            try:
                if file.endswith(".pdf"):
                    resume_texts.append(load_pdf_text(path))
                elif file.endswith(".docx"):
                    resume_texts.append(load_docx_text(path))
                print(f"Loaded: {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")

    resume_text = "\n".join(resume_texts)

    # Load job description
    job_folder = "data/job_description"
    job_texts = []
    if not os.path.exists(job_folder):
        print(f"Warning: {job_folder} does not exist!")
    else:
        for file in os.listdir(job_folder):
            path = os.path.join(job_folder, file)
            try:
                if file.endswith(".pdf"):
                    job_texts.append(load_pdf_text(path))
                elif file.endswith(".docx"):
                    job_texts.append(load_docx_text(path))
                elif file.endswith(".txt"):
                    job_texts.append(load_text_file(path))
                print(f"Loaded: {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    job_text = "\n".join(job_texts)

    # Combine resume + job desc
    combined_text = resume_text + "\n" + job_text
    
    if not combined_text.strip():
        print("Error: No text was loaded. Check your data folders.")
        exit(1)

    # Split to chunks
    print("\nChunking text...")
    chunks = chunk_text(combined_text)
    print(f"Created {len(chunks)} chunks")

    # Generate embeddings
    print(f"\nGenerating embeddings for {len(chunks)} chunks...")
    vectors = embed_chunks(chunks)

    # Build FAISS index
    print("\nBuilding FAISS index...")
    vectorstore = build_faiss_index(chunks, vectors)

    # Save locally
    print("\nSaving vector store...")
    save_index(vectorstore)
    print(f"✅ Vector DB created successfully with {len(chunks)} chunks!")
    
    # Optional: Test search
    print("\n--- Testing Search ---")
    test_query = "What are the key skills?"
    print(f"Query: {test_query}")
    results = vectorstore.similarity_search(test_query, k=3)
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(doc.page_content[:200] + "...")
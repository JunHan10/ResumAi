from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from ollama import Client

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
def query_llm(context, question, model="llama3.2"):
    """Query the LLM with context from the vectorstore"""
    client = Client()

    prompt = f"""You are a resume analysis assistant.

Context from resume and job description:
{context}

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
    
    # Check if user wants interactive mode or single query
    if len(sys.argv) > 1:
        # Single query mode
        question = " ".join(sys.argv[1:])
        print(f"Question: {question}\n")
        
        top_chunks = search(vectorstore, question, k=5)
        context = "\n\n".join(top_chunks)
        
        answer = query_llm(context, question)
        print("Answer:")
        print("-" * 60)
        print(answer)
        print("-" * 60)
    else:
        # Interactive mode
        interactive_query(vectorstore)
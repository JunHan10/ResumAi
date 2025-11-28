# ResumAi — Local Resume Feedback AI Agent

**resumAi** is a locally hosted AI agent that analyzes your resume, compares it against **100 real job descriptions**, and provides targeted, practical feedback to help you stand out.  
Built using **Ollama (Llama 3.2)**, **LangChain**, **Streamlit**, and a **vector store** for retrieval-augmented generation (RAG).  
Fast, private, and fully offline.

---

## Features

- Upload your resume (PDF or text)
- RAG-powered feedback using 100 real-world job descriptions
- Fully private — runs entirely on your machine using Ollama
- Vectorstore-backed similarity search for relevant job matches
- Clean and simple Streamlit UI
- Uses **uv** for environment management (lightweight + fast)
- No API keys or cloud services required

---

## Architecture
User Upload → Resume Parser → Embedding Model → Vectorstore Search
→ Context Builder → Llama 3.2 (Ollama) → Feedback Output

---

## Tech Stack
- **Backend LLM**: Llama 3.2 (via Ollama)
- **Frameworks**: LangChain, Streamlit
- **Vectorstore**: Chroma / FAISS
- **Environment**: uv
- **Parsing**: pypdf or pdfplumber

---

## Installation

#### 1. Clone the repository
```bash
git clone https://github.com/yourusername/resumAi.git
cd resumAi
```
#### 2. Install UV
```bash
pip install uv
```
#### 3. Create Env and Install Dependencies
```bash
uv venv
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows

uv pip install -r requirements.txt
```

#### 4. Install Ollama
Download from: https://ollama.com/download

#### 5. Pull the LLM from Ollama
```bash
ollama pull llama3.2
```
#### 6. Pull the Embedding Model
```bash
ollama pull mxbai-embed-text
```
#### 7. Running ingest.py to process needed data
```bash
uv run ingest.py
```
#### 8. Run query.py aka the program
```bash
uv run streamlit run query.py
```




from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from ollama import Client
from rapidfuzz import process, fuzz
import streamlit as st
import pdfplumber
import os
import io
import json
import difflib
import time


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

# Build prompt + call Ollama
def query_llm(job_descriptions, json_schema, question, file_content, model="llama3.2"):
    """Query the LLM with context from the vectorstore"""
    client = Client()

    prompt = f"""You are a resume analysis assistant.

Resume Content:
{file_content}

Question to focus your analysis on:
{question}

Provide detailed and specific suggestions for changes to this resume. Do not make up any metrics not mentioned in the resume. Do not make suggestions that require a metric unless the resume already uses a metric in that section. Existing metrics should not be changed but can be reused.
Do not name a specific company or role unless it is mentioned in the resume or the question.
If the question asked mentions a specific job position, tailor your suggestion to that role using the corresponding job descrition from this list:
{job_descriptions}

Your response needs to follow this JSON schema exactly:
{json_schema}

IMPORTANT:
- Do NOT use, reference, or analyze the example JSON provided in context.
- The example JSON exists ONLY to show the structure of the output.
- Do NOT generate suggestions based on text found in the example JSON.
- Only produce suggestions based on the actual resume content supplied.
- Ignore any text retrieved from the vectorstore that resembles sample output formats.
Use the same keys, structure, and ordering as in the example json. Make sure all "," delimeters and brackets are in the correct place.
Do not invent new keys. Do not produce non-JSON text. Do not include ""type": "object"" or ""properties"" or ""required"" fields. in your response.
Do not reuse the suggestions in the example json. Create new suggestions based on the resume content and question.
"""
    
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

def suggestion_parser(response, full_text):
    data = json.loads(response)
    suggestions = []
    
    for idx, s in enumerate(data["suggestions"], start=1):
        orig = s["original_text"]
        
        line_num, start_char, end_char = find_text_span(full_text, orig)

        suggestions.append({
            "id": idx,
            "original_text": orig,
            "suggested_text": s["suggested_text"],
            "explanation": s["explanation"],
            "category": s["category"],
            "confidence": s["confidence"],
            "line_number": line_num,
            "start_char": start_char,
            "end_char": end_char,
        })
    
    return suggestions

def apply_suggestion(suggestion_id):
    st.session_state.applied_suggestions.add(suggestion_id)

def find_text_span(full_text, snippet):
    """
    Fuzzy match 'snippet' inside 'full_text'. 
    Returns (line_number, start_char, end_char).
    """

    lines = full_text.splitlines()
    best_score = -1
    best_line_idx = None
    best_line_text = None

    # 1. Find best matching line using fuzzy match
    for i, line in enumerate(lines):
        score = fuzz.partial_ratio(snippet, line)
        if score > best_score:
            best_score = score
            best_line_idx = i
            best_line_text = line

    # Reject weak matches
    if best_score < 65:
        return None, None, None

    # 2. Find exact substring location using difflib
    matcher = difflib.SequenceMatcher(None, best_line_text.lower(), snippet.lower())
    match = matcher.find_longest_match(0, len(best_line_text), 0, len(snippet))

    if match.size == 0:
        return None, None, None

    start_char = match.a
    end_char = match.a + match.size

    # Convert to 1-based line numbering
    line_number = best_line_idx + 1

    return line_number, start_char, end_char



# --- Start of Streamlit UI ---
if 'applied_suggestions' not in st.session_state:
    st.session_state.applied_suggestions = set()
if 'selected_suggestion_id' not in st.session_state:
    st.session_state.selected_suggestion_id = None
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = None
if 'file_content' not in st.session_state:
    st.session_state.file_content = None

st.set_page_config(page_title="Resume Editor", page_icon=":briefcase:", layout="wide")

# CSS for floating suggestions panel
st.markdown("""
<style>
.floating-card {
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 350px;
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    padding: 24px;
    z-index: 1000;
    color: black;
}
.floating-card h3 {
    margin-top: 0;
    color: #1a1a1a;
    font-size: 18px;
}
.close-btn {
    position: absolute;
    top: 10px;
    right: 15px;
    background: none;
    border: none;
    font-size: 20px;
    cursor: pointer;
    color: #666;
}
.close-btn:hover {
    color: #000;
}
</style>
""", unsafe_allow_html=True)

st.title("Resume Editor (Powered by llama3.2)")

uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])
question = st.text_input("Your Question", "How can I improve my resume for a software engineering position?")

analyze = st.button("Critique Resume")

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
        # Create a placeholder for the status messages
        status_placeholder = st.empty()
        
        try:
            # Stage 1: Analyzing Resume (5 seconds)
            status_placeholder.markdown("### 📄 Analyzing Resume...")
            file_content = extract_text(uploaded_file)
            if not file_content.strip():
                st.error("The uploaded resume is empty.")
                st.stop()
            st.session_state.file_content = file_content
            time.sleep(8)
            
            # Stage 2: Searching relevant information (7 seconds)
            status_placeholder.markdown("### 🔍 Searching relevant information...")
            job_descriptions = search(vectorstore, question, k=5, filter={"source": "job_description"})
            job_description_context = "\n\n".join(job_descriptions)
            json_schema = search(vectorstore, question, k=3, filter={"source": "resume_schema"})
            json_schema_context = "\n\n".join(json_schema)
            time.sleep(10)

            # Stage 3: Generating answer (until completion)
            status_placeholder.markdown("### 💬 Generating answer...")
            answer = query_llm(job_description_context, json_schema_context, question, file_content)
            print(answer)
            suggestions = suggestion_parser(answer, file_content)
            st.session_state.suggestions = suggestions
            
            # Clear the status message once processing is complete
            status_placeholder.empty()
            
        except Exception as e:
            status_placeholder.empty()
            st.error(f"Error analyzing resume: {e}")
            st.stop()
    
    # Display results if we have suggestions in session state
    if st.session_state.suggestions is not None and st.session_state.file_content is not None:
        suggestions = st.session_state.suggestions
        file_content = st.session_state.file_content
        
        # Create columns for layout
        main_col, panel_col = st.columns([3, 1])
        
        # Use the main column for resume content
        with main_col:
            # Resume Display  
            st.markdown("### Annotated Resume:")
            st.caption("Click on 'View Suggestion' buttons to view details")
            
            # Control buttons at top
            button_cols = st.columns([1, 1, 3])
            with button_cols[0]:
                if st.button("Apply Selected", disabled=not st.session_state.selected_suggestion_id):
                    if st.session_state.selected_suggestion_id:
                        apply_suggestion(st.session_state.selected_suggestion_id)
                        st.rerun()
            with button_cols[1]:
                if st.button("🔄 Clear All"):
                    st.session_state.suggestions = None
                    st.session_state.file_content = None
                    st.session_state.applied_suggestions = set()
                    st.session_state.selected_suggestion_id = None
                    st.rerun()
            
            # Display resume with highlights
            lines = file_content.splitlines()  # safer than split('\n')

            for line_idx, line in enumerate(lines, start=1):
                # Find suggestions for this line
                line_suggestions = [s for s in suggestions if s['line_number'] == line_idx]

                if not line_suggestions:
                    st.text(line if line else " ")
                    continue

                # Each line may have multiple suggestions — highlight all
                rendered_line = line
                offset = 0  # Adjusts indices as we insert markup

                # Sort by start_char (safest)
                for s in sorted(line_suggestions, key=lambda x: x['start_char']):
                    sc = s['start_char'] + offset
                    ec = s['end_char'] + offset
                    
                    before = rendered_line[:sc]
                    target = rendered_line[sc:ec]
                    after = rendered_line[ec:]
                    
                    # Color coding
                    if s['id'] in st.session_state.applied_suggestions:
                        color = "green"
                        replacement = f":green-background[{s['suggested_text']}]"
                    else:
                        conf = s['confidence']
                        if conf >= 0.9:
                            color = "green"
                        elif conf >= 0.75:
                            color = "orange"
                        else:
                            color = "red"
                        replacement = f":{color}-background[{target}]"

                    # Replace target with markup
                    new_segment = replacement
                    offset += len(new_segment) - len(target)
                    rendered_line = before + new_segment + after

                st.markdown(rendered_line)

                # Add suggestion buttons under the line
                for s in line_suggestions:
                    if st.button(f"View Suggestion #{s['id']}", key=f"btn_{s['id']}"):
                        st.session_state.selected_suggestion_id = s['id']
            
            # Legend
            st.markdown("---")
            st.markdown("#### Confidence Legend")
            legend_cols = st.columns(4)
            with legend_cols[0]:
                st.markdown("🟢 High (≥90%)")
            with legend_cols[1]:
                st.markdown("🟡 Medium (75-89%)")
            with legend_cols[2]:
                st.markdown("🔴 Low (<75%)")
            with legend_cols[3]:
                st.markdown("✓ Applied")
        
        # Floating suggestion panel (outside columns)
        if st.session_state.selected_suggestion_id:
            selected_suggestion = next((s for s in suggestions if s['id'] == st.session_state.selected_suggestion_id), None)
            if selected_suggestion:
                panel_html = f"""
                <div class="floating-card">
                    <h3>💡 Suggestion #{selected_suggestion['id']}</h3>
                    <div style="margin-bottom: 15px;">
                        <strong>Confidence:</strong> {selected_suggestion['confidence']:.1%}<br>
                        <strong>Line:</strong> {selected_suggestion['line_number']}
                    </div>
                    <div style="margin-bottom: 15px;">
                        <strong>Original:</strong><br>
                        <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; font-family: monospace;">
                            {selected_suggestion['original_text']}
                        </div>
                    </div>
                    <div style="margin-bottom: 15px;">
                        <strong>Suggested:</strong><br>
                        <div style="background: #e8f5e8; padding: 10px; border-radius: 5px; font-family: monospace;">
                            {selected_suggestion['suggested_text']}
                        </div>
                    </div>
                    <div style="margin-bottom: 15px;">
                        <strong>Reason:</strong><br>
                        <div style="color: #666; font-style: italic;">
                            {selected_suggestion.get('reason', selected_suggestion.get('explanation', 'No explanation available'))}
                        </div>
                    </div>
                </div>
                """
                st.markdown(panel_html, unsafe_allow_html=True)
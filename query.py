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
        st.markdown("### 📄 Analyzing Resume...")
        try:
            file_content = extract_text(uploaded_file)
            if not file_content.strip():
                st.error("The uploaded resume is empty.")
                st.stop()
            st.session_state.file_content = file_content

            st.markdown("### 🔍 Searching relevant information...")
            job_descriptions = search(vectorstore, question, k=5, filter={"source": "job_description"})
            job_description_context = "\n\n".join(job_descriptions)
            json_schema = search(vectorstore, question, k=3, filter={"source": "resume_schema"})
            json_schema_context = "\n\n".join(json_schema)

            st.markdown("### 💬 Generating answer...")
            answer = query_llm(job_description_context, json_schema_context, question, file_content)
            print(answer)
            suggestions = suggestion_parser(answer, file_content)
            st.session_state.suggestions = suggestions
        except Exception as e:
            st.error(f"Error analyzing resume: {e}")
            st.stop()
    
    # Display results if we have suggestions in session state
    if st.session_state.suggestions is not None and st.session_state.file_content is not None:
        suggestions = st.session_state.suggestions
        file_content = st.session_state.file_content
        
        col1, col2 = st.columns([2, 1])

        with col1:
            # Resume Display
            st.markdown("### Annotated Resume:")
            st.caption("Click on 'View Suggestion' buttons to view details")
            
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
        
        with col2:
            # Suggestion Detail Panel
            selected_suggestion = None
            if st.session_state.selected_suggestion_id:
                selected_suggestion = next(
                    (s for s in suggestions if s['id'] == st.session_state.selected_suggestion_id),
                    None
                )
            
            if selected_suggestion:
                # Confidence badge
                confidence_pct = int(selected_suggestion['confidence'] * 100)
                if selected_suggestion['confidence'] >= 0.9:
                    badge_color = 'green'
                elif selected_suggestion['confidence'] >= 0.75:
                    badge_color = 'orange'
                else:
                    badge_color = 'red'
                
                st.markdown(f":{badge_color}[**{confidence_pct}% CONFIDENCE**]")
                st.markdown(f"**Category:** {selected_suggestion['category'].title()}")
                
                st.markdown("### Suggestion Details")
                
                # Original text
                st.markdown("**Original Text**")
                st.error(selected_suggestion['original_text'])
                
                # Suggested text
                st.markdown("**Suggested Text**")
                st.success(selected_suggestion['suggested_text'])
                
                # Explanation
                st.markdown("**Explanation**")
                st.info(selected_suggestion['explanation'])
                
                # Apply button
                is_applied = selected_suggestion['id'] in st.session_state.applied_suggestions
                if st.button(
                    "✓ Applied" if is_applied else "Apply Suggestion",
                    disabled=is_applied,
                    use_container_width=True,
                    type="primary" if not is_applied else "secondary",
                    key=f"apply_{selected_suggestion['id']}"
                ):
                    apply_suggestion(selected_suggestion['id'])
                    st.rerun()
            else:
                st.info("👆 Click on any 'View Suggestion' button to view details")
            
            # Summary Stats
            st.markdown("---")
            st.markdown("### Summary")
            st.metric("Total Suggestions", len(suggestions))
            st.metric("Applied", len(st.session_state.applied_suggestions))
            st.metric("Remaining", len(suggestions) - len(st.session_state.applied_suggestions))
            
            # Clear button to reset
            if st.button("🔄 Clear All", use_container_width=True):
                st.session_state.suggestions = None
                st.session_state.file_content = None
                st.session_state.applied_suggestions = set()
                st.session_state.selected_suggestion_id = None
                st.session_state.llm_answer = None
                st.rerun()
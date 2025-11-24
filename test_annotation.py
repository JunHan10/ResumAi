import streamlit as st

# Page configuration
st.set_page_config(page_title="Resume Analyzer", layout="wide")

# Initialize session state
if 'applied_suggestions' not in st.session_state:
    st.session_state.applied_suggestions = set()
if 'selected_suggestion_id' not in st.session_state:
    st.session_state.selected_suggestion_id = None

# Sample data
resume_text = """John Doe
Software Engineer

EXPERIENCE
ABC Tech Company - Software Developer (2020-2023)
- Developed web applications
- Worked with databases
- Collaborated with team members
- Fixed bugs and issues

XYZ Startup - Junior Developer (2018-2020)
- Built features for mobile app
- Participated in code reviews
- Helped with testing

EDUCATION
Bachelor of Science in Computer Science
State University (2014-2018)

SKILLS
Python, JavaScript, SQL, Git"""

suggestions = [
    {
        "id": 1,
        "start_char": 2,
        "end_char": 28,
        "line_number": 5,
        "original_text": "Developed web applications",
        "suggested_text": "Architected and deployed 5+ scalable web applications serving 10K+ users",
        "explanation": "Job requires quantifiable achievements and technical leadership. Adding metrics demonstrates impact and scale.",
        "category": "quantification",
        "confidence": 0.95
    },
    {
        "id": 2,
        "start_char": 2,
        "end_char": 23,
        "line_number": 6,
        "original_text": "Worked with databases",
        "suggested_text": "Optimized PostgreSQL database queries, improving response time by 40%",
        "explanation": "Vague statement. Specify database technology and add measurable improvement to show technical expertise.",
        "category": "specificity",
        "confidence": 0.92
    },
    {
        "id": 3,
        "start_char": 2,
        "end_char": 32,
        "line_number": 7,
        "original_text": "Collaborated with team members",
        "suggested_text": "Led cross-functional collaboration with 3 designers and 2 product managers",
        "explanation": "Job emphasizes teamwork and leadership. Quantify team size and specify roles to show leadership capacity.",
        "category": "leadership",
        "confidence": 0.88
    }
]

# Helper function to apply suggestion
def apply_suggestion(suggestion_id):
    st.session_state.applied_suggestions.add(suggestion_id)

# Title and description
st.title("Resume Analyzer")
st.markdown("Review AI-generated suggestions to tailor your resume for the target position")

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    # Resume Display
    st.markdown("### Your Resume")
    st.caption("Click on suggestions below to view details")
    
    # Display resume with highlights
    lines = resume_text.split('\n')
    
    for line_idx, line in enumerate(lines, start=1):
        # Find suggestions for this line
        line_suggestions = [s for s in suggestions if s['line_number'] == line_idx - 1]
        
        if not line_suggestions:
            st.text(line if line else " ")
        else:
            suggestion = line_suggestions[0]
            is_applied = suggestion['id'] in st.session_state.applied_suggestions
            
            if is_applied:
                # Show applied suggestion with precise replacement
                before = line[:suggestion['start_char']]
                after = line[suggestion['end_char']:]
                st.markdown(f"{before}:green-background[{suggestion['suggested_text']}]{after} **✓**")
            else:
                # Show original with precise highlight using start_char and end_char
                before = line[:suggestion['start_char']]
                highlighted = line[suggestion['start_char']:suggestion['end_char']]
                after = line[suggestion['end_char']:]
                
                # Color based on confidence
                if suggestion['confidence'] >= 0.9:
                    color = 'green'
                elif suggestion['confidence'] >= 0.75:
                    color = 'orange'
                else:
                    color = 'red'
                
                st.markdown(f"{before}:{color}-background[{highlighted}]{after}")
                
                # Add button to select this suggestion
                if st.button(f"View Suggestion #{suggestion['id']}", key=f"btn_{suggestion['id']}"):
                    st.session_state.selected_suggestion_id = suggestion['id']
    
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
            type="primary" if not is_applied else "secondary"
        ):
            apply_suggestion(selected_suggestion['id'])
            st.rerun()
    else:
        st.info("👆 Click on any highlighted text in the resume to view suggestion details")
    
    # Summary Stats
    st.markdown("---")
    st.markdown("### Summary")
    st.metric("Total Suggestions", len(suggestions))
    st.metric("Applied", len(st.session_state.applied_suggestions))
    st.metric("Remaining", len(suggestions) - len(st.session_state.applied_suggestions))
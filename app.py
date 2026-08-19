import os
import shutil
import tempfile
import streamlit as st
from dotenv import load_dotenv

from src.audio_downloader_and_processor import process_source_input
from src.audio_transcriber import transcribe_chunked_audio
from src.audio_translator import translate_chunked_audio
from src.legal_cleaner import clean_legal_dictation
from src.rag_pipeline import rag_engine

# Load environment variables from .env file if available
load_dotenv()

st.set_page_config(
    page_title="Judicial Dictation & Legal Audio RAG",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS for clean UI styling
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 950px;
    }
    h1 {
        font-weight: 700;
    }
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
    }
    .stTextArea textarea {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 14px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "transcript" not in st.session_state:
    st.session_state.transcript = None

if "audio_filename" not in st.session_state:
    st.session_state.audio_filename = ""

if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "⚖️ Legal Court Order Dictation (Verbatim Cleaning)"

if "error" not in st.session_state:
    st.session_state.error = None


def reset_session():
    st.session_state.messages = []
    st.session_state.rag_chain = None
    st.session_state.transcript = None
    st.session_state.audio_filename = ""
    st.session_state.error = None


# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    env_key = os.getenv("GROQ_API_KEY", "")
    groq_api_key = st.text_input(
        "Groq API Key",
        value=env_key,
        type="password",
        help="Enter your Groq API key from https://console.groq.com/"
    )
    
    st.subheader("Processing Mode")
    mode = st.radio(
        "Select Mode:",
        [
            "⚖️ Legal Court Order Dictation (Verbatim Cleaning)",
            "Transcription (Original Language)",
            "Translation (English)"
        ],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    if st.button("🗑️ Reset All", use_container_width=True):
        reset_session()
        st.rerun()

# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------
st.title("⚖️ Judicial Dictation & Legal Audio RAG")
st.caption("Upload court audio recordings to extract clean judicial orders and chat interactively with the document.")

# File Uploader
uploaded_file = st.file_uploader(
    "Upload Audio Recording",
    type=["mp3", "wav", "m4a", "ogg", "flac", "aac"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
    
    if st.button("🚀 Process Audio", type="primary", use_container_width=True):
        if not groq_api_key.strip():
            st.session_state.error = "⚠️ Please enter a Groq API Key in the sidebar before processing."
        else:
            st.session_state.error = None
            st.session_state.messages = []
            st.session_state.audio_filename = uploaded_file.name
            st.session_state.selected_mode = mode
            
            temp_dir = tempfile.mkdtemp(prefix="mp3_rag_")
            download_dir = os.path.join(temp_dir, "downloads")
            
            try:
                with st.status("Processing audio pipeline...", expanded=True) as status:
                    status.update(label="Step 1/4: Saving uploaded audio file...")
                    input_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(input_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    CHUNK_MINUTES = 10
                    status.update(label="Step 2/4: Downsampling & chunking audio...")
                    chunks = process_source_input(
                        source=input_path,
                        download_dir=download_dir,
                        chunk_minutes=CHUNK_MINUTES
                    )
                    st.write(f"✔️ Created {len(chunks)} audio chunk(s).")
                    
                    if "Legal" in mode:
                        status.update(label="Step 3/4: Processing legal speech & stripping Marathi/PA talk...")
                        raw_text = translate_chunked_audio(chunks, groq_api_key)
                        final_text = clean_legal_dictation(raw_text, groq_api_key)
                    elif "Transcription" in mode:
                        status.update(label="Step 3/4: Transcribing speech...")
                        final_text = transcribe_chunked_audio(chunks, groq_api_key)
                    else:
                        status.update(label="Step 3/4: Translating speech to English...")
                        final_text = translate_chunked_audio(chunks, groq_api_key)
                        
                    st.write("✔️ Speech processing completed.")
                    
                    status.update(label="Step 4/4: Building RAG search index...")
                    rag_chain = rag_engine(final_text, groq_api_key)
                    
                    st.session_state.transcript = final_text
                    st.session_state.rag_chain = rag_chain
                    
                    status.update(label="✅ Processing complete!", state="complete", expanded=False)
                    
            except Exception as e:
                st.session_state.error = f"❌ Processing error: {str(e)}"
            finally:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Force Streamlit to rerun so top-level widgets (text_area, chat_input) mount cleanly
            st.rerun()

# -----------------------------------------------------------------------------
# Error & Output Display
# -----------------------------------------------------------------------------
if st.session_state.get("error"):
    st.error(st.session_state.error)

if st.session_state.get("transcript"):
    st.markdown("---")
    st.subheader("📜 Output Document")
    
    st.text_area(
        label="Output Text",
        value=st.session_state.transcript,
        height=320,
        label_visibility="collapsed"
    )
    
    st.download_button(
        label="📥 Download Output (.txt)",
        data=st.session_state.transcript,
        file_name=f"{os.path.splitext(st.session_state.audio_filename)[0]}_clean.txt",
        mime="text/plain"
    )
    
    st.markdown("---")
    st.subheader("💬 Ask Questions")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_query := st.chat_input("Ask any question about this order..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                try:
                    response = st.session_state.rag_chain.invoke({"input": user_query})
                    answer = response.get("answer", "No answer generated.")
                    
                    st.markdown(answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                except Exception as e:
                    error_msg = f"❌ Query error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

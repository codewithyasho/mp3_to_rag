import os
import shutil
import tempfile
import streamlit as st
from dotenv import load_dotenv

from src.audio_downloader_and_processor import process_source_input
from src.audio_transcriber import transcribe_chunked_audio
from src.audio_translator import translate_chunked_audio
from src.rag_pipeline import rag_engine

# Load environment variables from .env file if available
load_dotenv()

st.set_page_config(
    page_title="MP3 to RAG Chatbot",
    page_icon="🎙️",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "transcript" not in st.session_state:
    st.session_state.transcript = None

if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

if "audio_filename" not in st.session_state:
    st.session_state.audio_filename = ""


def reset_session():
    st.session_state.messages = []
    st.session_state.rag_chain = None
    st.session_state.transcript = None
    st.session_state.file_processed = False
    st.session_state.audio_filename = ""


# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    env_key = os.getenv("GROQ_API_KEY", "")
    groq_api_key = st.text_input(
        "Groq API Key",
        value=env_key,
        type="password",
        help="Get your API key from https://console.groq.com/"
    )
    
    st.subheader("🎧 Select Your Audio Language")
    mode = st.radio(
        "Select Mode:",
        ["Transcription (English Audio)", "Translation (Non-English Audio)"],
        index=0,
        help="Transcription keeps original audio language. Translation converts non-English audio directly into English using Groq Whisper."
    )
    
    st.markdown("---")
    if st.button("🗑️ Reset Session & Chat", use_container_width=True):
        reset_session()
        st.rerun()

# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------
st.title("🎙️ MP3 / Audio to RAG Chatbot")
st.markdown(
    "Upload any MP3 or audio file to transcribe/translate it, index it into a vector store, "
    "and chat with an AI assistant about its content!"
)

# File Uploader
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["mp3", "wav", "m4a", "ogg", "flac", "aac"],
    help="Supported formats: MP3, WAV, M4A, OGG, FLAC, AAC"
)

if uploaded_file is not None:
    # Audio Preview
    st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
    
    # Process Button
    if st.button("🚀 Process Audio & Build RAG Index", type="primary", use_container_width=True):
        if not groq_api_key.strip():
            st.error("⚠️ Please provide a valid Groq API Key in the sidebar before proceeding.")
        else:
            reset_session()
            st.session_state.audio_filename = uploaded_file.name
            
            temp_dir = tempfile.mkdtemp(prefix="mp3_rag_")
            download_dir = os.path.join(temp_dir, "downloads")
            
            try:
                with st.status("Processing Audio Pipeline...", expanded=True) as status:
                    # Step 1: Save uploaded file
                    status.update(label="Step 1/4: Saving uploaded audio file...")
                    input_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(input_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Step 2: Convert and Chunk
                    CHUNK_MINUTES = 10
                    status.update(label=f"Step 2/4: Downsampling & chunking audio into {CHUNK_MINUTES}-min segments...")
                    chunks = process_source_input(
                        source=input_path,
                        download_dir=download_dir,
                        chunk_minutes=CHUNK_MINUTES
                    )
                    st.write(f"✔️ Created {len(chunks)} audio chunk(s).")
                    
                    # Step 3: Transcribe or Translate
                    if "Transcription" in mode:
                        status.update(label="Step 3/4: Transcribing audio chunks via Groq Whisper...")
                        transcript_text = transcribe_chunked_audio(chunks, groq_api_key)
                    else:
                        status.update(label="Step 3/4: Translating audio chunks to English via Groq Whisper...")
                        transcript_text = translate_chunked_audio(chunks, groq_api_key)
                        
                    st.write("✔️ Audio speech processing completed.")
                    
                    # Step 4: Build RAG Pipeline
                    status.update(label="Step 4/4: Generating embeddings & building Vector Store RAG chain...")
                    rag_chain = rag_engine(transcript_text, groq_api_key)
                    
                    st.session_state.transcript = transcript_text
                    st.session_state.rag_chain = rag_chain
                    st.session_state.file_processed = True
                    
                    status.update(label="🎉 RAG Pipeline Ready! You can now ask questions below.", state="complete")
                    
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
            finally:
                # Clean up temporary upload files
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)

# -----------------------------------------------------------------------------
# Results Section & RAG Chat
# -----------------------------------------------------------------------------
if st.session_state.file_processed and st.session_state.transcript:
    st.markdown("---")
    st.subheader(f"📄 Generated Transcript (`{st.session_state.audio_filename}`)")
    
    with st.expander("Click to view full transcript", expanded=False):
        st.write(st.session_state.transcript)
    
    st.download_button(
        label="📥 Download Transcript as TXT",
        data=st.session_state.transcript,
        file_name=f"{os.path.splitext(st.session_state.audio_filename)[0]}_transcript.txt",
        mime="text/plain"
    )
    
    st.markdown("---")
    st.subheader("💬 Chat with Audio Content")
    
    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "context" in msg and msg["context"]:
                with st.expander("🔍 View Retrieved Context Snippets"):
                    for idx, doc in enumerate(msg["context"], 1):
                        st.markdown(f"**Chunk {idx}:**\n> {doc.page_content}")

    # Chat Input
    if user_query := st.chat_input("Ask a question about the audio..."):
        # Display User Message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # Generate Answer
        with st.chat_message("assistant"):
            with st.spinner("Searching transcript & generating answer..."):
                try:
                    response = st.session_state.rag_chain.invoke({"input": user_query})
                    answer = response.get("answer", "No answer generated.")
                    context_docs = response.get("context", [])
                    
                    st.markdown(answer)
                    
                    if context_docs:
                        with st.expander("🔍 View Retrieved Context Snippets"):
                            for idx, doc in enumerate(context_docs, 1):
                                st.markdown(f"**Chunk {idx}:**\n> {doc.page_content}")
                                
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "context": context_docs
                    })
                except Exception as e:
                    error_msg = f"❌ Error querying RAG chain: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

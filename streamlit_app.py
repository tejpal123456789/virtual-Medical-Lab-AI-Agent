import streamlit as st
import requests
import os
import uuid
import tempfile
from io import BytesIO
import base64
from PIL import Image
import time
import json
from typing import Optional

# Configure Streamlit page
st.set_page_config(
    page_title="Medical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #0d6efd;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-tag {
        background-color: #ebf5ff;
        color: #0d6efd;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #4caf50;
    }
    .validation-container {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        border-radius: 20px;
    }
    .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'awaiting_validation' not in st.session_state:
    st.session_state.awaiting_validation = False
if 'validation_data' not in st.session_state:
    st.session_state.validation_data = None

# Configuration (you may want to move this to a config file)
API_BASE_URL = "http://localhost:8000"  # Adjust based on your FastAPI server

def call_chat_api(query: str) -> dict:
    """Call the FastAPI chat endpoint"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"query": query, "conversation_history": []},
            cookies={"session_id": st.session_state.session_id}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return {"status": "error", "response": "Failed to connect to API"}

def call_upload_api(image_file, text: str = "") -> dict:
    """Call the FastAPI upload endpoint"""
    try:
        files = {"image": image_file}
        data = {"text": text}
        cookies = {"session_id": st.session_state.session_id}
        
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files,
            data=data,
            cookies=cookies
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Upload API Error: {str(e)}")
        return {"status": "error", "response": "Failed to upload image"}

def call_validation_api(validation_result: str, comments: str = "") -> dict:
    """Call the FastAPI validation endpoint"""
    try:
        data = {
            "validation_result": validation_result,
            "comments": comments
        }
        cookies = {"session_id": st.session_state.session_id}
        
        response = requests.post(
            f"{API_BASE_URL}/validate",
            data=data,
            cookies=cookies
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Validation API Error: {str(e)}")
        return {"status": "error", "response": "Failed to submit validation"}

def transcribe_audio(audio_bytes) -> Optional[str]:
    """Call the FastAPI transcribe endpoint"""
    try:
        files = {"audio": ("recording.webm", audio_bytes, "audio/webm")}
        response = requests.post(f"{API_BASE_URL}/transcribe", files=files)
        response.raise_for_status()
        result = response.json()
        return result.get("transcript")
    except requests.exceptions.RequestException as e:
        st.error(f"Transcription Error: {str(e)}")
        return None

def generate_speech(text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM") -> Optional[bytes]:
    """Call the FastAPI speech generation endpoint"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate-speech",
            json={"text": text, "voice_id": voice_id}
        )
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        st.error(f"Speech Generation Error: {str(e)}")
        return None

def display_message(message: dict, is_user: bool = False):
    """Display a message in the chat"""
    if is_user:
        with st.container():
            st.markdown('<div class="user-message">', unsafe_allow_html=True)
            st.write("**You:**")
            st.write(message.get("content", ""))
            if message.get("image"):
                st.image(message["image"], width=300, caption="Uploaded Image")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        with st.container():
            st.markdown('<div class="assistant-message">', unsafe_allow_html=True)
            agent = message.get("agent", "Assistant")
            st.markdown(f'<span class="agent-tag">{agent}</span>', unsafe_allow_html=True)
            st.markdown(message.get("response", ""))
            
            # Display result image if available
            if message.get("result_image"):
                result_image_url = f"{API_BASE_URL}{message['result_image']}"
                st.image(result_image_url, width=400, caption="Analysis Result")
            
            # Add voice playback button
            if message.get("response"):
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("🔊 Play", key=f"play_{hash(message.get('response', ''))}"):
                        audio_bytes = generate_speech(message["response"])
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")
            
            st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🩺 Multi-Agent Medical Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🤖 Available Agents")
        
        with st.expander("💬 Conversation Agents", expanded=True):
            st.write("- Medical Conversation Agent")
            st.write("- Medical RAG Agent")
            st.write("- Web Search Agent")
        
        with st.expander("🔬 Computer Vision Agents", expanded=True):
            st.write("- Brain Tumor Detection")
            st.write("- Chest X-ray Covid-19 Classification")
            st.write("- Skin Lesion Segmentation")
        
        with st.expander("⚙️ Agent Capabilities", expanded=False):
            st.markdown("""
            **Medical RAG Agent:**
            - Docling based parsing
            - Embedding with structural boundaries
            - LLM semantic chunking
            - Query expansion with medical terms
            - Qdrant Vector DB hybrid search
            - Input-output guardrails
            - Confidence-based handoff
            
            **Human Validation:**
            - Human verification for CV agents
            - Multi-agent orchestration
            """)
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Conversation", type="secondary"):
            st.session_state.messages = []
            st.session_state.awaiting_validation = False
            st.session_state.validation_data = None
            st.rerun()
        
        # Audio recording section
        st.markdown("## 🎤 Voice Input")
        audio_bytes = None
        
        # Note: Streamlit doesn't have built-in audio recording
        # You would need to use a component like streamlit-audio-recorder
        # For now, we'll show a placeholder
        st.info("Voice recording feature requires additional setup with streamlit-audio-recorder component")
    
    # Main chat interface
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                display_message(message, is_user=True)
            else:
                display_message(message, is_user=False)
    
    # Handle validation if needed
    if st.session_state.awaiting_validation and st.session_state.validation_data:
        st.markdown('<div class="validation-container">', unsafe_allow_html=True)
        st.warning("⚠️ Human Validation Required: Do you agree with this result?")
        
        col1, col2, col3 = st.columns([1, 1, 3])
        
        with col1:
            if st.button("✅ Yes", key="validate_yes"):
                result = call_validation_api("yes")
                if result.get("status") != "error":
                    st.session_state.messages.append({
                        "role": "assistant",
                        "response": result.get("response", ""),
                        "agent": "HUMAN_VALIDATED"
                    })
                st.session_state.awaiting_validation = False
                st.session_state.validation_data = None
                st.rerun()
        
        with col2:
            if st.button("❌ No", key="validate_no"):
                comments = st.text_area("Comments (optional):", key="validation_comments")
                if st.button("Submit", key="submit_validation"):
                    result = call_validation_api("no", comments)
                    if result.get("status") != "error":
                        st.session_state.messages.append({
                            "role": "assistant",
                            "response": result.get("response", ""),
                            "agent": "HUMAN_VALIDATED"
                        })
                    st.session_state.awaiting_validation = False
                    st.session_state.validation_data = None
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Input section
    st.markdown("---")
    
    # Image upload
    uploaded_image = st.file_uploader(
        "📎 Upload Medical Image",
        type=["png", "jpg", "jpeg"],
        help="Upload medical images for analysis"
    )
    
    # Text input
    user_input = st.text_area(
        "💬 Ask a medical question or describe symptoms:",
        height=100,
        placeholder="Type your medical question here..."
    )
    
    # Submit button
    col1, col2 = st.columns([1, 4])
    with col1:
        submit_button = st.button("📤 Send", type="primary")
    
    # Handle submission
    if submit_button and (user_input.strip() or uploaded_image):
        # Add user message to chat
        user_message = {
            "role": "user",
            "content": user_input.strip() if user_input.strip() else "Image uploaded for analysis"
        }
        
        if uploaded_image:
            user_message["image"] = uploaded_image
        
        st.session_state.messages.append(user_message)
        
        # Show processing indicator
        with st.spinner("Processing your request..."):
            if uploaded_image:
                # Handle image upload
                result = call_upload_api(uploaded_image, user_input.strip())
            else:
                # Handle text-only query
                result = call_chat_api(user_input.strip())
        
        if result.get("status") != "error":
            assistant_message = {
                "role": "assistant",
                "response": result.get("response", ""),
                "agent": result.get("agent", "Assistant")
            }
            
            # Add result image if available
            if result.get("result_image"):
                assistant_message["result_image"] = result["result_image"]
            
            st.session_state.messages.append(assistant_message)
            
            # Check if validation is required
            if "HUMAN_VALIDATION" in result.get("agent", ""):
                st.session_state.awaiting_validation = True
                st.session_state.validation_data = result
        
        st.rerun()

if __name__ == "__main__":
    main()
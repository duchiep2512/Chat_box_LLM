"""
UI Components for Streamlit interface
"""

import streamlit as st
import json
import numpy as np
from datetime import datetime
from config import *

# Custom CSS
CUSTOM_CSS = """
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #333;
    }
    .execution-time {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
        margin-top: 0.5rem;
    }
    .relevant-sections {
        background-color: #fff3e0;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
        border: 1px solid #ffcc02;
    }
    .section-score {
        color: #ff6f00;
        font-weight: bold;
    }
    .match-content {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
        max-height: 100px;
        overflow-y: auto;
    }
    .status-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
    }
    .welcome-card {
        text-align: center;
        padding: 3rem;
        color: #666;
        background-color: #f8f9fa;
        border-radius: 1rem;
        margin: 2rem 0;
    }
</style>
"""

def apply_custom_css():
    """Apply custom CSS to the Streamlit app"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""

def display_message(message, is_user=True):
    """Display a chat message"""
    message_class = "user-message" if is_user else "bot-message"
    icon = "🧑‍💻" if is_user else "🤖"
    header = "You" if is_user else "Presight Assistant"
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <div class="message-header">{icon} {header}</div>
        <div>{message['content']}</div>
        {f'<div class="execution-time">⏱️ Execution time: {message.get("execution_time", 0):.4f} seconds</div>' if not is_user and message.get("execution_time") else ''}
    </div>
    """, unsafe_allow_html=True)
    
    # Show relevant sections for bot messages
    if not is_user and message.get("top_matches"):
        with st.expander("📊 View Top Relevant Sections", expanded=False):
            st.markdown('<div class="relevant-sections">', unsafe_allow_html=True)
            
            for i, match in enumerate(message["top_matches"], 1):
                st.markdown(f"""
                **{i}. {match['heading']}** 
                <span class="section-score">(Similarity Score: {match['similarity']:.4f})</span>
                """, unsafe_allow_html=True)
                
                # Show content preview
                content_preview = match['content'][:CONTENT_PREVIEW_LENGTH] + "..." if len(match['content']) > CONTENT_PREVIEW_LENGTH else match['content']
                st.markdown(f'<div class="match-content">{content_preview}</div>', unsafe_allow_html=True)
                
                # Show subheaders if available
                if match['section_data'].get('subheaders'):
                    with st.expander(f"📋 Subheaders for {match['heading']}", expanded=False):
                        for j, subheader in enumerate(match['section_data']['subheaders'][:3], 1):
                            st.markdown(f"**{j}. {subheader.get('Title', 'No Title')}**")
                            if subheader.get('Content'):
                                content = subheader['Content'][:SUBHEADER_PREVIEW_LENGTH] + "..." if len(subheader['Content']) > SUBHEADER_PREVIEW_LENGTH else subheader['Content']
                                st.markdown(f"_{content}_")
                            if subheader.get('List'):
                                st.markdown("**List items:**")
                                for item in subheader['List'][:3]:
                                    st.markdown(f"• {item}")
                
                if i < len(message["top_matches"]):
                    st.markdown("---")
            
            st.markdown('</div>', unsafe_allow_html=True)

def render_sidebar(rag_system):
    """Render the sidebar with controls and information"""
    with st.sidebar:
        st.markdown('<div class="status-card"><h3>🤖 Presight Chatbot</h3><p>Powered by RAG Technology</p></div>', unsafe_allow_html=True)
        
        # System status
        if rag_system:
            st.success("✅ RAG System Ready")
            if hasattr(rag_system, 'indexed_list'):
                st.info(f"📊 {len(rag_system.indexed_list)} sections loaded")
        else:
            st.error("❌ RAG System Not Available")
        
        st.markdown("---")
        
        # Configuration
        st.subheader("⚙️ Settings")
        top_k = st.slider("Relevant sections to analyze", min_value=MIN_TOP_K, max_value=MAX_TOP_K, value=DEFAULT_TOP_K,
                         help="How many most relevant sections to consider for answering")
        
        st.markdown("---")
        
        # Sample questions
        st.subheader("💡 Try These Questions")
        
        for i, question in enumerate(SAMPLE_QUESTIONS):
            if st.button(question, key=f"sample_{i}", use_container_width=True):
                st.session_state.query_input = question
                st.rerun()
        
        st.markdown("---")
        
        # Chat statistics
        st.subheader("📈 Chat Statistics")
        total_messages = len(st.session_state.messages)
        user_queries = len([msg for msg in st.session_state.messages if msg.get("is_user", False)])
        bot_responses = total_messages - user_queries
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("👤 Queries", user_queries)
        with col2:
            st.metric("🤖 Responses", bot_responses)
        
        # Average response time
        if st.session_state.messages:
            response_times = [msg.get("execution_time", 0) for msg in st.session_state.messages if not msg.get("is_user", False)]
            if response_times:
                avg_time = np.mean(response_times)
                st.metric("⚡ Avg Time", f"{avg_time:.2f}s")
        
        st.markdown("---")
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.query_input = ""
                st.rerun()
        
        with col2:
            if st.session_state.messages:
                # Convert numpy types to native Python types for JSON serialization
                def convert_for_json(obj):
                    """Convert numpy types to JSON serializable types"""
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_for_json(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_for_json(item) for item in obj]
                    else:
                        return obj
                
                # Convert messages to JSON-safe format
                json_safe_messages = convert_for_json(st.session_state.messages)
                chat_data = json.dumps(json_safe_messages, indent=2)
                
                st.download_button(
                    "📥 Export",
                    data=chat_data,
                    file_name=f"presight_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    return top_k

def render_welcome_message():
    """Render welcome message when no chat history"""
    st.markdown("""
    <div class="welcome-card">
        <h2>👋 Welcome to Presight Data Assistant!</h2>
        <p style="font-size: 1.1rem; margin: 1rem 0;">I'm here to help you understand Presight's data practices and policies.</p>
        <p>💡 Start by typing a question below or choose from the sample questions in the sidebar.</p>
        <p style="font-size: 0.9rem; color: #888;">Powered by advanced RAG technology with Google Gemini</p>
    </div>
    """, unsafe_allow_html=True)

def render_chat_interface():
    """Render the main chat interface"""
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.messages:
            render_welcome_message()
        else:
            for message in st.session_state.messages:
                display_message(message, message.get("is_user", False))

def render_query_input():
    """Render the query input form"""
    st.markdown("---")
    
    with st.form("query_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            query = st.text_input(
                "Ask your question:",
                value=st.session_state.query_input,
                placeholder="e.g., What personal data does Presight collect?",
                label_visibility="collapsed"
            )
        
        with col2:
            search_button = st.form_submit_button("🔍 Ask", type="primary", use_container_width=True)
    
    return query, search_button

def render_footer():
    """Render the footer"""
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.8rem; padding: 1rem;">
        <p>🚀 Built with Streamlit | 🧠 Powered by RAG + Google Gemini | 📊 Presight Privacy Policy Data</p>
        <p>Created by duchiep2512 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
    </div>
    """, unsafe_allow_html=True)
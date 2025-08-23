"""
Streamlit Web Interface for Enhanced RAG Chatbot

This module provides a modern web interface using Streamlit for the
enhanced chatbot with conversation history and source citations.
"""

import streamlit as st
import time
from datetime import datetime
from typing import Dict, List, Any
import json

# Import our chatbot components
from chatbot import EnhancedChatbot
from config import config

# Configure Streamlit page
st.set_page_config(
    page_title=config.ui.title,
    page_icon=config.ui.page_icon,
    layout=config.ui.layout,
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        color: #1f77b4;
    }
    
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        color: #666;
    }
    
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #9c27b0;
    }
    
    .source-info {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #ff9800;
        font-size: 0.9rem;
    }
    
    .system-info {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 5px;
        border-left: 3px solid #4caf50;
    }
    
    .error-message {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 5px;
        border-left: 3px solid #f44336;
        color: #c62828;
    }
    
    .metrics-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitChatInterface:
    """Streamlit web interface for the enhanced chatbot"""
    
    def __init__(self):
        self.initialize_session_state()
        self.chatbot = self.get_chatbot_instance()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = None
        
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
        
        if 'chat_session_started' not in st.session_state:
            st.session_state.chat_session_started = False
        
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        if 'system_status' not in st.session_state:
            st.session_state.system_status = {}
    
    def get_chatbot_instance(self) -> EnhancedChatbot:
        """Get or create chatbot instance"""
        if st.session_state.chatbot is None:
            with st.spinner("🚀 Initializing Enhanced RAG Chatbot..."):
                st.session_state.chatbot = EnhancedChatbot()
        return st.session_state.chatbot
    
    def display_header(self):
        """Display main header and subtitle"""
        st.markdown(f'<div class="main-header">{config.ui.title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="subtitle">{config.ui.subtitle}</div>', unsafe_allow_html=True)
    
    def display_sidebar(self):
        """Display sidebar with system information and controls"""
        with st.sidebar:
            st.header("🔧 System Controls")
            
            # Initialize system button
            if not st.session_state.system_initialized:
                if st.button("🚀 Initialize System", type="primary"):
                    with st.spinner("Initializing RAG system..."):
                        init_result = self.chatbot.initialize_system()
                        
                    if init_result['status'] == 'success':
                        st.session_state.system_initialized = True
                        st.session_state.system_status = init_result
                        st.success("✅ System initialized successfully!")
                        st.rerun()
                    else:
                        st.error(f"❌ Initialization failed: {init_result['message']}")
                
                st.warning("⚠️ Please initialize the system first")
                return
            
            # Start chat session
            if st.session_state.system_initialized and not st.session_state.chat_session_started:
                if st.button("💬 Start Chat Session", type="primary"):
                    session_id = self.chatbot.start_chat_session()
                    st.session_state.chat_session_started = True
                    st.success(f"✅ Started session: {session_id}")
                    st.rerun()
            
            # Chat controls
            if st.session_state.chat_session_started:
                st.divider()
                st.subheader("💬 Chat Controls")
                
                if st.button("🔄 New Conversation"):
                    self.chatbot.clear_conversation()
                    st.session_state.conversation_history = []
                    st.success("✅ Started new conversation")
                    st.rerun()
                
                if st.button("📥 Export Chat"):
                    export_result = self.chatbot.export_conversation()
                    st.success(f"✅ {export_result}")
                
                # Display conversation stats
                history = self.chatbot.get_conversation_history()
                if history:
                    st.metric("Messages", len(history))
                    
                    user_messages = sum(1 for msg in history if msg['role'] == 'user')
                    st.metric("User Messages", user_messages)
            
            # System information
            st.divider()
            st.subheader("📊 System Status")
            
            if st.button("🔄 Refresh Status"):
                st.session_state.system_status = self.chatbot.get_system_status()
            
            if st.session_state.system_status:
                status = st.session_state.system_status
                
                # Display system metrics
                if 'rag_system' in status:
                    rag_info = status['rag_system']
                    st.metric("Total Chunks", rag_info.get('total_chunks', 0))
                    st.metric("Cached Embeddings", rag_info.get('cached_embeddings', 0))
                    
                    if rag_info.get('is_indexed'):
                        st.success("🟢 RAG System Ready")
                    else:
                        st.error("🔴 RAG System Not Ready")
            
            # Configuration panel
            st.divider()
            st.subheader("⚙️ Configuration")
            
            show_sources = st.checkbox(
                "Show Sources", 
                value=config.chat.show_sources,
                help="Display source references in responses"
            )
            
            show_similarity_scores = st.checkbox(
                "Show Similarity Scores", 
                value=config.chat.show_similarity_scores,
                help="Display similarity scores for sources"
            )
            
            top_k = st.slider(
                "Number of Sources", 
                min_value=1, 
                max_value=10, 
                value=config.retrieval.top_k,
                help="Number of sources to retrieve for each query"
            )
            
            # Update chatbot settings
            self.chatbot.show_sources = show_sources
            self.chatbot.show_similarity_scores = show_similarity_scores
            config.retrieval.top_k = top_k
    
    def display_chat_interface(self):
        """Display main chat interface"""
        if not st.session_state.system_initialized:
            st.info("👈 Please initialize the system using the sidebar controls.")
            return
        
        if not st.session_state.chat_session_started:
            st.info("👈 Please start a chat session using the sidebar controls.")
            return
        
        # Display conversation history
        self.display_conversation_history()
        
        # Chat input
        self.display_chat_input()
    
    def display_conversation_history(self):
        """Display the conversation history"""
        history = self.chatbot.get_conversation_history()
        
        if not history:
            st.info("💬 Start a conversation by typing your question below!")
            return
        
        st.subheader("💬 Conversation")
        
        # Display messages in reverse order (newest first in terms of display)
        for message in history:
            if message['role'] == 'user':
                self.display_user_message(message)
            else:
                self.display_assistant_message(message)
    
    def display_user_message(self, message: Dict[str, Any]):
        """Display a user message"""
        timestamp = datetime.fromisoformat(message['timestamp']).strftime("%H:%M:%S")
        
        st.markdown(f"""
        <div class="user-message">
            <strong>👤 You</strong> <small>({timestamp})</small><br>
            {message['content']}
        </div>
        """, unsafe_allow_html=True)
    
    def display_assistant_message(self, message: Dict[str, Any]):
        """Display an assistant message with metadata"""
        timestamp = datetime.fromisoformat(message['timestamp']).strftime("%H:%M:%S")
        metadata = message.get('metadata', {})
        
        # Display main message
        st.markdown(f"""
        <div class="assistant-message">
            <strong>🤖 Assistant</strong> <small>({timestamp})</small><br>
            {message['content']}
        </div>
        """, unsafe_allow_html=True)
        
        # Display metadata if available
        if metadata and metadata.get('type') != 'welcome':
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'response_time' in metadata:
                    st.metric("⏱️ Response Time", f"{metadata['response_time']:.2f}s")
            
            with col2:
                if 'confidence' in metadata:
                    confidence = metadata['confidence']
                    confidence_label = "High" if confidence > 0.7 else "Medium" if confidence > 0.5 else "Low"
                    st.metric("🎯 Confidence", f"{confidence_label} ({confidence:.2f})")
            
            with col3:
                if 'sources_count' in metadata:
                    st.metric("📚 Sources", metadata['sources_count'])
            
            # Display detailed sources if available
            rag_result = metadata.get('rag_result', {})
            sources = rag_result.get('sources', [])
            
            if sources and self.chatbot.show_sources:
                with st.expander("📖 View Sources"):
                    for source in sources:
                        source_text = f"**{source['heading']}**"
                        if source.get('subheading'):
                            source_text += f" - {source['subheading']}"
                        
                        if self.chatbot.show_similarity_scores:
                            source_text += f" (Relevance: {source['similarity_score']:.3f})"
                        
                        st.markdown(f"""
                        <div class="source-info">
                            [{source['index']}] {source_text}<br>
                            <small>Type: {source.get('source_type', 'unknown')}</small>
                        </div>
                        """, unsafe_allow_html=True)
    
    def display_chat_input(self):
        """Display chat input and handle user messages"""
        st.divider()
        
        # Chat input form
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "💭 Your Question:",
                placeholder="Ask me anything about the available documents...",
                help="Type your question and press Ctrl+Enter or click Send",
                height=100
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                submit_button = st.form_submit_button("📤 Send", type="primary")
            
            if submit_button and user_input.strip():
                self.process_user_input(user_input.strip())
    
    def process_user_input(self, user_input: str):
        """Process user input and display response"""
        # Display user message immediately
        st.markdown(f"""
        <div class="user-message">
            <strong>👤 You</strong> <small>({datetime.now().strftime("%H:%M:%S")})</small><br>
            {user_input}
        </div>
        """, unsafe_allow_html=True)
        
        # Get response with progress indicator
        with st.spinner("🤔 Thinking..."):
            response = self.chatbot.chat(user_input)
        
        # Display response
        if response.get('status') == 'success':
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.markdown(f"""
            <div class="assistant-message">
                <strong>🤖 Assistant</strong> <small>({timestamp})</small><br>
                {response['response']}
            </div>
            """, unsafe_allow_html=True)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Response Time", f"{response.get('response_time', 0):.2f}s")
            with col2:
                confidence = response.get('confidence', 0)
                confidence_label = "High" if confidence > 0.7 else "Medium" if confidence > 0.5 else "Low"
                st.metric("🎯 Confidence", f"{confidence_label} ({confidence:.2f})")
            with col3:
                st.metric("📚 Sources", len(response.get('sources', [])))
        else:
            st.markdown(f"""
            <div class="error-message">
                <strong>❌ Error:</strong><br>
                {response.get('response', 'Unknown error occurred')}
            </div>
            """, unsafe_allow_html=True)
        
        # Force refresh to show new messages
        st.rerun()
    
    def display_system_info_page(self):
        """Display detailed system information"""
        st.header("📊 System Information")
        
        if not st.session_state.system_initialized:
            st.warning("System not initialized yet.")
            return
        
        status = self.chatbot.get_system_status()
        
        # RAG System Info
        st.subheader("🔍 RAG System")
        rag_info = status.get('rag_system', {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("System Status", "Ready" if rag_info.get('is_indexed') else "Not Ready")
            st.metric("Total Chunks", rag_info.get('total_chunks', 0))
            st.metric("Embedding Model", rag_info.get('embedding_model', 'Unknown'))
        
        with col2:
            st.metric("Vector Store Type", rag_info.get('vector_store_type', 'Unknown'))
            st.metric("Embedding Dimension", rag_info.get('embedding_dimension', 0))
            st.metric("Cached Embeddings", rag_info.get('cached_embeddings', 0))
        
        # Conversation Info
        st.subheader("💬 Conversation System")
        conv_info = status.get('conversation', {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Session", conv_info.get('current_session_id', 'None'))
            st.metric("Total Sessions", conv_info.get('total_sessions', 0))
        
        with col2:
            st.metric("Messages in Session", conv_info.get('current_session_messages', 0))
        
        # Configuration
        st.subheader("⚙️ Configuration")
        config_info = status.get('configuration', {})
        
        config_data = {
            "Max History Length": config_info.get('max_history_length', 0),
            "Response Timeout": f"{config_info.get('response_timeout', 0)}s",
            "Show Sources": config_info.get('show_sources', False),
            "Show Similarity Scores": config_info.get('show_similarity_scores', False)
        }
        
        st.json(config_data)
    
    def run(self):
        """Run the Streamlit application"""
        self.display_header()
        
        # Create tabs for different sections
        tab1, tab2 = st.tabs(["💬 Chat", "📊 System Info"])
        
        with tab1:
            # Main chat interface
            col1, col2 = st.columns([3, 1])
            
            with col1:
                self.display_chat_interface()
            
            with col2:
                self.display_sidebar()
        
        with tab2:
            self.display_system_info_page()

# Main application
def main():
    """Main application entry point"""
    interface = StreamlitChatInterface()
    interface.run()

if __name__ == "__main__":
    main()
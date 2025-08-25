"""
Main Streamlit Application for Presight Data Chatbot
Created by: duchiep2512
Date: 2025-08-23 13:56:25 UTC
"""

import streamlit as st
import os
from datetime import datetime

# Import custom modules
from config import *
from rag_system import initialize_rag_system, process_query_with_rag
from ui_components import (
    apply_custom_css, 
    initialize_session_state,
    render_sidebar,
    render_chat_interface,
    render_query_input,
    render_footer
)

# Configure Streamlit page
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

def check_prerequisites():
    """Check if all prerequisites are met"""
    errors = []
    
    # Check Google API key
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE" or len(GOOGLE_API_KEY) < 20:
        errors.append("🚨 **Google API Key Required**\n\nPlease set your Google Gemini API key in `config.py`")
    
    # Check data file
    if not os.path.exists(INDEXED_DATA_FILE):
        errors.append(f"🚨 **Data File Missing**\n\n`{INDEXED_DATA_FILE}` is required but not found.")
    
    return errors

def main():
    """Main application function"""
    # Apply custom styling
    apply_custom_css()
    
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.title("💬 Presight Data Assistant")
    st.markdown("*Ask me anything about Presight's data policies and practices!*")
    
    # Check prerequisites
    errors = check_prerequisites()
    if errors:
        for error in errors:
            st.error(error)
        
        with st.expander("🔧 Setup Instructions", expanded=True):
            st.markdown("""
            **To set up the application:**
            
            1. **Google API Key:**
               - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
               - Open `config.py` and replace `YOUR_GOOGLE_API_KEY_HERE` with your actual API key
            
            2. **Data File:**
               - Make sure you have processed your data and created `indexed_list.json`
               - Place the file in the same directory as this application
            
            3. **Install Dependencies:**
               ```bash
               pip install -r requirements.txt
               ```
            
            4. **Restart the application**
            """)
        st.stop()
    
    # Load RAG system
    rag_system = initialize_rag_system()
    
    # Render sidebar and get configuration
    top_k = render_sidebar(rag_system)
    
    # Render main chat interface
    render_chat_interface()
    
    # Render query input
    query, search_button = render_query_input()
    
    # Process query
    if search_button and query.strip():
        # Clear pending input
        st.session_state.query_input = ""
        
        # Add user message
        user_message = {
            "content": query,
            "is_user": True,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(user_message)
        
        # Process query
        with st.spinner("🤖 Analyzing your question..."):
            result = process_query_with_rag(rag_system, query, top_k=top_k)
            
            # Add bot response
            bot_message = {
                "content": result["answer"],
                "is_user": False,
                "timestamp": datetime.now().isoformat(),
                "top_matches": result.get("top_matches", []),
                "execution_time": result.get("execution_time", 0)
            }
            st.session_state.messages.append(bot_message)
        
        st.rerun()
    
    # Handle sample question selection
    if st.session_state.query_input and not search_button:
        st.info(f"💡 Selected: '{st.session_state.query_input}' - Click 'Ask' to submit or modify the question above.")
    
    # Render footer
    render_footer()

if __name__ == "__main__":
    main()
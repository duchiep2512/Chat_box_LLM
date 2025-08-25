"""
Configuration file for Presight Data Chatbot
"""

# API Configuration
GOOGLE_API_KEY = "Your_Google_API_Key_Here"

# File paths
INDEXED_DATA_FILE = "indexed_list.json"

# Model configurations
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "gemini-1.5-flash-8b-latest"

# Streamlit page configuration
PAGE_TITLE = "Presight Data Assistant"
PAGE_ICON = "💬"
LAYOUT = "wide"

# RAG Configuration
DEFAULT_TOP_K = 5
MIN_TOP_K = 1
MAX_TOP_K = 10

# UI Configuration
CONTENT_PREVIEW_LENGTH = 200
SUBHEADER_PREVIEW_LENGTH = 150
CHAT_HISTORY_MAX = 50

# Sample questions
SAMPLE_QUESTIONS = [
    "What personal data does Presight collect?",
    "How does Presight use my personal information?",
    "How long does Presight retain personal data?",
    "What security measures does Presight implement?",
    "How can I delete my personal data?",
    "Does Presight share data with third parties?",
    "What are my rights regarding personal data?",
    "How can I contact Presight about privacy concerns?"
]
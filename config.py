"""
Configuration management for the RAG Chatbot System
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class EmbeddingConfig:
    """Configuration for embedding model and vector search"""
    model_name: str = 'all-MiniLM-L6-v2'
    dimension: int = 384  # Dimension for all-MiniLM-L6-v2
    cache_embeddings: bool = True
    cache_file: str = 'embeddings_cache.pkl'

@dataclass
class FAISSConfig:
    """Configuration for FAISS vector database"""
    index_type: str = 'IndexFlatIP'  # Inner Product for cosine similarity
    use_gpu: bool = False
    nprobe: int = 10
    save_index: bool = True
    index_file: str = 'faiss_index.bin'

@dataclass
class LLMConfig:
    """Configuration for Language Model (Gemini)"""
    api_key: str = os.getenv('GEMINI_API_KEY', 'Your_API_Key')
    model_name: str = 'gemini-1.5-pro'
    temperature: float = 0.0
    max_output_tokens: int = 8192
    top_k: int = 24
    top_p: float = 0.8

@dataclass
class RetrievalConfig:
    """Configuration for document retrieval"""
    top_k: int = 5
    similarity_threshold: float = 0.3
    max_chunk_size: int = 512
    chunk_overlap: int = 50
    enable_reranking: bool = True

@dataclass
class ChatConfig:
    """Configuration for chat interface"""
    max_history_length: int = 10
    response_timeout: int = 30
    show_sources: bool = True
    show_similarity_scores: bool = False
    languages: list = None

    def __post_init__(self):
        if self.languages is None:
            self.languages = ['English', 'Vietnamese']

@dataclass
class UIConfig:
    """Configuration for UI interface"""
    title: str = "🤖 Enhanced RAG Chatbot"
    subtitle: str = "Powered by Google Gemini and FAISS"
    page_icon: str = "🤖"
    layout: str = "wide"
    theme: str = "light"

class Config:
    """Main configuration class that combines all configurations"""
    
    def __init__(self):
        self.embedding = EmbeddingConfig()
        self.faiss = FAISSConfig()
        self.llm = LLMConfig()
        self.retrieval = RetrievalConfig()
        self.chat = ChatConfig()
        self.ui = UIConfig()
        
        # Data paths
        self.data_file = 'indexed_list.json'
        self.cache_dir = 'cache'
        self.logs_dir = 'logs'
        
        # Create directories if they don't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def get_cache_path(self, filename: str) -> str:
        """Get full path for cache file"""
        return os.path.join(self.cache_dir, filename)
    
    def get_log_path(self, filename: str) -> str:
        """Get full path for log file"""
        return os.path.join(self.logs_dir, filename)
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary"""
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)

# Global configuration instance
config = Config()
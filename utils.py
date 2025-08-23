"""
Utility functions and helper classes for the RAG Chatbot System
"""

import os
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pickle
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileManager:
    """Utility class for file operations"""
    
    @staticmethod
    def ensure_directory(directory_path: str) -> None:
        """Ensure directory exists, create if it doesn't"""
        Path(directory_path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def safe_json_load(file_path: str, default: Any = None) -> Any:
        """Safely load JSON file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
            return default
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {e}")
            return default
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return default
    
    @staticmethod
    def safe_json_save(data: Any, file_path: str, indent: int = 2) -> bool:
        """Safely save data to JSON file"""
        try:
            # Ensure directory exists
            FileManager.ensure_directory(os.path.dirname(file_path))
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
            return True
        except Exception as e:
            logger.error(f"Error saving to {file_path}: {e}")
            return False
    
    @staticmethod
    def safe_pickle_load(file_path: str, default: Any = None) -> Any:
        """Safely load pickle file with error handling"""
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            logger.warning(f"Pickle file not found: {file_path}")
            return default
        except Exception as e:
            logger.error(f"Error loading pickle from {file_path}: {e}")
            return default
    
    @staticmethod
    def safe_pickle_save(data: Any, file_path: str) -> bool:
        """Safely save data to pickle file"""
        try:
            # Ensure directory exists
            FileManager.ensure_directory(os.path.dirname(file_path))
            
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            logger.error(f"Error saving pickle to {file_path}: {e}")
            return False
    
    @staticmethod
    def get_file_hash(file_path: str) -> Optional[str]:
        """Get MD5 hash of a file"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}")
            return None

class TextProcessor:
    """Utility class for text processing operations"""
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text"""
        import re
        return re.sub(r'\s+', ' ', text.strip())
    
    @staticmethod
    def clean_text(text: str, preserve_structure: bool = True) -> str:
        """Clean text while optionally preserving structure"""
        import re
        
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = TextProcessor.normalize_whitespace(text)
        
        if preserve_structure:
            # Keep basic punctuation for sentence structure
            text = re.sub(r'[^\w\s.,!?;:()\-\'"áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', '', text)
        else:
            # More aggressive cleaning
            text = re.sub(r'[^\w\s]', ' ', text)
            text = TextProcessor.normalize_whitespace(text)
        
        return text
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences"""
        import re
        
        # Simple sentence splitting (can be improved with spacy or nltk)
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract simple keywords from text"""
        import re
        from collections import Counter
        
        # Simple keyword extraction (can be improved with TF-IDF or other methods)
        words = re.findall(r'\b\w{3,}\b', text.lower())
        
        # Filter out common stop words (basic set)
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'this', 'that', 'these', 'those',
            'một', 'của', 'và', 'với', 'cho', 'từ', 'về', 'trong', 'trên', 'dưới'
        }
        
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count and return top keywords
        word_counts = Counter(filtered_words)
        return [word for word, _ in word_counts.most_common(max_keywords)]

class ValidationUtils:
    """Utility class for validation operations"""
    
    @staticmethod
    def validate_embedding_dimensions(embeddings: np.ndarray, expected_dim: int) -> bool:
        """Validate embedding dimensions"""
        if embeddings.ndim != 2:
            logger.error(f"Embeddings should be 2D array, got {embeddings.ndim}D")
            return False
        
        if embeddings.shape[1] != expected_dim:
            logger.error(f"Expected {expected_dim} dimensions, got {embeddings.shape[1]}")
            return False
        
        return True
    
    @staticmethod
    def validate_api_key(api_key: str, service: str = "API") -> bool:
        """Validate API key format"""
        if not api_key or api_key.strip() == "" or "Your_API" in api_key:
            logger.error(f"Invalid {service} key. Please set a valid API key.")
            return False
        
        if len(api_key.strip()) < 10:
            logger.warning(f"{service} key seems too short, please verify.")
            return False
        
        return True
    
    @staticmethod
    def validate_config(config_dict: Dict[str, Any], required_keys: List[str]) -> List[str]:
        """Validate configuration dictionary"""
        missing_keys = []
        for key in required_keys:
            if key not in config_dict:
                missing_keys.append(key)
        
        if missing_keys:
            logger.error(f"Missing required configuration keys: {missing_keys}")
        
        return missing_keys

class MetricsCalculator:
    """Utility class for calculating various metrics"""
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        return dot_product / (norm_vec1 * norm_vec2)
    
    @staticmethod
    def calculate_retrieval_metrics(retrieved_items: List[Any], 
                                   relevant_items: List[Any], 
                                   k: int = None) -> Dict[str, float]:
        """Calculate precision, recall, and F1 for retrieval"""
        if k is None:
            k = len(retrieved_items)
        
        retrieved_set = set(retrieved_items[:k])
        relevant_set = set(relevant_items)
        
        if len(retrieved_set) == 0:
            precision = 0.0
        else:
            precision = len(retrieved_set.intersection(relevant_set)) / len(retrieved_set)
        
        if len(relevant_set) == 0:
            recall = 0.0
        else:
            recall = len(retrieved_set.intersection(relevant_set)) / len(relevant_set)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'retrieved_count': len(retrieved_set),
            'relevant_count': len(relevant_set),
            'intersection_count': len(retrieved_set.intersection(relevant_set))
        }
    
    @staticmethod
    def calculate_response_quality_score(similarity_scores: List[float], 
                                       confidence: float = 1.0) -> float:
        """Calculate overall response quality score"""
        if not similarity_scores:
            return 0.0
        
        # Weighted average of similarity scores (higher weight for top results)
        weights = [1.0 / (i + 1) for i in range(len(similarity_scores))]
        weighted_avg = sum(score * weight for score, weight in zip(similarity_scores, weights))
        total_weight = sum(weights)
        
        if total_weight == 0:
            return 0.0
        
        base_score = weighted_avg / total_weight
        
        # Apply confidence multiplier
        quality_score = base_score * confidence
        
        return min(quality_score, 1.0)

class LoggingUtils:
    """Utility class for logging operations"""
    
    @staticmethod
    def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
        """Setup a logger with file and console handlers"""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # File handler if specified
        if log_file:
            FileManager.ensure_directory(os.path.dirname(log_file))
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Console formatter
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    @staticmethod
    def log_system_info(logger: logging.Logger) -> None:
        """Log system information"""
        import platform
        import sys
        
        logger.info(f"System: {platform.system()} {platform.release()}")
        logger.info(f"Python: {sys.version}")
        logger.info(f"Platform: {platform.platform()}")

class PerformanceMonitor:
    """Utility class for performance monitoring"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        logger.info(f"Starting {self.name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"Completed {self.name} in {duration:.2f} seconds")
    
    def get_duration(self) -> Optional[float]:
        """Get operation duration"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

class DataExporter:
    """Utility class for data export operations"""
    
    @staticmethod
    def export_to_csv(data: List[Dict[str, Any]], file_path: str) -> bool:
        """Export data to CSV file"""
        try:
            import pandas as pd
            
            df = pd.DataFrame(data)
            FileManager.ensure_directory(os.path.dirname(file_path))
            df.to_csv(file_path, index=False, encoding='utf-8')
            logger.info(f"Data exported to CSV: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False
    
    @staticmethod
    def export_chunks_analysis(chunks: List[Any], file_path: str) -> bool:
        """Export chunks analysis to JSON"""
        try:
            analysis_data = []
            
            for i, chunk in enumerate(chunks):
                chunk_analysis = {
                    'index': i,
                    'content_length': len(chunk.content) if hasattr(chunk, 'content') else 0,
                    'heading': getattr(chunk, 'heading', 'Unknown'),
                    'subheading': getattr(chunk, 'subheading', None),
                    'chunk_id': getattr(chunk, 'chunk_id', f'chunk_{i}'),
                    'metadata': getattr(chunk, 'metadata', {}),
                    'word_count': len(chunk.content.split()) if hasattr(chunk, 'content') else 0
                }
                analysis_data.append(chunk_analysis)
            
            return FileManager.safe_json_save(analysis_data, file_path)
        except Exception as e:
            logger.error(f"Error exporting chunks analysis: {e}")
            return False

# Convenience functions
def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Setup logging for the application"""
    log_file = os.path.join(log_dir, f"rag_chatbot_{datetime.now().strftime('%Y%m%d')}.log")
    return LoggingUtils.setup_logger("RAGChatbot", log_file, level)

def validate_system_requirements() -> Dict[str, bool]:
    """Validate system requirements"""
    requirements = {
        'numpy': False,
        'sentence_transformers': False,
        'faiss': False,
        'streamlit': False,
        'google_generativeai': False
    }
    
    try:
        import numpy
        requirements['numpy'] = True
    except ImportError:
        pass
    
    try:
        import sentence_transformers
        requirements['sentence_transformers'] = True
    except ImportError:
        pass
    
    try:
        import faiss
        requirements['faiss'] = True
    except ImportError:
        pass
    
    try:
        import streamlit
        requirements['streamlit'] = True
    except ImportError:
        pass
    
    try:
        import google.generativeai
        requirements['google_generativeai'] = True
    except ImportError:
        pass
    
    return requirements

def get_system_summary() -> Dict[str, Any]:
    """Get comprehensive system summary"""
    import platform
    import sys
    
    return {
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'processor': platform.processor()
        },
        'python': {
            'version': sys.version,
            'executable': sys.executable
        },
        'requirements': validate_system_requirements(),
        'timestamp': datetime.now().isoformat()
    }
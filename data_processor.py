"""
Enhanced Data Processor for RAG System

This module handles data processing, chunking, and document preparation
for the RAG pipeline with improved strategies for better retrieval.
"""

import json
import re
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of document with metadata"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    source: str
    heading: str
    subheading: Optional[str] = None
    chunk_index: int = 0
    chunk_size: int = 0
    similarity_score: Optional[float] = None

class DataProcessor:
    """Enhanced data processor with semantic chunking and metadata preservation"""
    
    def __init__(self, max_chunk_size: int = 512, chunk_overlap: int = 50):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        
    def load_indexed_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load indexed data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} indexed items from {file_path}")
            return data
        except FileNotFoundError:
            logger.error(f"File {file_path} not found")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {file_path}: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but preserve sentence structure
        text = re.sub(r'[^\w\s.,!?;:()\-\'"áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', '', text)
        
        return text

    def semantic_chunking(self, text: str, max_size: int = None) -> List[str]:
        """
        Advanced chunking strategy that preserves semantic meaning
        Splits on sentence boundaries when possible
        """
        if not text:
            return []
            
        max_size = max_size or self.max_chunk_size
        
        # If text is shorter than max_size, return as single chunk
        if len(text) <= max_size:
            return [text]
        
        # Split into sentences first
        sentences = re.split(r'[.!?]+\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed max_size
            potential_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if len(potential_chunk) > max_size:
                # If current_chunk is not empty, save it
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap if sentence is not too long
                    if len(sentence) <= max_size:
                        # Add some overlap from the end of previous chunk
                        overlap_words = current_chunk.split()[-self.chunk_overlap//10:] if self.chunk_overlap > 0 else []
                        current_chunk = " ".join(overlap_words + [sentence])
                    else:
                        # If sentence itself is too long, split it by character
                        current_chunk = sentence[:max_size]
                        chunks.append(current_chunk)
                        current_chunk = sentence[max_size-self.chunk_overlap:] if len(sentence) > max_size else ""
                else:
                    # If sentence itself is longer than max_size, split by character
                    current_chunk = sentence[:max_size]
            else:
                current_chunk = potential_chunk
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def create_document_chunks(self, indexed_data: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """
        Create document chunks with enhanced metadata for better retrieval
        """
        chunks = []
        chunk_id_counter = 0
        
        for item in indexed_data:
            heading = item.get('heading', '')
            content = item.get('content', '')
            subheaders = item.get('subheaders', [])
            
            # Process main content
            if content:
                cleaned_content = self.clean_text(content)
                content_chunks = self.semantic_chunking(cleaned_content)
                
                for i, chunk_text in enumerate(content_chunks):
                    chunk_id = self._generate_chunk_id(heading, chunk_text, i)
                    
                    chunk = DocumentChunk(
                        content=chunk_text,
                        metadata={
                            'source_type': 'main_content',
                            'heading': heading,
                            'original_index': len(chunks),
                            'total_chunks': len(content_chunks),
                            'char_count': len(chunk_text),
                            'word_count': len(chunk_text.split())
                        },
                        chunk_id=chunk_id,
                        source='indexed_data',
                        heading=heading,
                        chunk_index=i,
                        chunk_size=len(chunk_text)
                    )
                    chunks.append(chunk)
                    chunk_id_counter += 1
            
            # Process subheaders
            for subheader in subheaders:
                sub_title = subheader.get('Title', '')
                sub_content = subheader.get('Content', '')
                sub_list = subheader.get('List', [])
                
                # Combine subheader content
                combined_sub_content = sub_content
                if sub_list:
                    list_content = " ".join(sub_list)
                    combined_sub_content = f"{sub_content} {list_content}".strip()
                
                if combined_sub_content:
                    cleaned_sub_content = self.clean_text(combined_sub_content)
                    sub_chunks = self.semantic_chunking(cleaned_sub_content)
                    
                    for i, chunk_text in enumerate(sub_chunks):
                        chunk_id = self._generate_chunk_id(f"{heading}_{sub_title}", chunk_text, i)
                        
                        chunk = DocumentChunk(
                            content=chunk_text,
                            metadata={
                                'source_type': 'subheader',
                                'heading': heading,
                                'subheading': sub_title,
                                'original_index': len(chunks),
                                'total_chunks': len(sub_chunks),
                                'has_list': len(sub_list) > 0,
                                'char_count': len(chunk_text),
                                'word_count': len(chunk_text.split())
                            },
                            chunk_id=chunk_id,
                            source='indexed_data',
                            heading=heading,
                            subheading=sub_title,
                            chunk_index=i,
                            chunk_size=len(chunk_text)
                        )
                        chunks.append(chunk)
                        chunk_id_counter += 1
        
        logger.info(f"Created {len(chunks)} document chunks")
        return chunks

    def _generate_chunk_id(self, heading: str, content: str, index: int) -> str:
        """Generate a unique ID for each chunk"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        heading_clean = re.sub(r'[^\w]', '_', heading.lower())
        return f"{heading_clean}_{index}_{content_hash}"

    def prepare_text_for_embedding(self, chunks: List[DocumentChunk]) -> List[str]:
        """
        Prepare text for embedding by creating enhanced representations
        """
        prepared_texts = []
        
        for chunk in chunks:
            # Create enhanced text for better embedding
            enhanced_text = chunk.content
            
            # Add heading context for better semantic understanding
            if chunk.heading:
                enhanced_text = f"Topic: {chunk.heading}. Content: {enhanced_text}"
            
            # Add subheading context if available
            if chunk.subheading:
                enhanced_text = f"Subtopic: {chunk.subheading}. {enhanced_text}"
            
            prepared_texts.append(enhanced_text)
        
        return prepared_texts

    def get_chunk_metadata_summary(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Get summary statistics of the chunked data"""
        if not chunks:
            return {}
        
        total_chunks = len(chunks)
        total_chars = sum(chunk.chunk_size for chunk in chunks)
        total_words = sum(chunk.metadata.get('word_count', 0) for chunk in chunks)
        
        main_content_chunks = sum(1 for chunk in chunks if chunk.metadata.get('source_type') == 'main_content')
        subheader_chunks = sum(1 for chunk in chunks if chunk.metadata.get('source_type') == 'subheader')
        
        unique_headings = len(set(chunk.heading for chunk in chunks))
        unique_subheadings = len(set(chunk.subheading for chunk in chunks if chunk.subheading))
        
        avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0
        
        return {
            'total_chunks': total_chunks,
            'total_characters': total_chars,
            'total_words': total_words,
            'main_content_chunks': main_content_chunks,
            'subheader_chunks': subheader_chunks,
            'unique_headings': unique_headings,
            'unique_subheadings': unique_subheadings,
            'average_chunk_size': round(avg_chunk_size, 2),
            'max_chunk_size': self.max_chunk_size,
            'chunk_overlap': self.chunk_overlap
        }

    def export_chunks_for_analysis(self, chunks: List[DocumentChunk], output_file: str) -> None:
        """Export chunks to JSON file for analysis"""
        chunks_data = []
        for chunk in chunks:
            chunk_data = {
                'chunk_id': chunk.chunk_id,
                'content': chunk.content,
                'metadata': chunk.metadata,
                'source': chunk.source,
                'heading': chunk.heading,
                'subheading': chunk.subheading,
                'chunk_index': chunk.chunk_index,
                'chunk_size': chunk.chunk_size
            }
            chunks_data.append(chunk_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported {len(chunks)} chunks to {output_file}")
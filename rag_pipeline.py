"""
RAG Pipeline with FAISS Integration

This module implements a complete Retrieval-Augmented Generation pipeline
with FAISS vector database for efficient similarity search and ranking.
"""

import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import logging
from dataclasses import asdict

from config import config
from data_processor import DocumentChunk

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages embeddings with caching support"""
    
    def __init__(self, model_name: str = None, cache_embeddings: bool = True):
        self.model_name = model_name or config.embedding.model_name
        self.cache_embeddings = cache_embeddings
        self.cache_file = config.get_cache_path(config.embedding.cache_file)
        
        # Initialize the embedding model
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Load cached embeddings if available
        self.embedding_cache = self._load_embedding_cache()
    
    def _load_embedding_cache(self) -> Dict[str, np.ndarray]:
        """Load cached embeddings from file"""
        if self.cache_embeddings and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.info(f"Loaded {len(cache)} cached embeddings")
                return cache
            except Exception as e:
                logger.warning(f"Could not load embedding cache: {e}")
        return {}
    
    def _save_embedding_cache(self) -> None:
        """Save embeddings cache to file"""
        if self.cache_embeddings:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.embedding_cache, f)
                logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
            except Exception as e:
                logger.warning(f"Could not save embedding cache: {e}")
    
    def get_embeddings(self, texts: List[str], chunk_ids: List[str] = None) -> np.ndarray:
        """Get embeddings for texts with caching support"""
        if chunk_ids is None:
            chunk_ids = [f"temp_{i}" for i in range(len(texts))]
        
        embeddings = []
        texts_to_encode = []
        indices_to_encode = []
        
        # Check cache first
        for i, (text, chunk_id) in enumerate(zip(texts, chunk_ids)):
            if chunk_id in self.embedding_cache:
                embeddings.append(self.embedding_cache[chunk_id])
            else:
                embeddings.append(None)
                texts_to_encode.append(text)
                indices_to_encode.append(i)
        
        # Encode uncached texts
        if texts_to_encode:
            logger.info(f"Encoding {len(texts_to_encode)} new texts")
            new_embeddings = self.model.encode(texts_to_encode, show_progress_bar=True)
            
            # Store in cache and results
            for idx, embedding in zip(indices_to_encode, new_embeddings):
                embeddings[idx] = embedding
                if self.cache_embeddings:
                    self.embedding_cache[chunk_ids[idx]] = embedding
        
        # Save cache if updated
        if texts_to_encode and self.cache_embeddings:
            self._save_embedding_cache()
        
        return np.array(embeddings)

class FAISSVectorStore:
    """FAISS-based vector store for efficient similarity search"""
    
    def __init__(self, embedding_dim: int, index_type: str = 'IndexFlatIP'):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index_file = config.get_cache_path(config.faiss.index_file)
        
        # Initialize FAISS index
        self._create_index()
        self.chunks: List[DocumentChunk] = []
    
    def _create_index(self) -> None:
        """Create FAISS index based on configuration"""
        if self.index_type == 'IndexFlatIP':
            # Inner Product (for cosine similarity with normalized vectors)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == 'IndexFlatL2':
            # L2 distance
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == 'IndexIVFFlat':
            # Faster search with some accuracy tradeoff
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        else:
            # Default to flat IP
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        logger.info(f"Created FAISS index: {self.index_type}")
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[DocumentChunk]) -> None:
        """Add embeddings and corresponding chunks to the vector store"""
        # Normalize embeddings for cosine similarity (required for IndexFlatIP)
        if self.index_type == 'IndexFlatIP':
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Train index if needed (for IVF-based indices)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            logger.info("Training FAISS index...")
            self.index.train(embeddings.astype('float32'))
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        self.chunks.extend(chunks)
        
        logger.info(f"Added {len(embeddings)} embeddings to FAISS index")
        logger.info(f"Total vectors in index: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar vectors in the index"""
        # Normalize query embedding
        if self.index_type == 'IndexFlatIP':
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32').reshape(1, -1), top_k)
        
        # Return results with chunks and scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):  # Valid index
                chunk = self.chunks[idx]
                # Create a copy with similarity score
                chunk_copy = DocumentChunk(
                    content=chunk.content,
                    metadata=chunk.metadata.copy(),
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    heading=chunk.heading,
                    subheading=chunk.subheading,
                    chunk_index=chunk.chunk_index,
                    chunk_size=chunk.chunk_size,
                    similarity_score=float(score)
                )
                results.append((chunk_copy, float(score)))
        
        return results
    
    def save_index(self) -> None:
        """Save FAISS index to file"""
        if config.faiss.save_index:
            try:
                faiss.write_index(self.index, self.index_file)
                logger.info(f"Saved FAISS index to {self.index_file}")
                
                # Save chunks metadata
                chunks_file = self.index_file.replace('.bin', '_chunks.pkl')
                with open(chunks_file, 'wb') as f:
                    pickle.dump(self.chunks, f)
                logger.info(f"Saved chunks metadata to {chunks_file}")
            except Exception as e:
                logger.warning(f"Could not save FAISS index: {e}")
    
    def load_index(self) -> bool:
        """Load FAISS index from file"""
        try:
            if os.path.exists(self.index_file):
                self.index = faiss.read_index(self.index_file)
                logger.info(f"Loaded FAISS index from {self.index_file}")
                
                # Load chunks metadata
                chunks_file = self.index_file.replace('.bin', '_chunks.pkl')
                if os.path.exists(chunks_file):
                    with open(chunks_file, 'rb') as f:
                        self.chunks = pickle.load(f)
                    logger.info(f"Loaded {len(self.chunks)} chunks metadata")
                
                return True
        except Exception as e:
            logger.warning(f"Could not load FAISS index: {e}")
        return False

class RetrievalSystem:
    """Enhanced retrieval system with ranking and filtering"""
    
    def __init__(self, embedding_manager: EmbeddingManager, vector_store: FAISSVectorStore):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.similarity_threshold = config.retrieval.similarity_threshold
        self.enable_reranking = config.retrieval.enable_reranking
    
    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve relevant documents for a query"""
        top_k = top_k or config.retrieval.top_k
        
        # Get query embedding
        query_embedding = self.embedding_manager.get_embeddings([query], ['query'])[0]
        
        # Search in vector store
        results = self.vector_store.search(query_embedding, top_k * 2)  # Get more for reranking
        
        # Filter by similarity threshold
        filtered_results = [(chunk, score) for chunk, score in results 
                          if score >= self.similarity_threshold]
        
        # Rerank if enabled
        if self.enable_reranking and len(filtered_results) > top_k:
            filtered_results = self._rerank_results(query, filtered_results)
        
        # Return top_k results
        return filtered_results[:top_k]
    
    def _rerank_results(self, query: str, results: List[Tuple[DocumentChunk, float]]) -> List[Tuple[DocumentChunk, float]]:
        """Rerank results using additional heuristics"""
        # Simple reranking based on content length and heading relevance
        query_lower = query.lower()
        
        def rerank_score(chunk_score_pair):
            chunk, score = chunk_score_pair
            
            # Base score from similarity
            final_score = score
            
            # Boost if query terms appear in heading
            if any(term in chunk.heading.lower() for term in query_lower.split()):
                final_score *= 1.2
            
            # Boost if it's main content vs subheader
            if chunk.metadata.get('source_type') == 'main_content':
                final_score *= 1.1
            
            # Penalize very short chunks
            if chunk.chunk_size < 50:
                final_score *= 0.9
            
            return final_score
        
        # Sort by reranked score
        reranked = sorted(results, key=rerank_score, reverse=True)
        return reranked

class AnswerGenerator:
    """Generates answers using LLM with enhanced prompts"""
    
    def __init__(self):
        # Configure Gemini API
        genai.configure(api_key=config.llm.api_key)
        
        # Initialize model with configuration
        generation_config = {
            "temperature": config.llm.temperature,
            "top_k": config.llm.top_k,
            "top_p": config.llm.top_p,
            "max_output_tokens": config.llm.max_output_tokens,
        }
        
        self.model = genai.GenerativeModel(
            model_name=config.llm.model_name,
            generation_config=generation_config
        )
        
        logger.info(f"Initialized Gemini model: {config.llm.model_name}")
    
    def generate_answer(self, query: str, retrieved_chunks: List[Tuple[DocumentChunk, float]], 
                       show_sources: bool = True) -> Dict[str, Any]:
        """Generate answer with source citations"""
        
        if not retrieved_chunks:
            return {
                'answer': "I couldn't find relevant information to answer your question. Please try rephrasing your query or ask about topics related to the available documents.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Prepare context from retrieved chunks
        context_parts = []
        sources = []
        
        for i, (chunk, score) in enumerate(retrieved_chunks):
            source_info = {
                'index': i + 1,
                'heading': chunk.heading,
                'subheading': chunk.subheading,
                'similarity_score': score,
                'chunk_id': chunk.chunk_id,
                'source_type': chunk.metadata.get('source_type', 'unknown')
            }
            sources.append(source_info)
            
            # Format context with source reference
            context_part = f"[Source {i+1}] {chunk.content}"
            if show_sources:
                context_part += f" (From: {chunk.heading}"
                if chunk.subheading:
                    context_part += f" - {chunk.subheading}"
                context_part += f", Relevance: {score:.3f})"
            
            context_parts.append(context_part)
        
        context = "\n\n".join(context_parts)
        
        # Create enhanced prompt
        prompt = self._create_enhanced_prompt(query, context, show_sources)
        
        # Generate response
        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            
            # Calculate confidence based on similarity scores
            avg_similarity = sum(score for _, score in retrieved_chunks) / len(retrieved_chunks)
            confidence = min(avg_similarity, 1.0)
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'total_sources': len(sources)
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': f"I encountered an error while generating the answer: {str(e)}",
                'sources': sources,
                'confidence': 0.0
            }
    
    def _create_enhanced_prompt(self, query: str, context: str, show_sources: bool) -> str:
        """Create an enhanced prompt for better answer generation"""
        
        source_instruction = ""
        if show_sources:
            source_instruction = """
When referencing information, use the source numbers in brackets [Source X] to indicate where the information comes from.
At the end of your answer, list the sources you referenced.
"""
        
        prompt = f"""You are an intelligent assistant specializing in providing accurate information based on provided context.

CONTEXT INFORMATION:
{context}

USER QUESTION:
{query}

INSTRUCTIONS:
1. Answer the question comprehensively based ONLY on the provided context information.
2. If the information is not available in the context, clearly state that you don't have enough information.
3. Be precise and factual - do not add information that isn't in the context.
4. If providing lists or structured data, format them clearly.
5. Support bilingual queries (English and Vietnamese) - respond in the same language as the question.
{source_instruction}

ANSWER:"""
        
        return prompt

class RAGPipeline:
    """Complete RAG Pipeline combining all components"""
    
    def __init__(self):
        # Initialize components
        self.embedding_manager = EmbeddingManager()
        self.vector_store = FAISSVectorStore(self.embedding_manager.embedding_dim)
        self.retrieval_system = RetrievalSystem(self.embedding_manager, self.vector_store)
        self.answer_generator = AnswerGenerator()
        
        # Try to load existing index
        self.is_indexed = self.vector_store.load_index()
        
        logger.info("RAG Pipeline initialized successfully")
    
    def index_documents(self, chunks: List[DocumentChunk], texts: List[str]) -> None:
        """Index documents in the vector store"""
        logger.info(f"Indexing {len(chunks)} document chunks...")
        
        # Get embeddings
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        embeddings = self.embedding_manager.get_embeddings(texts, chunk_ids)
        
        # Add to vector store
        self.vector_store.add_embeddings(embeddings, chunks)
        
        # Save index
        self.vector_store.save_index()
        self.is_indexed = True
        
        logger.info("Document indexing completed")
    
    def query(self, question: str, top_k: int = None, show_sources: bool = None) -> Dict[str, Any]:
        """Process a query through the complete RAG pipeline"""
        
        if not self.is_indexed:
            return {
                'answer': "The system is not ready. Please index documents first.",
                'sources': [],
                'confidence': 0.0,
                'error': 'not_indexed'
            }
        
        top_k = top_k or config.retrieval.top_k
        show_sources = show_sources if show_sources is not None else config.chat.show_sources
        
        try:
            # Retrieve relevant chunks
            retrieved_chunks = self.retrieval_system.retrieve(question, top_k)
            
            # Generate answer
            result = self.answer_generator.generate_answer(question, retrieved_chunks, show_sources)
            
            # Add query metadata
            result['query'] = question
            result['retrieval_time'] = None  # Could add timing here
            result['generation_time'] = None  # Could add timing here
            
            return result
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                'answer': f"An error occurred while processing your question: {str(e)}",
                'sources': [],
                'confidence': 0.0,
                'error': str(e)
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the RAG system status"""
        return {
            'is_indexed': self.is_indexed,
            'total_chunks': len(self.vector_store.chunks),
            'embedding_model': self.embedding_manager.model_name,
            'embedding_dimension': self.embedding_manager.embedding_dim,
            'vector_store_type': self.vector_store.index_type,
            'cache_enabled': self.embedding_manager.cache_embeddings,
            'cached_embeddings': len(self.embedding_manager.embedding_cache)
        }
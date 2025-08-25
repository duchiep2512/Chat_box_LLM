"""
RAG System for Presight Data Chatbot
"""

import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import json
import os
import streamlit as st
from config import *

class RAGSystem:
    """Main RAG System class for processing queries"""
    
    def __init__(self, indexed_list, embeddings, embedding_model, generation_model, content_for_embedding):
        self.indexed_list = indexed_list
        self.embeddings = embeddings
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.content_for_embedding = content_for_embedding
    
    def get_query_embedding(self, query):
        """Get embedding for user query"""
        return self.embedding_model.encode([query])
    
    def find_best_match(self, query, top_k=DEFAULT_TOP_K):
        """Find best matching sections for the query"""
        query_embedding = self.get_query_embedding(query)
        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
        
        # Get top-k indices sorted by similarity (descending)
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        
        matches = []
        for idx in top_k_indices:
            matches.append({
                'index': idx,
                'content': self.content_for_embedding[idx],
                'similarity': similarities[idx],
                'heading': self.indexed_list[idx]['heading'],
                'section_data': self.indexed_list[idx]
            })
        
        return matches
    
    def generate_answer_gemini(self, top_matches, question):
        """Generate answer using Google Gemini"""
        # Create detailed context
        context = "Relevant information from Presight Privacy Policy:\n\n"
        
        for i, match in enumerate(top_matches, 1):
            context += f"Section {i}: {match['heading']}\n"
            context += f"Content: {match['section_data']['content']}\n"
            
            # Add subheaders if available
            if match['section_data'].get('subheaders'):
                for subheader in match['section_data']['subheaders']:
                    context += f"  - {subheader.get('Title', '')}: {subheader.get('Content', '')}\n"
                    if subheader.get('List'):
                        context += f"    List: {', '.join(subheader['List'])}\n"
            
            context += f"\n(Relevance Score: {match['similarity']:.4f})\n\n"
        
        prompt = f"""
        Act as a professional assistant at company Presight in answering the question provided.
        Your job is to provide a clear and concise answer based only on the information provided in the context.
        Do not add any details or information beyond what is provided in the context.

        Context:
        {context}

        Question: {question}

        Requirements:
        1. Answer the question as detailed as possible from the provided context, make sure to provide all the details.
        2. If the answer is not in the context provided, just say "Your question is not in the company's database, please ask another question." without any further answer.
        3. Provide specific examples or lists when available in the context.
        4. Mention relevant section titles for reference.

        Answer:
        """
        
        try:
            response = self.generation_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def query(self, user_question, top_k=DEFAULT_TOP_K, show_details=True):
        """Main query function"""
        start_time = time.time()
        
        # Step 1: Find best matches
        top_matches = self.find_best_match(user_question, top_k=top_k)
        
        # Step 2: Generate answer
        answer = self.generate_answer_gemini(top_matches, user_question)
        
        end_time = time.time()
        
        return {
            'query': user_question,
            'top_matches': top_matches,
            'answer': answer,
            'execution_time': end_time - start_time
        }

def initialize_rag_system():
    """Initialize RAG system with all components"""
    try:
        with st.spinner("🔄 Loading RAG system components..."):
            # Configure Google Gemini API
            if GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
                st.error("❌ Google API Key not configured")
                return None
            
            genai.configure(api_key=GOOGLE_API_KEY)
            
            # Load indexed data
            try:
                with open(INDEXED_DATA_FILE, 'r', encoding='utf-8') as f:
                    indexed_list = json.load(f)
                st.success(f"✅ Loaded {len(indexed_list)} indexed sections")
            except FileNotFoundError:
                st.error(f"❌ {INDEXED_DATA_FILE} not found")
                return None
            except Exception as e:
                st.error(f"❌ Error loading {INDEXED_DATA_FILE}: {e}")
                return None
            
            # Initialize embedding model
            try:
                embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                st.success("✅ Embedding model loaded")
            except Exception as e:
                st.error(f"❌ Error loading embedding model: {e}")
                return None
            
            # Initialize Gemini model
            try:
                generation_model = genai.GenerativeModel(GENERATION_MODEL_NAME)
                st.success("✅ Gemini model initialized")
            except Exception as e:
                st.error(f"❌ Error initializing Gemini model: {e}")
                return None
            
            # Create content for embedding
            content_for_embedding = []
            for item in indexed_list:
                content = item['content']
                if item.get('subheaders'):
                    for subheader in item['subheaders']:
                        content += f" {subheader.get('Title', '')} {subheader.get('Content', '')}"
                        if subheader.get('List'):
                            content += " " + " ".join(subheader['List'])
                content_for_embedding.append(content)
            
            # Generate embeddings
            with st.spinner("🧠 Generating embeddings..."):
                embeddings = embedding_model.encode(content_for_embedding)
                st.success(f"✅ Generated embeddings for {len(embeddings)} sections")
            
            # Create RAG system
            rag_system = RAGSystem(
                indexed_list=indexed_list,
                embeddings=embeddings,
                embedding_model=embedding_model,
                generation_model=generation_model,
                content_for_embedding=content_for_embedding
            )
            
            st.success("🎉 RAG System initialized successfully!")
            return rag_system
            
    except Exception as e:
        st.error(f"❌ Error initializing RAG system: {e}")
        return None

def process_query_with_rag(rag_system, query, top_k=DEFAULT_TOP_K):
    """Process query using RAG system"""
    if rag_system is None:
        return {
            'query': query,
            'top_matches': [],
            'answer': "❌ RAG system is not available. Please check the system setup.",
            'execution_time': 0
        }
    
    try:
        result = rag_system.query(query, top_k=top_k, show_details=False)
        return result
    except Exception as e:
        return {
            'query': query,
            'top_matches': [],
            'answer': f"❌ Error processing query: {str(e)}",
            'execution_time': 0
        }
#!/usr/bin/env python3
"""
SEMANTIC SEARCH ENGINE - Phase 1 Implementation
AI-powered semantic search with BERT embeddings and intelligent query expansion
"""

import asyncio
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pickle
import os
from pathlib import Path
import logging

# Deep Learning and NLP imports
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    from transformers import pipeline
    import torch
    from textblob import TextBlob
    import nltk
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError as e:
    logging.error(f"Semantic search dependencies not available: {e}")
    SEMANTIC_SEARCH_AVAILABLE = False

from loguru import logger

class SemanticSearchEngine:
    """Advanced semantic search with BERT embeddings and AI-powered query expansion"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder = None
        self.index = None
        self.documents = []
        self.document_embeddings = None
        self.document_metadata = []
        
        # Query expansion and enhancement
        self.query_expander = None
        self.sentiment_analyzer = None
        
        # Cache settings
        self.cache_dir = Path("semantic_search_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Performance metrics
        self.search_stats = {
            'total_searches': 0,
            'avg_response_time': 0,
            'cache_hits': 0,
            'semantic_searches': 0,
            'traditional_searches': 0
        }
        
        if SEMANTIC_SEARCH_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models for semantic search"""
        try:
            logger.info("Initializing semantic search models...")
            
            # Load sentence transformer model
            self.encoder = SentenceTransformer(self.model_name)
            logger.info(f"Loaded sentence transformer: {self.model_name}")
            
            # Initialize query expansion model
            try:
                self.query_expander = pipeline(
                    "text-generation",
                    model="microsoft/DialoGPT-medium",
                    tokenizer="microsoft/DialoGPT-medium",
                    device=-1  # CPU
                )
            except:
                logger.warning("Query expansion model not available, using basic expansion")
            
            # Initialize sentiment analyzer
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=-1  # CPU
                )
            except:
                logger.warning("Advanced sentiment analyzer not available, using TextBlob")
            
            # Download required NLTK data
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
            except:
                pass
            
            logger.info("Semantic search models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic search models: {e}")
            self.encoder = None
    
    async def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Index documents for semantic search"""
        
        if not SEMANTIC_SEARCH_AVAILABLE or not self.encoder:
            logger.error("Semantic search not available")
            return False
        
        try:
            logger.info(f"Indexing {len(documents)} documents for semantic search...")
            
            # Extract text content and metadata
            texts = []
            metadata = []
            
            for doc in documents:
                # Extract searchable text content
                text_content = self._extract_searchable_content(doc)
                if text_content.strip():
                    texts.append(text_content)
                    metadata.append(doc)
            
            if not texts:
                logger.warning("No valid text content found for indexing")
                return False
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.encoder.encode(texts, show_progress_bar=True, convert_to_tensor=False)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.index.add(embeddings_normalized.astype('float32'))
            
            # Store documents and metadata
            self.documents = texts
            self.document_embeddings = embeddings_normalized
            self.document_metadata = metadata
            
            # Cache the index
            await self._save_index_to_cache()
            
            logger.info(f"Successfully indexed {len(texts)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            return False
    
    async def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        use_query_expansion: bool = True,
        include_sentiment: bool = True,
        min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Perform semantic search with AI enhancements"""
        
        start_time = datetime.now()
        
        if not SEMANTIC_SEARCH_AVAILABLE or not self.encoder or not self.index:
            logger.warning("Semantic search not available, falling back to basic search")
            return await self._fallback_search(query, top_k)
        
        try:
            # Update search statistics
            self.search_stats['total_searches'] += 1
            self.search_stats['semantic_searches'] += 1
            
            # Analyze query sentiment
            query_sentiment = None
            if include_sentiment:
                query_sentiment = await self._analyze_query_sentiment(query)
            
            # Expand query if requested
            expanded_queries = [query]
            if use_query_expansion:
                expanded_queries = await self._expand_query(query)
            
            # Perform search for all query variants
            all_results = []
            
            for search_query in expanded_queries:
                # Generate query embedding
                query_embedding = self.encoder.encode([search_query], convert_to_tensor=False)
                query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
                
                # Search in FAISS index
                similarities, indices = self.index.search(query_embedding.astype('float32'), top_k * 2)
                
                # Process results
                for similarity, idx in zip(similarities[0], indices[0]):
                    if similarity >= min_similarity and idx < len(self.document_metadata):
                        result = {
                            'document': self.document_metadata[idx],
                            'content': self.documents[idx],
                            'similarity_score': float(similarity),
                            'search_query': search_query,
                            'query_sentiment': query_sentiment,
                            'search_type': 'semantic',
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Add content analysis
                        result['content_analysis'] = await self._analyze_content(result['content'])
                        
                        all_results.append(result)
            
            # Deduplicate and rank results
            final_results = await self._rank_and_deduplicate_results(all_results, query, top_k)
            
            # Update performance metrics
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_search_metrics(response_time)
            
            logger.info(f"Semantic search completed: {len(final_results)} results in {response_time:.2f}s")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return await self._fallback_search(query, top_k)
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining semantic and keyword search"""
        
        try:
            # Get semantic results
            semantic_results = await self.semantic_search(query, top_k=top_k)
            
            # Get keyword results (basic implementation)
            keyword_results = await self._keyword_search(query, top_k=top_k)
            
            # Combine and rerank results
            combined_results = await self._combine_search_results(
                semantic_results, keyword_results, semantic_weight, keyword_weight
            )
            
            return combined_results[:top_k]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return await self._fallback_search(query, top_k)
    
    async def _expand_query(self, query: str) -> List[str]:
        """Expand query with AI-powered suggestions"""
        
        expanded_queries = [query]
        
        try:
            # Basic synonym expansion using TextBlob
            blob = TextBlob(query)
            words = blob.words
            
            # Add variations
            variations = []
            for word in words:
                # Add lemmatized form
                variations.append(str(word.lemmatize()))
                
                # Add synonyms (basic implementation)
                try:
                    synonyms = word.synsets
                    if synonyms:
                        for syn in synonyms[:2]:  # Limit to 2 synonyms per word
                            variations.append(str(syn.lemmas()[0].name().replace('_', ' ')))
                except:
                    pass
            
            # Create expanded queries
            if variations:
                expanded_query = query + " " + " ".join(set(variations))
                expanded_queries.append(expanded_query)
            
            # Add contextual variations
            context_variations = [
                f"research about {query}",
                f"analysis of {query}",
                f"information on {query}",
                f"{query} trends",
                f"{query} insights"
            ]
            
            expanded_queries.extend(context_variations[:2])  # Add 2 contextual variations
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
        
        return expanded_queries[:5]  # Limit to 5 total queries
    
    async def _analyze_query_sentiment(self, query: str) -> Dict[str, Any]:
        """Analyze query sentiment and intent"""
        
        try:
            if self.sentiment_analyzer:
                # Use advanced sentiment analyzer
                result = self.sentiment_analyzer(query)
                return {
                    'sentiment': result[0]['label'],
                    'confidence': result[0]['score'],
                    'analyzer': 'roberta'
                }
            else:
                # Use TextBlob as fallback
                blob = TextBlob(query)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                sentiment_label = 'NEUTRAL'
                if polarity > 0.1:
                    sentiment_label = 'POSITIVE'
                elif polarity < -0.1:
                    sentiment_label = 'NEGATIVE'
                
                return {
                    'sentiment': sentiment_label,
                    'polarity': polarity,
                    'subjectivity': subjectivity,
                    'analyzer': 'textblob'
                }
        
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return {'sentiment': 'NEUTRAL', 'confidence': 0.5}
    
    async def _analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze content for additional insights"""
        
        try:
            analysis = {}
            
            # Basic content statistics
            analysis['length'] = len(content)
            analysis['word_count'] = len(content.split())
            analysis['sentence_count'] = len(content.split('.'))
            
            # Extract key phrases (simplified)
            blob = TextBlob(content)
            analysis['key_phrases'] = [str(phrase) for phrase in blob.noun_phrases[:5]]
            
            # Content sentiment
            sentiment = blob.sentiment
            analysis['content_sentiment'] = {
                'polarity': sentiment.polarity,
                'subjectivity': sentiment.subjectivity
            }
            
            # Reading difficulty (simplified)
            avg_sentence_length = analysis['word_count'] / max(analysis['sentence_count'], 1)
            if avg_sentence_length < 15:
                analysis['readability'] = 'Easy'
            elif avg_sentence_length < 25:
                analysis['readability'] = 'Medium'
            else:
                analysis['readability'] = 'Hard'
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Content analysis failed: {e}")
            return {'error': str(e)}
    
    async def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Basic keyword search as fallback"""
        
        results = []
        query_words = query.lower().split()
        
        for i, (doc_text, metadata) in enumerate(zip(self.documents, self.document_metadata)):
            doc_lower = doc_text.lower()
            score = 0
            
            # Simple keyword matching
            for word in query_words:
                if word in doc_lower:
                    score += doc_lower.count(word)
            
            if score > 0:
                results.append({
                    'document': metadata,
                    'content': doc_text,
                    'similarity_score': score / len(query_words),  # Normalize
                    'search_query': query,
                    'search_type': 'keyword',
                    'timestamp': datetime.now().isoformat()
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]
    
    async def _combine_search_results(
        self,
        semantic_results: List[Dict],
        keyword_results: List[Dict],
        semantic_weight: float,
        keyword_weight: float
    ) -> List[Dict[str, Any]]:
        """Combine semantic and keyword search results"""
        
        combined = {}
        
        # Add semantic results
        for result in semantic_results:
            doc_id = id(result['document'])
            if doc_id not in combined:
                combined[doc_id] = result.copy()
                combined[doc_id]['combined_score'] = result['similarity_score'] * semantic_weight
                combined[doc_id]['search_methods'] = ['semantic']
            
        # Add keyword results
        for result in keyword_results:
            doc_id = id(result['document'])
            if doc_id in combined:
                combined[doc_id]['combined_score'] += result['similarity_score'] * keyword_weight
                combined[doc_id]['search_methods'].append('keyword')
            else:
                combined[doc_id] = result.copy()
                combined[doc_id]['combined_score'] = result['similarity_score'] * keyword_weight
                combined[doc_id]['search_methods'] = ['keyword']
        
        # Sort by combined score
        results = list(combined.values())
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return results
    
    async def _rank_and_deduplicate_results(
        self,
        results: List[Dict],
        original_query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Rank and deduplicate search results"""
        
        seen_docs = set()
        final_results = []
        
        # Sort by similarity score first
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        for result in results:
            doc_id = id(result['document'])
            
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                
                # Add relevance boosting factors
                boost_factor = 1.0
                
                # Boost if original query terms appear in content
                query_words = original_query.lower().split()
                content_lower = result['content'].lower()
                
                matching_words = sum(1 for word in query_words if word in content_lower)
                if matching_words > 0:
                    boost_factor += 0.1 * (matching_words / len(query_words))
                
                # Apply boost
                result['final_score'] = result['similarity_score'] * boost_factor
                result['boost_factor'] = boost_factor
                
                final_results.append(result)
                
                if len(final_results) >= top_k:
                    break
        
        # Final sort by boosted score
        final_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return final_results
    
    async def _fallback_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback search when semantic search is not available"""
        
        self.search_stats['traditional_searches'] += 1
        
        # Simple text matching fallback
        results = []
        query_lower = query.lower()
        
        for i, (doc_text, metadata) in enumerate(zip(self.documents, self.document_metadata)):
            if query_lower in doc_text.lower():
                results.append({
                    'document': metadata,
                    'content': doc_text,
                    'similarity_score': 0.5,  # Default score
                    'search_query': query,
                    'search_type': 'fallback',
                    'timestamp': datetime.now().isoformat()
                })
        
        return results[:top_k]
    
    def _extract_searchable_content(self, document: Dict[str, Any]) -> str:
        """Extract searchable text content from document"""
        
        searchable_fields = ['title', 'content', 'description', 'summary', 'text', 'body']
        content_parts = []
        
        for field in searchable_fields:
            if field in document and document[field]:
                content_parts.append(str(document[field]))
        
        return " ".join(content_parts)
    
    async def _save_index_to_cache(self):
        """Save search index to cache"""
        
        try:
            cache_file = self.cache_dir / f"semantic_index_{self.model_name.replace('/', '_')}.pkl"
            
            cache_data = {
                'index': self.index,
                'documents': self.documents,
                'document_metadata': self.document_metadata,
                'document_embeddings': self.document_embeddings,
                'model_name': self.model_name,
                'created_at': datetime.now().isoformat()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Search index cached to {cache_file}")
            
        except Exception as e:
            logger.warning(f"Failed to cache search index: {e}")
    
    async def load_index_from_cache(self) -> bool:
        """Load search index from cache"""
        
        try:
            cache_file = self.cache_dir / f"semantic_index_{self.model_name.replace('/', '_')}.pkl"
            
            if not cache_file.exists():
                return False
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if cache is recent (within 7 days)
            created_at = datetime.fromisoformat(cache_data['created_at'])
            if datetime.now() - created_at > timedelta(days=7):
                logger.info("Search index cache is too old, will rebuild")
                return False
            
            self.index = cache_data['index']
            self.documents = cache_data['documents']
            self.document_metadata = cache_data['document_metadata']
            self.document_embeddings = cache_data['document_embeddings']
            
            logger.info(f"Loaded search index from cache: {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load search index from cache: {e}")
            return False
    
    def _update_search_metrics(self, response_time: float):
        """Update search performance metrics"""
        
        # Update average response time
        total_searches = self.search_stats['total_searches']
        current_avg = self.search_stats['avg_response_time']
        
        new_avg = ((current_avg * (total_searches - 1)) + response_time) / total_searches
        self.search_stats['avg_response_time'] = new_avg
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search performance statistics"""
        
        stats = self.search_stats.copy()
        
        if stats['total_searches'] > 0:
            stats['semantic_search_ratio'] = stats['semantic_searches'] / stats['total_searches']
            stats['traditional_search_ratio'] = stats['traditional_searches'] / stats['total_searches']
            stats['cache_hit_ratio'] = stats['cache_hits'] / stats['total_searches']
        else:
            stats['semantic_search_ratio'] = 0
            stats['traditional_search_ratio'] = 0
            stats['cache_hit_ratio'] = 0
        
        stats['model_name'] = self.model_name
        stats['index_size'] = len(self.documents) if self.documents else 0
        stats['semantic_search_available'] = SEMANTIC_SEARCH_AVAILABLE and self.encoder is not None
        
        return stats

# Global semantic search engine instance
semantic_search_engine = None

def get_semantic_search_engine() -> SemanticSearchEngine:
    """Get or create global semantic search engine instance"""
    global semantic_search_engine
    
    if semantic_search_engine is None:
        semantic_search_engine = SemanticSearchEngine()
    
    return semantic_search_engine

async def initialize_semantic_search(documents: List[Dict[str, Any]] = None) -> bool:
    """Initialize semantic search with optional document indexing"""
    
    engine = get_semantic_search_engine()
    
    # Try to load from cache first
    if await engine.load_index_from_cache():
        logger.info("Semantic search initialized from cache")
        return True
    
    # Index documents if provided
    if documents:
        success = await engine.index_documents(documents)
        if success:
            logger.info("Semantic search initialized with new documents")
            return True
    
    logger.warning("Semantic search not initialized - no documents available")
    return False
#!/usr/bin/env python3
"""
SENTIMENT ANALYSIS SERVICE - Phase 1 Implementation
Advanced sentiment analysis and content intelligence for research content
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import re

# NLP and sentiment analysis imports
try:
    from transformers import pipeline
    from textblob import TextBlob
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    import torch
    SENTIMENT_ANALYSIS_AVAILABLE = True
except ImportError as e:
    logging.error(f"Sentiment analysis dependencies not available: {e}")
    SENTIMENT_ANALYSIS_AVAILABLE = False

from loguru import logger

class AdvancedSentimentAnalyzer:
    """Advanced sentiment analysis with multiple models and content intelligence"""
    
    def __init__(self):
        self.sentiment_models = {}
        self.emotion_analyzer = None
        self.text_classifier = None
        self.lemmatizer = None
        
        # Analysis cache
        self.analysis_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Performance metrics
        self.analysis_stats = {
            'total_analyses': 0,
            'cache_hits': 0,
            'avg_processing_time': 0,
            'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
            'emotion_distribution': {}
        }
        
        if SENTIMENT_ANALYSIS_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize sentiment analysis models"""
        try:
            logger.info("Initializing sentiment analysis models...")
            
            # Download required NLTK data
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('vader_lexicon', quiet=True)
                self.lemmatizer = WordNetLemmatizer()
            except Exception as e:
                logger.warning(f"Failed to download NLTK data: {e}")
            
            # Initialize sentiment analysis models
            try:
                # RoBERTa-based sentiment analyzer (high accuracy)
                self.sentiment_models['roberta'] = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=-1  # CPU
                )
                logger.info("Loaded RoBERTa sentiment model")
            except Exception as e:
                logger.warning(f"Failed to load RoBERTa model: {e}")
            
            try:
                # DistilBERT sentiment analyzer (faster)
                self.sentiment_models['distilbert'] = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1  # CPU
                )
                logger.info("Loaded DistilBERT sentiment model")
            except Exception as e:
                logger.warning(f"Failed to load DistilBERT model: {e}")
            
            try:
                # Emotion detection model
                self.emotion_analyzer = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    device=-1  # CPU
                )
                logger.info("Loaded emotion detection model")
            except Exception as e:
                logger.warning(f"Failed to load emotion model: {e}")
            
            try:
                # Text classification for content type
                self.text_classifier = pipeline(
                    "text-classification",
                    model="microsoft/DialoGPT-medium",
                    device=-1  # CPU
                )
                logger.info("Loaded text classification model")
            except Exception as e:
                logger.warning(f"Failed to load text classifier: {e}")
            
            logger.info("Sentiment analysis models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize sentiment models: {e}")
    
    async def analyze_content(
        self,
        content: str,
        content_type: str = "general",
        include_emotions: bool = True,
        include_keywords: bool = True,
        include_topics: bool = True
    ) -> Dict[str, Any]:
        """Comprehensive content analysis with sentiment, emotions, and insights"""
        
        if not SENTIMENT_ANALYSIS_AVAILABLE:
            return await self._fallback_analysis(content)
        
        start_time = datetime.now()
        
        # Check cache first
        cache_key = hash(content + content_type + str(include_emotions))
        if cache_key in self.analysis_cache:
            cached_result = self.analysis_cache[cache_key]
            cache_time = datetime.fromisoformat(cached_result['timestamp'])
            
            if (datetime.now() - cache_time).seconds < self.cache_ttl:
                self.analysis_stats['cache_hits'] += 1
                return cached_result['analysis']
        
        try:
            # Update statistics
            self.analysis_stats['total_analyses'] += 1
            
            analysis_result = {
                'content_type': content_type,
                'content_length': len(content),
                'word_count': len(content.split()),
                'timestamp': datetime.now().isoformat()
            }
            
            # Basic content preprocessing
            cleaned_content = self._preprocess_content(content)
            
            # Sentiment Analysis
            sentiment_results = await self._analyze_sentiment(cleaned_content)
            analysis_result['sentiment'] = sentiment_results
            
            # Emotion Analysis
            if include_emotions:
                emotion_results = await self._analyze_emotions(cleaned_content)
                analysis_result['emotions'] = emotion_results
            
            # Keyword Extraction
            if include_keywords:
                keywords = await self._extract_keywords(cleaned_content)
                analysis_result['keywords'] = keywords
            
            # Topic Detection
            if include_topics:
                topics = await self._detect_topics(cleaned_content)
                analysis_result['topics'] = topics
            
            # Content Quality Assessment
            quality_score = await self._assess_content_quality(cleaned_content)
            analysis_result['quality_score'] = quality_score
            
            # Market Sentiment (specific to financial/business content)
            if content_type in ['financial', 'business', 'market']:
                market_sentiment = await self._analyze_market_sentiment(cleaned_content)
                analysis_result['market_sentiment'] = market_sentiment
            
            # Readability Analysis
            readability = await self._analyze_readability(cleaned_content)
            analysis_result['readability'] = readability
            
            # Cache the result
            self.analysis_cache[cache_key] = {
                'analysis': analysis_result,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_analysis_metrics(processing_time, sentiment_results)
            
            logger.info(f"Content analysis completed in {processing_time:.2f}s")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return await self._fallback_analysis(content)
    
    async def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Multi-model sentiment analysis"""
        
        sentiment_results = {
            'models': {},
            'consensus': {},
            'confidence': 0.0
        }
        
        try:
            # RoBERTa sentiment analysis
            if 'roberta' in self.sentiment_models:
                roberta_result = self.sentiment_models['roberta'](content)
                sentiment_results['models']['roberta'] = {
                    'label': roberta_result[0]['label'],
                    'score': roberta_result[0]['score']
                }
            
            # DistilBERT sentiment analysis
            if 'distilbert' in self.sentiment_models:
                distilbert_result = self.sentiment_models['distilbert'](content)
                sentiment_results['models']['distilbert'] = {
                    'label': distilbert_result[0]['label'],
                    'score': distilbert_result[0]['score']
                }
            
            # TextBlob sentiment analysis (fallback)
            try:
                blob = TextBlob(content)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                textblob_label = 'NEUTRAL'
                if polarity > 0.1:
                    textblob_label = 'POSITIVE'
                elif polarity < -0.1:
                    textblob_label = 'NEGATIVE'
                
                sentiment_results['models']['textblob'] = {
                    'label': textblob_label,
                    'polarity': polarity,
                    'subjectivity': subjectivity,
                    'score': abs(polarity)
                }
            except Exception as e:
                logger.warning(f"TextBlob analysis failed: {e}")
            
            # Calculate consensus
            sentiment_results['consensus'] = self._calculate_sentiment_consensus(sentiment_results['models'])
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
        
        return sentiment_results
    
    async def _analyze_emotions(self, content: str) -> Dict[str, Any]:
        """Emotion detection and analysis"""
        
        if not self.emotion_analyzer:
            return {'error': 'Emotion analyzer not available'}
        
        try:
            # Get emotion predictions
            emotions = self.emotion_analyzer(content)
            
            emotion_results = {
                'primary_emotion': emotions[0]['label'],
                'confidence': emotions[0]['score'],
                'all_emotions': emotions,
                'emotional_intensity': self._calculate_emotional_intensity(emotions)
            }
            
            # Update emotion statistics
            primary_emotion = emotions[0]['label']
            if primary_emotion not in self.analysis_stats['emotion_distribution']:
                self.analysis_stats['emotion_distribution'][primary_emotion] = 0
            self.analysis_stats['emotion_distribution'][primary_emotion] += 1
            
            return emotion_results
            
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            return {'error': str(e)}
    
    async def _extract_keywords(self, content: str) -> Dict[str, Any]:
        """Extract important keywords and phrases"""
        
        try:
            # Use TextBlob for basic keyword extraction
            blob = TextBlob(content)
            
            # Extract noun phrases
            noun_phrases = list(blob.noun_phrases)
            
            # Extract important words (remove stopwords)
            try:
                stop_words = set(stopwords.words('english'))
                words = word_tokenize(content.lower())
                important_words = [word for word in words if word.isalpha() and word not in stop_words]
            except:
                important_words = [word for word in content.lower().split() if word.isalpha()]
            
            # Count word frequencies
            from collections import Counter
            word_freq = Counter(important_words)
            
            # Extract entities (simplified)
            entities = self._extract_entities(content)
            
            return {
                'noun_phrases': noun_phrases[:10],  # Top 10 noun phrases
                'top_words': dict(word_freq.most_common(15)),  # Top 15 words
                'entities': entities,
                'keyword_count': len(important_words),
                'unique_words': len(set(important_words))
            }
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return {'error': str(e)}
    
    async def _detect_topics(self, content: str) -> Dict[str, Any]:
        """Detect main topics and themes in content"""
        
        try:
            # Define topic keywords for different domains
            topic_keywords = {
                'technology': ['ai', 'machine learning', 'software', 'data', 'algorithm', 'computer', 'digital', 'tech'],
                'business': ['market', 'revenue', 'profit', 'company', 'business', 'investment', 'growth', 'strategy'],
                'finance': ['financial', 'money', 'bank', 'investment', 'stock', 'trading', 'economy', 'capital'],
                'science': ['research', 'study', 'experiment', 'analysis', 'scientific', 'discovery', 'hypothesis'],
                'health': ['health', 'medical', 'treatment', 'patient', 'clinical', 'healthcare', 'medicine'],
                'space': ['space', 'satellite', 'orbit', 'rocket', 'aerospace', 'mission', 'launch', 'astronomy'],
                'politics': ['government', 'policy', 'political', 'election', 'law', 'regulation', 'public'],
                'environment': ['environmental', 'climate', 'green', 'sustainability', 'energy', 'carbon', 'pollution']
            }
            
            content_lower = content.lower()
            topic_scores = {}
            
            for topic, keywords in topic_keywords.items():
                score = sum(content_lower.count(keyword) for keyword in keywords)
                if score > 0:
                    topic_scores[topic] = score / len(keywords)  # Normalize by keyword count
            
            # Sort topics by relevance
            sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'primary_topic': sorted_topics[0][0] if sorted_topics else 'general',
                'topic_scores': dict(sorted_topics[:5]),  # Top 5 topics
                'topic_confidence': sorted_topics[0][1] if sorted_topics else 0,
                'detected_topics': len(sorted_topics)
            }
            
        except Exception as e:
            logger.error(f"Topic detection failed: {e}")
            return {'primary_topic': 'general', 'error': str(e)}
    
    async def _assess_content_quality(self, content: str) -> Dict[str, Any]:
        """Assess overall content quality"""
        
        try:
            quality_factors = {}
            
            # Length quality (not too short, not too long)
            length = len(content)
            if 100 <= length <= 5000:
                quality_factors['length'] = 1.0
            elif length < 100:
                quality_factors['length'] = length / 100
            else:
                quality_factors['length'] = max(0.5, 5000 / length)
            
            # Sentence structure quality
            sentences = sent_tokenize(content)
            avg_sentence_length = len(content.split()) / len(sentences)
            
            if 10 <= avg_sentence_length <= 25:
                quality_factors['sentence_structure'] = 1.0
            else:
                quality_factors['sentence_structure'] = max(0.3, 1 - abs(avg_sentence_length - 17.5) / 17.5)
            
            # Vocabulary diversity
            words = content.lower().split()
            unique_words = len(set(words))
            diversity_ratio = unique_words / len(words) if words else 0
            quality_factors['vocabulary_diversity'] = min(1.0, diversity_ratio * 2)
            
            # Grammar quality (simplified)
            blob = TextBlob(content)
            try:
                # Simple grammar check based on sentence structure
                grammar_score = 1.0 - (content.count('...') + content.count('???')) / len(sentences)
                quality_factors['grammar'] = max(0.5, grammar_score)
            except:
                quality_factors['grammar'] = 0.8  # Default
            
            # Overall quality score (weighted average)
            weights = {'length': 0.2, 'sentence_structure': 0.3, 'vocabulary_diversity': 0.3, 'grammar': 0.2}
            overall_score = sum(quality_factors[factor] * weights[factor] for factor in quality_factors)
            
            return {
                'overall_score': overall_score,
                'factors': quality_factors,
                'quality_rating': self._get_quality_rating(overall_score),
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_sentence_length': avg_sentence_length,
                'vocabulary_diversity': diversity_ratio
            }
            
        except Exception as e:
            logger.error(f"Content quality assessment failed: {e}")
            return {'overall_score': 0.5, 'error': str(e)}
    
    async def _analyze_market_sentiment(self, content: str) -> Dict[str, Any]:
        """Specialized market sentiment analysis for financial content"""
        
        try:
            # Financial sentiment indicators
            bullish_indicators = ['growth', 'profit', 'increase', 'rise', 'bullish', 'positive', 'gain', 'boost', 'strong']
            bearish_indicators = ['decline', 'loss', 'decrease', 'fall', 'bearish', 'negative', 'drop', 'weak', 'risk']
            
            content_lower = content.lower()
            
            bullish_score = sum(content_lower.count(indicator) for indicator in bullish_indicators)
            bearish_score = sum(content_lower.count(indicator) for indicator in bearish_indicators)
            
            total_indicators = bullish_score + bearish_score
            
            if total_indicators == 0:
                market_sentiment = 'neutral'
                confidence = 0.5
            else:
                if bullish_score > bearish_score:
                    market_sentiment = 'bullish'
                    confidence = bullish_score / total_indicators
                else:
                    market_sentiment = 'bearish'
                    confidence = bearish_score / total_indicators
            
            return {
                'sentiment': market_sentiment,
                'confidence': confidence,
                'bullish_indicators': bullish_score,
                'bearish_indicators': bearish_score,
                'market_strength': abs(bullish_score - bearish_score) / max(total_indicators, 1)
            }
            
        except Exception as e:
            logger.error(f"Market sentiment analysis failed: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.5, 'error': str(e)}
    
    async def _analyze_readability(self, content: str) -> Dict[str, Any]:
        """Analyze content readability"""
        
        try:
            sentences = sent_tokenize(content)
            words = content.split()
            
            # Basic readability metrics
            avg_sentence_length = len(words) / len(sentences) if sentences else 0
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            
            # Simplified readability score
            if avg_sentence_length < 15 and avg_word_length < 5:
                readability_level = 'Easy'
                score = 0.8
            elif avg_sentence_length < 25 and avg_word_length < 6:
                readability_level = 'Medium'
                score = 0.6
            else:
                readability_level = 'Hard'
                score = 0.4
            
            return {
                'level': readability_level,
                'score': score,
                'avg_sentence_length': avg_sentence_length,
                'avg_word_length': avg_word_length,
                'total_sentences': len(sentences),
                'total_words': len(words)
            }
            
        except Exception as e:
            logger.error(f"Readability analysis failed: {e}")
            return {'level': 'Medium', 'score': 0.6, 'error': str(e)}
    
    def _preprocess_content(self, content: str) -> str:
        """Preprocess content for analysis"""
        
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', ' ', content)
        
        # Remove extra whitespace
        content = re.sub(r'/s+', ' ', content)
        
        # Remove URLs
        content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*/(/),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)
        
        return content.strip()
    
    def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Simple entity extraction"""
        
        entities = []
        
        # Extract capitalized words (potential entities)
        words = content.split()
        capitalized_words = [word for word in words if word[0].isupper() and len(word) > 2]
        
        from collections import Counter
        entity_counts = Counter(capitalized_words)
        
        for entity, count in entity_counts.most_common(10):
            entities.append({
                'text': entity,
                'count': count,
                'type': 'UNKNOWN'  # Would need NER model for proper classification
            })
        
        return entities
    
    def _calculate_sentiment_consensus(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate consensus sentiment from multiple models"""
        
        if not model_results:
            return {'label': 'NEUTRAL', 'confidence': 0.5}
        
        # Collect all sentiments and scores
        sentiments = []
        scores = []
        
        for model, result in model_results.items():
            if 'label' in result and 'score' in result:
                sentiments.append(result['label'])
                scores.append(result['score'])
        
        if not sentiments:
            return {'label': 'NEUTRAL', 'confidence': 0.5}
        
        # Find most common sentiment
        from collections import Counter
        sentiment_counts = Counter(sentiments)
        consensus_sentiment = sentiment_counts.most_common(1)[0][0]
        
        # Calculate average confidence
        consensus_confidence = sum(scores) / len(scores)
        
        return {
            'label': consensus_sentiment,
            'confidence': consensus_confidence,
            'model_agreement': sentiment_counts[consensus_sentiment] / len(sentiments)
        }
    
    def _calculate_emotional_intensity(self, emotions: List[Dict]) -> float:
        """Calculate overall emotional intensity"""
        
        if not emotions:
            return 0.0
        
        # Calculate weighted average of emotion scores
        total_score = sum(emotion['score'] for emotion in emotions)
        return total_score / len(emotions)
    
    def _get_quality_rating(self, score: float) -> str:
        """Convert quality score to rating"""
        
        if score >= 0.8:
            return 'Excellent'
        elif score >= 0.6:
            return 'Good'
        elif score >= 0.4:
            return 'Fair'
        else:
            return 'Poor'
    
    def _update_analysis_metrics(self, processing_time: float, sentiment_results: Dict):
        """Update analysis performance metrics"""
        
        # Update average processing time
        total_analyses = self.analysis_stats['total_analyses']
        current_avg = self.analysis_stats['avg_processing_time']
        
        new_avg = ((current_avg * (total_analyses - 1)) + processing_time) / total_analyses
        self.analysis_stats['avg_processing_time'] = new_avg
        
        # Update sentiment distribution
        if 'consensus' in sentiment_results and 'label' in sentiment_results['consensus']:
            sentiment = sentiment_results['consensus']['label'].lower()
            if sentiment in self.analysis_stats['sentiment_distribution']:
                self.analysis_stats['sentiment_distribution'][sentiment] += 1
    
    async def _fallback_analysis(self, content: str) -> Dict[str, Any]:
        """Fallback analysis when advanced models are not available"""
        
        try:
            # Basic TextBlob analysis
            blob = TextBlob(content)
            sentiment = blob.sentiment
            
            sentiment_label = 'NEUTRAL'
            if sentiment.polarity > 0.1:
                sentiment_label = 'POSITIVE'
            elif sentiment.polarity < -0.1:
                sentiment_label = 'NEGATIVE'
            
            return {
                'sentiment': {
                    'models': {
                        'textblob': {
                            'label': sentiment_label,
                            'polarity': sentiment.polarity,
                            'subjectivity': sentiment.subjectivity
                        }
                    },
                    'consensus': {
                        'label': sentiment_label,
                        'confidence': abs(sentiment.polarity)
                    }
                },
                'content_length': len(content),
                'word_count': len(content.split()),
                'timestamp': datetime.now().isoformat(),
                'analysis_method': 'fallback'
            }
            
        except Exception as e:
            return {
                'error': f'Analysis failed: {str(e)}',
                'content_length': len(content),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get analysis performance statistics"""
        
        stats = self.analysis_stats.copy()
        stats.update({
            'sentiment_analysis_available': SENTIMENT_ANALYSIS_AVAILABLE,
            'models_loaded': len(self.sentiment_models),
            'emotion_analyzer_available': self.emotion_analyzer is not None,
            'cache_size': len(self.analysis_cache),
            'last_updated': datetime.now().isoformat()
        })
        
        return stats

# Global sentiment analyzer instance
sentiment_analyzer = None

def get_sentiment_analyzer() -> AdvancedSentimentAnalyzer:
    """Get or create global sentiment analyzer instance"""
    global sentiment_analyzer
    
    if sentiment_analyzer is None:
        sentiment_analyzer = AdvancedSentimentAnalyzer()
    
    return sentiment_analyzer
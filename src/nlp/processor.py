"""
NLP Processing Pipeline for Research Synthesis Engine
"""
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import openai
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import re
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor

from research_synthesis.config.settings import settings


class NLPProcessor:
    """Main NLP processing class"""
    
    def __init__(self):
        """Initialize NLP models and pipelines"""
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize models lazily
        self._spacy_model = None
        self._sentence_transformer = None
        self._summarizer = None
        self._ner_pipeline = None
        self._sentiment_pipeline = None
        self._language_detector = None
        
        # Language monitoring
        self.language_stats = {}
        self.non_english_threshold = 0.3  # Alert when >30% non-English content
        self.total_articles_processed = 0
        
        # Set OpenAI API key
        if settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY
    
    @property
    def spacy_model(self):
        """Lazy load spaCy model"""
        if self._spacy_model is None:
            try:
                self._spacy_model = spacy.load("en_core_web_sm")
            except:
                logger.warning("spaCy model not found, downloading...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self._spacy_model = spacy.load("en_core_web_sm")
        return self._spacy_model
    
    @property
    def sentence_transformer(self):
        """Lazy load sentence transformer"""
        if self._sentence_transformer is None:
            self._sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        return self._sentence_transformer
    
    @property
    def summarizer(self):
        """Lazy load summarization pipeline"""
        if self._summarizer is None:
            self._summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        return self._summarizer
    
    @property
    def ner_pipeline(self):
        """Lazy load NER pipeline"""
        if self._ner_pipeline is None:
            self._ner_pipeline = pipeline("ner", aggregation_strategy="simple")
        return self._ner_pipeline
    
    @property
    def sentiment_pipeline(self):
        """Lazy load sentiment analysis pipeline"""
        if self._sentiment_pipeline is None:
            self._sentiment_pipeline = pipeline("sentiment-analysis")
        return self._sentiment_pipeline
    
    @property
    def language_detector(self):
        """Lazy load language detection pipeline"""
        if self._language_detector is None:
            try:
                from transformers import pipeline
                self._language_detector = pipeline("text-classification", 
                                                 model="papluca/xlm-roberta-base-language-detection")
            except:
                logger.warning("Language detection model not available, using simple fallback")
                self._language_detector = "fallback"
        return self._language_detector
    
    async def process_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single article through the NLP pipeline"""
        try:
            # Run processing tasks in parallel
            tasks = [
                self.extract_entities_async(article['content']),
                self.generate_embeddings_async(article['title'], article['content']),
                self.analyze_sentiment_async(article['content']),
                self.extract_facts_async(article['content']),
                self.generate_summary_async(article['content']),
                self.detect_language_async(article.get('title', '') + ' ' + article.get('content', ''))
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            entities, embeddings, sentiment, facts, summary, language_info = results
            
            # Handle any errors
            if isinstance(entities, Exception):
                logger.error(f"Entity extraction failed: {entities}")
                entities = {'entities': [], 'space_entities': []}
            
            if isinstance(embeddings, Exception):
                logger.error(f"Embedding generation failed: {embeddings}")
                embeddings = {'title_embedding': None, 'content_embedding': None}
            
            if isinstance(sentiment, Exception):
                logger.error(f"Sentiment analysis failed: {sentiment}")
                sentiment = {'sentiment': 'neutral', 'score': 0.5}
            
            if isinstance(facts, Exception):
                logger.error(f"Fact extraction failed: {facts}")
                facts = {'facts': [], 'dates': [], 'numbers': []}
            
            if isinstance(summary, Exception):
                logger.error(f"Summary generation failed: {summary}")
                summary = article.get('summary', '')
            
            if isinstance(language_info, Exception):
                logger.error(f"Language detection failed: {language_info}")
                language_info = {'language': 'unknown', 'confidence': 0.0, 'is_english': True}
            
            # Update language monitoring
            self._update_language_stats(language_info)
            
            # Update article with NLP results
            processed_article = {
                **article,
                'entities': entities['entities'],
                'space_entities': entities['space_entities'],
                'title_embedding': embeddings['title_embedding'],
                'content_embedding': embeddings['content_embedding'],
                'sentiment': sentiment['sentiment'],
                'sentiment_score': sentiment['score'],
                'extracted_facts': facts['facts'],
                'extracted_dates': facts['dates'],
                'extracted_numbers': facts['numbers'],
                'ai_summary': summary,
                'detected_language': language_info['language'],
                'language_confidence': language_info['confidence'],
                'is_english': language_info['is_english'],
                'processed_at': datetime.utcnow().isoformat()
            }
            
            return processed_article
            
        except Exception as e:
            logger.error(f"Article processing failed: {e}")
            return article
    
    async def extract_entities_async(self, text: str) -> Dict[str, List]:
        """Extract named entities asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.extract_entities, text)
    
    def extract_entities(self, text: str) -> Dict[str, List]:
        """Extract named entities from text"""
        if not text:
            return {'entities': [], 'space_entities': []}
        
        # Use spaCy for entity extraction
        doc = self.spacy_model(text[:1000000])  # Limit text length
        
        entities = []
        space_entities = []
        
        # Space-related keywords for filtering
        space_keywords = [
            'space', 'orbit', 'satellite', 'rocket', 'launch', 'nasa', 'esa',
            'spacex', 'blue origin', 'virgin', 'asteroid', 'mars', 'moon',
            'iss', 'station', 'spacecraft', 'payload', 'mission'
        ]
        
        for ent in doc.ents:
            entity = {
                'text': ent.text,
                'type': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            }
            entities.append(entity)
            
            # Check if entity is space-related
            if any(keyword in ent.text.lower() for keyword in space_keywords):
                space_entities.append(entity)
        
        # Also use transformer-based NER for better accuracy
        try:
            transformer_entities = self.ner_pipeline(text[:512])  # Limit for performance
            for ent in transformer_entities:
                entity = {
                    'text': ent['word'],
                    'type': ent['entity_group'],
                    'score': ent['score']
                }
                if entity not in entities:
                    entities.append(entity)
        except:
            pass
        
        return {
            'entities': entities,
            'space_entities': space_entities
        }
    
    async def generate_embeddings_async(self, title: str, content: str) -> Dict[str, List[float]]:
        """Generate embeddings asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.generate_embeddings, title, content)
    
    def generate_embeddings(self, title: str, content: str) -> Dict[str, List[float]]:
        """Generate embeddings for title and content"""
        embeddings = {}
        
        try:
            # Generate title embedding
            if title:
                title_embedding = self.sentence_transformer.encode(title)
                embeddings['title_embedding'] = title_embedding.tolist()
            else:
                embeddings['title_embedding'] = None
            
            # Generate content embedding (use first 512 tokens for efficiency)
            if content:
                content_preview = content[:2000]  # Roughly 512 tokens
                content_embedding = self.sentence_transformer.encode(content_preview)
                embeddings['content_embedding'] = content_embedding.tolist()
            else:
                embeddings['content_embedding'] = None
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            embeddings = {'title_embedding': None, 'content_embedding': None}
        
        return embeddings
    
    async def analyze_sentiment_async(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.analyze_sentiment, text)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        if not text:
            return {'sentiment': 'neutral', 'score': 0.5}
        
        try:
            # Use first 512 tokens for sentiment analysis
            text_preview = text[:2000]
            result = self.sentiment_pipeline(text_preview)[0]
            
            # Convert to standard format
            sentiment = result['label'].lower()
            if sentiment == 'positive':
                score = result['score']
            elif sentiment == 'negative':
                score = 1 - result['score']
            else:
                score = 0.5
            
            return {'sentiment': sentiment, 'score': score}
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {'sentiment': 'neutral', 'score': 0.5}
    
    async def extract_facts_async(self, text: str) -> Dict[str, List]:
        """Extract facts asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.extract_facts, text)
    
    def extract_facts(self, text: str) -> Dict[str, List]:
        """Extract facts, dates, and numbers from text"""
        if not text:
            return {'facts': [], 'dates': [], 'numbers': []}
        
        facts = []
        dates = []
        numbers = []
        
        # Extract dates
        date_patterns = [
            r'/b/d{1,2}[/-]/d{1,2}[/-]/d{2,4}/b',
            r'/b/d{4}[/-]/d{1,2}[/-]/d{1,2}/b',
            r'/b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* /d{1,2},? /d{4}/b',
            r'/b/d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* /d{4}/b'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        # Extract numbers with context
        number_pattern = r'(/$?/d+(?:,/d{3})*(?:/./d+)?(?:/s*(?:million|billion|trillion|thousand|K|M|B))?)'
        number_matches = re.findall(number_pattern, text, re.IGNORECASE)
        
        for match in number_matches:
            # Get context around the number
            context_pattern = rf'(.{{0,50}}{re.escape(match)}.{{0,50}})'
            context_match = re.search(context_pattern, text)
            if context_match:
                numbers.append({
                    'value': match,
                    'context': context_match.group(1).strip()
                })
        
        # Extract key facts using pattern matching
        fact_patterns = [
            r'announced .*',
            r'launched .*',
            r'signed .*agreement',
            r'invested /$.*',
            r'acquired .*',
            r'partnership with .*',
            r'developed .*',
            r'tested .*',
            r'achieved .*',
            r'discovered .*'
        ]
        
        doc = self.spacy_model(text[:10000])  # Limit for performance
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            for pattern in fact_patterns:
                if re.search(pattern, sent_text, re.IGNORECASE):
                    facts.append(sent_text)
                    break
        
        return {
            'facts': facts[:20],  # Limit number of facts
            'dates': list(set(dates))[:10],
            'numbers': numbers[:15]
        }
    
    async def generate_summary_async(self, text: str) -> str:
        """Generate summary asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.generate_summary, text)
    
    def generate_summary(self, text: str, use_ai: bool = True) -> str:
        """Generate article summary"""
        if not text:
            return ""
        
        # Limit text length
        max_length = 1024
        if len(text) > max_length:
            text = text[:max_length]
        
        try:
            if use_ai and settings.OPENAI_API_KEY:
                # Use OpenAI for better summaries
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Summarize the following article in 2-3 sentences."},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=100,
                    temperature=0.5
                )
                return response.choices[0].message.content
            else:
                # Use transformer-based summarizer
                summary = self.summarizer(text, max_length=100, min_length=30, do_sample=False)
                return summary[0]['summary_text']
                
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            # Fallback to simple extraction
            sentences = text.split('.')[:3]
            return '. '.join(sentences) + '.'
    
    async def classify_content(self, text: str) -> Dict[str, Any]:
        """Classify content into categories"""
        categories = {
            'technology': ['technology', 'software', 'hardware', 'ai', 'machine learning'],
            'business': ['business', 'market', 'investment', 'funding', 'acquisition'],
            'science': ['research', 'study', 'experiment', 'discovery', 'scientific'],
            'manufacturing': ['manufacturing', 'production', 'factory', 'assembly', '3d printing'],
            'launch': ['launch', 'mission', 'rocket', 'payload', 'orbit'],
            'satellite': ['satellite', 'constellation', 'communications', 'earth observation'],
            'exploration': ['mars', 'moon', 'asteroid', 'exploration', 'planetary']
        }
        
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[category] = score
        
        # Get top categories
        sorted_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_categories = [cat for cat, score in sorted_categories[:3] if score > 0]
        
        return {
            'categories': top_categories,
            'scores': scores
        }
    
    async def detect_language_async(self, text: str) -> Dict[str, Any]:
        """Detect language asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.detect_language, text)
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect language of text"""
        if not text or len(text.strip()) < 10:
            return {'language': 'unknown', 'confidence': 0.0, 'is_english': True}
        
        try:
            # Use advanced language detector if available
            if self.language_detector != "fallback":
                # Use first 512 characters for detection
                text_sample = text[:512].strip()
                result = self.language_detector(text_sample)[0]
                
                language = result['label'].lower()
                confidence = result['score']
                is_english = language in ['en', 'english']
                
                return {
                    'language': language,
                    'confidence': confidence,
                    'is_english': is_english
                }
            else:
                # Use simple fallback detection
                return self._simple_language_detect(text)
                
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return self._simple_language_detect(text)
    
    def _simple_language_detect(self, text: str) -> Dict[str, Any]:
        """Simple language detection fallback"""
        # Count common English words vs non-English indicators
        common_english_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'must', 'shall', 'this', 'that', 'these', 'those', 'a', 'an'
        }
        
        # Japanese indicators
        japanese_chars = any(ord(char) > 0x3040 for char in text[:200])
        
        # Chinese indicators  
        chinese_chars = any(0x4e00 <= ord(char) <= 0x9fff for char in text[:200])
        
        # Count English words
        words = text.lower().split()[:50]  # Check first 50 words
        english_word_count = sum(1 for word in words if word.strip('.,!?";:') in common_english_words)
        
        if japanese_chars:
            return {'language': 'ja', 'confidence': 0.8, 'is_english': False}
        elif chinese_chars:
            return {'language': 'zh', 'confidence': 0.8, 'is_english': False}
        elif len(words) > 0 and english_word_count / len(words) > 0.3:
            return {'language': 'en', 'confidence': 0.7, 'is_english': True}
        else:
            return {'language': 'unknown', 'confidence': 0.5, 'is_english': False}
    
    def _update_language_stats(self, language_info: Dict[str, Any]):
        """Update language statistics and check for alerts"""
        self.total_articles_processed += 1
        language = language_info.get('language', 'unknown')
        is_english = language_info.get('is_english', True)
        
        # Update language counts
        if language not in self.language_stats:
            self.language_stats[language] = 0
        self.language_stats[language] += 1
        
        # Check if non-English content threshold is exceeded
        non_english_count = sum(count for lang, count in self.language_stats.items() 
                               if lang not in ['en', 'english', 'unknown'])
        
        if self.total_articles_processed > 10:  # Only alert after processing at least 10 articles
            non_english_ratio = non_english_count / self.total_articles_processed
            
            if non_english_ratio > self.non_english_threshold:
                logger.warning(
                    f"High non-English content detected: {non_english_ratio:.2%} "
                    f"({non_english_count}/{self.total_articles_processed} articles). "
                    f"Languages found: {dict(self.language_stats)}"
                )
    
    def get_language_stats(self) -> Dict[str, Any]:
        """Get current language statistics"""
        non_english_count = sum(count for lang, count in self.language_stats.items() 
                               if lang not in ['en', 'english', 'unknown'])
        
        return {
            'total_articles_processed': self.total_articles_processed,
            'language_distribution': dict(self.language_stats),
            'non_english_count': non_english_count,
            'non_english_ratio': non_english_count / max(self.total_articles_processed, 1),
            'threshold_exceeded': (non_english_count / max(self.total_articles_processed, 1)) > self.non_english_threshold,
            'threshold': self.non_english_threshold
        }
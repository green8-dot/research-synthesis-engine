#!/usr/bin/env python3
"""
ADVANCED NLP PROCESSOR - Phase 2 Implementation
Industry-specific Named Entity Recognition and Intent Analysis
"""

import asyncio
import json
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

# NLP and ML imports
try:
    import spacy
    from spacy import displacy
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    import torch
    from textblob import TextBlob
    import networkx as nx
    ADVANCED_NLP_AVAILABLE = True
except ImportError as e:
    logging.error(f"Advanced NLP dependencies not available: {e}")
    ADVANCED_NLP_AVAILABLE = False

from loguru import logger

class ContentType(Enum):
    """Content type classification"""
    NEWS = "news"
    RESEARCH = "research" 
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    SPACE = "space"
    GENERAL = "general"

class IntentType(Enum):
    """Intent classification for content"""
    INFORMATIONAL = "informational"
    ANALYTICAL = "analytical" 
    PREDICTIVE = "predictive"
    ACTIONABLE = "actionable"
    URGENT = "urgent"
    MONITORING = "monitoring"

@dataclass
class EntityExtraction:
    """Enhanced entity extraction result"""
    text: str
    label: str
    confidence: float
    start: int
    end: int
    context: str
    industry_specific: bool = False
    semantic_type: Optional[str] = None

@dataclass
class IntentAnalysis:
    """Content intent analysis result"""
    primary_intent: IntentType
    confidence: float
    urgency_score: float  # 0-1 scale
    actionability_score: float  # 0-1 scale
    insights: List[str]

@dataclass
class AdvancedNLPResult:
    """Comprehensive NLP analysis result"""
    entities: List[EntityExtraction]
    intent: IntentAnalysis
    content_type: ContentType
    semantic_summary: str
    key_concepts: List[str]
    sentiment: Dict[str, float]
    readability_score: float
    technical_complexity: float
    processing_time: float

class AdvancedNLPProcessor:
    """Advanced NLP processing with industry-specific capabilities"""
    
    def __init__(self):
        self.nlp_model = None
        self.nlp_lg_model = None  # Large model for better accuracy
        self.ner_model = None
        self.intent_classifier = None
        self.finance_ner = None
        self.xgb_classifier = None  # XGBoost for classification
        self.lgbm_classifier = None  # LightGBM for classification
        
        # Industry-specific entity patterns
        self.space_entities = {
            'SPACECRAFT': r'/b(?:Dragon|Falcon|Starship|ISS|International Space Station|Crew Dragon|Starlink)/b',
            'SPACE_AGENCY': r'/b(?:NASA|SpaceX|ESA|Roscosmos|CNSA|ISRO|Blue Origin)/b',
            'SPACE_MISSION': r'/b(?:Artemis|Apollo|Mars 2020|Europa Clipper|James Webb)/b',
            'CELESTIAL_BODY': r'/b(?:Mars|Moon|Jupiter|Saturn|Europa|Titan|asteroid|comet)/b'
        }
        
        self.finance_entities = {
            'CRYPTO': r'/b(?:Bitcoin|BTC|Ethereum|ETH|cryptocurrency|crypto|blockchain|DeFi)/b',
            'STOCK_SYMBOL': r'/b[A-Z]{1,5}(?:/.[A-Z]{1,2})?/b',
            'FINANCIAL_METRIC': r'/b(?:GDP|inflation|yield|dividend|P/E ratio|market cap)/b',
            'FINANCIAL_INST': r'/b(?:Federal Reserve|Fed|ECB|Bank of America|Goldman Sachs|JPMorgan)/b'
        }
        
        self.tech_entities = {
            'AI_ML': r'/b(?:artificial intelligence|AI|machine learning|ML|deep learning|neural network|transformer|GPT|LLM)/b',
            'TECH_COMPANY': r'/b(?:Google|Microsoft|Apple|Amazon|Meta|Tesla|NVIDIA|OpenAI|Anthropic)/b',
            'PROGRAMMING': r'/b(?:Python|JavaScript|TypeScript|React|FastAPI|Docker|Kubernetes|AWS|Azure|GCP)/b'
        }
        
        if ADVANCED_NLP_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models and pipelines"""
        try:
            logger.info("Initializing advanced NLP models...")
            
            # Load spaCy model for general NLP
            self.nlp_model = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy English model")
            
            # Initialize transformer-based NER for better accuracy
            try:
                self.ner_model = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple",
                    device=-1  # CPU
                )
                logger.info("Loaded BERT-based NER model")
            except:
                logger.warning("BERT NER model not available, using spaCy NER")
            
            # Initialize intent classification
            try:
                self.intent_classifier = pipeline(
                    "text-classification",
                    model="microsoft/DialoGPT-medium",
                    device=-1
                )
            except:
                logger.warning("Intent classifier not available, using rule-based approach")
            
            # Initialize financial NER
            try:
                self.finance_ner = pipeline(
                    "ner",
                    model="ProsusAI/finbert",
                    aggregation_strategy="simple", 
                    device=-1
                )
                logger.info("Loaded FinBERT for financial NER")
            except:
                logger.warning("FinBERT not available, using pattern matching")
                
        except Exception as e:
            logger.error(f"Error initializing NLP models: {e}")
    
    async def process_content(self, text: str, content_hint: Optional[str] = None) -> AdvancedNLPResult:
        """
        Comprehensive NLP processing of content
        
        Args:
            text: Input text to process
            content_hint: Optional hint about content domain (space, finance, tech)
            
        Returns:
            AdvancedNLPResult with comprehensive analysis
        """
        if not ADVANCED_NLP_AVAILABLE:
            return self._fallback_processing(text)
        
        start_time = datetime.now()
        
        try:
            # 1. Content type classification
            content_type = await self._classify_content_type(text, content_hint)
            
            # 2. Enhanced entity extraction
            entities = await self._extract_entities(text, content_type)
            
            # 3. Intent analysis
            intent = await self._analyze_intent(text, content_type)
            
            # 4. Semantic analysis
            semantic_summary = await self._generate_semantic_summary(text)
            key_concepts = await self._extract_key_concepts(text)
            
            # 5. Sentiment and complexity analysis
            sentiment = await self._analyze_sentiment(text)
            readability = self._calculate_readability(text)
            complexity = self._calculate_technical_complexity(text, content_type)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AdvancedNLPResult(
                entities=entities,
                intent=intent,
                content_type=content_type,
                semantic_summary=semantic_summary,
                key_concepts=key_concepts,
                sentiment=sentiment,
                readability_score=readability,
                technical_complexity=complexity,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in advanced NLP processing: {e}")
            return self._fallback_processing(text)
    
    async def _classify_content_type(self, text: str, hint: Optional[str] = None) -> ContentType:
        """Classify content type based on patterns and context"""
        text_lower = text.lower()
        
        # Use hint if provided
        if hint:
            if hint.lower() in ['space', 'aerospace', 'satellite']:
                return ContentType.SPACE
            elif hint.lower() in ['finance', 'trading', 'investment']:
                return ContentType.FINANCIAL
            elif hint.lower() in ['tech', 'technology', 'software']:
                return ContentType.TECHNICAL
        
        # Pattern-based classification
        space_keywords = ['space', 'rocket', 'satellite', 'nasa', 'spacex', 'mars', 'moon', 'orbit', 'mission']
        finance_keywords = ['stock', 'market', 'trading', 'investment', 'economy', 'financial', 'revenue', 'profit']
        tech_keywords = ['technology', 'software', 'ai', 'algorithm', 'data', 'computing', 'digital', 'platform']
        research_keywords = ['study', 'research', 'analysis', 'findings', 'methodology', 'experiment', 'paper']
        
        space_score = sum(1 for kw in space_keywords if kw in text_lower)
        finance_score = sum(1 for kw in finance_keywords if kw in text_lower)
        tech_score = sum(1 for kw in tech_keywords if kw in text_lower)
        research_score = sum(1 for kw in research_keywords if kw in text_lower)
        
        scores = {
            ContentType.SPACE: space_score,
            ContentType.FINANCIAL: finance_score,
            ContentType.TECHNICAL: tech_score,
            ContentType.RESEARCH: research_score
        }
        
        max_score = max(scores.values())
        if max_score >= 2:  # Threshold for classification
            return max(scores, key=scores.get)
        
        return ContentType.GENERAL
    
    async def _extract_entities(self, text: str, content_type: ContentType) -> List[EntityExtraction]:
        """Enhanced entity extraction with industry-specific recognition"""
        entities = []
        
        # 1. Standard NER using spaCy
        if self.nlp_model:
            doc = self.nlp_model(text)
            for ent in doc.ents:
                entities.append(EntityExtraction(
                    text=ent.text,
                    label=ent.label_,
                    confidence=0.8,  # spaCy doesn't provide confidence scores
                    start=ent.start_char,
                    end=ent.end_char,
                    context=text[max(0, ent.start_char-50):ent.end_char+50],
                    industry_specific=False
                ))
        
        # 2. Enhanced NER using BERT (if available)
        if self.ner_model:
            try:
                bert_entities = self.ner_model(text)
                for ent in bert_entities:
                    entities.append(EntityExtraction(
                        text=ent['word'],
                        label=ent['entity_group'],
                        confidence=ent['score'],
                        start=ent['start'],
                        end=ent['end'],
                        context=text[max(0, ent['start']-50):ent['end']+50],
                        industry_specific=False
                    ))
            except Exception as e:
                logger.warning(f"BERT NER failed: {e}")
        
        # 3. Industry-specific entity extraction
        industry_entities = await self._extract_industry_entities(text, content_type)
        entities.extend(industry_entities)
        
        # 4. Financial entities (if financial content)
        if content_type == ContentType.FINANCIAL and self.finance_ner:
            try:
                fin_entities = self.finance_ner(text)
                for ent in fin_entities:
                    entities.append(EntityExtraction(
                        text=ent['word'],
                        label=f"FIN_{ent['entity_group']}",
                        confidence=ent['score'],
                        start=ent['start'],
                        end=ent['end'],
                        context=text[max(0, ent['start']-50):ent['end']+50],
                        industry_specific=True,
                        semantic_type="financial"
                    ))
            except Exception as e:
                logger.warning(f"Financial NER failed: {e}")
        
        # Remove duplicates and sort by confidence
        entities = self._deduplicate_entities(entities)
        return sorted(entities, key=lambda x: x.confidence, reverse=True)
    
    async def _extract_industry_entities(self, text: str, content_type: ContentType) -> List[EntityExtraction]:
        """Extract industry-specific entities using pattern matching"""
        entities = []
        
        patterns = {}
        semantic_type = None
        
        if content_type == ContentType.SPACE:
            patterns = self.space_entities
            semantic_type = "space"
        elif content_type == ContentType.FINANCIAL:
            patterns = self.finance_entities
            semantic_type = "financial"
        elif content_type == ContentType.TECHNICAL:
            patterns = self.tech_entities
            semantic_type = "technology"
        
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(EntityExtraction(
                    text=match.group(),
                    label=entity_type,
                    confidence=0.9,  # High confidence for pattern matching
                    start=match.start(),
                    end=match.end(),
                    context=text[max(0, match.start()-50):match.end()+50],
                    industry_specific=True,
                    semantic_type=semantic_type
                ))
        
        return entities
    
    def _deduplicate_entities(self, entities: List[EntityExtraction]) -> List[EntityExtraction]:
        """Remove overlapping and duplicate entities"""
        # Sort by start position
        entities.sort(key=lambda x: x.start)
        
        deduplicated = []
        for entity in entities:
            # Check for overlap with existing entities
            overlap = False
            for existing in deduplicated:
                if (entity.start < existing.end and entity.end > existing.start):
                    # Keep the one with higher confidence
                    if entity.confidence > existing.confidence:
                        deduplicated.remove(existing)
                    else:
                        overlap = True
                        break
            
            if not overlap:
                deduplicated.append(entity)
        
        return deduplicated
    
    async def _analyze_intent(self, text: str, content_type: ContentType) -> IntentAnalysis:
        """Analyze content intent and urgency"""
        text_lower = text.lower()
        
        # Rule-based intent classification
        intent_patterns = {
            IntentType.URGENT: ['breaking', 'urgent', 'alert', 'immediate', 'critical', 'emergency'],
            IntentType.PREDICTIVE: ['predict', 'forecast', 'expect', 'will', 'future', 'outlook', 'projection'],
            IntentType.ANALYTICAL: ['analysis', 'study', 'examine', 'investigate', 'research', 'findings'],
            IntentType.ACTIONABLE: ['should', 'recommend', 'suggest', 'propose', 'action', 'decision'],
            IntentType.MONITORING: ['track', 'monitor', 'watch', 'observe', 'update', 'status', 'progress']
        }
        
        intent_scores = {}
        for intent_type, patterns in intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            intent_scores[intent_type] = score
        
        # Default to informational if no clear intent
        primary_intent = max(intent_scores, key=intent_scores.get) if max(intent_scores.values()) > 0 else IntentType.INFORMATIONAL
        confidence = min(intent_scores[primary_intent] / 10.0, 1.0)  # Normalize to 0-1
        
        # Calculate urgency score
        urgency_keywords = ['breaking', 'urgent', 'immediate', 'now', 'alert', 'critical']
        urgency_score = min(sum(1 for kw in urgency_keywords if kw in text_lower) / 5.0, 1.0)
        
        # Calculate actionability score  
        action_keywords = ['should', 'must', 'recommend', 'buy', 'sell', 'invest', 'avoid', 'consider']
        actionability_score = min(sum(1 for kw in action_keywords if kw in text_lower) / 5.0, 1.0)
        
        # Generate insights
        insights = []
        if urgency_score > 0.6:
            insights.append("High urgency content requiring immediate attention")
        if actionability_score > 0.5:
            insights.append("Contains actionable recommendations")
        if primary_intent == IntentType.PREDICTIVE:
            insights.append("Forward-looking content with predictions")
        
        return IntentAnalysis(
            primary_intent=primary_intent,
            confidence=confidence,
            urgency_score=urgency_score,
            actionability_score=actionability_score,
            insights=insights
        )
    
    async def _generate_semantic_summary(self, text: str) -> str:
        """Generate semantic summary of content"""
        # Simple extractive summarization
        sentences = text.split('. ')
        if len(sentences) <= 3:
            return text
        
        # Score sentences based on key information
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            score = 0
            # Prefer sentences with entities, numbers, and key concepts
            if re.search(r'/d+', sentence):
                score += 1
            if any(word in sentence.lower() for word in ['new', 'first', 'major', 'significant', 'important']):
                score += 1
            if len(sentence.split()) > 10:  # Prefer informative sentences
                score += 1
                
            sentence_scores[i] = score
        
        # Select top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        top_indices = sorted([idx for idx, _ in top_sentences])
        
        return '. '.join([sentences[i] for i in top_indices]) + '.'
    
    async def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts using NLP"""
        if not self.nlp_model:
            # Fallback to simple keyword extraction
            words = text.lower().split()
            # Filter out common words and short words
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
            keywords = [word for word in words if len(word) > 4 and word not in stopwords]
            return list(set(keywords))[:10]
        
        doc = self.nlp_model(text)
        
        # Extract noun phrases and important terms
        concepts = set()
        
        # Add noun chunks
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 3:
                concepts.add(chunk.text.lower())
        
        # Add entities
        for ent in doc.ents:
            concepts.add(ent.text.lower())
        
        # Add important adjectives and nouns
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and len(token.text) > 4:
                concepts.add(token.lemma_.lower())
        
        return list(concepts)[:15]
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Enhanced sentiment analysis"""
        blob = TextBlob(text)
        
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'compound_score': blob.sentiment.polarity * (1 - blob.sentiment.subjectivity)
        }
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (Flesch Reading Ease approximation)"""
        sentences = len(text.split('.'))
        words = len(text.split())
        syllables = sum([self._count_syllables(word) for word in text.split()])
        
        if sentences == 0 or words == 0:
            return 0.5
        
        # Simplified Flesch Reading Ease
        score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        
        # Normalize to 0-1 scale
        return max(0.0, min(1.0, score / 100.0))
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count for a word"""
        word = word.lower()
        syllables = 0
        vowels = 'aeiouy'
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllables += 1
            previous_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e'):
            syllables -= 1
        
        return max(1, syllables)
    
    def _calculate_technical_complexity(self, text: str, content_type: ContentType) -> float:
        """Calculate technical complexity score"""
        technical_indicators = 0
        total_words = len(text.split())
        
        if total_words == 0:
            return 0.0
        
        # Count technical terms
        technical_patterns = [
            r'/b/w+(?:tion|sion|ment|ance|ence)/b',  # Technical suffixes
            r'/b[A-Z]{2,}/b',  # Acronyms
            r'/b/d+/.?/d*%?/b',  # Numbers and percentages
            r'/b/w+(?:ly|al|ic|ous)/b',  # Technical adjectives
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, text)
            technical_indicators += len(matches)
        
        # Domain-specific complexity
        domain_multiplier = 1.0
        if content_type in [ContentType.TECHNICAL, ContentType.RESEARCH]:
            domain_multiplier = 1.5
        elif content_type == ContentType.FINANCIAL:
            domain_multiplier = 1.3
        
        complexity = (technical_indicators / total_words) * domain_multiplier
        return min(1.0, complexity)
    
    def _fallback_processing(self, text: str) -> AdvancedNLPResult:
        """Fallback processing when advanced models aren't available"""
        blob = TextBlob(text)
        
        # Basic entity extraction using simple patterns
        entities = []
        # Extract potential entities (capitalized words)
        words = text.split()
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 3:
                entities.append(EntityExtraction(
                    text=word,
                    label="UNKNOWN",
                    confidence=0.5,
                    start=text.find(word),
                    end=text.find(word) + len(word),
                    context="",
                    industry_specific=False
                ))
        
        # Basic intent
        intent = IntentAnalysis(
            primary_intent=IntentType.INFORMATIONAL,
            confidence=0.5,
            urgency_score=0.0,
            actionability_score=0.0,
            insights=["Basic analysis - advanced models not available"]
        )
        
        return AdvancedNLPResult(
            entities=entities[:10],  # Limit to 10
            intent=intent,
            content_type=ContentType.GENERAL,
            semantic_summary=text[:200] + "..." if len(text) > 200 else text,
            key_concepts=[],
            sentiment={
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'compound_score': blob.sentiment.polarity * (1 - blob.sentiment.subjectivity)
            },
            readability_score=0.5,
            technical_complexity=0.5,
            processing_time=0.1
        )

# Global instance
advanced_nlp_processor = AdvancedNLPProcessor()

def get_advanced_nlp_processor() -> AdvancedNLPProcessor:
    """Get the global advanced NLP processor instance"""
    return advanced_nlp_processor
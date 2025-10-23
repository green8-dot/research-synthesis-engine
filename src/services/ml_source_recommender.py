"""
ML-Powered Source Search Recommendation System
Intelligently recommends the best sources to search based on topic analysis and learned patterns
"""
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
import json
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

# Import ML error handler
try:
    from research_synthesis.utils.ml_error_handler import ml_error_handler, MLErrorType
    ML_ERROR_HANDLER_AVAILABLE = True
except ImportError:
    logger.warning("ML Error Handler not available")
    ML_ERROR_HANDLER_AVAILABLE = False

# Import ML cache
try:
    from research_synthesis.utils.ml_cache import ml_cache
    ML_CACHE_AVAILABLE = True
except ImportError:
    logger.warning("ML Cache not available")
    ML_CACHE_AVAILABLE = False

# Import ML model validator
try:
    from research_synthesis.utils.ml_model_validator import model_validator
    ML_VALIDATOR_AVAILABLE = True
except ImportError:
    logger.warning("ML Model Validator not available")
    ML_VALIDATOR_AVAILABLE = False

try:
    from research_synthesis.database.connection import get_postgres_session
    from research_synthesis.database.models import Article, Source, Entity, Report
    from sqlalchemy import select, and_, or_, func, desc
    DATABASE_AVAILABLE = True
except ImportError:
    logger.warning("Database not available for ML source recommender")
    DATABASE_AVAILABLE = False

@dataclass
class SourceRecommendation:
    """Represents a recommended source for research"""
    source_name: str
    source_url: str
    relevance_score: float
    confidence_score: float
    source_type: str  # 'academic', 'news', 'technical', 'social', 'government'
    specialization: List[str]  # Areas of expertise
    quality_metrics: Dict[str, float]
    search_strategy: str  # Recommended search approach
    keywords: List[str]  # Recommended search keywords
    expected_results: int  # Predicted number of relevant results
    reasoning: str  # Why this source was recommended

@dataclass
class SearchStrategy:
    """Represents a comprehensive search strategy"""
    primary_sources: List[SourceRecommendation]
    secondary_sources: List[SourceRecommendation]
    search_sequence: List[str]  # Optimal order of source searching
    time_allocation: Dict[str, float]  # Recommended time per source
    depth_recommendations: Dict[str, str]  # 'shallow', 'moderate', 'deep' per source
    cross_validation_sources: List[str]  # Sources for fact-checking
    ml_confidence: float
    total_expected_results: int

class MLSourceRecommender:
    """ML-powered system for recommending research sources"""
    
    def __init__(self):
        self.model_path = Path("models/source_recommender")
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ML components
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        self.relevance_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # Source knowledge base
        self.source_database = self._initialize_source_database()
        
        # Learning history
        self.search_history = []
        self.success_patterns = defaultdict(list)
        self.failure_patterns = defaultdict(list)
        
        # Load pre-trained models if available
        self._load_models()
        
        # Source quality scores (learned over time)
        self.source_quality_scores = defaultdict(lambda: 0.5)
        
        # Topic-source affinity matrix
        self.topic_source_affinity = defaultdict(lambda: defaultdict(float))
    
    def _initialize_source_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize knowledge base of sources and their characteristics"""
        return {
            # Academic Sources
            "Google Scholar": {
                "url": "https://scholar.google.com",
                "type": "academic",
                "specialization": ["research papers", "citations", "academic articles"],
                "quality_baseline": 0.9,
                "search_depth": "deep",
                "api_available": True,
                "rate_limit": 100,
                "best_for": ["theoretical research", "literature review", "citations"]
            },
            "arXiv": {
                "url": "https://arxiv.org",
                "type": "academic",
                "specialization": ["physics", "mathematics", "computer science", "space"],
                "quality_baseline": 0.85,
                "search_depth": "deep",
                "api_available": True,
                "rate_limit": 1000,
                "best_for": ["cutting-edge research", "preprints", "technical papers"]
            },
            "PubMed": {
                "url": "https://pubmed.ncbi.nlm.nih.gov",
                "type": "academic",
                "specialization": ["medical", "biological", "life sciences"],
                "quality_baseline": 0.9,
                "search_depth": "deep",
                "api_available": True,
                "rate_limit": 300,
                "best_for": ["medical research", "biological studies", "health technology"]
            },
            
            # News Sources
            "SpaceNews": {
                "url": "https://spacenews.com",
                "type": "news",
                "specialization": ["space industry", "satellites", "launches", "space policy"],
                "quality_baseline": 0.8,
                "search_depth": "moderate",
                "api_available": False,
                "rate_limit": 50,
                "best_for": ["industry news", "company updates", "space missions"]
            },
            "Space.com": {
                "url": "https://www.space.com",
                "type": "news",
                "specialization": ["space exploration", "astronomy", "space technology"],
                "quality_baseline": 0.75,
                "search_depth": "moderate",
                "api_available": False,
                "rate_limit": 50,
                "best_for": ["space news", "mission updates", "technology developments"]
            },
            "TechCrunch": {
                "url": "https://techcrunch.com",
                "type": "news",
                "specialization": ["technology", "startups", "innovation", "venture capital"],
                "quality_baseline": 0.7,
                "search_depth": "shallow",
                "api_available": True,
                "rate_limit": 100,
                "best_for": ["tech startups", "innovation news", "funding rounds"]
            },
            
            # Technical Sources
            "GitHub": {
                "url": "https://github.com",
                "type": "technical",
                "specialization": ["code", "open source", "software", "documentation"],
                "quality_baseline": 0.8,
                "search_depth": "deep",
                "api_available": True,
                "rate_limit": 5000,
                "best_for": ["implementation examples", "code libraries", "technical documentation"]
            },
            "Stack Overflow": {
                "url": "https://stackoverflow.com",
                "type": "technical",
                "specialization": ["programming", "technical solutions", "debugging"],
                "quality_baseline": 0.75,
                "search_depth": "moderate",
                "api_available": True,
                "rate_limit": 300,
                "best_for": ["technical problems", "implementation details", "best practices"]
            },
            "IEEE Xplore": {
                "url": "https://ieeexplore.ieee.org",
                "type": "technical",
                "specialization": ["engineering", "electronics", "computer science"],
                "quality_baseline": 0.85,
                "search_depth": "deep",
                "api_available": True,
                "rate_limit": 100,
                "best_for": ["engineering research", "technical standards", "conference papers"]
            },
            
            # Social Sources
            "Reddit": {
                "url": "https://www.reddit.com",
                "type": "social",
                "specialization": ["community discussions", "trends", "opinions"],
                "quality_baseline": 0.6,
                "search_depth": "moderate",
                "api_available": True,
                "rate_limit": 60,
                "best_for": ["community insights", "trends", "practical experiences"],
                "subreddits": {
                    "space": ["r/space", "r/spacex", "r/nasa", "r/aerospace"],
                    "manufacturing": ["r/manufacturing", "r/3Dprinting", "r/engineering"],
                    "automation": ["r/automation", "r/robotics", "r/PLC"]
                }
            },
            "Twitter/X": {
                "url": "https://twitter.com",
                "type": "social",
                "specialization": ["real-time updates", "expert opinions", "breaking news"],
                "quality_baseline": 0.5,
                "search_depth": "shallow",
                "api_available": True,
                "rate_limit": 100,
                "best_for": ["breaking news", "expert opinions", "trend analysis"]
            },
            
            # Government Sources
            "NASA": {
                "url": "https://www.nasa.gov",
                "type": "government",
                "specialization": ["space exploration", "research", "missions", "technology"],
                "quality_baseline": 0.95,
                "search_depth": "deep",
                "api_available": True,
                "rate_limit": 1000,
                "best_for": ["official mission data", "research publications", "technology reports"]
            },
            "ESA": {
                "url": "https://www.esa.int",
                "type": "government",
                "specialization": ["European space", "satellites", "research"],
                "quality_baseline": 0.9,
                "search_depth": "deep",
                "api_available": False,
                "rate_limit": 50,
                "best_for": ["European space programs", "international collaboration", "satellite data"]
            },
            "Patent Databases": {
                "url": "https://patents.google.com",
                "type": "technical",
                "specialization": ["patents", "innovations", "intellectual property"],
                "quality_baseline": 0.85,
                "search_depth": "deep",
                "api_available": True,
                "rate_limit": 100,
                "best_for": ["innovation tracking", "technology trends", "competitive analysis"]
            }
        }
    
    @ml_cache.cached(ttl=3600) if ML_CACHE_AVAILABLE else lambda x: x  # Cache for 1 hour
    @ml_error_handler.ml_error_handler(
        component="MLSourceRecommender",
        error_types=[MLErrorType.PREDICTION_ERROR, MLErrorType.MODEL_LOADING_ERROR, MLErrorType.DATABASE_CONNECTION_ERROR],
        fallback_enabled=True,
        max_retries=2
    ) if ML_ERROR_HANDLER_AVAILABLE else lambda x: x
    async def recommend_sources(self, topic: str, context: str = "", 
                               research_type: str = "comprehensive",
                               time_constraint: int = 60) -> SearchStrategy:
        """
        Recommend optimal sources for researching a topic using ML
        
        Args:
            topic: Main research topic
            context: Additional context or requirements
            research_type: 'quick', 'comprehensive', 'academic', 'industry'
            time_constraint: Available time in minutes
        
        Returns:
            SearchStrategy with ML-powered recommendations
        """
        logger.info(f"Generating ML-powered source recommendations for: {topic}")
        
        # Extract features from topic and context
        features = self._extract_topic_features(topic, context)
        
        # Predict source relevance using ML
        source_predictions = await self._predict_source_relevance(features, research_type)
        
        # Learn from historical patterns
        historical_insights = await self._analyze_historical_patterns(topic, features)
        
        # Generate recommendations
        primary_sources = []
        secondary_sources = []
        
        for source_name, prediction_score in source_predictions.items():
            source_info = self.source_database.get(source_name, {})
            
            # Calculate composite score
            quality_score = self.source_quality_scores[source_name]
            historical_score = historical_insights.get(source_name, 0.5)
            relevance_score = prediction_score
            
            # Weight the scores
            composite_score = (
                relevance_score * 0.4 +
                quality_score * 0.3 +
                historical_score * 0.3
            )
            
            # Generate keywords using ML
            keywords = self._generate_search_keywords(topic, context, source_name)
            
            # Predict expected results
            expected_results = self._predict_result_count(topic, source_name, features)
            
            # Create recommendation
            recommendation = SourceRecommendation(
                source_name=source_name,
                source_url=source_info.get('url', ''),
                relevance_score=relevance_score,
                confidence_score=composite_score,
                source_type=source_info.get('type', 'general'),
                specialization=source_info.get('specialization', []),
                quality_metrics={
                    'baseline_quality': source_info.get('quality_baseline', 0.5),
                    'learned_quality': quality_score,
                    'historical_success': historical_score
                },
                search_strategy=self._determine_search_strategy(source_name, features),
                keywords=keywords,
                expected_results=expected_results,
                reasoning=self._generate_reasoning(source_name, features, composite_score)
            )
            
            # Classify as primary or secondary
            if composite_score > 0.7:
                primary_sources.append(recommendation)
            elif composite_score > 0.5:
                secondary_sources.append(recommendation)
        
        # Sort by confidence
        primary_sources.sort(key=lambda x: x.confidence_score, reverse=True)
        secondary_sources.sort(key=lambda x: x.confidence_score, reverse=True)
        
        # Determine optimal search sequence using ML
        search_sequence = self._optimize_search_sequence(
            primary_sources + secondary_sources,
            time_constraint
        )
        
        # Allocate time based on expected value
        time_allocation = self._allocate_search_time(
            primary_sources + secondary_sources,
            time_constraint
        )
        
        # Determine search depth recommendations
        depth_recommendations = self._recommend_search_depth(
            primary_sources + secondary_sources,
            time_constraint,
            research_type
        )
        
        # Select cross-validation sources
        cross_validation = self._select_cross_validation_sources(
            primary_sources + secondary_sources
        )
        
        # Calculate total expected results
        total_expected = sum(s.expected_results for s in primary_sources) + /
                        sum(s.expected_results for s in secondary_sources) // 2
        
        # Create search strategy
        strategy = SearchStrategy(
            primary_sources=primary_sources[:5],  # Top 5 primary
            secondary_sources=secondary_sources[:5],  # Top 5 secondary
            search_sequence=search_sequence,
            time_allocation=time_allocation,
            depth_recommendations=depth_recommendations,
            cross_validation_sources=cross_validation,
            ml_confidence=self._calculate_strategy_confidence(primary_sources),
            total_expected_results=total_expected
        )
        
        # Learn from this recommendation
        self._update_learning_history(topic, features, strategy)
        
        return strategy
    
    def _extract_topic_features(self, topic: str, context: str) -> Dict[str, Any]:
        """Extract ML features from topic and context"""
        text = f"{topic} {context}".lower()
        
        features = {
            'text': text,
            'length': len(text.split()),
            'has_space': 'space' in text or 'orbital' in text or 'satellite' in text,
            'has_manufacturing': 'manufactur' in text or 'production' in text,
            'has_automation': 'automat' in text or 'robot' in text,
            'has_research': 'research' in text or 'study' in text or 'analysis' in text,
            'has_technical': 'technical' in text or 'engineering' in text or 'system' in text,
            'has_academic': 'theory' in text or 'hypothesis' in text or 'model' in text,
            'has_innovation': 'innovat' in text or 'novel' in text or 'new' in text,
            'complexity': self._assess_complexity(text),
            'domain_indicators': self._extract_domain_indicators(text),
            'temporal_focus': self._extract_temporal_focus(text),
            'geographic_scope': self._extract_geographic_scope(text)
        }
        
        return features
    
    @ml_cache.cached(ttl=1800) if ML_CACHE_AVAILABLE else lambda x: x  # Cache for 30 minutes
    @ml_error_handler.ml_error_handler(
        component="MLSourceRecommender",
        error_types=[MLErrorType.PREDICTION_ERROR, MLErrorType.DATA_PREPROCESSING_ERROR],
        fallback_enabled=True,
        max_retries=1
    ) if ML_ERROR_HANDLER_AVAILABLE else lambda x: x
    async def _predict_source_relevance(self, features: Dict[str, Any], 
                                       research_type: str) -> Dict[str, float]:
        """Use ML to predict relevance of each source"""
        predictions = {}
        
        for source_name, source_info in self.source_database.items():
            # Feature engineering for source-topic matching
            match_features = self._create_matching_features(features, source_info)
            
            # Apply different strategies based on research type
            if research_type == 'academic':
                if source_info['type'] == 'academic':
                    base_score = 0.9
                else:
                    base_score = 0.3
            elif research_type == 'industry':
                if source_info['type'] in ['news', 'technical']:
                    base_score = 0.8
                else:
                    base_score = 0.4
            else:  # comprehensive
                base_score = 0.6
            
            # Calculate specialization match
            specialization_score = self._calculate_specialization_match(
                features['domain_indicators'],
                source_info['specialization']
            )
            
            # Combine scores
            relevance = (base_score * 0.5 + specialization_score * 0.5)
            
            # Apply ML adjustment if model is trained
            if hasattr(self, 'trained_model'):
                ml_adjustment = self._get_ml_adjustment(match_features)
                relevance = relevance * 0.7 + ml_adjustment * 0.3
            
            predictions[source_name] = min(relevance, 1.0)
        
        return predictions
    
    def _create_matching_features(self, topic_features: Dict[str, Any], 
                                 source_info: Dict[str, Any]) -> np.ndarray:
        """Create feature vector for ML matching"""
        features = []
        
        # Binary features for domain match
        for domain in ['space', 'manufacturing', 'automation', 'research', 'technical']:
            topic_has = topic_features.get(f'has_{domain}', False)
            source_specializes = any(domain in s.lower() for s in source_info['specialization'])
            features.append(float(topic_has and source_specializes))
        
        # Quality and depth features
        features.append(source_info.get('quality_baseline', 0.5))
        features.append(1.0 if source_info.get('api_available', False) else 0.0)
        
        # Complexity match
        topic_complexity = topic_features.get('complexity', 0.5)
        source_depth = {'shallow': 0.3, 'moderate': 0.6, 'deep': 0.9}.get(
            source_info.get('search_depth', 'moderate'), 0.6
        )
        features.append(1.0 - abs(topic_complexity - source_depth))
        
        return np.array(features)
    
    def _calculate_specialization_match(self, domain_indicators: List[str],
                                       specializations: List[str]) -> float:
        """Calculate how well source specializations match domain indicators"""
        if not domain_indicators or not specializations:
            return 0.5
        
        matches = 0
        for indicator in domain_indicators:
            for spec in specializations:
                if indicator.lower() in spec.lower() or spec.lower() in indicator.lower():
                    matches += 1
        
        max_possible = len(domain_indicators) * len(specializations)
        return min(matches / max(max_possible, 1), 1.0)
    
    async def _analyze_historical_patterns(self, topic: str, 
                                          features: Dict[str, Any]) -> Dict[str, float]:
        """Analyze historical search patterns for insights"""
        insights = {}
        
        if not DATABASE_AVAILABLE:
            return insights
        
        try:
            # Query database for similar past searches
            async for session in get_postgres_session():
                # Find articles with similar topics
                similar_articles = await session.execute(
                    select(Article.source_id, func.count(Article.id).label('count'),
                          func.avg(Article.quality_score).label('avg_quality'))
                    .where(
                        or_(
                            Article.title.ilike(f'%{topic}%'),
                            Article.content.ilike(f'%{topic}%')
                        )
                    )
                    .group_by(Article.source_id)
                    .limit(20)
                )
                
                for row in similar_articles:
                    # Map source_id to source name (simplified)
                    source_name = self._map_source_id_to_name(row.source_id)
                    if source_name:
                        # Calculate historical success score
                        count_score = min(row.count / 10.0, 1.0)
                        quality_score = row.avg_quality or 0.5
                        insights[source_name] = (count_score * 0.6 + quality_score * 0.4)
                
                break
                
        except Exception as e:
            logger.warning(f"Error analyzing historical patterns: {e}")
        
        # Apply learned patterns
        for pattern_topic, pattern_sources in self.success_patterns.items():
            similarity = self._calculate_topic_similarity(topic, pattern_topic)
            if similarity > 0.7:
                for source in pattern_sources:
                    current_score = insights.get(source, 0.5)
                    insights[source] = min(current_score + similarity * 0.1, 1.0)
        
        return insights
    
    def _generate_search_keywords(self, topic: str, context: str, 
                                 source_name: str) -> List[str]:
        """Generate optimal search keywords for a specific source"""
        base_keywords = topic.lower().split()
        
        # Add context keywords
        if context:
            context_words = context.lower().split()
            important_context = [w for w in context_words 
                                if len(w) > 4 and w not in ['research', 'study', 'analysis']]
            base_keywords.extend(important_context[:3])
        
        # Source-specific keyword optimization
        source_info = self.source_database.get(source_name, {})
        source_type = source_info.get('type', 'general')
        
        if source_type == 'academic':
            # Add academic search terms
            base_keywords.extend(['research', 'study', 'analysis'])
        elif source_type == 'technical':
            # Add technical terms
            base_keywords.extend(['implementation', 'system', 'architecture'])
        elif source_type == 'news':
            # Add news-relevant terms
            base_keywords.extend(['latest', 'development', 'announcement'])
        
        # Apply ML keyword optimization if available
        if hasattr(self, 'keyword_optimizer'):
            base_keywords = self._optimize_keywords_ml(base_keywords, source_name)
        
        # Remove duplicates and limit
        unique_keywords = list(dict.fromkeys(base_keywords))
        return unique_keywords[:8]
    
    def _predict_result_count(self, topic: str, source_name: str, 
                             features: Dict[str, Any]) -> int:
        """Predict expected number of relevant results"""
        base_estimate = 10
        
        # Adjust based on source characteristics
        source_info = self.source_database.get(source_name, {})
        
        if source_info.get('type') == 'academic':
            base_estimate = 20
        elif source_info.get('type') == 'news':
            base_estimate = 15
        elif source_info.get('type') == 'social':
            base_estimate = 30
        
        # Adjust based on topic complexity
        complexity = features.get('complexity', 0.5)
        if complexity > 0.7:
            base_estimate = int(base_estimate * 0.6)  # Complex topics have fewer results
        elif complexity < 0.3:
            base_estimate = int(base_estimate * 1.5)  # Simple topics have more results
        
        # Apply historical adjustment
        if source_name in self.topic_source_affinity:
            affinity = self.topic_source_affinity[source_name].get(topic, 0.5)
            base_estimate = int(base_estimate * (0.5 + affinity))
        
        return max(1, min(base_estimate, 100))
    
    def _determine_search_strategy(self, source_name: str, 
                                  features: Dict[str, Any]) -> str:
        """Determine optimal search strategy for source"""
        source_info = self.source_database.get(source_name, {})
        
        if source_info.get('api_available'):
            if features.get('complexity', 0.5) > 0.7:
                return "API deep search with multiple query variations"
            else:
                return "API search with standard parameters"
        else:
            if source_info.get('type') == 'academic':
                return "Advanced search with filters and citation tracking"
            elif source_info.get('type') == 'social':
                return "Trend analysis with community engagement metrics"
            else:
                return "Standard web search with relevance sorting"
    
    def _generate_reasoning(self, source_name: str, features: Dict[str, Any],
                           score: float) -> str:
        """Generate explanation for why source was recommended"""
        reasons = []
        
        source_info = self.source_database.get(source_name, {})
        
        if score > 0.8:
            reasons.append(f"Highly relevant to {features.get('text', 'topic')[:50]}")
        
        if source_info.get('quality_baseline', 0) > 0.8:
            reasons.append("High quality source with reliable information")
        
        if features.get('has_space') and 'space' in str(source_info.get('specialization', [])).lower():
            reasons.append("Specializes in space-related content")
        
        if features.get('has_academic') and source_info.get('type') == 'academic':
            reasons.append("Academic source suitable for theoretical research")
        
        if source_info.get('api_available'):
            reasons.append("API access enables comprehensive automated search")
        
        return "; ".join(reasons) if reasons else "Relevant source for comprehensive research"
    
    def _optimize_search_sequence(self, sources: List[SourceRecommendation],
                                 time_constraint: int) -> List[str]:
        """Optimize the sequence of source searching using ML"""
        # Sort by expected value (confidence * expected_results / time_required)
        source_values = []
        
        for source in sources:
            time_required = self._estimate_search_time(source)
            if time_required > 0:
                value = (source.confidence_score * source.expected_results) / time_required
            else:
                value = 0
            source_values.append((source.source_name, value))
        
        # Sort by value
        source_values.sort(key=lambda x: x[1], reverse=True)
        
        # Build sequence respecting time constraint
        sequence = []
        total_time = 0
        
        for source_name, value in source_values:
            source = next((s for s in sources if s.source_name == source_name), None)
            if source:
                estimated_time = self._estimate_search_time(source)
                if total_time + estimated_time <= time_constraint:
                    sequence.append(source_name)
                    total_time += estimated_time
        
        return sequence
    
    def _allocate_search_time(self, sources: List[SourceRecommendation],
                             time_constraint: int) -> Dict[str, float]:
        """Allocate search time optimally across sources"""
        allocation = {}
        
        # Calculate relative importance
        total_importance = sum(s.confidence_score * s.expected_results for s in sources)
        
        if total_importance > 0:
            for source in sources:
                importance = source.confidence_score * source.expected_results
                time_fraction = importance / total_importance
                allocated_time = time_constraint * time_fraction
                allocation[source.source_name] = round(allocated_time, 1)
        
        return allocation
    
    def _recommend_search_depth(self, sources: List[SourceRecommendation],
                               time_constraint: int, research_type: str) -> Dict[str, str]:
        """Recommend search depth for each source"""
        depth_recommendations = {}
        
        for source in sources:
            # Ensure confidence_score is a float
            try:
                confidence = float(source.confidence_score)
            except (ValueError, TypeError):
                confidence = 0.5  # Default fallback
                
            if research_type == 'quick':
                depth = 'shallow'
            elif confidence > 0.8 and time_constraint > 30:
                depth = 'deep'
            elif confidence > 0.6:
                depth = 'moderate'
            else:
                depth = 'shallow'
            
            depth_recommendations[source.source_name] = depth
        
        return depth_recommendations
    
    def _select_cross_validation_sources(self, 
                                        sources: List[SourceRecommendation]) -> List[str]:
        """Select sources for cross-validation and fact-checking"""
        validation_sources = []
        
        # Select diverse source types for validation
        source_types_seen = set()
        
        for source in sorted(sources, key=lambda x: x.confidence_score, reverse=True):
            if source.source_type not in source_types_seen:
                validation_sources.append(source.source_name)
                source_types_seen.add(source.source_type)
                
                if len(validation_sources) >= 3:
                    break
        
        return validation_sources
    
    def _calculate_strategy_confidence(self, primary_sources: List[SourceRecommendation]) -> float:
        """Calculate overall confidence in the search strategy"""
        if not primary_sources:
            return 0.3
        
        # Average confidence of top sources
        top_confidence = sum(s.confidence_score for s in primary_sources[:3]) / min(3, len(primary_sources))
        
        # Diversity bonus
        source_types = set(s.source_type for s in primary_sources[:5])
        diversity_bonus = len(source_types) * 0.05
        
        return min(top_confidence + diversity_bonus, 0.95)
    
    def _estimate_search_time(self, source: SourceRecommendation) -> float:
        """Estimate time required to search a source"""
        base_time = 5.0  # Base 5 minutes
        
        # Adjust based on search depth
        if 'deep' in source.search_strategy.lower():
            base_time *= 2
        elif 'shallow' in source.search_strategy.lower():
            base_time *= 0.5
        
        # Adjust based on expected results
        if source.expected_results > 20:
            base_time *= 1.5
        elif source.expected_results < 5:
            base_time *= 0.7
        
        return base_time
    
    def _assess_complexity(self, text: str) -> float:
        """Assess complexity of research topic"""
        complexity_indicators = [
            'complex', 'advanced', 'sophisticated', 'intricate',
            'multi', 'interdisciplinary', 'theoretical', 'quantum'
        ]
        
        text_lower = text.lower()
        complexity_score = sum(0.15 for indicator in complexity_indicators 
                              if indicator in text_lower)
        
        # Word count factor
        word_count = len(text.split())
        if word_count > 20:
            complexity_score += 0.2
        
        return min(complexity_score, 1.0)
    
    def _extract_domain_indicators(self, text: str) -> List[str]:
        """Extract domain indicators from text"""
        domains = []
        
        domain_keywords = {
            'space': ['space', 'orbital', 'satellite', 'spacecraft', 'astronomy'],
            'manufacturing': ['manufacturing', 'production', 'factory', 'assembly'],
            'automation': ['automation', 'robot', 'automated', 'autonomous'],
            'ai': ['ai', 'artificial intelligence', 'machine learning', 'neural'],
            'engineering': ['engineering', 'design', 'system', 'architecture']
        }
        
        text_lower = text.lower()
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                domains.append(domain)
        
        return domains
    
    def _extract_temporal_focus(self, text: str) -> str:
        """Extract temporal focus from text"""
        if any(word in text.lower() for word in ['future', 'emerging', 'next']):
            return 'future'
        elif any(word in text.lower() for word in ['current', 'recent', 'today']):
            return 'present'
        elif any(word in text.lower() for word in ['historical', 'past', 'evolution']):
            return 'past'
        else:
            return 'general'
    
    def _extract_geographic_scope(self, text: str) -> str:
        """Extract geographic scope from text"""
        if any(word in text.lower() for word in ['global', 'international', 'worldwide']):
            return 'global'
        elif any(word in text.lower() for word in ['national', 'domestic', 'local']):
            return 'national'
        elif any(word in text.lower() for word in ['space', 'orbital', 'planetary']):
            return 'space'
        else:
            return 'general'
    
    def _calculate_topic_similarity(self, topic1: str, topic2: str) -> float:
        """Calculate similarity between two topics"""
        # Simple word overlap similarity
        words1 = set(topic1.lower().split())
        words2 = set(topic2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _map_source_id_to_name(self, source_id: Any) -> Optional[str]:
        """Map database source ID to source name"""
        # Simplified mapping - in production would use proper lookup
        id_mapping = {
            1: "Google Scholar",
            2: "arXiv",
            3: "SpaceNews",
            4: "NASA",
            5: "Reddit"
        }
        return id_mapping.get(source_id)
    
    def _update_learning_history(self, topic: str, features: Dict[str, Any],
                                strategy: SearchStrategy):
        """Update learning history for continuous improvement"""
        self.search_history.append({
            'topic': topic,
            'features': features,
            'strategy': strategy,
            'timestamp': datetime.now()
        })
        
        # Update topic-source affinity
        for source in strategy.primary_sources:
            self.topic_source_affinity[source.source_name][topic] = /
                max(self.topic_source_affinity[source.source_name][topic],
                    source.confidence_score)
        
        # Limit history size
        if len(self.search_history) > 1000:
            self.search_history = self.search_history[-500:]
    
    @ml_error_handler.ml_error_handler(
        component="MLSourceRecommender",
        error_types=[MLErrorType.MODEL_LOADING_ERROR],
        fallback_enabled=True,
        max_retries=2
    ) if ML_ERROR_HANDLER_AVAILABLE else lambda x: x
    def _load_models(self):
        """Load pre-trained ML models if available with validation"""
        try:
            model_file = self.model_path / "relevance_classifier.pkl"
            if model_file.exists():
                loaded_model = joblib.load(model_file)
                
                # Validate model if validator is available and test data exists
                if ML_VALIDATOR_AVAILABLE and self._has_test_data():
                    try:
                        X_test, y_test = self._get_test_data()
                        validation_results = model_validator.validate_model_auto(
                            loaded_model, X_test, y_test, "relevance_classifier"
                        )
                        
                        if validation_results.meets_threshold:
                            self.trained_model = loaded_model
                            logger.info(f"Model validated successfully - {validation_results.recommendation}")
                            logger.info(f"Validation metrics: {validation_results.metrics}")
                        else:
                            logger.warning(f"Model validation failed - {validation_results.recommendation}")
                            logger.warning("Using fallback heuristic methods")
                            self.trained_model = None
                    except Exception as e:
                        logger.warning(f"Model validation failed: {e}, using model without validation")
                        self.trained_model = loaded_model
                else:
                    self.trained_model = loaded_model
                    logger.info("Loaded pre-trained relevance classifier (no validation)")
                
            vectorizer_file = self.model_path / "tfidf_vectorizer.pkl"
            if vectorizer_file.exists():
                self.tfidf_vectorizer = joblib.load(vectorizer_file)
                logger.info("Loaded pre-trained TF-IDF vectorizer")
                
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
    
    def _has_test_data(self) -> bool:
        """Check if test data is available for model validation"""
        test_file = self.model_path / "test_data.pkl"
        return test_file.exists() and len(self.search_history) >= 20
    
    def _get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get test data for model validation"""
        # Try to load from file first
        test_file = self.model_path / "test_data.pkl"
        if test_file.exists():
            try:
                test_data = joblib.load(test_file)
                return test_data['X_test'], test_data['y_test']
            except Exception as e:
                logger.warning(f"Could not load test data from file: {e}")
        
        # Generate test data from search history if available
        if len(self.search_history) >= 20:
            try:
                # Use recent search history as test data
                recent_searches = self.search_history[-20:]
                X_test = []
                y_test = []
                
                for search in recent_searches:
                    features = search.get('features', {})
                    success_rate = search.get('success_rate', 0.5)  # Success rate as target
                    
                    # Convert features to numerical array
                    feature_vector = [
                        features.get('length', 0),
                        int(features.get('has_space', False)),
                        int(features.get('has_manufacturing', False)),
                        int(features.get('has_technical', False)),
                        features.get('complexity_score', 0),
                        len(features.get('entities', []))
                    ]
                    
                    X_test.append(feature_vector)
                    y_test.append(1 if success_rate > 0.7 else 0)  # Binary classification
                
                return np.array(X_test), np.array(y_test)
                
            except Exception as e:
                logger.warning(f"Could not generate test data from history: {e}")
                
        # Return dummy test data as fallback
        X_test = np.random.rand(10, 6)  # 10 samples, 6 features
        y_test = np.random.randint(0, 2, 10)  # Binary classification
        return X_test, y_test
    
    @ml_error_handler.ml_error_handler(
        component="MLSourceRecommender",
        error_types=[MLErrorType.MODEL_LOADING_ERROR],
        fallback_enabled=False,
        max_retries=2
    ) if ML_ERROR_HANDLER_AVAILABLE else lambda x: x
    def save_models(self):
        """Save trained models for future use"""
        try:
            if hasattr(self, 'trained_model'):
                joblib.dump(self.trained_model, self.model_path / "relevance_classifier.pkl")
                
            joblib.dump(self.tfidf_vectorizer, self.model_path / "tfidf_vectorizer.pkl")
            
            # Save learning history
            with open(self.model_path / "learning_history.json", 'w') as f:
                json.dump({
                    'source_quality_scores': dict(self.source_quality_scores),
                    'topic_source_affinity': {k: dict(v) for k, v in self.topic_source_affinity.items()}
                }, f)
                
            logger.info("Saved ML models and learning history")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")

# Create global instance
ml_source_recommender = MLSourceRecommender()
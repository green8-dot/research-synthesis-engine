"""
Comprehensive Learning System
Manages all learning capabilities with knowledge graph integration
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
import hashlib
import uuid

# Import ML error handler
try:
    from research_synthesis.utils.ml_error_handler import ml_error_handler, MLErrorType
    ML_ERROR_HANDLER_AVAILABLE = True
except ImportError:
    print("ML Error Handler not available for learning system")
    ML_ERROR_HANDLER_AVAILABLE = False

# Database imports
from research_synthesis.database.connection import get_postgres_session, get_redis
from research_synthesis.database.models import Report, Article, Entity, SourceConfiguration
from sqlalchemy import select, func, and_, or_, desc, text
from sqlalchemy.orm import sessionmaker

class LearningPattern:
    """Represents a learned pattern from data analysis"""
    def __init__(self, pattern_id: str, pattern_type: str, source_data: Dict[str, Any], 
                 confidence: float, frequency: int, created_at: datetime):
        self.pattern_id = pattern_id
        self.pattern_type = pattern_type  # entity_relation, topic_trend, report_insight, etc.
        self.source_data = source_data
        self.confidence = confidence
        self.frequency = frequency
        self.created_at = created_at
        self.last_seen = created_at
        self.applications = []

class ReportInsight:
    """Represents insights learned from report analysis"""
    def __init__(self, insight_id: str, report_ids: List[str], insight_type: str,
                 summary: str, key_findings: List[str], recommendations: List[str],
                 confidence_score: float):
        self.insight_id = insight_id
        self.report_ids = report_ids
        self.insight_type = insight_type
        self.summary = summary
        self.key_findings = key_findings
        self.recommendations = recommendations
        self.confidence_score = confidence_score
        self.created_at = datetime.now()

class KnowledgeNode:
    """Represents a node in the knowledge graph with learned connections"""
    def __init__(self, node_id: str, node_type: str, properties: Dict[str, Any]):
        self.node_id = node_id
        self.node_type = node_type
        self.properties = properties
        self.connections = {}  # node_id -> {relationship_type: str, weight: float}
        self.learned_insights = []
        self.mention_history = []

class LearningSystem:
    """Comprehensive Learning System for Research Synthesis"""
    
    def __init__(self):
        self.patterns = {}  # pattern_id -> LearningPattern
        self.insights = {}  # insight_id -> ReportInsight
        self.knowledge_graph = {}  # node_id -> KnowledgeNode
        self.learning_rules = {}
        self.performance_metrics = {
            'patterns_discovered': 0,
            'insights_generated': 0,
            'accuracy_improvements': [],
            'last_learning_cycle': None
        }
        
        # Learning configuration
        self.learning_config = {
            'min_pattern_frequency': 3,
            'min_confidence_threshold': 0.7,
            'max_pattern_age_days': 30,
            'learning_cycle_interval_hours': 6,
            'report_analysis_depth': 'comprehensive',  # basic, detailed, comprehensive
            'enable_predictive_insights': True,
            'auto_source_discovery': True
        }
        
    @ml_error_handler.ml_error_handler(
        component="LearningSystem",
        error_types=[MLErrorType.DATABASE_CONNECTION_ERROR, MLErrorType.MODEL_LOADING_ERROR],
        fallback_enabled=True,
        max_retries=2
    ) if ML_ERROR_HANDLER_AVAILABLE else lambda x: x
    async def initialize(self):
        """Initialize the learning system"""
        print("Initializing Learning System...")
        
        # Load existing patterns and insights from database/cache
        await self._load_existing_patterns()
        await self._build_knowledge_graph()
        await self._initialize_learning_rules()
        
        # Start background learning cycle
        asyncio.create_task(self._continuous_learning_cycle())
        
        print(f"Learning System initialized with {len(self.patterns)} patterns and {len(self.insights)} insights")
        
    async def _load_existing_patterns(self):
        """Load existing learning patterns from storage"""
        try:
            redis_client = get_redis()
            if redis_client:
                # Load patterns from Redis cache
                patterns_data = await redis_client.get("learning_patterns")
                if patterns_data:
                    patterns_dict = json.loads(patterns_data)
                    for pattern_id, pattern_data in patterns_dict.items():
                        self.patterns[pattern_id] = LearningPattern(
                            pattern_id=pattern_data['pattern_id'],
                            pattern_type=pattern_data['pattern_type'],
                            source_data=pattern_data['source_data'],
                            confidence=pattern_data['confidence'],
                            frequency=pattern_data['frequency'],
                            created_at=datetime.fromisoformat(pattern_data['created_at'])
                        )
                        
                # Load insights from cache
                insights_data = await redis_client.get("learning_insights")
                if insights_data:
                    insights_dict = json.loads(insights_data)
                    for insight_id, insight_data in insights_dict.items():
                        self.insights[insight_id] = ReportInsight(
                            insight_id=insight_data['insight_id'],
                            report_ids=insight_data['report_ids'],
                            insight_type=insight_data['insight_type'],
                            summary=insight_data['summary'],
                            key_findings=insight_data['key_findings'],
                            recommendations=insight_data['recommendations'],
                            confidence_score=insight_data['confidence_score']
                        )
        except Exception as e:
            print(f"Error loading existing patterns: {e}")
            
    async def _build_knowledge_graph(self):
        """Build knowledge graph from entities and their relationships"""
        try:
            async for session in get_postgres_session():
                # Load entities
                result = await session.execute(select(Entity))
                entities = result.scalars().all()
                
                for entity in entities:
                    node = KnowledgeNode(
                        node_id=str(entity.id),
                        node_type=entity.entity_type,
                        properties={
                            'name': entity.name,
                            'description': entity.description,
                            'space_sector': entity.space_sector,
                            'mention_count': entity.mention_count,
                            'sentiment_average': entity.sentiment_average
                        }
                    )
                    self.knowledge_graph[str(entity.id)] = node
                    
                # Analyze entity co-occurrences to build connections
                await self._analyze_entity_relationships(session)
                break
                
        except Exception as e:
            print(f"Error building knowledge graph: {e}")
            
    async def _analyze_entity_relationships(self, session):
        """Analyze entity relationships from articles"""
        try:
            # Get articles with entity mentions
            result = await session.execute(
                select(Article.id, Article.title, Article.content)
                .limit(100)  # Limit for performance
            )
            articles = result.fetchall()
            
            entity_cooccurrence = defaultdict(lambda: defaultdict(int))
            
            for article in articles:
                # Find entities mentioned in this article
                mentioned_entities = []
                for node_id, node in self.knowledge_graph.items():
                    entity_name = node.properties.get('name', '').lower()
                    content = (article.title + " " + (article.content or "")).lower()
                    if entity_name in content:
                        mentioned_entities.append(node_id)
                        
                # Create co-occurrence connections
                for i, entity1 in enumerate(mentioned_entities):
                    for entity2 in mentioned_entities[i+1:]:
                        entity_cooccurrence[entity1][entity2] += 1
                        entity_cooccurrence[entity2][entity1] += 1
                        
            # Add relationships to knowledge graph
            for entity1, connections in entity_cooccurrence.items():
                if entity1 in self.knowledge_graph:
                    for entity2, count in connections.items():
                        if count >= 2:  # Minimum co-occurrence threshold
                            weight = min(count / 10.0, 1.0)  # Normalize weight
                            self.knowledge_graph[entity1].connections[entity2] = {
                                'relationship_type': 'co_mentioned',
                                'weight': weight
                            }
                            
        except Exception as e:
            print(f"Error analyzing entity relationships: {e}")
            
    async def _initialize_learning_rules(self):
        """Initialize learning rules for pattern recognition"""
        self.learning_rules = {
            'entity_trending': {
                'description': 'Detect trending entities based on mention frequency',
                'pattern_type': 'entity_trend',
                'min_mentions': 5,
                'time_window_days': 7,
                'enabled': True
            },
            'topic_correlation': {
                'description': 'Find correlated topics in reports',
                'pattern_type': 'topic_correlation',
                'min_correlation': 0.6,
                'min_reports': 3,
                'enabled': True
            },
            'report_quality_patterns': {
                'description': 'Learn what makes high-quality reports',
                'pattern_type': 'quality_pattern',
                'min_confidence_score': 0.8,
                'min_word_count': 1000,
                'enabled': True
            },
            'source_effectiveness': {
                'description': 'Learn which sources produce best insights',
                'pattern_type': 'source_effectiveness',
                'min_articles': 10,
                'quality_threshold': 0.7,
                'enabled': True
            },
            'knowledge_gaps': {
                'description': 'Identify gaps in knowledge coverage',
                'pattern_type': 'knowledge_gap',
                'coverage_threshold': 0.3,
                'enabled': True
            }
        }
        
    async def _continuous_learning_cycle(self):
        """Background continuous learning process"""
        while True:
            try:
                print("Starting learning cycle...")
                
                # Analyze reports for insights
                await self.analyze_reports_for_insights()
                
                # Discover patterns from entities and articles
                await self.discover_patterns()
                
                # Update knowledge graph connections
                await self.update_knowledge_graph()
                
                # Generate predictive insights
                await self.generate_predictive_insights()
                
                # Clean old patterns
                await self.cleanup_old_patterns()
                
                # Save learned data
                await self.save_learning_state()
                
                self.performance_metrics['last_learning_cycle'] = datetime.now()
                
                # Wait for next cycle
                await asyncio.sleep(self.learning_config['learning_cycle_interval_hours'] * 3600)
                
            except Exception as e:
                print(f"Error in learning cycle: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry
                
    async def analyze_reports_for_insights(self) -> List[ReportInsight]:
        """Analyze existing reports to generate insights"""
        insights_generated = []
        
        try:
            async for session in get_postgres_session():
                # Get recent reports
                result = await session.execute(
                    select(Report)
                    .order_by(desc(Report.created_at))
                    .limit(20)
                )
                reports = result.scalars().all()
                
                if len(reports) < 2:
                    print("Not enough reports for analysis")
                    return insights_generated
                    
                # Analyze report patterns
                report_topics = {}
                report_quality_scores = []
                common_findings = defaultdict(list)
                
                for report in reports:
                    # Extract topics from report
                    topics = self._extract_topics_from_report(report)
                    report_topics[str(report.id)] = topics
                    
                    # Track quality metrics
                    if report.confidence_score:
                        report_quality_scores.append({
                            'score': report.confidence_score,
                            'word_count': report.word_count,
                            'type': report.report_type
                        })
                    
                    # Extract common findings
                    if report.key_findings:
                        try:
                            findings = json.loads(report.key_findings)
                            for finding in findings:
                                if isinstance(finding, str):
                                    common_findings[finding.lower()].append(str(report.id))
                        except:
                            pass
                            
                # Generate topic correlation insights
                topic_correlations = self._analyze_topic_correlations(report_topics)
                for correlation in topic_correlations:
                    insight_id = str(uuid.uuid4())
                    insight = ReportInsight(
                        insight_id=insight_id,
                        report_ids=correlation['report_ids'],
                        insight_type='topic_correlation',
                        summary=f"Strong correlation between {correlation['topic1']} and {correlation['topic2']}",
                        key_findings=[
                            f"Correlation strength: {correlation['strength']:.2f}",
                            f"Found in {len(correlation['report_ids'])} reports"
                        ],
                        recommendations=[
                            "Consider creating combined analysis reports",
                            "Explore deeper connections between these topics"
                        ],
                        confidence_score=correlation['strength']
                    )
                    self.insights[insight_id] = insight
                    insights_generated.append(insight)
                    
                # Generate quality pattern insights
                if len(report_quality_scores) >= 5:
                    quality_insight = self._analyze_quality_patterns(report_quality_scores)
                    if quality_insight:
                        insight_id = str(uuid.uuid4())
                        self.insights[insight_id] = quality_insight
                        insights_generated.append(quality_insight)
                        
                # Generate common findings insights
                frequent_findings = {k: v for k, v in common_findings.items() if len(v) >= 2}
                if frequent_findings:
                    insight_id = str(uuid.uuid4())
                    insight = ReportInsight(
                        insight_id=insight_id,
                        report_ids=list(set([report_id for findings in frequent_findings.values() for report_id in findings])),
                        insight_type='common_findings',
                        summary=f"Identified {len(frequent_findings)} recurring findings across reports",
                        key_findings=list(frequent_findings.keys())[:5],
                        recommendations=[
                            "Create standardized finding templates",
                            "Develop deeper analysis for recurring themes"
                        ],
                        confidence_score=0.8
                    )
                    self.insights[insight_id] = insight
                    insights_generated.append(insight)
                    
                break
                
        except Exception as e:
            print(f"Error analyzing reports for insights: {e}")
            
        self.performance_metrics['insights_generated'] += len(insights_generated)
        print(f"Generated {len(insights_generated)} new insights from report analysis")
        return insights_generated
        
    def _extract_topics_from_report(self, report: Report) -> List[str]:
        """Extract topics from a report"""
        topics = []
        
        # Extract from report type
        if report.report_type:
            topics.append(report.report_type.replace('_', ' '))
            
        # Extract from title
        if report.title:
            # Simple keyword extraction
            keywords = report.title.lower().split()
            topics.extend([word for word in keywords if len(word) > 4])
            
        # Extract from key findings
        if report.key_findings:
            try:
                findings = json.loads(report.key_findings)
                for finding in findings:
                    if isinstance(finding, str):
                        words = finding.lower().split()
                        topics.extend([word for word in words if len(word) > 4])
            except:
                pass
                
        return list(set(topics))
        
    def _analyze_topic_correlations(self, report_topics: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Analyze correlations between topics across reports"""
        correlations = []
        
        # Create topic co-occurrence matrix
        all_topics = list(set([topic for topics in report_topics.values() for topic in topics]))
        
        if len(all_topics) < 2:
            return correlations
            
        topic_cooccurrence = defaultdict(lambda: defaultdict(int))
        topic_counts = defaultdict(int)
        
        for report_id, topics in report_topics.items():
            for topic in topics:
                topic_counts[topic] += 1
                for other_topic in topics:
                    if topic != other_topic:
                        topic_cooccurrence[topic][other_topic] += 1
                        
        # Calculate correlations
        for topic1 in all_topics:
            for topic2 in all_topics:
                if topic1 < topic2:  # Avoid duplicates
                    cooccur = topic_cooccurrence[topic1][topic2]
                    if cooccur >= 2:
                        # Simple correlation metric
                        strength = cooccur / min(topic_counts[topic1], topic_counts[topic2])
                        if strength >= 0.5:
                            # Find reports containing both topics
                            related_reports = [
                                report_id for report_id, topics in report_topics.items()
                                if topic1 in topics and topic2 in topics
                            ]
                            correlations.append({
                                'topic1': topic1,
                                'topic2': topic2,
                                'strength': strength,
                                'report_ids': related_reports
                            })
                            
        return correlations
        
    def _analyze_quality_patterns(self, quality_scores: List[Dict[str, Any]]) -> Optional[ReportInsight]:
        """Analyze patterns in high-quality reports"""
        try:
            high_quality = [s for s in quality_scores if s['score'] >= 0.8]
            if len(high_quality) < 3:
                return None
                
            avg_word_count = sum(r['word_count'] for r in high_quality if r['word_count']) / len(high_quality)
            common_types = Counter(r['type'] for r in high_quality).most_common(3)
            
            return ReportInsight(
                insight_id=str(uuid.uuid4()),
                report_ids=[],  # General pattern, not tied to specific reports
                insight_type='quality_pattern',
                summary=f"High-quality reports pattern identified",
                key_findings=[
                    f"Average word count of high-quality reports: {avg_word_count:.0f}",
                    f"Most common high-quality report types: {[t[0] for t in common_types]}"
                ],
                recommendations=[
                    f"Target {avg_word_count:.0f}+ words for better quality",
                    f"Focus on {common_types[0][0]} report types for best results"
                ],
                confidence_score=0.85
            )
            
        except Exception as e:
            print(f"Error analyzing quality patterns: {e}")
            return None
            
    async def discover_patterns(self) -> List[LearningPattern]:
        """Discover new patterns from data"""
        patterns_found = []
        
        try:
            # Entity trending patterns
            if self.learning_rules['entity_trending']['enabled']:
                entity_patterns = await self._discover_entity_trends()
                patterns_found.extend(entity_patterns)
                
            # Source effectiveness patterns
            if self.learning_rules['source_effectiveness']['enabled']:
                source_patterns = await self._discover_source_patterns()
                patterns_found.extend(source_patterns)
                
            # Knowledge gap patterns
            if self.learning_rules['knowledge_gaps']['enabled']:
                gap_patterns = await self._discover_knowledge_gaps()
                patterns_found.extend(gap_patterns)
                
        except Exception as e:
            print(f"Error discovering patterns: {e}")
            
        # Add new patterns to system
        for pattern in patterns_found:
            self.patterns[pattern.pattern_id] = pattern
            
        self.performance_metrics['patterns_discovered'] += len(patterns_found)
        print(f"Discovered {len(patterns_found)} new patterns")
        return patterns_found
        
    async def _discover_entity_trends(self) -> List[LearningPattern]:
        """Discover trending entity patterns"""
        patterns = []
        
        try:
            async for session in get_postgres_session():
                # Get entities with recent high activity
                cutoff_date = datetime.now() - timedelta(days=7)
                result = await session.execute(
                    select(Entity.name, Entity.mention_count, Entity.entity_type)
                    .filter(Entity.last_mentioned >= cutoff_date)
                    .filter(Entity.mention_count >= 5)
                    .order_by(desc(Entity.mention_count))
                    .limit(10)
                )
                trending_entities = result.fetchall()
                
                for entity in trending_entities:
                    pattern_id = hashlib.md5(f"entity_trend_{entity.name}".encode()).hexdigest()
                    pattern = LearningPattern(
                        pattern_id=pattern_id,
                        pattern_type='entity_trend',
                        source_data={
                            'entity_name': entity.name,
                            'entity_type': entity.entity_type,
                            'mention_count': entity.mention_count,
                            'time_window': '7_days'
                        },
                        confidence=min(entity.mention_count / 20.0, 1.0),
                        frequency=entity.mention_count,
                        created_at=datetime.now()
                    )
                    patterns.append(pattern)
                    
                break
                
        except Exception as e:
            print(f"Error discovering entity trends: {e}")
            
        return patterns
        
    async def _discover_source_patterns(self) -> List[LearningPattern]:
        """Discover source effectiveness patterns"""
        patterns = []
        
        try:
            async for session in get_postgres_session():
                # Analyze source configurations and their effectiveness
                result = await session.execute(
                    select(SourceConfiguration.source_name, SourceConfiguration.enabled,
                           func.count(Article.id).label('article_count'),
                           func.avg(Article.quality_score).label('avg_quality'))
                    .outerjoin(Article, SourceConfiguration.source_name == Article.source_id)
                    .group_by(SourceConfiguration.source_name, SourceConfiguration.enabled)
                    .having(func.count(Article.id) >= 5)
                )
                source_stats = result.fetchall()
                
                for source in source_stats:
                    if source.avg_quality and source.avg_quality >= 0.7:
                        pattern_id = hashlib.md5(f"source_effective_{source.source_name}".encode()).hexdigest()
                        pattern = LearningPattern(
                            pattern_id=pattern_id,
                            pattern_type='source_effectiveness',
                            source_data={
                                'source_name': source.source_name,
                                'article_count': source.article_count,
                                'avg_quality': float(source.avg_quality),
                                'effectiveness_score': min(source.avg_quality * (source.article_count / 10), 1.0)
                            },
                            confidence=source.avg_quality,
                            frequency=source.article_count,
                            created_at=datetime.now()
                        )
                        patterns.append(pattern)
                        
                break
                
        except Exception as e:
            print(f"Error discovering source patterns: {e}")
            
        return patterns
        
    async def _discover_knowledge_gaps(self) -> List[LearningPattern]:
        """Discover knowledge gaps in coverage"""
        patterns = []
        
        try:
            # Analyze entity types and coverage
            entity_type_coverage = {}
            for node in self.knowledge_graph.values():
                entity_type = node.node_type
                if entity_type not in entity_type_coverage:
                    entity_type_coverage[entity_type] = {
                        'count': 0,
                        'avg_mentions': 0,
                        'total_mentions': 0
                    }
                    
                entity_type_coverage[entity_type]['count'] += 1
                mentions = node.properties.get('mention_count', 0)
                entity_type_coverage[entity_type]['total_mentions'] += mentions
                
            # Calculate coverage gaps
            total_entities = sum(data['count'] for data in entity_type_coverage.values())
            for entity_type, data in entity_type_coverage.items():
                coverage_ratio = data['count'] / total_entities
                avg_mentions = data['total_mentions'] / max(data['count'], 1)
                
                if coverage_ratio < 0.1 and avg_mentions < 5:  # Low coverage gap
                    pattern_id = hashlib.md5(f"knowledge_gap_{entity_type}".encode()).hexdigest()
                    pattern = LearningPattern(
                        pattern_id=pattern_id,
                        pattern_type='knowledge_gap',
                        source_data={
                            'entity_type': entity_type,
                            'coverage_ratio': coverage_ratio,
                            'avg_mentions': avg_mentions,
                            'entity_count': data['count'],
                            'gap_severity': 'high' if coverage_ratio < 0.05 else 'medium'
                        },
                        confidence=1.0 - coverage_ratio,  # Higher confidence for bigger gaps
                        frequency=1,
                        created_at=datetime.now()
                    )
                    patterns.append(pattern)
                    
        except Exception as e:
            print(f"Error discovering knowledge gaps: {e}")
            
        return patterns
        
    async def update_knowledge_graph(self):
        """Update knowledge graph with new learnings"""
        try:
            # Update node properties based on patterns
            for pattern in self.patterns.values():
                if pattern.pattern_type == 'entity_trend':
                    entity_name = pattern.source_data.get('entity_name')
                    # Find corresponding node and update trending status
                    for node in self.knowledge_graph.values():
                        if node.properties.get('name') == entity_name:
                            node.properties['is_trending'] = True
                            node.properties['trend_confidence'] = pattern.confidence
                            node.learned_insights.append({
                                'type': 'trending',
                                'confidence': pattern.confidence,
                                'detected_at': pattern.created_at.isoformat()
                            })
                            
            # Update connections based on insights
            for insight in self.insights.values():
                if insight.insight_type == 'topic_correlation':
                    # This would connect related topic nodes in a more sophisticated implementation
                    pass
                    
            print(f"Updated knowledge graph with {len(self.patterns)} patterns")
            
        except Exception as e:
            print(f"Error updating knowledge graph: {e}")
            
    async def generate_predictive_insights(self) -> List[Dict[str, Any]]:
        """Generate predictive insights based on learned patterns"""
        predictions = []
        
        if not self.learning_config['enable_predictive_insights']:
            return predictions
            
        try:
            # Predict trending topics
            entity_trends = [p for p in self.patterns.values() if p.pattern_type == 'entity_trend']
            if len(entity_trends) >= 3:
                trend_prediction = {
                    'type': 'topic_trend_prediction',
                    'prediction': f"Based on current patterns, expect increased activity in {entity_trends[0].source_data.get('entity_type')} sector",
                    'confidence': sum(p.confidence for p in entity_trends[:3]) / 3,
                    'timeframe': '2-4 weeks',
                    'supporting_patterns': [p.pattern_id for p in entity_trends[:3]]
                }
                predictions.append(trend_prediction)
                
            # Predict source recommendations
            effective_sources = [p for p in self.patterns.values() if p.pattern_type == 'source_effectiveness']
            if len(effective_sources) >= 2:
                source_prediction = {
                    'type': 'source_optimization',
                    'prediction': f"Recommend prioritizing {effective_sources[0].source_data.get('source_name')} for higher quality content",
                    'confidence': effective_sources[0].confidence,
                    'expected_improvement': f"{(effective_sources[0].source_data.get('avg_quality', 0) - 0.5) * 100:.1f}% quality increase",
                    'supporting_patterns': [p.pattern_id for p in effective_sources]
                }
                predictions.append(source_prediction)
                
            # Predict knowledge gap filling
            knowledge_gaps = [p for p in self.patterns.values() if p.pattern_type == 'knowledge_gap']
            if knowledge_gaps:
                gap_prediction = {
                    'type': 'knowledge_gap_opportunity',
                    'prediction': f"Addressing {knowledge_gaps[0].source_data.get('entity_type')} coverage could improve overall system intelligence",
                    'confidence': knowledge_gaps[0].confidence,
                    'priority': knowledge_gaps[0].source_data.get('gap_severity', 'medium'),
                    'suggested_actions': [
                        f"Add more sources covering {knowledge_gaps[0].source_data.get('entity_type')} topics",
                        "Increase scraping frequency for this domain"
                    ]
                }
                predictions.append(gap_prediction)
                
            print(f"Generated {len(predictions)} predictive insights")
            
        except Exception as e:
            print(f"Error generating predictive insights: {e}")
            
        return predictions
        
    async def cleanup_old_patterns(self):
        """Remove old or low-confidence patterns"""
        cutoff_date = datetime.now() - timedelta(days=self.learning_config['max_pattern_age_days'])
        
        patterns_to_remove = []
        for pattern_id, pattern in self.patterns.items():
            if (pattern.created_at < cutoff_date or 
                pattern.confidence < self.learning_config['min_confidence_threshold']):
                patterns_to_remove.append(pattern_id)
                
        for pattern_id in patterns_to_remove:
            del self.patterns[pattern_id]
            
        if patterns_to_remove:
            print(f"Cleaned up {len(patterns_to_remove)} old patterns")
            
    async def save_learning_state(self):
        """Save current learning state to storage"""
        try:
            redis_client = get_redis()
            if redis_client:
                # Save patterns
                patterns_data = {}
                for pattern_id, pattern in self.patterns.items():
                    patterns_data[pattern_id] = {
                        'pattern_id': pattern.pattern_id,
                        'pattern_type': pattern.pattern_type,
                        'source_data': pattern.source_data,
                        'confidence': pattern.confidence,
                        'frequency': pattern.frequency,
                        'created_at': pattern.created_at.isoformat()
                    }
                    
                await redis_client.setex(
                    "learning_patterns", 
                    86400 * 7,  # 7 days TTL
                    json.dumps(patterns_data, default=str)
                )
                
                # Save insights
                insights_data = {}
                for insight_id, insight in self.insights.items():
                    insights_data[insight_id] = {
                        'insight_id': insight.insight_id,
                        'report_ids': insight.report_ids,
                        'insight_type': insight.insight_type,
                        'summary': insight.summary,
                        'key_findings': insight.key_findings,
                        'recommendations': insight.recommendations,
                        'confidence_score': insight.confidence_score
                    }
                    
                await redis_client.setex(
                    "learning_insights",
                    86400 * 7,  # 7 days TTL
                    json.dumps(insights_data, default=str)
                )
                
                # Save performance metrics
                await redis_client.setex(
                    "learning_metrics",
                    86400,  # 1 day TTL
                    json.dumps(self.performance_metrics, default=str)
                )
                
        except Exception as e:
            print(f"Error saving learning state: {e}")
            
    async def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning system status"""
        return {
            'system_status': 'active',
            'patterns_count': len(self.patterns),
            'insights_count': len(self.insights),
            'knowledge_graph_nodes': len(self.knowledge_graph),
            'performance_metrics': self.performance_metrics,
            'learning_config': self.learning_config,
            'pattern_types': list(set([p.pattern_type for p in self.patterns.values()])),
            'insight_types': list(set([i.insight_type for i in self.insights.values()])),
            'last_learning_cycle': self.performance_metrics.get('last_learning_cycle'),
            'predictive_insights_enabled': self.learning_config['enable_predictive_insights']
        }
        
    async def get_insights_for_report_generation(self, report_type: str) -> Dict[str, Any]:
        """Get insights to improve report generation for specific type"""
        relevant_insights = []
        relevant_patterns = []
        
        # Find relevant insights
        for insight in self.insights.values():
            if (report_type in insight.summary.lower() or 
                any(report_type in finding.lower() for finding in insight.key_findings)):
                relevant_insights.append({
                    'type': insight.insight_type,
                    'summary': insight.summary,
                    'recommendations': insight.recommendations,
                    'confidence': insight.confidence_score
                })
                
        # Find relevant patterns
        for pattern in self.patterns.values():
            if (report_type in str(pattern.source_data).lower()):
                relevant_patterns.append({
                    'type': pattern.pattern_type,
                    'data': pattern.source_data,
                    'confidence': pattern.confidence
                })
                
        return {
            'report_type': report_type,
            'relevant_insights': relevant_insights,
            'relevant_patterns': relevant_patterns,
            'recommendations': self._generate_report_recommendations(report_type, relevant_insights, relevant_patterns)
        }
        
    def _generate_report_recommendations(self, report_type: str, insights: List[Dict], patterns: List[Dict]) -> List[str]:
        """Generate recommendations for report improvement"""
        recommendations = []
        
        # Quality-based recommendations
        quality_patterns = [p for p in patterns if p['type'] == 'quality_pattern']
        if quality_patterns:
            recommendations.append(f"Target {quality_patterns[0]['data'].get('min_word_count', 1000)}+ words for better quality")
            
        # Source-based recommendations
        effective_sources = [p for p in patterns if p['type'] == 'source_effectiveness']
        if effective_sources:
            best_source = max(effective_sources, key=lambda x: x['confidence'])
            recommendations.append(f"Prioritize {best_source['data'].get('source_name')} for {report_type} reports")
            
        # Topic-based recommendations
        topic_insights = [i for i in insights if i['type'] == 'topic_correlation']
        if topic_insights:
            recommendations.append("Consider including related topics based on learned correlations")
            
        if not recommendations:
            recommendations = [
                f"Focus on comprehensive analysis for {report_type} reports",
                "Include diverse source perspectives",
                "Provide actionable insights and recommendations"
            ]
            
        return recommendations

# Global learning system instance
learning_system = LearningSystem()
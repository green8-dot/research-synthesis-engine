"""
Report Analyzer Logic
Reads reports, analyzes content, and learns from compiled reports to generate insights
"""

import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
import re
import hashlib
from dataclasses import dataclass

# Database imports
from research_synthesis.database.connection import get_postgres_session, get_redis
from research_synthesis.database.models import Report, Article, Entity
from sqlalchemy import select, func, and_, or_, desc

# NLP and analysis imports
import numpy as np

@dataclass
class ReportAnalysis:
    """Represents the analysis results of a report"""
    report_id: str
    quality_score: float
    complexity_score: float
    topic_coverage: List[str]
    entity_mentions: Dict[str, int]
    key_themes: List[str]
    readability_score: float
    insight_density: float
    source_diversity: int
    factual_accuracy: float
    recommendations_quality: float
    
@dataclass
class ReportPattern:
    """Represents a learned pattern from report analysis"""
    pattern_id: str
    pattern_type: str
    reports_analyzed: List[str]
    pattern_data: Dict[str, Any]
    confidence_score: float
    frequency: int
    discovered_at: datetime

class ReportAnalyzer:
    """Advanced Report Analysis System with Learning Capabilities"""
    
    def __init__(self):
        self.analysis_cache = {}  # report_id -> ReportAnalysis
        self.learned_patterns = {}  # pattern_id -> ReportPattern
        self.topic_vocabulary = set()
        self.entity_vocabulary = set()
        self.quality_benchmarks = {
            'excellent': {'min_score': 0.9, 'examples': []},
            'good': {'min_score': 0.7, 'examples': []},
            'average': {'min_score': 0.5, 'examples': []},
            'poor': {'min_score': 0.0, 'examples': []}
        }
        
        # Analysis configuration
        self.config = {
            'min_word_count_quality': 500,
            'optimal_word_count': 1500,
            'max_word_count_efficiency': 5000,
            'topic_extraction_threshold': 0.3,
            'entity_extraction_threshold': 2,
            'learning_batch_size': 10,
            'pattern_confidence_threshold': 0.6,
            'analysis_depth': 'comprehensive',  # basic, detailed, comprehensive
            'enable_predictive_scoring': True
        }
        
        # Initialize vocabularies
        asyncio.create_task(self._initialize_vocabularies())
        
    async def _initialize_vocabularies(self):
        """Initialize topic and entity vocabularies from database"""
        try:
            async for session in get_postgres_session():
                # Build entity vocabulary
                result = await session.execute(select(Entity.name, Entity.entity_type))
                entities = result.fetchall()
                for entity in entities:
                    self.entity_vocabulary.add(entity.name.lower())
                    
                # Build topic vocabulary from reports
                result = await session.execute(select(Report.title, Report.key_findings, Report.report_type))
                reports = result.fetchall()
                for report in reports:
                    if report.title:
                        self.topic_vocabulary.update(self._extract_keywords(report.title))
                    if report.key_findings:
                        try:
                            findings = json.loads(report.key_findings)
                            for finding in findings:
                                if isinstance(finding, str):
                                    self.topic_vocabulary.update(self._extract_keywords(finding))
                        except:
                            pass
                    if report.report_type:
                        self.topic_vocabulary.add(report.report_type.replace('_', ' '))
                        
                break
                
            print(f"Initialized vocabularies: {len(self.entity_vocabulary)} entities, {len(self.topic_vocabulary)} topics")
            
        except Exception as e:
            print(f"Error initializing vocabularies: {e}")
            
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        if not text:
            return []
            
        # Remove special characters and convert to lowercase
        clean_text = re.sub(r'[^a-zA-Z/s]', '', text.lower())
        
        # Split into words and filter
        words = [word for word in clean_text.split() if len(word) > 3]
        
        # Remove common stop words
        stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'what', 'make'}
        keywords = [word for word in words if word not in stop_words]
        
        return keywords
        
    async def analyze_report(self, report_id: str) -> ReportAnalysis:
        """Perform comprehensive analysis of a single report"""
        try:
            async for session in get_postgres_session():
                # Get report data
                result = await session.execute(
                    select(Report).filter(Report.id == int(report_id))
                )
                report = result.scalar_one_or_none()
                
                if not report:
                    raise ValueError(f"Report {report_id} not found")
                    
                # Perform analysis
                analysis = await self._perform_report_analysis(report)
                
                # Cache the analysis
                self.analysis_cache[report_id] = analysis
                
                return analysis
                
        except Exception as e:
            print(f"Error analyzing report {report_id}: {e}")
            # Return default analysis
            return ReportAnalysis(
                report_id=report_id,
                quality_score=0.5,
                complexity_score=0.5,
                topic_coverage=[],
                entity_mentions={},
                key_themes=[],
                readability_score=0.5,
                insight_density=0.5,
                source_diversity=0,
                factual_accuracy=0.5,
                recommendations_quality=0.5
            )
            
    async def _perform_report_analysis(self, report: Report) -> ReportAnalysis:
        """Perform detailed analysis of report content"""
        
        # Extract content for analysis
        content_text = ""
        if report.full_content:
            try:
                content_data = json.loads(report.full_content)
                content_text = self._extract_text_from_content(content_data)
            except:
                content_text = str(report.full_content)
        
        title_text = report.title or ""
        findings_text = ""
        if report.key_findings:
            try:
                findings = json.loads(report.key_findings)
                findings_text = " ".join([str(f) for f in findings if isinstance(f, str)])
            except:
                findings_text = str(report.key_findings)
                
        full_text = f"{title_text} {content_text} {findings_text}"
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(report, full_text)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(full_text)
        
        # Extract topic coverage
        topic_coverage = self._extract_topic_coverage(full_text)
        
        # Extract entity mentions
        entity_mentions = self._extract_entity_mentions(full_text)
        
        # Extract key themes
        key_themes = self._extract_key_themes(full_text)
        
        # Calculate readability score
        readability_score = self._calculate_readability_score(full_text)
        
        # Calculate insight density
        insight_density = self._calculate_insight_density(full_text, findings_text)
        
        # Calculate source diversity (from content analysis)
        source_diversity = self._calculate_source_diversity(content_text)
        
        # Calculate factual accuracy (based on entity mentions and known facts)
        factual_accuracy = self._estimate_factual_accuracy(entity_mentions, full_text)
        
        # Calculate recommendations quality
        recommendations_quality = self._analyze_recommendations_quality(report)
        
        return ReportAnalysis(
            report_id=str(report.id),
            quality_score=quality_score,
            complexity_score=complexity_score,
            topic_coverage=topic_coverage,
            entity_mentions=entity_mentions,
            key_themes=key_themes,
            readability_score=readability_score,
            insight_density=insight_density,
            source_diversity=source_diversity,
            factual_accuracy=factual_accuracy,
            recommendations_quality=recommendations_quality
        )
        
    def _extract_text_from_content(self, content_data: Dict) -> str:
        """Extract readable text from content structure"""
        text_parts = []
        
        if isinstance(content_data, dict):
            # Extract from common content fields
            for key in ['summary', 'executive_summary', 'sections', 'content', 'analysis']:
                if key in content_data:
                    value = content_data[key]
                    if isinstance(value, str):
                        text_parts.append(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict) and 'content' in item:
                                text_parts.append(str(item['content']))
                            elif isinstance(item, str):
                                text_parts.append(item)
                    elif isinstance(value, dict):
                        text_parts.append(str(value))
                        
        return " ".join(text_parts)
        
    def _calculate_quality_score(self, report: Report, full_text: str) -> float:
        """Calculate overall quality score of the report"""
        scores = []
        
        # Word count score (0-1)
        word_count = len(full_text.split()) if full_text else 0
        if word_count < self.config['min_word_count_quality']:
            word_score = word_count / self.config['min_word_count_quality']
        elif word_count <= self.config['optimal_word_count']:
            word_score = 0.8 + 0.2 * ((word_count - self.config['min_word_count_quality']) / 
                                     (self.config['optimal_word_count'] - self.config['min_word_count_quality']))
        else:
            # Diminishing returns after optimal length
            excess = word_count - self.config['optimal_word_count']
            max_excess = self.config['max_word_count_efficiency'] - self.config['optimal_word_count']
            word_score = 1.0 - 0.3 * min(excess / max_excess, 1.0)
            
        scores.append(word_score)
        
        # Structure score (based on sections, findings, etc.)
        structure_score = 0.5  # Base score
        if report.executive_summary:
            structure_score += 0.2
        if report.key_findings:
            try:
                findings = json.loads(report.key_findings)
                if len(findings) >= 3:
                    structure_score += 0.2
                else:
                    structure_score += 0.1
            except:
                structure_score += 0.05
        if report.recommendations:
            structure_score += 0.1
            
        scores.append(min(structure_score, 1.0))
        
        # Content richness score
        content_score = 0.3  # Base score
        if word_count > 0:
            unique_words = len(set(full_text.lower().split()))
            vocabulary_richness = unique_words / word_count if word_count > 0 else 0
            content_score += 0.4 * vocabulary_richness
            
            # Sentence variety
            sentences = full_text.split('.')
            if len(sentences) > 1:
                avg_sentence_length = word_count / len(sentences)
                if 10 <= avg_sentence_length <= 25:  # Good range
                    content_score += 0.3
                else:
                    content_score += 0.1
                    
        scores.append(min(content_score, 1.0))
        
        # Confidence score from report metadata
        if report.confidence_score:
            scores.append(report.confidence_score)
        else:
            scores.append(0.6)  # Default if not available
            
        # Return weighted average
        return sum(scores) / len(scores)
        
    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate complexity score based on vocabulary and structure"""
        if not text or len(text.strip()) == 0:
            return 0.0
            
        words = text.split()
        if len(words) == 0:
            return 0.0
            
        # Lexical complexity
        unique_words = set(word.lower() for word in words)
        lexical_diversity = len(unique_words) / len(words)
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        word_length_score = min(avg_word_length / 8.0, 1.0)  # Normalize to 0-1
        
        # Sentence complexity
        sentences = text.split('.')
        avg_sentence_length = len(words) / max(len(sentences), 1)
        sentence_complexity = min(avg_sentence_length / 20.0, 1.0)  # Normalize to 0-1
        
        # Technical vocabulary (entities, technical terms)
        technical_words = 0
        for word in words:
            if (word.lower() in self.entity_vocabulary or 
                word.lower() in self.topic_vocabulary or
                len(word) > 10):  # Long words often technical
                technical_words += 1
                
        technical_density = technical_words / len(words)
        
        # Combine scores
        complexity = (lexical_diversity * 0.3 + 
                     word_length_score * 0.2 + 
                     sentence_complexity * 0.3 + 
                     technical_density * 0.2)
                     
        return min(complexity, 1.0)
        
    def _extract_topic_coverage(self, text: str) -> List[str]:
        """Extract topics covered in the report"""
        if not text:
            return []
            
        topics_found = []
        text_lower = text.lower()
        
        # Check against topic vocabulary
        for topic in self.topic_vocabulary:
            if topic in text_lower:
                topics_found.append(topic)
                
        # Extract from word frequency
        words = self._extract_keywords(text)
        word_freq = Counter(words)
        frequent_topics = [word for word, count in word_freq.most_common(10) 
                          if count >= 2 and len(word) > 4]
        
        topics_found.extend(frequent_topics)
        
        return list(set(topics_found))
        
    def _extract_entity_mentions(self, text: str) -> Dict[str, int]:
        """Extract and count entity mentions"""
        if not text:
            return {}
            
        entity_counts = {}
        text_lower = text.lower()
        
        for entity in self.entity_vocabulary:
            count = text_lower.count(entity)
            if count >= self.config['entity_extraction_threshold']:
                entity_counts[entity] = count
                
        return entity_counts
        
    def _extract_key_themes(self, text: str) -> List[str]:
        """Extract key themes from the report"""
        if not text:
            return []
            
        # Simple theme extraction based on word clusters
        words = self._extract_keywords(text)
        word_freq = Counter(words)
        
        # Group related words as themes
        themes = []
        high_freq_words = [word for word, count in word_freq.most_common(20) if count >= 3]
        
        # Look for theme patterns
        space_words = [w for w in high_freq_words if any(space in w for space in ['space', 'orbit', 'satellite', 'rocket'])]
        if space_words:
            themes.append('space_technology')
            
        business_words = [w for w in high_freq_words if any(biz in w for biz in ['market', 'company', 'business', 'investment'])]
        if business_words:
            themes.append('business_analysis')
            
        tech_words = [w for w in high_freq_words if any(tech in w for tech in ['technology', 'innovation', 'development', 'research'])]
        if tech_words:
            themes.append('technology_trends')
            
        return themes
        
    def _calculate_readability_score(self, text: str) -> float:
        """Calculate readability score (simplified Flesch-Kincaid style)"""
        if not text or len(text.strip()) == 0:
            return 0.5
            
        sentences = [s for s in text.split('.') if s.strip()]
        words = text.split()
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.5
            
        avg_sentence_length = len(words) / len(sentences)
        
        # Count syllables (simplified - count vowel groups)
        total_syllables = 0
        for word in words:
            syllables = max(1, len(re.findall(r'[aeiouAEIOU]+', word)))
            total_syllables += syllables
            
        avg_syllables_per_word = total_syllables / len(words)
        
        # Simplified Flesch Reading Ease
        # 206.835 - (1.015 x ASL) - (84.6 x ASW)
        # Where ASL = average sentence length, ASW = average syllables per word
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Normalize to 0-1 scale (0 = very difficult, 1 = very easy)
        # Flesch scores typically range from 0-100
        normalized_score = max(0, min(100, flesch_score)) / 100
        
        return normalized_score
        
    def _calculate_insight_density(self, full_text: str, findings_text: str) -> float:
        """Calculate the density of insights and actionable content"""
        if not full_text:
            return 0.0
            
        total_words = len(full_text.split())
        insight_indicators = 0
        
        # Count insight keywords
        insight_keywords = ['analysis', 'insight', 'conclusion', 'finding', 'discovery', 
                           'trend', 'pattern', 'implication', 'significance', 'impact']
        
        full_text_lower = full_text.lower()
        for keyword in insight_keywords:
            insight_indicators += full_text_lower.count(keyword)
            
        # Bonus for structured findings
        if findings_text:
            findings_words = len(findings_text.split())
            insight_indicators += findings_words * 0.5  # Weight findings higher
            
        # Calculate density
        density = min(insight_indicators / max(total_words, 1) * 100, 1.0)
        
        return density
        
    def _calculate_source_diversity(self, content_text: str) -> int:
        """Estimate source diversity from content analysis"""
        if not content_text:
            return 0
            
        # Look for source indicators
        source_indicators = ['according to', 'source:', 'cited', 'reported by', 
                           'study by', 'research from', 'data from', 'via']
        
        diversity_count = 0
        content_lower = content_text.lower()
        
        for indicator in source_indicators:
            if indicator in content_lower:
                diversity_count += content_lower.count(indicator)
                
        return min(diversity_count, 10)  # Cap at reasonable number
        
    def _estimate_factual_accuracy(self, entity_mentions: Dict[str, int], text: str) -> float:
        """Estimate factual accuracy based on entity consistency and known facts"""
        if not entity_mentions and not text:
            return 0.5  # Default neutral score
            
        accuracy_score = 0.7  # Base score
        
        # Check entity mention consistency
        total_mentions = sum(entity_mentions.values())
        if total_mentions > 0:
            # Higher entity density often indicates more factual content
            entity_density = len(entity_mentions) / max(len(text.split()), 1) * 100
            if entity_density > 0.05:  # Good entity coverage
                accuracy_score += 0.1
            if entity_density > 0.1:  # Excellent entity coverage
                accuracy_score += 0.1
                
        # Check for factual language patterns
        factual_indicators = ['data shows', 'statistics', 'reported', 'measured', 
                            'according to', 'study found', 'research indicates']
        
        text_lower = text.lower() if text else ""
        factual_language_count = sum(text_lower.count(indicator) for indicator in factual_indicators)
        
        if factual_language_count > 0:
            accuracy_score += min(factual_language_count * 0.02, 0.2)
            
        return min(accuracy_score, 1.0)
        
    def _analyze_recommendations_quality(self, report: Report) -> float:
        """Analyze the quality of recommendations in the report"""
        if not report.recommendations:
            return 0.3  # Low score for missing recommendations
            
        try:
            recommendations = json.loads(report.recommendations)
        except:
            recommendations = [str(report.recommendations)]
            
        if not recommendations:
            return 0.3
            
        quality_score = 0.5  # Base score for having recommendations
        
        # Check recommendation characteristics
        for rec in recommendations:
            if isinstance(rec, str):
                rec_length = len(rec.split())
                
                # Good length recommendations (not too short or long)
                if 10 <= rec_length <= 50:
                    quality_score += 0.1
                    
                # Action-oriented language
                action_words = ['should', 'recommend', 'suggest', 'implement', 
                              'consider', 'develop', 'improve', 'focus on']
                if any(word in rec.lower() for word in action_words):
                    quality_score += 0.05
                    
                # Specific rather than generic
                specific_indicators = ['%', 'by', 'within', 'target', 'metric']
                if any(indicator in rec.lower() for indicator in specific_indicators):
                    quality_score += 0.05
                    
        # Average across recommendations
        quality_score = min(quality_score, 1.0)
        return quality_score
        
    async def analyze_batch_reports(self, report_ids: List[str]) -> Dict[str, ReportAnalysis]:
        """Analyze multiple reports in batch for efficiency"""
        analyses = {}
        
        for report_id in report_ids:
            try:
                analysis = await self.analyze_report(report_id)
                analyses[report_id] = analysis
            except Exception as e:
                print(f"Error analyzing report {report_id}: {e}")
                
        # Learn patterns from batch
        if len(analyses) >= self.config['learning_batch_size']:
            await self._learn_patterns_from_batch(list(analyses.values()))
            
        return analyses
        
    async def _learn_patterns_from_batch(self, analyses: List[ReportAnalysis]):
        """Learn patterns from a batch of report analyses"""
        try:
            # Quality patterns
            high_quality = [a for a in analyses if a.quality_score >= 0.8]
            if len(high_quality) >= 3:
                pattern_id = hashlib.md5(f"high_quality_pattern_{datetime.now()}".encode()).hexdigest()
                
                avg_complexity = sum(a.complexity_score for a in high_quality) / len(high_quality)
                common_themes = []
                theme_counter = Counter()
                for analysis in high_quality:
                    theme_counter.update(analysis.key_themes)
                common_themes = [theme for theme, count in theme_counter.most_common(3)]
                
                pattern = ReportPattern(
                    pattern_id=pattern_id,
                    pattern_type='high_quality_characteristics',
                    reports_analyzed=[a.report_id for a in high_quality],
                    pattern_data={
                        'avg_quality_score': sum(a.quality_score for a in high_quality) / len(high_quality),
                        'avg_complexity': avg_complexity,
                        'common_themes': common_themes,
                        'avg_word_count_indicator': sum(len(a.topic_coverage) for a in high_quality) / len(high_quality),
                        'avg_entity_mentions': sum(len(a.entity_mentions) for a in high_quality) / len(high_quality)
                    },
                    confidence_score=0.8,
                    frequency=len(high_quality),
                    discovered_at=datetime.now()
                )
                
                self.learned_patterns[pattern_id] = pattern
                
            # Topic coverage patterns
            topic_coverage = defaultdict(list)
            for analysis in analyses:
                for topic in analysis.topic_coverage:
                    topic_coverage[topic].append(analysis.quality_score)
                    
            effective_topics = {}
            for topic, scores in topic_coverage.items():
                if len(scores) >= 3:
                    avg_score = sum(scores) / len(scores)
                    if avg_score >= 0.7:
                        effective_topics[topic] = avg_score
                        
            if effective_topics:
                pattern_id = hashlib.md5(f"effective_topics_{datetime.now()}".encode()).hexdigest()
                pattern = ReportPattern(
                    pattern_id=pattern_id,
                    pattern_type='effective_topic_coverage',
                    reports_analyzed=[a.report_id for a in analyses],
                    pattern_data={
                        'effective_topics': effective_topics,
                        'top_topic': max(effective_topics.items(), key=lambda x: x[1])[0],
                        'topic_quality_correlation': True
                    },
                    confidence_score=0.75,
                    frequency=len(effective_topics),
                    discovered_at=datetime.now()
                )
                
                self.learned_patterns[pattern_id] = pattern
                
            print(f"Learned {len(self.learned_patterns)} patterns from batch analysis")
            
        except Exception as e:
            print(f"Error learning patterns from batch: {e}")
            
    async def get_improvement_suggestions(self, report_id: str) -> Dict[str, Any]:
        """Get suggestions for improving a specific report"""
        if report_id not in self.analysis_cache:
            await self.analyze_report(report_id)
            
        analysis = self.analysis_cache[report_id]
        suggestions = {
            'overall_score': analysis.quality_score,
            'areas_for_improvement': [],
            'specific_suggestions': [],
            'learned_recommendations': []
        }
        
        # Quality improvement suggestions
        if analysis.quality_score < 0.7:
            suggestions['areas_for_improvement'].append('overall_quality')
            if len(analysis.topic_coverage) < 3:
                suggestions['specific_suggestions'].append('Expand topic coverage to include more diverse themes')
            if analysis.insight_density < 0.3:
                suggestions['specific_suggestions'].append('Include more analytical insights and conclusions')
                
        # Complexity suggestions
        if analysis.complexity_score < 0.4:
            suggestions['areas_for_improvement'].append('content_depth')
            suggestions['specific_suggestions'].append('Add more technical detail and sophisticated analysis')
        elif analysis.complexity_score > 0.8 and analysis.readability_score < 0.5:
            suggestions['areas_for_improvement'].append('readability')
            suggestions['specific_suggestions'].append('Simplify language while maintaining technical accuracy')
            
        # Source diversity
        if analysis.source_diversity < 3:
            suggestions['areas_for_improvement'].append('source_diversity')
            suggestions['specific_suggestions'].append('Include more diverse sources and citations')
            
        # Entity coverage
        if len(analysis.entity_mentions) < 5:
            suggestions['areas_for_improvement'].append('entity_coverage')
            suggestions['specific_suggestions'].append('Reference more relevant entities and organizations')
            
        # Add learned recommendations
        for pattern in self.learned_patterns.values():
            if pattern.pattern_type == 'high_quality_characteristics':
                if analysis.quality_score < pattern.pattern_data['avg_quality_score']:
                    suggestions['learned_recommendations'].extend([
                        f"Target complexity score around {pattern.pattern_data['avg_complexity']:.2f}",
                        f"Include topics like: {', '.join(pattern.pattern_data['common_themes'])}",
                        f"Aim for {pattern.pattern_data['avg_entity_mentions']:.0f}+ entity mentions"
                    ])
                    
        return suggestions
        
    async def get_analyzer_status(self) -> Dict[str, Any]:
        """Get current status of the report analyzer"""
        return {
            'system_status': 'active',
            'reports_analyzed': len(self.analysis_cache),
            'patterns_learned': len(self.learned_patterns),
            'vocabulary_size': {
                'entities': len(self.entity_vocabulary),
                'topics': len(self.topic_vocabulary)
            },
            'quality_benchmarks': {
                level: {'min_score': data['min_score'], 'example_count': len(data['examples'])}
                for level, data in self.quality_benchmarks.items()
            },
            'analysis_config': self.config,
            'pattern_types': list(set([p.pattern_type for p in self.learned_patterns.values()]))
        }

# Global report analyzer instance
report_analyzer = ReportAnalyzer()
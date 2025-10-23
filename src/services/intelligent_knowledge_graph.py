#!/usr/bin/env python3
"""
INTELLIGENT KNOWLEDGE GRAPH - Phase 2 Implementation
AI-enhanced knowledge graph with learning and reasoning capabilities
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# ML and graph processing imports
try:
    import networkx as nx
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    import torch
    INTELLIGENT_GRAPH_AVAILABLE = True
except ImportError as e:
    logging.error(f"Intelligent knowledge graph dependencies not available: {e}")
    INTELLIGENT_GRAPH_AVAILABLE = False

from loguru import logger

class EntityType(Enum):
    """Enhanced entity types"""
    PERSON = "person"
    ORGANIZATION = "organization"
    TECHNOLOGY = "technology"
    CONCEPT = "concept"
    EVENT = "event"
    LOCATION = "location"
    PRODUCT = "product"
    FINANCIAL = "financial"
    SPACE_OBJECT = "space_object"
    RESEARCH = "research"

class RelationshipType(Enum):
    """Enhanced relationship types"""
    WORKS_FOR = "works_for"
    DEVELOPS = "develops"
    COMPETES_WITH = "competes_with"
    COLLABORATES_WITH = "collaborates_with"
    INFLUENCES = "influences"
    DEPENDS_ON = "depends_on"
    SIMILAR_TO = "similar_to"
    PART_OF = "part_of"
    CAUSES = "causes"
    ENABLES = "enables"

class ConfidenceLevel(Enum):
    """Confidence levels for relationships"""
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.9

@dataclass
class EnhancedEntity:
    """Enhanced entity with AI-generated features"""
    entity_id: str
    name: str
    entity_type: EntityType
    description: str
    properties: Dict[str, Any]
    embedding: Optional[List[float]]
    importance_score: float
    topic_cluster: Optional[int]
    created_at: datetime
    last_updated: datetime

@dataclass
class EnhancedRelationship:
    """Enhanced relationship with AI-inferred properties"""
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    confidence: float
    strength: float  # 0-1 scale
    context: str
    evidence: List[str]
    inferred: bool
    created_at: datetime
    last_verified: datetime

@dataclass
class GraphInsight:
    """AI-generated insight from graph analysis"""
    insight_type: str
    description: str
    entities_involved: List[str]
    confidence: float
    impact_score: float
    actionable: bool
    generated_at: datetime

@dataclass
class CommunityDetection:
    """Community/cluster detection result"""
    community_id: int
    entities: List[str]
    core_topic: str
    coherence_score: float
    representative_entities: List[str]

class IntelligentKnowledgeGraph:
    """AI-enhanced knowledge graph with learning capabilities"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, EnhancedEntity] = {}
        self.relationships: List[EnhancedRelationship] = []
        
        # AI components
        self.text_vectorizer = None
        self.entity_embeddings = {}
        self.topic_clusters = {}
        self.similarity_threshold = 0.7
        
        # Analysis cache
        self.insights_cache = []
        self.communities_cache = []
        self.last_analysis = None
        
        if INTELLIGENT_GRAPH_AVAILABLE:
            self._initialize_ai_components()
    
    def _initialize_ai_components(self):
        """Initialize AI components for graph intelligence"""
        try:
            logger.info("Initializing intelligent knowledge graph components...")
            
            # Text vectorizer for semantic similarity
            self.text_vectorizer = TfidfVectorizer(
                max_features=500,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            logger.info("Knowledge graph AI components initialized")
            
        except Exception as e:
            logger.error(f"Error initializing knowledge graph AI: {e}")
    
    async def add_entity(
        self, 
        name: str, 
        entity_type: EntityType, 
        description: str = "",
        properties: Dict[str, Any] = None
    ) -> str:
        """Add an enhanced entity to the knowledge graph"""
        
        entity_id = f"{entity_type.value}_{name.lower().replace(' ', '_')}"
        
        # Check if entity already exists
        if entity_id in self.entities:
            await self._update_entity(entity_id, description, properties or {})
            return entity_id
        
        # Generate embedding
        embedding = await self._generate_entity_embedding(name, description)
        
        # Calculate importance score
        importance_score = await self._calculate_importance_score(name, entity_type, description)
        
        entity = EnhancedEntity(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            description=description,
            properties=properties or {},
            embedding=embedding,
            importance_score=importance_score,
            topic_cluster=None,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        self.entities[entity_id] = entity
        self.graph.add_node(entity_id, **asdict(entity))
        
        logger.debug(f"Added entity: {name} ({entity_type.value})")
        return entity_id
    
    async def _update_entity(self, entity_id: str, description: str, properties: Dict[str, Any]):
        """Update existing entity with new information"""
        if entity_id not in self.entities:
            return
        
        entity = self.entities[entity_id]
        
        # Update description if more detailed
        if len(description) > len(entity.description):
            entity.description = description
            # Regenerate embedding
            entity.embedding = await self._generate_entity_embedding(entity.name, description)
        
        # Merge properties
        entity.properties.update(properties)
        entity.last_updated = datetime.now()
        
        # Update graph node
        self.graph.nodes[entity_id].update(asdict(entity))
    
    async def add_relationship(
        self,
        source_name: str,
        target_name: str,
        relationship_type: RelationshipType,
        context: str = "",
        evidence: List[str] = None,
        confidence: float = 0.8
    ) -> bool:
        """Add an enhanced relationship between entities"""
        
        # Find entities (fuzzy matching if needed)
        source_id = await self._find_entity_id(source_name)
        target_id = await self._find_entity_id(target_name)
        
        if not source_id or not target_id:
            logger.warning(f"Could not find entities for relationship: {source_name} -> {target_name}")
            return False
        
        # Calculate relationship strength
        strength = await self._calculate_relationship_strength(
            source_id, target_id, relationship_type, context
        )
        
        relationship = EnhancedRelationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            confidence=confidence,
            strength=strength,
            context=context,
            evidence=evidence or [],
            inferred=False,
            created_at=datetime.now(),
            last_verified=datetime.now()
        )
        
        self.relationships.append(relationship)
        self.graph.add_edge(
            source_id, 
            target_id, 
            relationship=relationship_type.value,
            weight=strength,
            **asdict(relationship)
        )
        
        logger.debug(f"Added relationship: {source_name} -{relationship_type.value}-> {target_name}")
        return True
    
    async def _find_entity_id(self, name: str) -> Optional[str]:
        """Find entity ID by name (with fuzzy matching)"""
        # Exact match first
        for entity_id, entity in self.entities.items():
            if entity.name.lower() == name.lower():
                return entity_id
        
        # Fuzzy match
        best_match = None
        best_score = 0
        
        for entity_id, entity in self.entities.items():
            # Simple similarity score
            similarity = len(set(name.lower().split()) & set(entity.name.lower().split()))
            if similarity > best_score:
                best_score = similarity
                best_match = entity_id
        
        return best_match if best_score > 0 else None
    
    async def _generate_entity_embedding(self, name: str, description: str) -> List[float]:
        """Generate semantic embedding for entity"""
        if not INTELLIGENT_GRAPH_AVAILABLE or not self.text_vectorizer:
            return []
        
        try:
            # Combine name and description
            text = f"{name} {description}".strip()
            
            if not text:
                return []
            
            # Simple TF-IDF embedding (in production, would use transformer models)
            if hasattr(self.text_vectorizer, 'vocabulary_'):
                # Vectorizer is already fitted
                embedding = self.text_vectorizer.transform([text])
            else:
                # First entity - fit and transform
                embedding = self.text_vectorizer.fit_transform([text])
            
            return embedding.toarray()[0].tolist()
            
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return []
    
    async def _calculate_importance_score(
        self, 
        name: str, 
        entity_type: EntityType, 
        description: str
    ) -> float:
        """Calculate importance score for entity"""
        
        score = 0.5  # Base score
        
        # Type-based scoring
        type_scores = {
            EntityType.PERSON: 0.8,
            EntityType.ORGANIZATION: 0.9,
            EntityType.TECHNOLOGY: 0.7,
            EntityType.CONCEPT: 0.6,
            EntityType.EVENT: 0.7
        }
        score += type_scores.get(entity_type, 0.5) * 0.3
        
        # Name-based scoring (longer names often more specific/important)
        name_factor = min(1.0, len(name.split()) / 5.0)
        score += name_factor * 0.1
        
        # Description-based scoring
        desc_factor = min(1.0, len(description.split()) / 20.0)
        score += desc_factor * 0.1
        
        return min(1.0, score)
    
    async def _calculate_relationship_strength(
        self,
        source_id: str,
        target_id: str,
        relationship_type: RelationshipType,
        context: str
    ) -> float:
        """Calculate strength of relationship"""
        
        strength = 0.5  # Base strength
        
        # Type-based strength
        strong_relations = [
            RelationshipType.WORKS_FOR,
            RelationshipType.DEVELOPS,
            RelationshipType.PART_OF
        ]
        
        if relationship_type in strong_relations:
            strength += 0.3
        
        # Context-based strength
        if context:
            context_words = context.lower().split()
            strong_indicators = ['directly', 'primarily', 'exclusively', 'founded', 'created', 'owns']
            weak_indicators = ['possibly', 'might', 'could', 'reportedly', 'allegedly']
            
            for word in strong_indicators:
                if word in context_words:
                    strength += 0.1
            
            for word in weak_indicators:
                if word in context_words:
                    strength -= 0.1
        
        return min(1.0, max(0.1, strength))
    
    async def infer_missing_relationships(self) -> List[EnhancedRelationship]:
        """Use AI to infer missing relationships"""
        if not INTELLIGENT_GRAPH_AVAILABLE:
            return []
        
        inferred_relationships = []
        
        try:
            # Get all entity pairs without direct relationships
            entity_ids = list(self.entities.keys())
            
            for i, source_id in enumerate(entity_ids):
                for target_id in entity_ids[i+1:]:
                    
                    # Skip if relationship already exists
                    if self.graph.has_edge(source_id, target_id):
                        continue
                    
                    # Infer relationship based on similarity and properties
                    relationship = await self._infer_relationship(source_id, target_id)
                    
                    if relationship and relationship.confidence > 0.6:
                        relationship.inferred = True
                        inferred_relationships.append(relationship)
                        
                        # Add to graph
                        self.relationships.append(relationship)
                        self.graph.add_edge(
                            source_id, 
                            target_id,
                            relationship=relationship.relationship_type.value,
                            weight=relationship.strength,
                            **asdict(relationship)
                        )
            
            logger.info(f"Inferred {len(inferred_relationships)} new relationships")
            return inferred_relationships
            
        except Exception as e:
            logger.error(f"Error inferring relationships: {e}")
            return []
    
    async def _infer_relationship(self, source_id: str, target_id: str) -> Optional[EnhancedRelationship]:
        """Infer relationship between two entities"""
        
        source_entity = self.entities[source_id]
        target_entity = self.entities[target_id]
        
        # Same type entities might be similar
        if source_entity.entity_type == target_entity.entity_type:
            # Calculate semantic similarity
            similarity = await self._calculate_semantic_similarity(source_entity, target_entity)
            
            if similarity > self.similarity_threshold:
                return EnhancedRelationship(
                    source_id=source_id,
                    target_id=target_id,
                    relationship_type=RelationshipType.SIMILAR_TO,
                    confidence=similarity,
                    strength=similarity,
                    context=f"Inferred from semantic similarity ({similarity:.2f})",
                    evidence=["semantic_similarity"],
                    inferred=True,
                    created_at=datetime.now(),
                    last_verified=datetime.now()
                )
        
        # Organization-Person relationships
        if (source_entity.entity_type == EntityType.ORGANIZATION and 
            target_entity.entity_type == EntityType.PERSON):
            
            # Look for name mentions in descriptions
            if target_entity.name.lower() in source_entity.description.lower():
                return EnhancedRelationship(
                    source_id=target_id,  # Person works for organization
                    target_id=source_id,
                    relationship_type=RelationshipType.WORKS_FOR,
                    confidence=0.7,
                    strength=0.7,
                    context="Inferred from entity description mention",
                    evidence=["description_mention"],
                    inferred=True,
                    created_at=datetime.now(),
                    last_verified=datetime.now()
                )
        
        return None
    
    async def _calculate_semantic_similarity(
        self, 
        entity1: EnhancedEntity, 
        entity2: EnhancedEntity
    ) -> float:
        """Calculate semantic similarity between entities"""
        
        if not entity1.embedding or not entity2.embedding:
            return 0.0
        
        try:
            # Cosine similarity
            embedding1 = np.array(entity1.embedding).reshape(1, -1)
            embedding2 = np.array(entity2.embedding).reshape(1, -1)
            
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Failed to calculate semantic similarity: {e}")
            return 0.0
    
    async def detect_communities(self) -> List[CommunityDetection]:
        """Detect communities/clusters in the knowledge graph"""
        if not INTELLIGENT_GRAPH_AVAILABLE:
            return []
        
        try:
            # Use NetworkX community detection
            communities = nx.community.greedy_modularity_communities(self.graph.to_undirected())
            
            community_results = []
            
            for i, community in enumerate(communities):
                if len(community) < 2:  # Skip single-node communities
                    continue
                
                entity_names = [self.entities[entity_id].name for entity_id in community if entity_id in self.entities]
                
                # Find core topic using most common entity type or properties
                entity_types = [self.entities[entity_id].entity_type for entity_id in community if entity_id in self.entities]
                core_topic = max(set(entity_types), key=entity_types.count).value if entity_types else "unknown"
                
                # Calculate coherence (average edge weight within community)
                coherence_score = 0.0
                edge_count = 0
                for u, v in self.graph.edges(community):
                    if u in community and v in community:
                        weight = self.graph[u][v].get('weight', 0.5)
                        coherence_score += weight
                        edge_count += 1
                
                if edge_count > 0:
                    coherence_score /= edge_count
                
                # Select representative entities (highest importance scores)
                community_entities = [(entity_id, self.entities[entity_id].importance_score) 
                                    for entity_id in community if entity_id in self.entities]
                community_entities.sort(key=lambda x: x[1], reverse=True)
                representative_entities = [self.entities[entity_id].name for entity_id, _ in community_entities[:3]]
                
                community_result = CommunityDetection(
                    community_id=i,
                    entities=list(community),
                    core_topic=core_topic,
                    coherence_score=coherence_score,
                    representative_entities=representative_entities
                )
                
                community_results.append(community_result)
            
            self.communities_cache = community_results
            logger.info(f"Detected {len(community_results)} communities")
            
            return community_results
            
        except Exception as e:
            logger.error(f"Error detecting communities: {e}")
            return []
    
    async def generate_insights(self) -> List[GraphInsight]:
        """Generate AI insights from graph analysis"""
        insights = []
        
        try:
            # Network structure insights
            centrality_insight = await self._analyze_centrality()
            if centrality_insight:
                insights.append(centrality_insight)
            
            # Community insights
            community_insights = await self._analyze_communities()
            insights.extend(community_insights)
            
            # Growth pattern insights
            growth_insight = await self._analyze_growth_patterns()
            if growth_insight:
                insights.append(growth_insight)
            
            # Relationship pattern insights
            relationship_insights = await self._analyze_relationship_patterns()
            insights.extend(relationship_insights)
            
            self.insights_cache = insights
            self.last_analysis = datetime.now()
            
            logger.info(f"Generated {len(insights)} graph insights")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return []
    
    async def _analyze_centrality(self) -> Optional[GraphInsight]:
        """Analyze network centrality to find key entities"""
        try:
            centrality = nx.degree_centrality(self.graph)
            
            if not centrality:
                return None
            
            # Find most central entity
            most_central = max(centrality, key=centrality.get)
            centrality_score = centrality[most_central]
            
            if most_central in self.entities:
                entity_name = self.entities[most_central].name
                
                return GraphInsight(
                    insight_type="centrality_analysis",
                    description=f"{entity_name} is the most connected entity in the knowledge graph with {centrality_score:.1%} centrality",
                    entities_involved=[most_central],
                    confidence=0.9,
                    impact_score=centrality_score,
                    actionable=True,
                    generated_at=datetime.now()
                )
            
        except Exception as e:
            logger.warning(f"Centrality analysis failed: {e}")
        
        return None
    
    async def _analyze_communities(self) -> List[GraphInsight]:
        """Analyze detected communities for insights"""
        insights = []
        
        if not self.communities_cache:
            await self.detect_communities()
        
        for community in self.communities_cache:
            if community.coherence_score > 0.7:
                insight = GraphInsight(
                    insight_type="strong_community",
                    description=f"Strong {community.core_topic} community detected with {len(community.entities)} entities and {community.coherence_score:.1%} coherence",
                    entities_involved=community.entities[:5],  # Top 5 entities
                    confidence=community.coherence_score,
                    impact_score=len(community.entities) / len(self.entities) if self.entities else 0,
                    actionable=True,
                    generated_at=datetime.now()
                )
                insights.append(insight)
        
        return insights
    
    async def _analyze_growth_patterns(self) -> Optional[GraphInsight]:
        """Analyze temporal growth patterns"""
        if len(self.entities) < 10:
            return None
        
        # Analyze entity creation dates
        creation_dates = [entity.created_at for entity in self.entities.values()]
        recent_entities = [d for d in creation_dates if d > datetime.now() - timedelta(days=7)]
        
        if len(recent_entities) > len(creation_dates) * 0.5:  # More than 50% created recently
            return GraphInsight(
                insight_type="rapid_growth",
                description=f"Rapid knowledge graph growth: {len(recent_entities)} entities added in the last 7 days ({len(recent_entities)/len(creation_dates):.1%} of total)",
                entities_involved=[],
                confidence=0.8,
                impact_score=len(recent_entities) / len(creation_dates),
                actionable=True,
                generated_at=datetime.now()
            )
        
        return None
    
    async def _analyze_relationship_patterns(self) -> List[GraphInsight]:
        """Analyze relationship patterns for insights"""
        insights = []
        
        if not self.relationships:
            return insights
        
        # Analyze relationship types distribution
        relationship_counts = {}
        for rel in self.relationships:
            rel_type = rel.relationship_type.value
            relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1
        
        # Find dominant relationship type
        if relationship_counts:
            dominant_type = max(relationship_counts, key=relationship_counts.get)
            dominant_count = relationship_counts[dominant_type]
            
            if dominant_count > len(self.relationships) * 0.4:  # More than 40% of relationships
                insight = GraphInsight(
                    insight_type="dominant_relationship",
                    description=f"'{dominant_type}' relationships dominate the graph ({dominant_count} out of {len(self.relationships)} relationships)",
                    entities_involved=[],
                    confidence=0.7,
                    impact_score=dominant_count / len(self.relationships),
                    actionable=False,
                    generated_at=datetime.now()
                )
                insights.append(insight)
        
        return insights
    
    async def get_entity_recommendations(self, entity_id: str, max_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Get entity recommendations based on graph structure"""
        if entity_id not in self.entities:
            return []
        
        recommendations = []
        
        try:
            # Get entities 2 hops away (friends of friends)
            two_hop_entities = set()
            
            # Get direct neighbors
            neighbors = set(self.graph.neighbors(entity_id))
            
            # Get neighbors of neighbors
            for neighbor in neighbors:
                two_hop_entities.update(self.graph.neighbors(neighbor))
            
            # Remove direct neighbors and self
            two_hop_entities -= neighbors
            two_hop_entities.discard(entity_id)
            
            # Calculate recommendation scores
            for candidate_id in two_hop_entities:
                if candidate_id not in self.entities:
                    continue
                
                candidate = self.entities[candidate_id]
                
                # Calculate score based on shared neighbors and similarity
                shared_neighbors = neighbors & set(self.graph.neighbors(candidate_id))
                shared_count = len(shared_neighbors)
                
                # Semantic similarity if embeddings available
                source_entity = self.entities[entity_id]
                similarity = await self._calculate_semantic_similarity(source_entity, candidate)
                
                score = (shared_count * 0.6) + (similarity * 0.4) + (candidate.importance_score * 0.2)
                
                recommendations.append({
                    'entity_id': candidate_id,
                    'name': candidate.name,
                    'type': candidate.entity_type.value,
                    'score': score,
                    'shared_connections': shared_count,
                    'similarity': similarity,
                    'reason': f"Connected through {shared_count} mutual connections"
                })
            
            # Sort by score and return top recommendations
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations[:max_recommendations]
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        stats = {
            'total_entities': len(self.entities),
            'total_relationships': len(self.relationships),
            'graph_density': nx.density(self.graph) if len(self.entities) > 1 else 0,
            'average_degree': sum(dict(self.graph.degree()).values()) / len(self.entities) if self.entities else 0,
            'entity_types': {},
            'relationship_types': {},
            'communities': len(self.communities_cache),
            'insights_generated': len(self.insights_cache),
            'last_analysis': self.last_analysis
        }
        
        # Entity type distribution
        for entity in self.entities.values():
            entity_type = entity.entity_type.value
            stats['entity_types'][entity_type] = stats['entity_types'].get(entity_type, 0) + 1
        
        # Relationship type distribution
        for relationship in self.relationships:
            rel_type = relationship.relationship_type.value
            stats['relationship_types'][rel_type] = stats['relationship_types'].get(rel_type, 0) + 1
        
        return stats

# Global instance
intelligent_knowledge_graph = IntelligentKnowledgeGraph()

def get_intelligent_knowledge_graph() -> IntelligentKnowledgeGraph:
    """Get the global intelligent knowledge graph instance"""
    return intelligent_knowledge_graph
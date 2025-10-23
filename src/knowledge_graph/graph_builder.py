"""
Knowledge Graph Builder for Research Synthesis Engine
"""
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional
import json
from datetime import datetime
from loguru import logger
import numpy as np
from collections import defaultdict


class KnowledgeGraphBuilder:
    """Build and manage knowledge graphs from extracted entities and relationships"""
    
    def __init__(self):
        """Initialize knowledge graph"""
        self.graph = nx.DiGraph()
        self.entity_index = {}
        self.relationship_types = set()
        self.space_entities = set()
        
    def add_article_to_graph(self, article: Dict[str, Any]):
        """Add article entities and relationships to the knowledge graph"""
        article_id = article.get('id')
        entities = article.get('entities', [])
        
        # Add article node
        self.graph.add_node(
            f"article_{article_id}",
            node_type='article',
            title=article.get('title'),
            url=article.get('url'),
            date=article.get('published_date'),
            source=article.get('source')
        )
        
        # Process entities
        for entity in entities:
            self.add_entity(entity, article_id)
        
        # Extract and add relationships
        relationships = self.extract_relationships(article)
        for rel in relationships:
            self.add_relationship(rel, article_id)
        
        # Connect article to its entities
        for entity in entities:
            entity_id = self.get_entity_id(entity)
            self.graph.add_edge(
                f"article_{article_id}",
                entity_id,
                relationship='mentions',
                weight=1
            )
    
    def add_entity(self, entity: Dict[str, Any], article_id: str):
        """Add entity to the graph"""
        entity_id = self.get_entity_id(entity)
        
        if entity_id not in self.graph:
            # Add new entity node
            self.graph.add_node(
                entity_id,
                node_type='entity',
                entity_type=entity.get('type'),
                text=entity.get('text'),
                first_seen=datetime.utcnow().isoformat(),
                mention_count=1,
                articles=[article_id]
            )
            
            # Check if space-related
            if self.is_space_entity(entity):
                self.space_entities.add(entity_id)
        else:
            # Update existing entity
            node_data = self.graph.nodes[entity_id]
            node_data['mention_count'] += 1
            if article_id not in node_data['articles']:
                node_data['articles'].append(article_id)
        
        # Update entity index
        self.entity_index[entity.get('text', '').lower()] = entity_id
    
    def extract_relationships(self, article: Dict[str, Any]) -> List[Dict]:
        """Extract relationships between entities from article content"""
        relationships = []
        content = article.get('content', '')
        entities = article.get('entities', [])
        
        # Relationship patterns
        patterns = {
            'partnership': ['partnered with', 'partnership with', 'collaboration with', 'joint venture'],
            'acquisition': ['acquired', 'bought', 'purchased', 'takeover'],
            'investment': ['invested in', 'funding from', 'raised', 'secured funding'],
            'competition': ['competes with', 'rival', 'competitor', 'competing'],
            'supply': ['supplies', 'provides', 'delivers to', 'contractor for'],
            'customer': ['customer of', 'client', 'uses', 'purchases from'],
            'subsidiary': ['subsidiary of', 'owned by', 'division of', 'part of'],
            'leadership': ['CEO of', 'founded', 'leads', 'heads', 'director of']
        }
        
        # Find relationships between entities
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Check if entities appear near each other
                text1 = entity1.get('text', '')
                text2 = entity2.get('text', '')
                
                # Find sentences containing both entities
                sentences = content.split('.')
                for sentence in sentences:
                    if text1 in sentence and text2 in sentence:
                        # Check for relationship patterns
                        for rel_type, keywords in patterns.items():
                            for keyword in keywords:
                                if keyword in sentence.lower():
                                    relationships.append({
                                        'entity1': entity1,
                                        'entity2': entity2,
                                        'type': rel_type,
                                        'evidence': sentence.strip(),
                                        'confidence': 0.8
                                    })
                                    break
        
        return relationships
    
    def add_relationship(self, relationship: Dict[str, Any], article_id: str):
        """Add relationship between entities"""
        entity1_id = self.get_entity_id(relationship['entity1'])
        entity2_id = self.get_entity_id(relationship['entity2'])
        rel_type = relationship.get('type', 'related')
        
        # Add or update edge
        if self.graph.has_edge(entity1_id, entity2_id):
            # Update existing relationship
            edge_data = self.graph[entity1_id][entity2_id]
            edge_data['weight'] = edge_data.get('weight', 0) + 1
            edge_data['articles'].append(article_id)
            edge_data['evidence'].append(relationship.get('evidence', ''))
        else:
            # Add new relationship
            self.graph.add_edge(
                entity1_id,
                entity2_id,
                relationship=rel_type,
                weight=1,
                confidence=relationship.get('confidence', 0.5),
                articles=[article_id],
                evidence=[relationship.get('evidence', '')]
            )
        
        self.relationship_types.add(rel_type)
    
    def get_entity_id(self, entity: Dict[str, Any]) -> str:
        """Generate unique entity ID"""
        text = entity.get('text', '')
        entity_type = entity.get('type', 'unknown')
        return f"entity_{entity_type}_{text.lower().replace(' ', '_')}"
    
    def is_space_entity(self, entity: Dict[str, Any]) -> bool:
        """Check if entity is space-related"""
        space_keywords = [
            'space', 'satellite', 'rocket', 'orbit', 'launch',
            'nasa', 'esa', 'spacex', 'blue origin', 'virgin galactic',
            'iss', 'mars', 'moon', 'asteroid', 'spacecraft'
        ]
        
        text = entity.get('text', '').lower()
        return any(keyword in text for keyword in space_keywords)
    
    def find_related_entities(self, entity_text: str, max_distance: int = 2) -> List[Dict]:
        """Find entities related to a given entity"""
        entity_id = self.entity_index.get(entity_text.lower())
        
        if not entity_id or entity_id not in self.graph:
            return []
        
        related = []
        
        # Find directly connected entities
        for neighbor in self.graph.neighbors(entity_id):
            if neighbor.startswith('entity_'):
                edge_data = self.graph[entity_id][neighbor]
                node_data = self.graph.nodes[neighbor]
                
                related.append({
                    'entity': node_data.get('text'),
                    'type': node_data.get('entity_type'),
                    'relationship': edge_data.get('relationship'),
                    'weight': edge_data.get('weight'),
                    'distance': 1
                })
        
        # Find entities within max_distance
        if max_distance > 1:
            distances = nx.single_source_shortest_path_length(
                self.graph, entity_id, cutoff=max_distance
            )
            
            for node, distance in distances.items():
                if node.startswith('entity_') and distance > 1:
                    node_data = self.graph.nodes[node]
                    related.append({
                        'entity': node_data.get('text'),
                        'type': node_data.get('entity_type'),
                        'relationship': 'indirect',
                        'distance': distance
                    })
        
        return sorted(related, key=lambda x: (x['distance'], -x.get('weight', 0)))
    
    def get_entity_timeline(self, entity_text: str) -> List[Dict]:
        """Get timeline of mentions for an entity"""
        entity_id = self.entity_index.get(entity_text.lower())
        
        if not entity_id or entity_id not in self.graph:
            return []
        
        node_data = self.graph.nodes[entity_id]
        article_ids = node_data.get('articles', [])
        
        timeline = []
        for article_id in article_ids:
            article_node = f"article_{article_id}"
            if article_node in self.graph:
                article_data = self.graph.nodes[article_node]
                timeline.append({
                    'date': article_data.get('date'),
                    'title': article_data.get('title'),
                    'source': article_data.get('source'),
                    'url': article_data.get('url')
                })
        
        # Sort by date
        timeline.sort(key=lambda x: x.get('date', ''))
        return timeline
    
    def identify_key_entities(self, min_mentions: int = 5) -> List[Dict]:
        """Identify most important entities in the graph"""
        key_entities = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('node_type') == 'entity':
                mention_count = node_data.get('mention_count', 0)
                
                if mention_count >= min_mentions:
                    # Calculate importance metrics
                    degree = self.graph.degree(node_id)
                    betweenness = nx.betweenness_centrality(self.graph).get(node_id, 0)
                    pagerank = nx.pagerank(self.graph).get(node_id, 0)
                    
                    key_entities.append({
                        'entity': node_data.get('text'),
                        'type': node_data.get('entity_type'),
                        'mentions': mention_count,
                        'connections': degree,
                        'betweenness': betweenness,
                        'pagerank': pagerank,
                        'importance_score': mention_count * 0.3 + degree * 0.3 + betweenness * 20 + pagerank * 100
                    })
        
        # Sort by importance score
        key_entities.sort(key=lambda x: x['importance_score'], reverse=True)
        return key_entities
    
    def detect_communities(self) -> List[List[str]]:
        """Detect communities/clusters in the knowledge graph"""
        # Convert to undirected graph for community detection
        undirected = self.graph.to_undirected()
        
        # Filter to only entity nodes
        entity_graph = undirected.subgraph(
            [n for n in undirected.nodes() if n.startswith('entity_')]
        )
        
        if len(entity_graph) == 0:
            return []
        
        # Detect communities using Louvain method
        import community
        communities = community.best_partition(entity_graph)
        
        # Group entities by community
        community_groups = defaultdict(list)
        for node, comm_id in communities.items():
            node_data = self.graph.nodes[node]
            community_groups[comm_id].append(node_data.get('text'))
        
        return list(community_groups.values())
    
    def find_trends(self, time_window: int = 30) -> List[Dict]:
        """Identify trending entities and topics"""
        from datetime import datetime, timedelta
        
        trends = []
        current_date = datetime.utcnow()
        window_start = current_date - timedelta(days=time_window)
        
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('node_type') == 'entity':
                # Count recent mentions
                recent_mentions = 0
                article_ids = node_data.get('articles', [])
                
                for article_id in article_ids:
                    article_node = f"article_{article_id}"
                    if article_node in self.graph:
                        article_date = self.graph.nodes[article_node].get('date')
                        if article_date:
                            try:
                                date_obj = datetime.fromisoformat(article_date)
                                if date_obj >= window_start:
                                    recent_mentions += 1
                            except:
                                pass
                
                if recent_mentions > 0:
                    trends.append({
                        'entity': node_data.get('text'),
                        'type': node_data.get('entity_type'),
                        'recent_mentions': recent_mentions,
                        'total_mentions': node_data.get('mention_count', 0),
                        'trend_score': recent_mentions / max(time_window, 1)
                    })
        
        # Sort by trend score
        trends.sort(key=lambda x: x['trend_score'], reverse=True)
        return trends[:50]  # Return top 50 trends
    
    def export_graph(self, format: str = 'json') -> Any:
        """Export knowledge graph in various formats"""
        if format == 'json':
            # Convert to JSON-serializable format
            data = {
                'nodes': [],
                'edges': []
            }
            
            for node_id, node_data in self.graph.nodes(data=True):
                data['nodes'].append({
                    'id': node_id,
                    **node_data
                })
            
            for source, target, edge_data in self.graph.edges(data=True):
                data['edges'].append({
                    'source': source,
                    'target': target,
                    **edge_data
                })
            
            return data
        
        elif format == 'gexf':
            # Export as GEXF for Gephi
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.gexf', delete=False) as f:
                nx.write_gexf(self.graph, f.name)
                return f.name
        
        elif format == 'graphml':
            # Export as GraphML
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.graphml', delete=False) as f:
                nx.write_graphml(self.graph, f.name)
                return f.name
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        entity_nodes = [n for n in self.graph.nodes() if n.startswith('entity_')]
        article_nodes = [n for n in self.graph.nodes() if n.startswith('article_')]
        
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'entity_count': len(entity_nodes),
            'article_count': len(article_nodes),
            'relationship_types': list(self.relationship_types),
            'space_entity_count': len(self.space_entities),
            'graph_density': nx.density(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1),
            'connected_components': nx.number_weakly_connected_components(self.graph)
        }
        
        # Add top entities
        key_entities = self.identify_key_entities(min_mentions=3)
        stats['top_entities'] = key_entities[:10]
        
        return stats
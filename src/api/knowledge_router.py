from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, func
import json

router = APIRouter()

# Database connection
try:
    from research_synthesis.database.connection import get_postgres_session
    from research_synthesis.database.models import Entity, EntityRelationship, Article
    DATABASE_AVAILABLE = True
except ImportError as e:
    print(f"Database not available: {e}")
    DATABASE_AVAILABLE = False

class EntityInfo(BaseModel):
    name: str
    type: str
    description: str
    relationships: List[Dict[str, Any]]
    metadata: Dict[str, Any]

def generate_logical_relationships(nodes):
    """Generate meaningful business and technical relationships between entities"""
    edges = []
    relationships_added = set()  # Avoid duplicates
    
    # Create entity lookup for easier access
    node_by_id = {node["id"]: node for node in nodes}
    
    # Define comprehensive relationship rules with business context
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes[i+1:], i+1):  # Avoid self-connections and duplicates
            relationship_key = f"{min(node1['id'], node2['id'])}_{max(node1['id'], node2['id'])}"
            if relationship_key in relationships_added:
                continue
                
            relationship_type = None
            confidence = 0.0
            relationship_context = ""
            
            name1, name2 = node1["name"].lower(), node2["name"].lower()
            type1, type2 = node1["type"], node2["type"]
            
            # 1. SUPPLY CHAIN & MANUFACTURING RELATIONSHIPS
            if any(keyword in name1 for keyword in ["supplier", "manufacturer", "factory"]) and type2 == "company":
                relationship_type = "supplies_to"
                confidence = 0.8
                relationship_context = "Supply chain partnership"
            elif any(keyword in name2 for keyword in ["supplier", "manufacturer", "factory"]) and type1 == "company":
                relationship_type = "sources_from"
                confidence = 0.8
                relationship_context = "Supply chain dependency"
            
            # 2. INVESTMENT & FUNDING RELATIONSHIPS
            elif any(keyword in name1 for keyword in ["venture", "capital", "fund", "investment"]) and type2 == "company":
                relationship_type = "invests_in"
                confidence = 0.9
                relationship_context = "Financial backing"
            elif any(keyword in name1 for keyword in ["startup", "founded"]) and type2 == "person":
                relationship_type = "funded_by"
                confidence = 0.7
                relationship_context = "Startup funding"
            
            # 3. TECHNOLOGY & R&D PARTNERSHIPS
            elif any(tech in name1 for tech in ["ai", "automation", "robotics", "iot", "blockchain"]) and /
                 any(tech in name2 for tech in ["ai", "automation", "robotics", "iot", "blockchain"]):
                relationship_type = "collaborates_on_tech"
                confidence = 0.8
                relationship_context = "Technology development partnership"
            
            # 4. MARKET COMPETITION ANALYSIS
            elif (node1.get("mention_count", 0) >= 3 and node2.get("mention_count", 0) >= 3 and 
                  type1 == "company" and type2 == "company" and 
                  node1.get("space_sector") == node2.get("space_sector")):
                if node1.get("space_sector") in ["aerospace", "manufacturing", "defense"]:
                    relationship_type = "competes_in_market"
                    confidence = 0.7
                    relationship_context = f"Market competition in {node1.get('space_sector', 'sector')}"
                else:
                    relationship_type = "operates_in_same_sector" 
                    confidence = 0.6
                    relationship_context = "Same industry operations"
            
            # 5. LEADERSHIP & EXECUTIVE CONNECTIONS
            elif type1 == "person" and type2 == "company":
                if any(title in name1 for title in ["ceo", "cto", "founder", "president", "director"]):
                    relationship_type = "leads"
                    confidence = 0.9
                    relationship_context = "Executive leadership"
                elif any(title in name1 for title in ["engineer", "scientist", "researcher"]):
                    relationship_type = "contributes_expertise"
                    confidence = 0.7
                    relationship_context = "Technical expertise"
                else:
                    relationship_type = "employed_by"
                    confidence = 0.6
                    relationship_context = "Employment relationship"
            
            # 6. GEOGRAPHIC & OPERATIONAL HUBS
            elif type1 == "location" and type2 == "company":
                if any(keyword in name1 for keyword in ["silicon valley", "boston", "seattle", "austin"]):
                    relationship_type = "tech_hub_for"
                    confidence = 0.8
                    relationship_context = "Technology cluster location"
                else:
                    relationship_type = "headquarters_of"
                    confidence = 0.7
                    relationship_context = "Business location"
            
            # 7. INDUSTRY-SPECIFIC PARTNERSHIPS
            elif ("space" in name1 or "aerospace" in name1) and ("space" in name2 or "aerospace" in name2):
                relationship_type = "space_industry_partners"
                confidence = 0.8
                relationship_context = "Aerospace industry collaboration"
            elif ("manufacturing" in name1 and "automation" in name2) or ("automation" in name1 and "manufacturing" in name2):
                relationship_type = "automation_integration"
                confidence = 0.8
                relationship_context = "Manufacturing automation partnership"
            
            # 8. RESEARCH & ACADEMIC CONNECTIONS
            elif any(inst in name1 for inst in ["university", "institute", "lab", "research"]) and type2 == "company":
                relationship_type = "research_collaboration"
                confidence = 0.7
                relationship_context = "Academic-industry partnership"
            
            # 9. GOVERNMENT & REGULATORY RELATIONSHIPS
            elif any(gov in name1 for gov in ["nasa", "faa", "dot", "government"]) and type2 == "company":
                relationship_type = "regulatory_oversight"
                confidence = 0.8
                relationship_context = "Government regulation/partnership"
            
            # 10. PRODUCT & SERVICE ECOSYSTEMS
            elif any(product in name1 for product in ["software", "platform", "service"]) and /
                 any(product in name2 for product in ["software", "platform", "service"]):
                relationship_type = "ecosystem_partners"
                confidence = 0.6
                relationship_context = "Product/service ecosystem"
                
            # 11. SAME SECTOR COLLABORATION (Enhanced)
            elif node1.get("space_sector") == node2.get("space_sector") and node1["space_sector"] != "general":
                sector = node1["space_sector"]
                if sector in ["defense", "aerospace"]:
                    relationship_type = "defense_contractors"
                    confidence = 0.8
                    relationship_context = f"Defense/aerospace contracting network"
                elif sector == "manufacturing":
                    relationship_type = "manufacturing_network"
                    confidence = 0.7
                    relationship_context = "Manufacturing industry network"
                else:
                    relationship_type = "industry_peers"
                    confidence = 0.6
                    relationship_context = f"{sector.title()} industry peers"
            
            # Add relationship if one was determined with sufficient confidence
            if relationship_type and confidence > 0.5:  # Raised threshold for more meaningful relationships
                edges.append({
                    "source": node1["id"],
                    "target": node2["id"],
                    "relationship": relationship_type,
                    "confidence": confidence,
                    "context": relationship_context,
                    "active": True,
                    "business_value": "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
                })
                relationships_added.add(relationship_key)
                
                # Limit to prevent too many relationships but allow more meaningful ones
                if len(edges) >= 35:  # Increased limit for richer relationships
                    break
        
        if len(edges) >= 25:
            break
    
    return edges

@router.get("/")
async def get_knowledge_status():
    """Get knowledge graph status"""
    if DATABASE_AVAILABLE:
        try:
            async for db_session in get_postgres_session():
                # Get real counts from database
                # Note: For SQLite with SQLAlchemy 2.0, we need to use execute instead of query
                from sqlalchemy import select
                
                total_entities_result = await db_session.execute(select(func.count(Entity.id)))
                total_entities = total_entities_result.scalar() or 0
                
                total_relationships_result = await db_session.execute(select(func.count(EntityRelationship.id)))
                total_relationships = total_relationships_result.scalar() or 0
                
                # Get entity types distribution
                entity_types_query = await db_session.execute(
                    select(Entity.entity_type, func.count(Entity.id).label('count'))
                    .group_by(Entity.entity_type)
                )
                
                entity_types = [{"type": et[0], "count": et[1]} for et in entity_types_query.fetchall()]
                
                return {
                    "status": "Knowledge Graph API active",
                    "service": "ready", 
                    "total_entities": total_entities,
                    "total_relationships": total_relationships,
                    "entity_types": ["companies", "technologies", "people", "locations", "products"],
                    "entity_distribution": entity_types
                }
        except Exception as e:
            print(f"Database query error: {e}")
    
    # Fallback to static response
    return {
        "status": "Knowledge Graph API active",
        "service": "ready",
        "total_entities": 0,
        "total_relationships": 0,
        "entity_types": [
            "companies",
            "technologies", 
            "people",
            "locations",
            "products"
        ]
    }

@router.get("/entities/top")
async def get_top_entities(limit: int = 20):
    """Get top entities by mention count, grouped by type with caching and profiling"""
    from research_synthesis.utils.cache_manager import cache_manager
    from research_synthesis.utils.profiler import profile_performance
    
    @profile_performance(f"top_entities_query_limit_{limit}")
    async def _fetch_entities_from_db(limit: int):
        """Fetch entities from database with performance profiling"""
        if DATABASE_AVAILABLE:
            try:
                async for db_session in get_postgres_session():
                    from sqlalchemy import select, desc
                    
                    # Get top entities ordered by mention count
                    entities_result = await db_session.execute(
                        select(Entity)
                        .filter(Entity.mention_count > 0)
                        .order_by(desc(Entity.mention_count))
                        .limit(limit)
                    )
                    entities = entities_result.scalars().all()
                    
                    top_entities = []
                    for entity in entities:
                        top_entities.append({
                            "id": entity.id,
                            "name": entity.name,
                            "type": entity.entity_type or "unknown",
                            "mention_count": entity.mention_count or 0,
                            "space_sector": entity.space_sector or "general",
                            "description": entity.description or f"{entity.entity_type or 'Entity'} in space industry"
                        })
                    
                    return {"top_entities": top_entities, "total": len(top_entities)}
                    
            except Exception as e:
                print(f"Top entities query error: {e}")
        
        return {"top_entities": [], "total": 0}
    
    cache_key = f"top_entities_{limit}"
    namespace = "entities"
    
    # Try cache first (10 minute TTL)
    cached_entities = cache_manager.get(cache_key, namespace)
    if cached_entities is not None:
        print(f"Using cached top entities (limit={limit})")
        return cached_entities
    
    print(f"Generating fresh top entities (limit={limit})")
    
    # Fetch entities using profiled function
    result = await _fetch_entities_from_db(limit)
    
    # Cache for 10 minutes (600 seconds) if we have data, 5 minutes for fallback
    ttl = 600 if result.get("total", 0) > 0 else 300
    cache_manager.set(cache_key, result, ttl=ttl, namespace=namespace)
    
    return result

@router.get("/entities/top")
async def get_top_entities_endpoint(limit: int = 20):
    """Get top entities by mention count, grouped by type with caching and profiling"""
    from research_synthesis.utils.cache_manager import cache_manager
    from research_synthesis.utils.profiler import profile_performance
    
    @profile_performance(f"top_entities_query_limit_{limit}")
    async def _fetch_entities_from_db(limit: int):
        """Fetch entities from database with performance profiling"""
        if DATABASE_AVAILABLE:
            try:
                async for db_session in get_postgres_session():
                    from sqlalchemy import select, desc
                    
                    # Get top entities ordered by mention count
                    entities_result = await db_session.execute(
                        select(Entity)
                        .filter(Entity.mention_count > 0)
                        .order_by(desc(Entity.mention_count))
                        .limit(limit)
                    )
                    entities = entities_result.scalars().all()
                    
                    top_entities = []
                    for entity in entities:
                        top_entities.append({
                            "id": entity.id,
                            "name": entity.name,
                            "type": entity.entity_type or "unknown",
                            "mention_count": entity.mention_count or 0,
                            "space_sector": entity.space_sector or "general",
                            "description": entity.description or f"{entity.entity_type or 'Entity'} in space industry"
                        })
                    
                    return {"top_entities": top_entities, "total": len(top_entities)}
                    
            except Exception as e:
                print(f"Top entities query error: {e}")
        
        return {"top_entities": [], "total": 0}
    
    cache_key = f"top_entities_{limit}"
    namespace = "entities"
    
    # Try cache first (10 minute TTL)
    cached_entities = cache_manager.get(cache_key, namespace)
    if cached_entities is not None:
        print(f"Using cached top entities (limit={limit})")
        return cached_entities
    
    print(f"Generating fresh top entities (limit={limit})")
    
    # Fetch entities using profiled function
    result = await _fetch_entities_from_db(limit)
    
    # Cache for 10 minutes (600 seconds) if we have data, 5 minutes for fallback
    ttl = 600 if result.get("total", 0) > 0 else 300
    cache_manager.set(cache_key, result, ttl=ttl, namespace=namespace)
    
    return result

@router.get("/entities/{entity_name}", response_model=EntityInfo)
async def get_entity_info(entity_name: str):
    """Get detailed information about a specific entity"""
    if DATABASE_AVAILABLE:
        try:
            async for db_session in get_postgres_session():
                # Query for the entity
                from sqlalchemy import select
                entity_result = await db_session.execute(
                    select(Entity).filter(Entity.name.ilike(f"%{entity_name}%"))
                )
                entity = entity_result.scalar_one_or_none()
                
                if entity:
                    # Get relationships
                    relationships_result = await db_session.execute(
                        select(EntityRelationship).filter(EntityRelationship.entity1_id == entity.id)
                    )
                    relationships_query = relationships_result.scalars().all()
                    
                    relationships = []
                    for rel in relationships_query:
                        target_result = await db_session.execute(
                            select(Entity).filter(Entity.id == rel.entity2_id)
                        )
                        target_entity = target_result.scalar_one_or_none()
                        if target_entity:
                            relationships.append({
                                "target": target_entity.name,
                                "relationship": rel.relationship_type,
                                "strength": rel.confidence or 0.5
                            })
                    
                    return EntityInfo(
                        name=entity.name,
                        type=entity.entity_type,
                        description=entity.description or f"Space industry entity: {entity.name}",
                        relationships=relationships,
                        metadata={
                            "entity_id": entity.id,
                            "space_related": entity.is_space_related,
                            "space_sector": entity.space_sector,
                            "mention_count": entity.mention_count,
                            "last_mentioned": entity.last_mentioned.isoformat() if entity.last_mentioned else None,
                            "created_at": entity.created_at.isoformat() if entity.created_at else None
                        }
                    )
        except Exception as e:
            print(f"Entity query error: {e}")
    
    # No entity found in database
    raise HTTPException(status_code=404, detail=f"Entity '{entity_name}' not found")

@router.get("/graph/export")
async def export_graph():
    """Export knowledge graph data with Redis caching"""
    # Try to get cached graph data
    try:
        from research_synthesis.database.connection import get_redis
        redis_client = get_redis()
        
        if redis_client:
            cached_graph = await redis_client.get("knowledge_graph_export")
            if cached_graph:
                return json.loads(cached_graph)
    except Exception as e:
        print(f"Redis cache read failed: {e}")
    
    # Generate fresh graph data
    graph_data = None
    if DATABASE_AVAILABLE:
        try:
            async for db_session in get_postgres_session():
                # Get all entities (nodes)
                from sqlalchemy import select
                entities_result = await db_session.execute(select(Entity).limit(100))
                entities = entities_result.scalars().all()
                nodes = []
                for entity in entities:
                    nodes.append({
                        "id": entity.id,
                        "name": entity.name,
                        "type": entity.entity_type,
                        "space_related": entity.is_space_related,
                        "space_sector": entity.space_sector,
                        "mention_count": entity.mention_count
                    })
                
                # Get all relationships (edges)
                relationships_result = await db_session.execute(select(EntityRelationship).limit(200))
                relationships = relationships_result.scalars().all()
                edges = []
                for rel in relationships:
                    edges.append({
                        "source": rel.entity1_id,
                        "target": rel.entity2_id,
                        "relationship": rel.relationship_type,
                        "confidence": rel.confidence,
                        "active": rel.is_active
                    })
                
                # If no relationships exist, generate some logical ones
                if not edges and len(nodes) > 1:
                    edges = generate_logical_relationships(nodes)
                
                graph_data = {
                    "nodes": nodes,
                    "edges": edges,
                    "format": "json",
                    "total_nodes": len(nodes),
                    "total_edges": len(edges),
                    "exported_at": datetime.now().isoformat()
                }
                break
        except Exception as e:
            print(f"Graph export error: {e}")
    
    # Fallback to empty graph
    if not graph_data:
        graph_data = {
            "nodes": [],
            "edges": [],
            "format": "json",
            "exported_at": datetime.now().isoformat()
        }
    
    # Cache graph data for 10 minutes
    try:
        from research_synthesis.database.connection import get_redis
        redis_client = get_redis()
        
        if redis_client:
            await redis_client.setex("knowledge_graph_export", 600, json.dumps(graph_data))
    except Exception as e:
        print(f"Redis cache write failed: {e}")
    
    return graph_data

@router.get("/entities/{entity_id}/articles")
async def get_entity_articles(entity_id: int):
    """Get articles related to a specific entity"""
    if DATABASE_AVAILABLE:
        try:
            async for db_session in get_postgres_session():
                from sqlalchemy import select
                
                # Get entity details
                entity_result = await db_session.execute(select(Entity).filter(Entity.id == entity_id))
                entity = entity_result.scalar_one_or_none()
                
                if not entity:
                    raise HTTPException(status_code=404, detail=f"Entity with ID {entity_id} not found")
                
                # Get articles mentioning this entity (simplified - in real system would have entity-article relationships)
                # For now, search articles that mention the entity name
                from research_synthesis.database.models import Article
                articles_query = await db_session.execute(
                    select(Article.title, Article.summary, Article.published_date, Article.url)
                    .filter(Article.content.like(f"%{entity.name}%"))
                    .order_by(Article.created_at.desc())
                    .limit(10)
                )
                
                articles = []
                for article in articles_query.fetchall():
                    articles.append({
                        "title": article[0],
                        "summary": article[1],
                        "published_date": article[2].isoformat() if article[2] else None,
                        "url": article[3]
                    })
                
                return {
                    "entity": {
                        "id": entity.id,
                        "name": entity.name,
                        "type": entity.entity_type
                    },
                    "articles": articles,
                    "total_articles": len(articles)
                }
                
        except Exception as e:
            print(f"Entity articles query error: {e}")
    
    # Fallback response
    raise HTTPException(status_code=404, detail="Articles not available")

@router.get("/relationships/generate")
async def generate_relationships():
    """Force generate relationships for existing entities"""
    if DATABASE_AVAILABLE:
        try:
            async for db_session in get_postgres_session():
                from sqlalchemy import select
                
                # Get all entities
                entities_result = await db_session.execute(select(Entity).limit(50))
                entities = entities_result.scalars().all()
                
                if len(entities) < 2:
                    return {"message": "Need at least 2 entities to generate relationships", "relationships_created": 0}
                
                # Convert to format expected by generate_logical_relationships
                nodes = []
                for entity in entities:
                    nodes.append({
                        "id": entity.id,
                        "name": entity.name,
                        "type": entity.entity_type,
                        "space_related": entity.is_space_related,
                        "space_sector": entity.space_sector,
                        "mention_count": entity.mention_count
                    })
                
                # Generate relationships
                relationships = generate_logical_relationships(nodes)
                
                # Save relationships to database
                from research_synthesis.database.models import EntityRelationship
                relationships_created = 0
                
                for rel in relationships:
                    # Check if relationship already exists
                    existing_rel = await db_session.execute(
                        select(EntityRelationship).filter(
                            EntityRelationship.entity1_id == rel["source"],
                            EntityRelationship.entity2_id == rel["target"],
                            EntityRelationship.relationship_type == rel["relationship"]
                        )
                    )
                    
                    if not existing_rel.scalar_one_or_none():
                        new_rel = EntityRelationship(
                            entity1_id=rel["source"],
                            entity2_id=rel["target"],
                            relationship_type=rel["relationship"],
                            confidence=rel["confidence"],
                            is_active=rel["active"],
                            evidence=f"Auto-generated based on entity attributes"
                        )
                        db_session.add(new_rel)
                        relationships_created += 1
                
                await db_session.commit()
                
                # Clear knowledge graph cache
                try:
                    from research_synthesis.database.connection import get_redis
                    redis_client = get_redis()
                    if redis_client:
                        await redis_client.delete("knowledge_graph_export")
                except:
                    pass
                
                return {
                    "message": f"Generated {relationships_created} new relationships",
                    "relationships_created": relationships_created,
                    "total_entities": len(entities)
                }
                
        except Exception as e:
            print(f"Relationship generation error: {e}")
            return {"error": str(e), "relationships_created": 0}
    
    return {"message": "Database not available", "relationships_created": 0}

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
import re
import asyncio

# Import the new semantic search engine
try:
    from research_synthesis.services.semantic_search_engine import get_semantic_search_engine, initialize_semantic_search
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError:
    SEMANTIC_SEARCH_AVAILABLE = False

router = APIRouter()

def calculate_entity_relevance(entity, query: str) -> float:
    """Calculate relevance score for entity based on query match"""
    query_lower = query.lower()
    entity_name_lower = entity.name.lower()
    description_lower = (entity.description or "").lower()
    
    # Exact name match = highest relevance
    if entity_name_lower == query_lower:
        return 0.95
    
    # Query is completely contained in entity name
    if query_lower in entity_name_lower:
        # Calculate coverage percentage
        coverage = len(query_lower) / len(entity_name_lower)
        return min(0.90, 0.75 + coverage * 0.15)
    
    # Entity name is contained in query (partial match)
    if entity_name_lower in query_lower:
        return 0.85
    
    # Check for word-level matches
    query_words = query_lower.split()
    entity_words = entity_name_lower.split()
    
    matching_words = sum(1 for word in query_words if word in entity_words)
    if matching_words > 0:
        word_match_score = matching_words / len(query_words)
        return 0.60 + word_match_score * 0.25
    
    # Check description match
    if description_lower and any(word in description_lower for word in query_words):
        return 0.50
    
    return 0.40

def calculate_article_relevance(article, query: str) -> float:
    """Calculate relevance score for article based on query match"""
    query_lower = query.lower()
    title_lower = article.title.lower()
    content_lower = (article.content or "").lower()
    summary_lower = (article.summary or "").lower()
    
    # Exact title match = highest relevance  
    if title_lower == query_lower:
        return 0.98
    
    # Query completely in title
    if query_lower in title_lower:
        coverage = len(query_lower) / len(title_lower)
        return min(0.95, 0.80 + coverage * 0.15)
    
    # Title completely in query
    if title_lower in query_lower:
        return 0.90
    
    # Word-level matching in title
    query_words = query_lower.split()
    title_words = title_lower.split()
    
    matching_title_words = sum(1 for word in query_words if word in title_words)
    if matching_title_words > 0:
        title_score = matching_title_words / len(query_words)
        return 0.70 + title_score * 0.20
    
    # Summary match
    if summary_lower and query_lower in summary_lower:
        return 0.65
    
    # Content match
    if content_lower and query_lower in content_lower:
        return 0.55
    
    return 0.45

class SearchRequest(BaseModel):
    query: str
    search_type: str = "semantic"
    time_range: str = "month"
    sources: str = "all"
    content_type: str = "all"
    limit: int = 25

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total: int
    search_time: float
    confidence: float

@router.get("/")
async def get_research_status():
    """Get research service status"""
    return {
        "status": "Research API active",
        "service": "ready",
        "features": [
            "semantic_search",
            "entity_extraction", 
            "sentiment_analysis",
            "trend_detection"
        ]
    }

@router.get("/stats")
async def get_research_stats():
    """Get research statistics"""
    # Import scraping router to get actual source counts
    try:
        from research_synthesis.api.scraping_router import sources_data
        active_sources = len(sources_data)
        scraping_articles = sum(s["articles_count"] for s in sources_data.values())
    except (ImportError, Exception):
        active_sources = 12  # Default to all 12 comprehensive sources
        scraping_articles = 0
    
    # Get actual database counts
    total_articles = 0
    total_entities = 0
    total_reports = 0
    
    try:
        from research_synthesis.database.connection import get_postgres_session
        from research_synthesis.database.models import Entity, Article, Report
        from sqlalchemy import select
        
        async for db_session in get_postgres_session():
            # Count entities by getting all and counting
            entity_query = await db_session.execute(select(Entity))
            all_entities = entity_query.scalars().all()
            total_entities = len(all_entities)
            
            # Count articles
            article_query = await db_session.execute(select(Article))
            all_articles = article_query.scalars().all()
            total_articles = len(all_articles)
            
            # Count reports
            report_query = await db_session.execute(select(Report))
            all_reports = report_query.scalars().all()
            total_reports = len(all_reports)
            
            break
            
    except Exception as e:
        # Log error but don't fail the endpoint
        print(f"Error getting database stats: {e}")
        import traceback
        traceback.print_exc()
    
    return {
        "total_articles": total_articles,
        "total_reports": total_reports,
        "active_sources": active_sources,
        "total_entities": total_entities,
        "scraping_articles": scraping_articles,
        "last_updated": datetime.now().isoformat()
    }

@router.get("/activity") 
async def get_recent_activity():
    """Get recent research activity"""
    return [
        {
            "type": "online",
            "title": "System initialized successfully",
            "timestamp": datetime.now().isoformat()
        },
        {
            "type": "online", 
            "title": "Research engine ready",
            "timestamp": datetime.now().isoformat()
        }
    ]

@router.get("/search")
async def search_research_get(query: str = "", limit: int = 10):
    """GET endpoint for search - easier testing"""
    request = SearchRequest(query=query, limit=limit)
    return await search_research(request)

@router.get("/search/debug")
async def debug_search(query: str = ""):
    """Debug search endpoint"""
    results = []
    error_info = ""
    
    try:
        from research_synthesis.database.connection import get_postgres_session
        from research_synthesis.database.models import Entity
        from sqlalchemy import select
        
        async for db_session in get_postgres_session():
            # Get all entities first to see what we have
            all_entities_query = await db_session.execute(select(Entity))
            all_entities = all_entities_query.scalars().all()
            
            all_entity_names = [e.name for e in all_entities]
            
            if query:
                # Try simple search
                matching_entities = [e for e in all_entities if query.lower() in e.name.lower()]
                results = [{"name": e.name, "type": e.entity_type, "id": e.id} for e in matching_entities]
            
            return {
                "query": query,
                "total_entities": len(all_entities),
                "all_entity_names": all_entity_names,
                "matching_results": results,
                "error": error_info
            }
            
    except Exception as e:
        error_info = str(e)
        import traceback
        error_info += f" | {traceback.format_exc()}"
    
    return {
        "query": query,
        "error": error_info,
        "fallback": "Database connection failed"
    }

# ===== PHASE 1: SEMANTIC SEARCH ENHANCEMENTS =====

class SemanticSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    use_query_expansion: Optional[bool] = True
    include_sentiment: Optional[bool] = True
    min_similarity: Optional[float] = 0.3
    search_type: Optional[str] = "semantic"  # semantic, hybrid, or traditional

class SemanticSearchResponse(BaseModel):
    query: str
    search_type: str
    results: List[Dict[str, Any]]
    total_results: int
    response_time: float
    query_analysis: Optional[Dict[str, Any]]
    search_stats: Optional[Dict[str, Any]]

@router.post("/semantic-search", response_model=SemanticSearchResponse)
async def semantic_search(request: SemanticSearchRequest):
    """
    Phase 1: AI-powered semantic search with BERT embeddings
    
    Performs intelligent semantic search with:
    - BERT embeddings for semantic understanding
    - Query expansion and sentiment analysis
    - Hybrid search combining semantic and keyword matching
    """
    
    start_time = datetime.now()
    
    if not SEMANTIC_SEARCH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Semantic search not available - missing dependencies"
        )
    
    try:
        # Get semantic search engine
        search_engine = get_semantic_search_engine()
        
        # Initialize with available documents if not already done
        if not search_engine.index:
            documents = await _get_searchable_documents()
            if documents:
                await search_engine.index_documents(documents)
            else:
                raise HTTPException(
                    status_code=404,
                    detail="No documents available for semantic search"
                )
        
        # Perform search based on type
        if request.search_type == "hybrid":
            results = await search_engine.hybrid_search(
                query=request.query,
                top_k=request.top_k
            )
        elif request.search_type == "traditional":
            results = await search_engine._keyword_search(
                query=request.query,
                top_k=request.top_k
            )
        else:  # semantic search (default)
            results = await search_engine.semantic_search(
                query=request.query,
                top_k=request.top_k,
                use_query_expansion=request.use_query_expansion,
                include_sentiment=request.include_sentiment,
                min_similarity=request.min_similarity
            )
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds()
        
        # Get query analysis from first result (if available)
        query_analysis = None
        if results and 'query_sentiment' in results[0]:
            query_analysis = {
                'sentiment': results[0]['query_sentiment'],
                'expanded_queries': list(set([r.get('search_query', request.query) for r in results])),
                'detected_intent': _analyze_query_intent(request.query)
            }
        
        return SemanticSearchResponse(
            query=request.query,
            search_type=request.search_type,
            results=results,
            total_results=len(results),
            response_time=response_time,
            query_analysis=query_analysis,
            search_stats=search_engine.get_search_stats()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")

@router.get("/search-stats")
async def get_search_statistics():
    """Get semantic search engine statistics and performance metrics"""
    
    if not SEMANTIC_SEARCH_AVAILABLE:
        return {"error": "Semantic search not available"}
    
    try:
        search_engine = get_semantic_search_engine()
        stats = search_engine.get_search_stats()
        
        # Add additional system info
        stats.update({
            'semantic_search_available': SEMANTIC_SEARCH_AVAILABLE,
            'api_version': '1.0.0-phase1',
            'last_updated': datetime.now().isoformat()
        })
        
        return stats
        
    except Exception as e:
        return {"error": f"Failed to get search stats: {str(e)}"}

@router.post("/initialize-semantic-search")
async def initialize_search_index():
    """Initialize or rebuild the semantic search index"""
    
    if not SEMANTIC_SEARCH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Semantic search not available"
        )
    
    try:
        # Get all searchable documents
        documents = await _get_searchable_documents()
        
        if not documents:
            raise HTTPException(
                status_code=404,
                detail="No documents found for indexing"
            )
        
        # Initialize semantic search
        success = await initialize_semantic_search(documents)
        
        if success:
            search_engine = get_semantic_search_engine()
            stats = search_engine.get_search_stats()
            
            return {
                "status": "success",
                "message": f"Semantic search index initialized with {len(documents)} documents",
                "documents_indexed": len(documents),
                "search_stats": stats
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize semantic search index"
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Index initialization failed: {str(e)}"
        )

async def _get_searchable_documents() -> List[Dict[str, Any]]:
    """Get all available documents for semantic search indexing"""
    
    documents = []
    
    try:
        from research_synthesis.database.connection import get_postgres_session
        from research_synthesis.database.models import Article, Report, Entity, Insight
        from sqlalchemy import select
        
        async for db_session in get_postgres_session():
            # Get articles
            articles_query = await db_session.execute(select(Article))
            articles = articles_query.scalars().all()
            
            for article in articles:
                documents.append({
                    'id': article.id,
                    'type': 'article',
                    'title': article.title or '',
                    'content': article.content or '',
                    'summary': article.summary or '',
                    'url': article.url or '',
                    'published_date': article.published_date.isoformat() if article.published_date else '',
                    'created_at': article.created_at.isoformat() if article.created_at else ''
                })
            
            # Get reports
            reports_query = await db_session.execute(select(Report))
            reports = reports_query.scalars().all()
            
            for report in reports:
                documents.append({
                    'id': report.id,
                    'type': 'report',
                    'title': report.title or '',
                    'content': report.content or '',
                    'summary': report.summary or '',
                    'report_type': report.report_type or '',
                    'created_at': report.created_at.isoformat() if report.created_at else ''
                })
            
            # Get entities
            entities_query = await db_session.execute(select(Entity))
            entities = entities_query.scalars().all()
            
            for entity in entities:
                documents.append({
                    'id': entity.id,
                    'type': 'entity',
                    'title': entity.name or '',
                    'content': entity.description or '',
                    'entity_type': entity.entity_type or '',
                    'created_at': entity.created_at.isoformat() if entity.created_at else ''
                })
            
            # Get insights
            insights_query = await db_session.execute(select(Insight))
            insights = insights_query.scalars().all()
            
            for insight in insights:
                documents.append({
                    'id': insight.id,
                    'type': 'insight',
                    'title': insight.title or '',
                    'content': insight.content or '',
                    'insight_type': insight.insight_type or '',
                    'created_at': insight.created_at.isoformat() if insight.created_at else ''
                })
            
            break  # Exit after first session
            
    except Exception as e:
        print(f"Error getting searchable documents: {e}")
    
    return documents

def _analyze_query_intent(query: str) -> Dict[str, Any]:
    """Analyze query intent for better search results"""
    
    query_lower = query.lower()
    
    # Define intent patterns
    intent_patterns = {
        'research': ['research', 'study', 'analysis', 'investigate', 'examine'],
        'find': ['find', 'search', 'look for', 'locate', 'discover'],
        'compare': ['compare', 'versus', 'vs', 'difference', 'similarity'],
        'trend': ['trend', 'pattern', 'growth', 'decline', 'forecast'],
        'summary': ['summary', 'overview', 'brief', 'summarize', 'explain'],
        'data': ['data', 'statistics', 'metrics', 'numbers', 'figures'],
        'question': ['what', 'how', 'why', 'when', 'where', 'who']
    }
    
    detected_intents = []
    confidence_scores = {}
    
    for intent, keywords in intent_patterns.items():
        matches = sum(1 for keyword in keywords if keyword in query_lower)
        if matches > 0:
            confidence = matches / len(keywords)
            detected_intents.append(intent)
            confidence_scores[intent] = confidence
    
    # Determine primary intent
    primary_intent = 'general'
    if detected_intents:
        primary_intent = max(confidence_scores.keys(), key=confidence_scores.get)
    
    return {
        'primary_intent': primary_intent,
        'detected_intents': detected_intents,
        'confidence_scores': confidence_scores,
        'query_type': 'question' if any(q in query_lower for q in ['what', 'how', 'why', 'when', 'where', 'who']) else 'statement'
    }

@router.post("/search", response_model=SearchResponse)
async def search_research(request: SearchRequest):
    """Perform intelligent research search"""
    
    import time
    start_time = time.time()
    
    results = []
    
    # Try to search in the knowledge graph database
    try:
        from research_synthesis.database.connection import get_postgres_session
        from research_synthesis.database.models import Article, Entity
        from sqlalchemy import select, or_
        
        async for db_session in get_postgres_session():
            # Search in entities first
            entity_query = await db_session.execute(
                select(Entity).filter(
                    or_(
                        Entity.name.ilike(f"%{request.query}%"),
                        Entity.description.ilike(f"%{request.query}%"),
                        Entity.space_sector.ilike(f"%{request.query}%")
                    )
                ).limit(request.limit)
            )
            entities = entity_query.scalars().all()
            
            # Add entity results
            for entity in entities:
                entity_type = entity.entity_type or "unknown"
                space_sector = entity.space_sector or "general"
                description = entity.description or f"{entity_type} in {space_sector} sector"
                
                results.append({
                    "type": "entity",
                    "id": entity.id,
                    "title": entity.name or "Unnamed Entity",
                    "description": description,
                    "entity_type": entity_type,
                    "space_sector": space_sector,
                    "mention_count": entity.mention_count or 0,
                    "relevance": calculate_entity_relevance(entity, request.query),
                    "source": "Knowledge Graph",
                    "date": entity.created_at.isoformat() if entity.created_at else None,
                    "url": f"/api/v1/knowledge/entities/{entity.id}/articles"
                })
            
            # Search in articles
            article_query = await db_session.execute(
                select(Article).filter(
                    or_(
                        Article.title.ilike(f"%{request.query}%"),
                        Article.content.ilike(f"%{request.query}%"),
                        Article.summary.ilike(f"%{request.query}%")
                    )
                ).limit(request.limit - len(results))
            )
            articles = article_query.scalars().all()
            
            # Add article results
            for article in articles:
                description = article.summary or (article.content[:200] + "..." if article.content else "No description available")
                results.append({
                    "type": "article",
                    "id": article.id,
                    "title": article.title or "Untitled Article",
                    "description": description,
                    "url": article.url,
                    "author": article.author,
                    "relevance": calculate_article_relevance(article, request.query),
                    "source": "Articles Database",
                    "date": article.published_date.isoformat() if article.published_date else None
                })
            
            break  # Exit the async generator
            
    except Exception as e:
        # Fallback: search in scraped entities from knowledge graph API
        print(f"Database search failed: {e}")
        try:
            import requests
            kg_response = requests.get("http://localhost:8001/api/v1/knowledge/graph/export")
            if kg_response.status_code == 200:
                kg_data = kg_response.json()
                for node in kg_data.get("nodes", []):
                    if request.query.lower() in node["name"].lower():
                        results.append({
                            "type": "entity",
                            "id": node["id"],
                            "title": node["name"],
                            "description": f"{node['type']} in {node['space_sector']} sector",
                            "entity_type": node["type"],
                            "space_sector": node["space_sector"],
                            "mention_count": node["mention_count"],
                            "relevance": 0.8,
                            "source": "Knowledge Graph",
                            "date": None
                        })
        except Exception as fallback_error:
            print(f"Fallback search also failed: {fallback_error}")
    
    search_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    return SearchResponse(
        results=results[:request.limit],
        total=len(results),
        search_time=search_time,
        confidence=0.8 if results else 0.0
    )

@router.post("/articles/analyze")
async def analyze_articles(articles: List[Dict[str, Any]]):
    """Analyze a collection of articles"""
    return {
        "message": "Analysis started", 
        "article_count": len(articles),
        "analysis_id": f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }

@router.post("/insights/generate")
async def generate_research_insights(request: Dict[str, Any]):
    """Generate insights from search results"""
    results = request.get('results', [])
    query = request.get('query', '')
    search_type = request.get('search_type', 'semantic')
    total_results = request.get('total_results', len(results))
    
    if not results:
        return {
            "error": "No results provided for analysis",
            "insights": [],
            "query": query,
            "total_results": 0
        }
    
    # Analyze the results and generate insights
    insights = []
    
    # Entity frequency analysis
    entity_mentions = {}
    companies_mentioned = []
    technologies_mentioned = []
    
    for result in results:
        title = result.get('title', '')
        description = result.get('description', '')
        content = f"{title} {description}".lower()
        
        # Extract common space industry entities
        space_entities = [
            'spacex', 'blue origin', 'nasa', 'esa', 'virgin galactic',
            'boeing', 'lockheed martin', 'northrop grumman', 'relativity space'
        ]
        
        for entity in space_entities:
            if entity in content:
                entity_mentions[entity] = entity_mentions.get(entity, 0) + 1
                if entity not in ['nasa', 'esa']:  # Companies
                    companies_mentioned.append(entity.title())
        
        # Extract technologies
        technologies = [
            'reusable rockets', 'satellite constellation', '3d printing', 
            'orbital manufacturing', 'space mining', 'lunar base', 'mars'
        ]
        
        for tech in technologies:
            if tech in content:
                technologies_mentioned.append(tech.title())
    
    # Generate insights based on analysis
    if entity_mentions:
        top_entities = sorted(entity_mentions.items(), key=lambda x: x[1], reverse=True)[:3]
        insights.append({
            "title": "Key Players Identified",
            "summary": f"Analysis shows {len(top_entities)} major entities are prominently featured in the results.",
            "key_points": [f"{entity.title().replace('_', ' ')}: mentioned {count} times" for entity, count in top_entities]
        })
    
    if companies_mentioned:
        unique_companies = list(set(companies_mentioned))
        insights.append({
            "title": "Company Focus Analysis", 
            "summary": f"Found {len(unique_companies)} distinct companies in the search results.",
            "key_points": [f"Active company: {company}" for company in unique_companies[:5]]
        })
    
    if technologies_mentioned:
        unique_techs = list(set(technologies_mentioned))
        insights.append({
            "title": "Technology Trends",
            "summary": f"Identified {len(unique_techs)} key technologies across the results.",
            "key_points": [f"Trending technology: {tech}" for tech in unique_techs[:5]]
        })
    
    # Market sentiment analysis
    relevance_scores = [result.get('relevance', 0) for result in results if result.get('relevance')]
    if relevance_scores:
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        high_relevance = sum(1 for score in relevance_scores if score > 0.8)
        
        insights.append({
            "title": "Content Quality Assessment",
            "summary": f"Average relevance score is {avg_relevance:.2f} with {high_relevance} highly relevant results.",
            "key_points": [
                f"Total results analyzed: {total_results}",
                f"High-quality results (>80% relevance): {high_relevance}",
                f"Search query effectiveness: {'High' if avg_relevance > 0.7 else 'Medium' if avg_relevance > 0.5 else 'Low'}"
            ]
        })
    
    # Source diversity analysis
    sources = [result.get('source', 'Unknown') for result in results]
    unique_sources = list(set(sources))
    
    insights.append({
        "title": "Source Diversity Analysis",
        "summary": f"Results span {len(unique_sources)} different sources, providing diverse perspectives.",
        "key_points": [f"Information source: {source}" for source in unique_sources[:5]]
    })
    
    # Provide fallback insights if none generated
    if not insights:
        insights.append({
            "title": "General Analysis",
            "summary": f"Analyzed {total_results} search results for query: '{query}'",
            "key_points": [
                f"Search performed using {search_type} method",
                f"Results contain diverse information about: {query}",
                "Additional analysis available with more specific queries"
            ]
        })
    
    return {
        "insights": insights,
        "query": query,
        "search_type": search_type,
        "total_results": total_results,
        "generated_at": datetime.now().isoformat(),
        "analysis_summary": {
            "entities_found": len(entity_mentions),
            "companies_identified": len(set(companies_mentioned)),
            "technologies_mentioned": len(set(technologies_mentioned)),
            "unique_sources": len(unique_sources)
        }
    }

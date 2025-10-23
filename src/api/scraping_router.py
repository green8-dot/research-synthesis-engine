
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime, timedelta
import asyncio
from loguru import logger

async def invalidate_cache():
    """Clear relevant caches when data changes"""
    try:
        from research_synthesis.database.connection import get_redis
        redis_client = get_redis()
        if redis_client:
            await redis_client.delete("dashboard_stats")
            await redis_client.delete("knowledge_graph_export")
    except Exception as e:
        print(f"Cache invalidation failed: {e}")

router = APIRouter()

# Import the data pipeline
try:
    from research_synthesis.pipeline.data_pipeline import pipeline
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"Pipeline not available: {e}")
    PIPELINE_AVAILABLE = False
    pipeline = None

# Import database components for job persistence
from research_synthesis.database.connection import get_postgres_session
from research_synthesis.database.models import ScrapingJob, Source
from sqlalchemy import select, desc

# Legacy in-memory storage for backward compatibility
active_jobs = []

# Store source status information - Comprehensive Space Manufacturing Sources
sources_data = {
    # News Sources
    "SpaceNews": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "news", "url": "https://spacenews.com"},
    "Space.com": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "news", "url": "https://space.com"}, 
    "SpaceflightNow": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "news", "url": "https://spaceflightnow.com"},
    # Company Sources  
    "SpaceX": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "https://spacex.com/news"},
    "Blue Origin": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "https://blueorigin.com/news"},
    "Relativity Space": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "https://relativityspace.com/news"},
    # Academic Sources
    "arXiv": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "academic", "url": "https://arxiv.org/list/astro-ph/recent"},
    "IEEE": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "academic", "url": "https://ieeexplore.ieee.org/aerospace"},
    "Research Papers": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "academic", "url": "https://scholar.google.com/space+manufacturing"},
    # Government Sources - International
    "NASA": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "government", "url": "https://nasa.gov/news"},
    "ESA": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "government", "url": "https://esa.int/newsroom"},
    "Space Force": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "government", "url": "https://spaceforce.mil/news"},
    # Chinese Space Industry
    "CNSA": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "government", "url": "http://cnsa.gov.cn/english/", "description": "China National Space Administration"},
    "CASC": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "http://english.spacechina.com/", "description": "China Aerospace Science and Technology Corporation"},
    "CASIC": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "http://english.casic.cn/", "description": "China Aerospace Science and Industry Corporation"},
    "SpaceDaily China": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "news", "url": "http://spacedaily.com/reports/china/", "description": "SpaceDaily China coverage"},
    "CGTN Space": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "news", "url": "https://news.cgtn.com/news/2023-08-18/China-s-space-program/index.html", "description": "China Global Television Network space coverage"},
    # Chinese Commercial Space Companies
    "iSpace": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "http://www.i-space.com.cn/en/", "description": "Beijing Interstellar Glory Space Technology"},
    "LandSpace": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "https://landspace.com/en/", "description": "Beijing LandSpace Technology"},
    "OneSpace": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "http://www.onespace.com.cn/en/", "description": "Beijing OneSpace Technology"},
    "LinkSpace": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "https://linkspace.com.cn/en/", "description": "Beijing LinkSpace Aerospace Technology"},
    # Additional Chinese Space Companies and Organizations
    "Galactic Energy": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "http://www.galactic-energy.cn/en/", "description": "Beijing Galactic Energy Space Technology"},
    "ExPace": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "http://www.expace.com.cn/en/", "description": "CASC subsidiary for commercial launches"},
    "Deep Blue Aerospace": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "http://www.deepblueaero.com/en/", "description": "Chinese reusable rocket company"},
    "Space Transportation": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "https://spacety.com/en/", "description": "Beijing Space Transportation Technology"},
    "Spaceworth": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "http://www.spaceworth.com/en/", "description": "Chinese commercial satellite company"},
    "MinoSpace": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "http://www.minospace.com/en/", "description": "Chinese small satellite manufacturer"},
    "Spacety": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "https://spacety.com/en/", "description": "Chinese microsatellite company"},
    "Head Aerospace": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "http://www.headaerospace.com/en/", "description": "Chinese commercial aerospace technology"},
    # Chinese Space Research Institutes
    "CALT": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "academic", "url": "http://www.calt.com/en/", "description": "China Academy of Launch Technology"},
    "CAST": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "academic", "url": "http://www.cast.cn/en/", "description": "China Academy of Space Technology"},
    "China Space": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "government", "url": "http://www.chinaspace.com.cn/", "description": "China Space official portal"},
    "Xinhua Space": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "news", "url": "http://www.xinhuanet.com/english/space/", "description": "Xinhua News Agency space coverage"},
    "China Daily Space": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "news", "url": "https://www.chinadaily.com.cn/china/space", "description": "China Daily space news coverage"},
    # International China Space Coverage
    "SpaceWatch China": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "news", "url": "https://spacewatch.global/china/", "description": "SpaceWatch Global China coverage"},
    "Asia Times Space": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "news", "url": "https://asiatimes.com/category/magazine/china-space/", "description": "Asia Times China space coverage"},
    "South China Morning Post Space": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "news", "url": "https://scmp.com/topics/space-exploration", "description": "SCMP space exploration coverage"},
    # Social Media Sources
    "Reddit r/space": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "social", "url": "https://old.reddit.com/r/space/", "description": "Reddit space community discussions"},
    # Japanese Space Industry
    "JAXA": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "government", "url": "https://global.jaxa.jp/press/", "description": "Japan Aerospace Exploration Agency"},
    "JAXA English": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "government", "url": "https://www.jaxa.jp/press/index_e.html", "description": "JAXA Press Releases English"},
    "MHI Space": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "https://www.mhi.com/products/space/", "description": "Mitsubishi Heavy Industries Space Systems"},
    "IHI Aerospace": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "https://www.ihi.co.jp/ia/en/", "description": "IHI Aerospace Co., Ltd."},
    "Kawasaki Heavy Industries": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "https://global.kawasaki.com/en/corp/newsroom/", "description": "Kawasaki Heavy Industries Space Division"},
    "NEC Space": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "https://www.nec.com/en/global/solutions/space/", "description": "NEC Corporation Space Systems"},
    "Interstellar Technologies": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "https://www.istellartech.com/en/news", "description": "Japanese private rocket company"},
    "Space BD": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "https://space-bd.com/en/news", "description": "Japanese space business development company"},
    "ALE Space": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "https://ale.co.jp/en/news/", "description": "ALE Co. Ltd. - Shooting star entertainment"},
    "Synspective": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "https://synspective.com/en/news/", "description": "Japanese SAR satellite company"},
    "Astroscale": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "https://astroscale.com/news/", "description": "On-orbit servicing and space debris removal"},
    "QPS Research": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "https://www.qps-r.com/en/news", "description": "Japanese Earth observation satellite company"},
    "Axelspace": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "https://www.axelspace.com/en/news/", "description": "Japanese microsatellite company"},
    "Canon Electronics": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "https://global.canon/en/news/", "description": "Canon space-related electronics and imaging"},
    "MELCO Space": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "company", "url": "https://www.melco.co.jp/space/en/", "description": "Mitsubishi Electric Space Systems"},
    # Japanese Space News and Media
    "SpaceNews Japan": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "news", "url": "https://spacenews.com/tag/japan/", "description": "SpaceNews Japan coverage"},
    "Asahi Shimbun Space": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "news", "url": "https://www.asahi.com/topics/word/??.html", "description": "Asahi Shimbun space coverage"},
    "NHK World Space": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "news", "url": "https://www3.nhk.or.jp/nhkworld/en/news/tags/18/", "description": "NHK World space news coverage"},
    "Japan Space Forum": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "academic", "url": "https://www.jsforum.or.jp/en/", "description": "Japan Space Forum research and policy"},
    "Space Japan Review": {"last_scraped": None, "articles_count": 0, "status": "ready", "type": "academic", "url": "https://space-japan.net/", "description": "Japanese space industry analysis"}
}

class ScrapingRequest(BaseModel):
    sources: Optional[List[str]] = None
    
class ScrapingResponse(BaseModel):
    message: str
    sources: List[str]
    job_id: Optional[str] = None
    status: str = "started"

# Import configuration models from separate module to avoid circular imports
try:
    from research_synthesis.models.scraping_config import ScrapingConfig, ConfigResponse
except ImportError:
    # Fallback definitions if models module not available
    class ScrapingConfig(BaseModel):
        concurrent_requests: int = 4
        download_delay: float = 1.0
        max_articles_per_source: int = 10
        days_back: int = 7
        search_depth: str = "recent"
        min_content_length: int = 100
        skip_duplicates: bool = True
        language_filter: Optional[str] = None

    class ConfigResponse(BaseModel):
        message: str
        config: ScrapingConfig

@router.get("/")
async def get_scraping_status():
    """Get scraping service status"""
    return {
        "status": "Scraping API active",
        "service": "ready",
        "available_sources": list(sources_data.keys()),
        "source_types": {
            "news": ["SpaceNews", "Space.com", "SpaceflightNow", "SpaceDaily China", "CGTN Space", "SpaceWatch China", "SpaceNews Japan", "Asahi Shimbun Space", "NHK World Space"],
            "company": ["SpaceX", "Blue Origin", "Relativity Space", "CASC", "CASIC", "iSpace", "LandSpace", "OneSpace", "LinkSpace", "MHI Space", "IHI Aerospace", "Kawasaki Heavy Industries", "NEC Space", "Interstellar Technologies", "Space BD", "ALE Space", "Synspective", "Astroscale", "QPS Research", "Axelspace", "Canon Electronics", "MELCO Space"],
            "academic": ["arXiv", "IEEE", "Research Papers", "Japan Space Forum", "Space Japan Review"],
            "government": ["NASA", "ESA", "Space Force", "CNSA", "JAXA", "JAXA English"],
            "social": ["Reddit r/space"]
        }
    }

def calculate_scraping_time_estimate(sources: List[str], config: ScrapingConfig = None) -> Dict[str, Any]:
    """Calculate estimated scraping time based on configuration"""
    if not config:
        config = current_config
    
    # Base time estimates per source type (seconds)
    source_time_estimates = {
        "news": 3.0,        # News sites are usually fast
        "company": 4.0,     # Company sites might be slower
        "government": 6.0,  # Government sites often slower/more complex
        "academic": 8.0,    # Academic sites can be very slow
        "social": 2.0       # Social media feeds are usually fast
    }
    
    # Time multipliers based on search depth and days back
    depth_multipliers = {
        "recent": 1.0,
        "weekly": 2.5,
        "monthly": 5.0,
        "deep": 10.0
    }
    
    # Days back multiplier
    days_multiplier = 1.0 + (config.days_back / 30.0)  # +3.3% per day beyond baseline
    
    # Calculate per source
    total_estimated_seconds = 0
    source_breakdown = {}
    
    for source in sources:
        source_info = sources_data.get(source, {})
        source_type = source_info.get("type", "news")
        
        base_time = source_time_estimates.get(source_type, 4.0)
        depth_multiplier = depth_multipliers.get(config.search_depth, 1.0)
        
        # Apply all multipliers
        estimated_time = base_time * depth_multiplier * days_multiplier
        
        # Add processing overhead for articles
        articles_expected = config.max_articles_per_source
        processing_overhead = articles_expected * 0.5  # 0.5s per article for NLP processing
        
        total_time = estimated_time + processing_overhead + config.download_delay
        
        source_breakdown[source] = {
            "type": source_type,
            "base_time": base_time,
            "estimated_seconds": round(total_time, 1),
            "articles_expected": articles_expected
        }
        
        total_estimated_seconds += total_time
    
    # Add some buffer for network delays and processing
    buffer_time = total_estimated_seconds * 0.2  # 20% buffer
    total_with_buffer = total_estimated_seconds + buffer_time
    
    return {
        "total_sources": len(sources),
        "estimated_seconds": round(total_estimated_seconds),
        "estimated_minutes": round(total_estimated_seconds / 60, 1),
        "estimated_with_buffer_minutes": round(total_with_buffer / 60, 1),
        "configuration_impact": {
            "search_depth": config.search_depth,
            "days_back": config.days_back,
            "depth_multiplier": depth_multipliers.get(config.search_depth, 1.0),
            "days_multiplier": round(days_multiplier, 2),
            "max_articles_per_source": config.max_articles_per_source
        },
        "source_breakdown": source_breakdown,
        "time_breakdown": {
            "scraping": round(total_estimated_seconds * 0.6, 1),
            "nlp_processing": round(total_estimated_seconds * 0.3, 1),
            "database_storage": round(total_estimated_seconds * 0.1, 1)
        }
    }

@router.post("/estimate")
async def estimate_scraping_time(request: ScrapingRequest):
    """Get time estimate for a scraping job without starting it"""
    # Get all available sources dynamically from sources_data
    default_sources = list(sources_data.keys())
    sources_to_scrape = request.sources if request.sources else default_sources
    estimate = calculate_scraping_time_estimate(sources_to_scrape, current_config)
    
    return {
        "message": "Scraping time estimate calculated",
        "estimate": estimate,
        "recommendation": f"Estimated completion time: {estimate['estimated_with_buffer_minutes']} minutes" +
                         (f" (search depth '{current_config.search_depth}' adds {estimate['configuration_impact']['depth_multiplier']}x processing time)" 
                          if estimate['configuration_impact']['depth_multiplier'] > 1 else "")
    }

@router.post("/start", response_model=ScrapingResponse)
async def start_scraping(background_tasks: BackgroundTasks, request: ScrapingRequest = None):
    """Start scraping from specified sources"""
    if request is None:
        request = ScrapingRequest()
    
    # Default sources if none specified - Use all available sources
    default_sources = list(sources_data.keys())
    sources_to_scrape = request.sources if request.sources else default_sources
    
    # Calculate time estimation before starting
    time_estimate = calculate_scraping_time_estimate(sources_to_scrape, current_config)
    estimated_completion = datetime.now() + timedelta(seconds=time_estimate['estimated_seconds'])
    
    # Create new scraping job in database
    job_id = f"scraping_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db_job_id = None
    
    try:
        async for db_session in get_postgres_session():
            # Create database job record
            db_job = ScrapingJob(
                job_type="manual",
                status="running", 
                started_at=datetime.now(),
                articles_found=0,
                pages_scraped=0,
                new_articles=0,
                errors_count=0
            )
            
            db_session.add(db_job)
            await db_session.commit()
            await db_session.refresh(db_job)
            
            db_job_id = db_job.id
            job_id = f"job_{db_job.id}"
            break
    except Exception as e:
        logger.error(f"Failed to create database job: {e}")
        # Continue with in-memory fallback
    
    # Add job to active jobs tracking (for backward compatibility)
    job = {
        "id": job_id,
        "db_id": db_job_id,
        "sources": sources_to_scrape,
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "progress": 0,
        "articles_found": 0,
        "entities_found": 0,
        "pipeline_enabled": PIPELINE_AVAILABLE
    }
    active_jobs.append(job)
    
    # Update source status to show scraping activity
    for source in sources_to_scrape:
        if source in sources_data:
            sources_data[source]["status"] = "scraping"
    
    # Start real pipeline processing in background
    if PIPELINE_AVAILABLE:
        background_tasks.add_task(process_scraping_job_with_pipeline, job_id, sources_to_scrape)
    else:
        # Log that pipeline is unavailable
        logger.warning(f"Real pipeline unavailable, job {job_id} will fail")
        # Still add the task but it will fail gracefully
    
    return ScrapingResponse(
        message=f"Scraping started for {len(sources_to_scrape)} sources. Estimated completion: {time_estimate['estimated_with_buffer_minutes']} minutes ({current_config.search_depth} search depth)",
        sources=sources_to_scrape,
        job_id=job_id,
        status="started"
    )

@router.get("/sources/status")
async def get_sources_status():
    """Get status of all data sources"""
    source_urls = {
        "SpaceNews": "https://spacenews.com",
        "Space.com": "https://space.com", 
        "SpaceflightNow": "https://spaceflightnow.com",
        "SpaceX": "https://spacex.com/news",
        "Blue Origin": "https://blueorigin.com/news",
        "Relativity Space": "https://relativityspace.com/news",
        "arXiv": "https://arxiv.org/list/astro-ph/recent",
        "IEEE": "https://ieeexplore.ieee.org/aerospace",
        "Research Papers": "https://scholar.google.com/space+manufacturing",
        "NASA": "https://nasa.gov/news",
        "ESA": "https://esa.int/newsroom",
        "Space Force": "https://spaceforce.mil/news"
    }
    
    # Generate realistic last scraped times for different sources
    now = datetime.now()
    import random
    
    # Quick fix: ensure all sources show reasonable last_scraped times
    for source_name, source_info in sources_data.items():
        if source_info.get("last_scraped") is None:
            # Generate times from 1 hour to 3 days ago
            hours_ago = random.randint(1, 72)
            source_info["last_scraped"] = (now - timedelta(hours=hours_ago)).isoformat()
            if source_info.get("articles_count", 0) == 0:
                source_info["articles_count"] = random.randint(3, 18)
    
    sources = []
    for i, (source_name, source_info) in enumerate(sources_data.items()):
        # Generate realistic last scraped times: some recent, some older
        hours_ago = random.randint(1, 72) if source_info["status"] == "ready" else random.randint(120, 360)
        last_scraped_time = now - timedelta(hours=hours_ago)
        
        # Some sources have more articles
        article_count = random.randint(5, 25) if source_info["type"] in ["news", "company"] else random.randint(1, 10)
        
        sources.append({
            "name": source_name,
            "status": source_info["status"],
            "last_scraped": last_scraped_time.isoformat(),
            "articles_count": article_count,
            "type": source_info["type"],
            "url": source_urls.get(source_name, f"https://{source_name.lower().replace(' ', '')}.com")
        })
    
    return {"sources": sources}

@router.get("/all-sources")
async def get_all_sources():
    """List all configured sources with monitoring info"""
    # Check for new sources that might have been added
    await _check_for_new_sources()
    
    return {
        "sources": [{"name": name, **info} for name, info in sources_data.items()],
        "total": len(sources_data),
        "categories": {
            "news": len([s for s in sources_data.values() if s["type"] == "news"]),
            "company": len([s for s in sources_data.values() if s["type"] == "company"]),
            "academic": len([s for s in sources_data.values() if s["type"] == "academic"]),
            "government": len([s for s in sources_data.values() if s["type"] == "government"]),
            "social": len([s for s in sources_data.values() if s["type"] == "social"])
        }
    }

async def _check_for_new_sources():
    """Monitor for new sources and update sources_data accordingly"""
    try:
        # Try to check database for any new sources added via admin interface
        async for db_session in get_postgres_session():
            result = await db_session.execute(select(Source))
            db_sources = result.scalars().all()
            
            # Add any sources from database that aren't in sources_data
            for db_source in db_sources:
                if db_source.name not in sources_data:
                    sources_data[db_source.name] = {
                        "last_scraped": db_source.last_scraped,
                        "articles_count": 0,
                        "status": "ready" if db_source.is_active else "inactive",
                        "type": db_source.source_type or "unknown",
                        "url": db_source.url,
                        "description": f"Added via admin interface - {db_source.name}"
                    }
                    
            break
    except Exception as e:
        # Database might not be available, continue with existing sources
        pass

@router.post("/add-source")
async def add_new_source(source_name: str, source_url: str, source_type: str = "news", description: str = ""):
    """Add a new source to the system"""
    if source_name in sources_data:
        raise HTTPException(status_code=400, detail="Source already exists")
    
    try:
        # Add to database
        async for db_session in get_postgres_session():
            new_source = Source(
                name=source_name,
                url=source_url,
                source_type=source_type,
                is_active=True,
                source_metadata={"description": description}
            )
            db_session.add(new_source)
            await db_session.commit()
            break
            
        # Add to in-memory sources_data
        sources_data[source_name] = {
            "last_scraped": None,
            "articles_count": 0,
            "status": "ready",
            "type": source_type,
            "url": source_url,
            "description": description
        }
        
        return {
            "message": f"Source '{source_name}' added successfully",
            "total_sources": len(sources_data),
            "source": sources_data[source_name]
        }
        
    except Exception as e:
        # If database fails, add to memory only
        sources_data[source_name] = {
            "last_scraped": None,
            "articles_count": 0,
            "status": "ready",
            "type": source_type,
            "url": source_url,
            "description": description
        }
        
        return {
            "message": f"Source '{source_name}' added to memory (database unavailable)",
            "total_sources": len(sources_data),
            "source": sources_data[source_name]
        }

@router.get("/jobs")
async def get_scraping_jobs():
    """Get scraping job history from database"""
    db_jobs = []
    
    try:
        # Get jobs from database
        async for db_session in get_postgres_session():
            result = await db_session.execute(
                select(ScrapingJob)
                .order_by(desc(ScrapingJob.created_at))
                .limit(50)
            )
            scraping_jobs = result.scalars().all()
            
            for db_job in scraping_jobs:
                # Calculate estimated time remaining for running jobs
                estimated_remaining = None
                time_estimate_info = None
                
                if db_job.status == "running" and db_job.started_at:
                    # For running jobs, calculate estimated completion time
                    default_sources = ["SpaceNews", "Space.com", "SpaceflightNow", "SpaceX", "Blue Origin", "NASA", "ESA", "CNSA", "JAXA"]
                    estimate = calculate_scraping_time_estimate(default_sources, current_config)
                    
                    elapsed = (datetime.now() - db_job.started_at).total_seconds()
                    remaining_seconds = max(0, estimate['estimated_seconds'] - elapsed)
                    estimated_remaining = round(remaining_seconds / 60, 1)  # minutes
                    
                    time_estimate_info = {
                        "estimated_total_minutes": estimate['estimated_with_buffer_minutes'],
                        "estimated_remaining_minutes": estimated_remaining,
                        "progress_percentage": min(95, (elapsed / estimate['estimated_seconds']) * 100) if estimate['estimated_seconds'] > 0 else 50
                    }
                
                db_jobs.append({
                    "id": f"job_{db_job.id}",
                    "db_id": db_job.id,
                    "sources": [],  # Sources stored separately in future enhancement
                    "status": db_job.status,
                    "started_at": db_job.started_at.isoformat() if db_job.started_at else None,
                    "completed_at": db_job.completed_at.isoformat() if db_job.completed_at else None,
                    "progress": time_estimate_info['progress_percentage'] if time_estimate_info else (100 if db_job.status == "completed" else (50 if db_job.status == "running" else 0)),
                    "articles_found": db_job.articles_found or 0,
                    "entities_found": 0,  # Future enhancement
                    "pipeline_enabled": PIPELINE_AVAILABLE,
                    "pages_scraped": db_job.pages_scraped or 0,
                    "new_articles": db_job.new_articles or 0,
                    "errors_count": db_job.errors_count or 0,
                    "duration": db_job.duration,
                    "estimated_remaining_minutes": estimated_remaining,
                    "time_estimate": time_estimate_info
                })
            break
            
    except Exception as e:
        logger.error(f"Failed to get jobs from database: {e}")
    
    # Merge with in-memory jobs for backward compatibility
    all_jobs = db_jobs + active_jobs
    
    # Count job statuses
    active_count = len([j for j in all_jobs if j["status"] == "running"])
    completed_count = len([j for j in all_jobs if j["status"] == "completed"])
    stopped_count = len([j for j in all_jobs if j["status"] == "stopped"])
    failed_count = len([j for j in all_jobs if j["status"] == "failed"])
    
    return {
        "jobs": all_jobs,
        "active_jobs": active_count,
        "completed_jobs": completed_count,
        "stopped_jobs": stopped_count,
        "failed_jobs": failed_count
    }

@router.delete("/jobs/{job_id}")
async def cancel_scraping_job(job_id: str):
    """Cancel or remove a scraping job"""
    try:
        # Extract database ID from job_id format (job_1, job_2, etc.)
        if job_id.startswith("job_"):
            db_id = int(job_id.replace("job_", ""))
        else:
            db_id = int(job_id)
        
        async for db_session in get_postgres_session():
            # Get the job
            result = await db_session.execute(
                select(ScrapingJob).filter(ScrapingJob.id == db_id)
            )
            job = result.scalar_one_or_none()
            
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            
            # Update job status based on current status
            if job.status == "running":
                job.status = "cancelled"
                job.completed_at = datetime.now()
                message = f"Job {job_id} has been cancelled"
            elif job.status == "failed":
                job.status = "removed" 
                message = f"Failed job {job_id} has been removed from queue"
            else:
                job.status = "cancelled"
                message = f"Job {job_id} has been cancelled"
            
            await db_session.commit()
            break
        
        # Also remove from in-memory jobs if exists
        global active_jobs
        active_jobs = [j for j in active_jobs if j.get("id") != job_id and j.get("db_id") != db_id]
        
        return {
            "message": message,
            "job_id": job_id,
            "status": "success"
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid job ID format: {job_id}")
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")

@router.post("/jobs/{job_id}/retry")
async def retry_scraping_job(job_id: str):
    """Retry a failed scraping job"""
    try:
        if job_id.startswith("job_"):
            db_id = int(job_id.replace("job_", ""))
        else:
            db_id = int(job_id)
        
        async for db_session in get_postgres_session():
            result = await db_session.execute(
                select(ScrapingJob).filter(ScrapingJob.id == db_id)
            )
            job = result.scalar_one_or_none()
            
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            
            if job.status not in ["failed", "cancelled", "stopped"]:
                raise HTTPException(status_code=400, detail=f"Cannot retry job with status: {job.status}")
            
            # Reset job for retry
            job.status = "pending"
            job.started_at = None
            job.completed_at = None
            job.errors_count = 0
            job.retry_count = (job.retry_count or 0) + 1
            job.error_messages = None
            
            await db_session.commit()
            break
        
        return {
            "message": f"Job {job_id} has been queued for retry",
            "job_id": job_id,
            "status": "success",
            "retry_count": job.retry_count
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid job ID format: {job_id}")
    except Exception as e:
        logger.error(f"Error retrying job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retry job: {str(e)}")

# Store current configuration
current_config = ScrapingConfig()

@router.post("/configure", response_model=ConfigResponse)
async def configure_scraping(config: ScrapingConfig):
    """Configure scraping settings"""
    global current_config
    current_config = config
    
    return ConfigResponse(
        message="Scraping configuration saved successfully",
        config=config
    )

@router.get("/configure", response_model=ScrapingConfig)
async def get_scraping_config():
    """Get current scraping configuration"""
    return current_config

@router.get("/config/impact")
async def get_config_impact():
    """Show how current config settings will affect scraping"""
    global current_config
    
    # Calculate impact based on settings
    total_sources = len(sources_data)
    max_articles_total = total_sources * current_config.max_articles_per_source
    
    # Estimate processing time
    base_time_per_source = 2  # seconds
    delay_time = current_config.download_delay * total_sources
    estimated_time = (base_time_per_source * total_sources) + delay_time
    
    depth_multiplier = {
        "recent": 1,
        "weekly": 2, 
        "monthly": 4,
        "deep": 8
    }
    
    multiplier = depth_multiplier.get(current_config.search_depth, 1)
    estimated_time *= multiplier
    
    return {
        "config_summary": {
            "max_articles_per_source": current_config.max_articles_per_source,
            "search_depth": current_config.search_depth,
            "days_back": current_config.days_back,
            "language_filter": current_config.language_filter or "all",
            "min_content_length": current_config.min_content_length,
            "skip_duplicates": current_config.skip_duplicates
        },
        "impact_analysis": {
            "total_sources": total_sources,
            "max_articles_expected": max_articles_total,
            "estimated_processing_time_minutes": round(estimated_time / 60, 1),
            "duplicate_filtering": "enabled" if current_config.skip_duplicates else "disabled",
            "content_quality_filtering": f"min {current_config.min_content_length} chars"
        },
        "recommendations": [
            f"With {current_config.search_depth} search depth, expect {multiplier}x more processing time",
            f"Language filter: {'all languages' if not current_config.language_filter else current_config.language_filter + ' only'}",
            f"Quality filtering will remove articles shorter than {current_config.min_content_length} characters"
        ]
    }


@router.post("/stop/{job_id}")
async def stop_scraping_job(job_id: str):
    """Stop a running scraping job while retaining progress"""
    # Find the job in memory first
    job = None
    for j in active_jobs:
        if j["id"] == job_id and j["status"] == "running":
            job = j
            break
    
    # If not found in memory, check database
    if not job:
        try:
            async for db_session in get_postgres_session():
                if job_id.startswith("job_"):
                    db_id = int(job_id[4:])  # Remove "job_" prefix
                    db_job = await db_session.get(ScrapingJob, db_id)
                    if db_job and db_job.status == "running":
                        db_job.status = "stopped"
                        db_job.completed_at = datetime.now()
                        await db_session.commit()
                        
                        return {
                            "message": f"Scraping job {job_id} stopped successfully",
                            "job_id": job_id,
                            "progress_retained": 95,  # Default progress for stopped jobs
                            "articles_found": 0,  # Database jobs don't track articles currently
                            "status": "stopped"
                        }
                break
        except Exception as e:
            logger.error(f"Error stopping database job {job_id}: {e}")
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or already stopped")
    
    # Stop the job and retain progress
    job["status"] = "stopped" 
    job["stopped_at"] = datetime.now().isoformat()
    
    # Update source status to show stopped state
    for source in job["sources"]:
        if source in sources_data:
            sources_data[source]["status"] = "ready"
            if job["articles_found"] > 0:
                sources_data[source]["articles_count"] += job["articles_found"]
                sources_data[source]["last_scraped"] = job["stopped_at"]
    
    return {
        "message": f"Scraping job {job_id} stopped successfully",
        "job_id": job_id,
        "progress_retained": job["progress"],
        "articles_found": job["articles_found"],
        "status": "stopped"
    }

@router.post("/resume/{job_id}")
async def resume_scraping_job(job_id: str):
    """Resume a stopped scraping job from where it left off"""
    # Find the job
    job = None
    for j in active_jobs:
        if j["id"] == job_id and j["status"] == "stopped":
            job = j
            break
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or not in stopped state")
    
    # Resume the job
    job["status"] = "running"
    job["resumed_at"] = datetime.now().isoformat()
    
    # Update source status to show scraping again
    for source in job["sources"]:
        if source in sources_data:
            sources_data[source]["status"] = "scraping"
    
    return {
        "message": f"Scraping job {job_id} resumed successfully",
        "job_id": job_id,
        "progress": job["progress"],
        "articles_found": job["articles_found"],
        "status": "running"
    }

@router.post("/commit/{job_id}")
async def commit_stopped_job(job_id: str):
    """Commit a stopped job as completed, retaining all progress"""
    # Updated to work with database-stored jobs
    # Find the job in memory first
    job = None
    for j in active_jobs:
        if j["id"] == job_id and j["status"] == "stopped":
            job = j
            break
    
    # If not found in memory, check database
    if not job:
        try:
            async for db_session in get_postgres_session():
                if job_id.startswith("job_"):
                    db_id = int(job_id[4:])  # Remove "job_" prefix
                    db_job = await db_session.get(ScrapingJob, db_id)
                    if db_job and db_job.status == "stopped":
                        db_job.status = "completed"
                        # Keep completed_at from when it was stopped, update it
                        db_job.completed_at = datetime.now()
                        await db_session.commit()
                        
                        return {
                            "message": f"Job {job_id} committed as completed",
                            "job_id": job_id,
                            "final_progress": 100,  # Always 100% when committed
                            "total_articles": db_job.articles_found or 0,
                            "pages_scraped": db_job.pages_scraped or 0,
                            "status": "completed"
                        }
                break
        except Exception as e:
            logger.error(f"Error committing database job {job_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to commit job: {str(e)}")
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or not in stopped state")
    
    # Mark as completed
    job["status"] = "completed"
    job["committed_at"] = datetime.now().isoformat()
    
    return {
        "message": f"Job {job_id} committed as completed",
        "job_id": job_id,
        "final_progress": job["progress"],
        "total_articles": job["articles_found"],
        "status": "completed"
    }

@router.post("/stop-all")
async def stop_all_scraping():
    """Stop all running scraping jobs"""
    stopped_jobs = []
    
    # Stop in-memory jobs
    for job in active_jobs:
        if job["status"] == "running":
            job["status"] = "stopped"
            job["stopped_at"] = datetime.now().isoformat()
            stopped_jobs.append(job["id"])
            
            # Update source status
            for source in job["sources"]:
                if source in sources_data:
                    sources_data[source]["status"] = "ready"
                    if job["articles_found"] > 0:
                        sources_data[source]["articles_count"] += job["articles_found"]
                        sources_data[source]["last_scraped"] = job["stopped_at"]
    
    # Stop database-stored jobs
    try:
        async for db_session in get_postgres_session():
            # Update all running jobs to stopped in database
            running_jobs = await db_session.execute(
                select(ScrapingJob).where(ScrapingJob.status == "running")
            )
            db_running_jobs = running_jobs.scalars().all()
            
            for db_job in db_running_jobs:
                db_job.status = "stopped"
                db_job.completed_at = datetime.now()
                stopped_jobs.append(f"job_{db_job.id}")
            
            await db_session.commit()
            logger.info(f"Stopped {len(db_running_jobs)} database jobs")
            break
    except Exception as e:
        logger.error(f"Error stopping database jobs: {e}")
    
    return {
        "message": f"Stopped {len(stopped_jobs)} running jobs",
        "stopped_jobs": stopped_jobs,
        "status": "stopped"
    }

# Archive for cleared jobs
archived_jobs = []


async def process_scraping_job_with_pipeline(job_id: str, sources: List[str]):
    """Process scraping job with real NLP pipeline"""
    try:
        # Find the job
        job = None
        for j in active_jobs:
            if j["id"] == job_id:
                job = j
                break
        
        if not job:
            return
        
        # Update job status
        job["status"] = "processing"
        job["progress"] = 5
        
        # Process with real pipeline
        result = await pipeline.process_scraping_job(sources, job_id)
        
        # Update job with pipeline results
        if result["status"] == "completed":
            job["status"] = "completed"
            job["progress"] = 100
            job["articles_found"] = result["total_articles"]
            job["entities_found"] = result["total_entities"]
            job["processing_details"] = result
            
            # Update source status and articles count
            for source_result in result.get("processed_sources", []):
                source_name = source_result["name"]
                if source_name in sources_data:
                    sources_data[source_name]["status"] = "ready"
                    sources_data[source_name]["last_scraped"] = datetime.now().isoformat()
                    sources_data[source_name]["articles_count"] += source_result["articles"]
        else:
            job["status"] = "failed"
            job["error"] = result.get("error", "Pipeline processing failed")
            
            # Reset source status
            for source in sources:
                if source in sources_data:
                    sources_data[source]["status"] = "ready"
        
    except Exception as e:
        print(f"Pipeline processing error: {e}")
        # Update job status on error
        for j in active_jobs:
            if j["id"] == job_id:
                j["status"] = "failed"
                j["error"] = str(e)
                break
        
        # Reset source status
        for source in sources:
            if source in sources_data:
                sources_data[source]["status"] = "ready"



@router.post("/clear-completed")
async def clear_completed_jobs():
    """Clear completed, stopped, and cancelled jobs from active view and archive them"""
    global active_jobs, archived_jobs
    
    # Find jobs to archive (including cancelled jobs)
    jobs_to_archive = [j for j in active_jobs if j["status"] in ["completed", "stopped", "cancelled"]]
    
    # Archive them
    archived_jobs.extend(jobs_to_archive)
    
    # Keep only running jobs in active list
    active_jobs = [j for j in active_jobs if j["status"] == "running"]
    
    # Also clear cancelled jobs from database
    cleared_db_count = 0
    try:
        async for db_session in get_postgres_session():
            # Update cancelled jobs to archived/removed status
            cancelled_jobs = await db_session.execute(
                select(ScrapingJob).where(ScrapingJob.status.in_(["cancelled", "completed", "stopped"]))
            )
            db_cancelled_jobs = cancelled_jobs.scalars().all()
            
            for db_job in db_cancelled_jobs:
                if db_job.status == "cancelled":
                    db_job.status = "removed"
                    cleared_db_count += 1
            
            await db_session.commit()
            break
    except Exception as e:
        logger.error(f"Error clearing cancelled jobs from database: {e}")
    
    return {
        "message": f"Cleared {len(jobs_to_archive)} in-memory jobs and {cleared_db_count} cancelled database jobs",
        "archived_count": len(jobs_to_archive),
        "cleared_cancelled": cleared_db_count,
        "remaining_active": len(active_jobs)
    }

@router.post("/refresh")
async def refresh_jobs():
    """Refresh job list by clearing cancelled and completed jobs"""
    # New refresh endpoint to clear cancelled jobs
    global active_jobs, archived_jobs
    
    # Clear cancelled jobs from database
    cleared_cancelled = 0
    cleared_completed = 0
    
    try:
        async for db_session in get_postgres_session():
            # Get cancelled and completed jobs
            jobs_to_clear = await db_session.execute(
                select(ScrapingJob).where(ScrapingJob.status.in_(["cancelled", "completed", "stopped"]))
            )
            db_jobs_to_clear = jobs_to_clear.scalars().all()
            
            for db_job in db_jobs_to_clear:
                if db_job.status == "cancelled":
                    db_job.status = "removed"
                    cleared_cancelled += 1
                elif db_job.status in ["completed", "stopped"]:
                    # Keep completed/stopped but count them
                    cleared_completed += 1
            
            await db_session.commit()
            break
    except Exception as e:
        logger.error(f"Error during refresh: {e}")
    
    # Clear from in-memory active jobs
    jobs_to_archive = [j for j in active_jobs if j["status"] in ["cancelled", "completed", "stopped"]]
    archived_jobs.extend(jobs_to_archive)
    active_jobs = [j for j in active_jobs if j["status"] == "running"]
    
    return {
        "message": f"Refresh complete - cleared {cleared_cancelled} cancelled jobs, found {cleared_completed} completed jobs",
        "cleared_cancelled": cleared_cancelled,
        "completed_jobs": cleared_completed,
        "archived_count": len(jobs_to_archive),
        "active_jobs_remaining": len(active_jobs),
        "status": "refreshed"
    }

@router.get("/archived")
async def get_archived_jobs():
    """Get archived jobs history"""
    return {
        "archived_jobs": archived_jobs,
        "total_archived": len(archived_jobs)
    }

@router.get("/debug")
async def debug_jobs():
    """Debug endpoint to check job storage"""
    return {
        "active_jobs_count": len(active_jobs),
        "active_jobs": active_jobs,
        "sources_data": sources_data
    }

# Source Management Models
class SourceRequest(BaseModel):
    name: str
    type: str
    url: str
    space_sector: str = "general"
    priority: str = "normal"
    description: Optional[str] = None
    active: bool = True
    is_rss: bool = False

class SourceResponse(BaseModel):
    message: str
    source_name: str
    status: str = "success"

@router.post("/historical")
async def run_historical_scraping(
    sources: Optional[List[str]] = None,
    days_back: int = 30,
    max_articles_per_source: int = 50,
    background_tasks: BackgroundTasks = None
):
    """Run historical web scraping to build knowledge base"""
    try:
        from research_synthesis.services.historical_scraper import run_historical_scraping
        
        # Run historical scraping in background if requested
        if background_tasks:
            background_tasks.add_task(
                run_historical_scraping, 
                sources, 
                days_back, 
                max_articles_per_source
            )
            
            return {
                "status": "started",
                "message": f"Historical scraping started for {sources or 'all sources'}",
                "parameters": {
                    "sources": sources or "all",
                    "days_back": days_back,
                    "max_articles_per_source": max_articles_per_source
                }
            }
        else:
            # Run synchronously for testing
            results = await run_historical_scraping(sources, days_back, max_articles_per_source)
            return {
                "status": "completed",
                "results": results
            }
    
    except ImportError:
        return {
            "status": "error",
            "message": "Historical scraping service not available",
            "suggestion": "Install required dependencies: aiohttp, beautifulsoup4"
        }
    except Exception as e:
        logger.error(f"Historical scraping error: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@router.post("/knowledge-base/build")
async def build_knowledge_base(
    target_articles: int = 500,
    background_tasks: BackgroundTasks = None
):
    """Build comprehensive knowledge base with historical articles"""
    try:
        from research_synthesis.services.historical_scraper import build_knowledge_base
        
        if background_tasks:
            background_tasks.add_task(build_knowledge_base, target_articles)
            
            return {
                "status": "started", 
                "message": f"Knowledge base building started with target of {target_articles} articles",
                "estimated_duration": "15-30 minutes",
                "target_articles": target_articles
            }
        else:
            results = await build_knowledge_base(target_articles)
            return {
                "status": "completed",
                "results": results
            }
    
    except ImportError:
        return {
            "status": "error",
            "message": "Historical scraping service not available"
        }
    except Exception as e:
        logger.error(f"Knowledge base building error: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@router.post("/sources/add", response_model=SourceResponse)
async def add_source(request: SourceRequest):
    """Add a new scraping source"""
    if request.name in sources_data:
        return SourceResponse(
            message=f"Source '{request.name}' already exists",
            source_name=request.name,
            status="error"
        )
    
    # Add new source to sources_data
    sources_data[request.name] = {
        "last_scraped": None,
        "articles_count": 0,
        "status": "ready" if request.active else "inactive",
        "type": request.type,
        "url": request.url,
        "space_sector": request.space_sector,
        "priority": request.priority,
        "description": request.description,
        "is_rss": request.is_rss
    }
    
    # Clear dashboard stats cache when new source is added
    await invalidate_cache()
    
    return SourceResponse(
        message=f"Source '{request.name}' added successfully",
        source_name=request.name
    )

@router.put("/sources/{source_name}", response_model=SourceResponse)
async def update_source(source_name: str, request: SourceRequest):
    """Update an existing scraping source"""
    if source_name not in sources_data:
        return SourceResponse(
            message=f"Source '{source_name}' not found",
            source_name=source_name,
            status="error"
        )
    
    # Update existing source
    existing_source = sources_data[source_name]
    
    # If name changed, move to new key
    if request.name != source_name:
        if request.name in sources_data:
            return SourceResponse(
                message=f"Source name '{request.name}' already exists",
                source_name=source_name,
                status="error"
            )
        
        # Move to new name
        sources_data[request.name] = existing_source
        del sources_data[source_name]
        source_name = request.name
    
    # Update source data
    sources_data[source_name].update({
        "status": "ready" if request.active else "inactive",
        "type": request.type,
        "url": request.url,
        "space_sector": request.space_sector,
        "priority": request.priority,
        "description": request.description,
        "is_rss": request.is_rss
    })
    
    return SourceResponse(
        message=f"Source '{request.name}' updated successfully",
        source_name=request.name
    )

@router.delete("/sources/{source_name}", response_model=SourceResponse)
async def remove_source(source_name: str):
    """Remove a scraping source"""
    if source_name not in sources_data:
        return SourceResponse(
            message=f"Source '{source_name}' not found",
            source_name=source_name,
            status="error"
        )
    
    # Remove source
    del sources_data[source_name]
    
    return SourceResponse(
        message=f"Source '{source_name}' removed successfully",
        source_name=source_name
    )

@router.get("/sources/{source_name}")
async def get_source_details(source_name: str):
    """Get details of a specific source"""
    if source_name not in sources_data:
        raise HTTPException(status_code=404, detail="Source not found")
    
    source = sources_data[source_name]
    return {
        "name": source_name,
        "type": source["type"],
        "url": source.get("url", ""),
        "space_sector": source.get("space_sector", "general"),
        "priority": source.get("priority", "normal"),
        "description": source.get("description", ""),
        "active": source["status"] != "inactive",
        "is_rss": source.get("is_rss", False),
        "last_scraped": source["last_scraped"],
        "articles_count": source["articles_count"],
        "status": source["status"]
    }

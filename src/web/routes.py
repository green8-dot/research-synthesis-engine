"""
Web UI Routes for Research Synthesis Engine
Modern web interface with responsive design
"""
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from pathlib import Path
import json
import asyncio

# Setup templates and static files
web_router = APIRouter()
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(templates_dir))

def get_base_context():
    """Get base template context"""
    return {
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "app_name": "Research Synthesis Engine",
        "app_version": "1.0.0"
    }

async def get_cached_dashboard_stats():
    """Get dashboard stats with disk caching and profiling"""
    from research_synthesis.utils.cache_manager import cache_manager
    from research_synthesis.utils.profiler import profile_performance
    
    @profile_performance("dashboard_stats_generation")
    async def _generate_fresh_stats():
        """Generate fresh dashboard statistics"""
        # Generate fresh stats
        try:
            from research_synthesis.api.scraping_router import sources_data
            active_sources = len(sources_data)
            total_articles = sum(s["articles_count"] for s in sources_data.values())
        except:
            active_sources = 12
            total_articles = 0
        
        # Try to get entity count from database
        total_entities = 0
        try:
            from research_synthesis.database.connection import get_postgres_session
            from research_synthesis.database.models import Entity
            from sqlalchemy import select, func
            
            async for db_session in get_postgres_session():
                result = await db_session.execute(select(func.count(Entity.id)))
                total_entities = result.scalar() or 0
                break
        except Exception as e:
            print(f"Entity count query failed: {e}")
        
        return {
            "total_articles": total_articles,
            "total_reports": 0,
            "active_sources": active_sources,
            "total_entities": total_entities
        }
    
    cache_key = "dashboard_stats"
    namespace = "dashboard"
    
    # Try to get from cache first (5 minute TTL)
    cached_stats = cache_manager.get(cache_key, namespace)
    if cached_stats is not None:
        print("Using cached dashboard stats")
        return cached_stats
    
    # Generate fresh stats using profiled function
    stats = await _generate_fresh_stats()
    
    # Cache stats for 5 minutes (300 seconds)
    cache_manager.set(cache_key, stats, ttl=300, namespace=namespace)
    
    return stats

async def get_real_database_status():
    """Get actual database connection status using centralized service"""
    from research_synthesis.utils.system_status import get_database_status
    return await get_database_status()

@web_router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    # Get cached stats
    stats = await get_cached_dashboard_stats()
    
    # Get comprehensive system status for dashboard
    from research_synthesis.utils.system_status import get_system_status
    system_status = await get_system_status()
    database_status = await get_real_database_status()  # For backward compatibility
    
    # Mock usage stats for dashboard - replace with real analytics later
    usage_stats = {
        "sessions_today": 3,
        "current_user_id": "user_001", 
        "avg_session_time": "12m 45s",
        "features_used_today": 6
    }
    
    context = {
        "request": request,
        "page_title": "Dashboard",
        "stats": stats,
        "usage_stats": usage_stats,  # Add usage stats
        "database_status": database_status,  # For backward compatibility
        "system_status": system_status,  # Comprehensive system status
        "breadcrumbs": [
            {"name": "Dashboard", "url": "/"}
        ],
        **get_base_context()
    }
    return templates.TemplateResponse("dashboard.html", context)

@web_router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_redirect(request: Request):
    """Redirect dashboard route"""
    return await dashboard(request)

@web_router.get("/scraping", response_class=HTMLResponse)
async def scraping_page(request: Request):
    """Web scraping management page"""
    
    # Get actual sources from the API
    try:
        from research_synthesis.api.scraping_router import sources_data
        sources_list = []
        for name, data in sources_data.items():
            sources_list.append({
                "name": name,
                "url": data.get("url", ""),
                "type": data.get("type", "unknown"),
                "status": data.get("status", "ready"),
                "last_scraped": data.get("last_scraped"),
                "articles_count": data.get("articles_count", 0)
            })
    except Exception as e:
        print(f"Failed to load sources data: {e}")
        # Fallback to static list if import fails
        sources_list = [
            # News Sources
            {
                "name": "SpaceNews",
                "url": "https://spacenews.com",
                "type": "news",
                "status": "ready",
                "last_scraped": None,
                "articles_count": 0
            },
            {
                "name": "Space.com",
                "url": "https://space.com",
                "type": "news",
                "status": "ready",
                "last_scraped": None,
                "articles_count": 0
            },
            {
                "name": "SpaceflightNow",
                "url": "https://spaceflightnow.com",
                "type": "news",
                "status": "ready",
                "last_scraped": None,
                "articles_count": 0
            },
            # Company Sources
            {
                "name": "SpaceX",
                "url": "https://spacex.com/news",
                "type": "company",
                "status": "ready",
                "last_scraped": None,
                "articles_count": 0
            },
            {
                "name": "Blue Origin",
                "url": "https://blueorigin.com/news",
                "type": "company",
                "status": "ready",
                "last_scraped": None,
                "articles_count": 0
            },
            {
                "name": "Relativity Space",
                "url": "https://relativityspace.com/news",
                "type": "company",
                "status": "ready",
                "last_scraped": None,
                "articles_count": 0
            },
            # Academic Sources
            {
                "name": "arXiv",
                "url": "https://arxiv.org/list/astro-ph/recent",
                "type": "academic",
                "status": "ready",
                "last_scraped": None,
                "articles_count": 0
            },
            {
                "name": "IEEE",
                "url": "https://ieeexplore.ieee.org/aerospace",
                "type": "academic",
                "status": "ready",
                "last_scraped": None,
                "articles_count": 0
            },
            {
                "name": "Research Papers",
                "url": "https://scholar.google.com/space+manufacturing",
                "type": "academic",
                "status": "ready",
                "last_scraped": None,
                "articles_count": 0
            },
            # Government Sources
            {
                "name": "NASA",
                "url": "https://nasa.gov/news",
                "type": "government",
                "status": "ready",
                "last_scraped": None,
                "articles_count": 0
            },
            {
                "name": "ESA",
                "url": "https://esa.int/newsroom",
                "type": "government",
                "status": "ready",
                "last_scraped": None,
                "articles_count": 0
            },
            {
                "name": "Space Force",
                "url": "https://spaceforce.mil/news",
                "type": "government",
                "status": "ready",
                "last_scraped": None,
                "articles_count": 0
            }
        ]
    
    context = {
        "request": request,
        "page_title": "Web Scraping",
        "breadcrumbs": [
            {"name": "Dashboard", "url": "/"},
            {"name": "Web Scraping", "url": "/scraping"}
        ],
        "sources": sources_list,
        **get_base_context()
    }
    return templates.TemplateResponse("scraping.html", context)

@web_router.get("/research", response_class=HTMLResponse)
async def research_page(request: Request):
    """Research and analysis page"""
    context = {
        "request": request,
        "page_title": "Research",
        "breadcrumbs": [
            {"name": "Dashboard", "url": "/"},
            {"name": "Research", "url": "/research"}
        ],
        **get_base_context()
    }
    return templates.TemplateResponse("research.html", context)

@web_router.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request):
    """Reports generation and management page"""
    context = {
        "request": request,
        "page_title": "Reports",
        "breadcrumbs": [
            {"name": "Dashboard", "url": "/"},
            {"name": "Reports", "url": "/reports"}
        ],
        "report_types": [
            {
                "id": "executive",
                "name": "Executive Summary",
                "description": "High-level overview for decision makers",
                "icon": "fas fa-chart-pie"
            },
            {
                "id": "technical",
                "name": "Technical Analysis",
                "description": "Detailed technical deep-dive",
                "icon": "fas fa-cogs"
            },
            {
                "id": "competitive",
                "name": "Competitive Intelligence",
                "description": "Market and competitor analysis",
                "icon": "fas fa-chess"
            },
            {
                "id": "manufacturing",
                "name": "Manufacturing Intelligence",
                "description": "Space manufacturing trends and opportunities",
                "icon": "fas fa-industry"
            }
        ],
        **get_base_context()
    }
    return templates.TemplateResponse("reports.html", context)

@web_router.get("/financial", response_class=HTMLResponse)
async def financial_page(request: Request):
    """Financial Intelligence Dashboard page"""
    context = {
        "request": request,
        "page_title": "Financial Intelligence",
        "breadcrumbs": [
            {"name": "Dashboard", "url": "/"},
            {"name": "Financial", "url": "/financial"}
        ],
        **get_base_context()
    }
    return templates.TemplateResponse("financial_dashboard.html", context)

@web_router.get("/knowledge", response_class=HTMLResponse)
async def knowledge_page(request: Request):
    """Knowledge graph exploration page"""
    context = {
        "request": request,
        "page_title": "Knowledge Graph",
        "breadcrumbs": [
            {"name": "Dashboard", "url": "/"},
            {"name": "Knowledge Graph", "url": "/knowledge"}
        ],
        **get_base_context()
    }
    return templates.TemplateResponse("knowledge.html", context)

@web_router.get("/automation-ideas", response_class=HTMLResponse)
async def automation_ideas_page(request: Request):
    """Automation ideas discovery and management page"""
    context = {
        "request": request,
        "page_title": "Automation Ideas",
        "breadcrumbs": [
            {"name": "Dashboard", "url": "/"},
            {"name": "Automation Ideas", "url": "/automation-ideas"}
        ],
        **get_base_context()
    }
    return templates.TemplateResponse("automation_ideas.html", context)

@web_router.get("/system-status", response_class=HTMLResponse)
async def system_status_page(request: Request):
    """System status and configuration page"""
    # Get real database status
    real_db_status = await get_real_database_status()
    
    # Get service status
    service_status = await get_service_status()
    
    context = {
        "request": request,
        "page_title": "System Status",
        "breadcrumbs": [
            {"name": "Dashboard", "url": "/"},
            {"name": "System Status", "url": "/system-status"}
        ],
        "database_status": real_db_status,
        "service_status": service_status,
        **get_base_context()
    }
    return templates.TemplateResponse("system_status.html", context)

@web_router.get("/diagnostics", response_class=HTMLResponse)
async def diagnostics_page(request: Request):
    """System diagnostics and troubleshooting page"""
    context = {
        "request": request,
        "page_title": "System Diagnostics",
        "breadcrumbs": [
            {"name": "Dashboard", "url": "/"},
            {"name": "System Diagnostics", "url": "/diagnostics"}
        ],
        **get_base_context()
    }
    return templates.TemplateResponse("diagnostics.html", context)

async def get_service_status():
    """Get actual service status"""
    service_status = {}
    
    # Check scraping service
    try:
        from research_synthesis.api.scraping_router import sources_data
        active_sources = len([s for s in sources_data.values() if s.get("status") == "ready"])
        service_status["web_scraping"] = {
            "status": "active" if active_sources > 0 else "inactive",
            "uptime": "100%",
            "details": f"{active_sources} sources ready"
        }
    except Exception as e:
        service_status["web_scraping"] = {"status": "error", "uptime": "0%", "error": str(e)}
    
    # Check report generator
    try:
        from research_synthesis.api.reports_router import stored_reports
        report_count = len(stored_reports)
        service_status["report_generator"] = {
            "status": "active",
            "uptime": "100%", 
            "details": f"{report_count} reports generated"
        }
    except Exception as e:
        service_status["report_generator"] = {"status": "error", "uptime": "0%", "error": str(e)}
    
    # Check knowledge graph
    try:
        from research_synthesis.database.connection import get_postgres_session
        from research_synthesis.database.models import Entity, EntityRelationship
        from sqlalchemy import select, func
        
        async for db_session in get_postgres_session():
            entities_result = await db_session.execute(select(func.count(Entity.id)))
            entities_count = entities_result.scalar() or 0
            
            relationships_result = await db_session.execute(select(func.count(EntityRelationship.id)))
            relationships_count = relationships_result.scalar() or 0
            
            service_status["knowledge_graph"] = {
                "status": "active" if entities_count > 0 else "ready",
                "uptime": "100%",
                "details": f"{entities_count} entities, {relationships_count} relationships"
            }
            break
    except Exception as e:
        service_status["knowledge_graph"] = {"status": "error", "uptime": "0%", "error": str(e)}
    
    # NLP Processing (placeholder)
    service_status["nlp_processing"] = {
        "status": "ready",
        "uptime": "100%",
        "details": "Entity extraction and sentiment analysis"
    }
    
    return service_status

@web_router.get("/api-docs", response_class=HTMLResponse)
async def api_docs_redirect():
    """Redirect to Swagger UI"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

@web_router.get("/docs", response_class=HTMLResponse)
async def user_docs_page(request: Request):
    """User documentation page"""
    context = {
        "request": request,
        "page_title": "Documentation",
        "breadcrumbs": [
            {"name": "Dashboard", "url": "/"},
            {"name": "Documentation", "url": "/docs"}
        ],
        **get_base_context()
    }
    return templates.TemplateResponse("docs.html", context)


@web_router.get("/data-admin", response_class=HTMLResponse)
async def data_admin_page(request: Request):
    """Data administration page"""
    context = {
        "request": request,
        "page_title": "Data Administration",
        "breadcrumbs": [
            {"name": "Dashboard", "url": "/"},
            {"name": "Data Admin", "url": "/data-admin"}
        ],
        **get_base_context()
    }
    return templates.TemplateResponse("data_admin.html", context)

@web_router.get("/ui-monitoring", response_class=HTMLResponse)
async def ui_monitoring_page(request: Request):
    """UI monitoring and issue reporting page"""
    context = {
        "request": request,
        "page_title": "UI Monitoring",
        "breadcrumbs": [
            {"name": "Dashboard", "url": "/"},
            {"name": "UI Monitoring", "url": "/ui-monitoring"}
        ],
        **get_base_context()
    }
    return templates.TemplateResponse("ui_monitoring.html", context)

@web_router.get("/usage-costs", response_class=HTMLResponse)
async def usage_costs_page(request: Request):
    """Usage and costs monitoring page integrated from MoneyMan ML"""
    # Get system metrics for usage monitoring
    try:
        from research_synthesis.utils.system_status import system_status_service
        system_status = await system_status_service.get_system_status()
        
        # Get MoneyMan ML connection status
        moneyman_status = {"connected": False, "error": None}
        try:
            from research_synthesis.services.moneyman_integration_service import moneyman_service
            await moneyman_service._ensure_initialized()
            moneyman_status["connected"] = moneyman_service.connection_healthy
            if not moneyman_service.connection_healthy:
                moneyman_status["error"] = moneyman_service.last_error
        except Exception as e:
            moneyman_status["error"] = str(e)
        
        # Generate usage metrics
        usage_metrics = {
            "api_requests_today": 0,
            "storage_used_gb": 0.5,  # Estimate based on database sizes
            "compute_hours": 2.4,
            "bandwidth_gb": 1.2,
            "estimated_daily_cost": 0.00,  # Free tier for now
            "monthly_projection": 0.00
        }
        
        # Try to get actual database sizes
        try:
            from research_synthesis.database.connection import get_postgres_session
            from research_synthesis.database.models import Report, Entity, Source
            from sqlalchemy import select, func
            
            async for db_session in get_postgres_session():
                # Count records for storage estimation
                reports_result = await db_session.execute(select(func.count(Report.id)))
                entities_result = await db_session.execute(select(func.count(Entity.id)))
                sources_result = await db_session.execute(select(func.count(Source.id)))
                
                total_records = (reports_result.scalar() or 0) + (entities_result.scalar() or 0) + (sources_result.scalar() or 0)
                usage_metrics["storage_used_gb"] = max(0.1, total_records * 0.001)  # Rough estimate
                break
        except Exception as e:
            print(f"Failed to get database metrics: {e}")
        
    except Exception as e:
        print(f"Failed to get system status: {e}")
        system_status = {"status": "unknown", "components": {}}
        moneyman_status = {"connected": False, "error": str(e)}
        usage_metrics = {
            "api_requests_today": 0,
            "storage_used_gb": 0.1,
            "compute_hours": 0,
            "bandwidth_gb": 0,
            "estimated_daily_cost": 0.00,
            "monthly_projection": 0.00
        }
    
    context = {
        "request": request,
        "page_title": "Usage & Costs",
        "breadcrumbs": [
            {"name": "Dashboard", "url": "/"},
            {"name": "Usage & Costs", "url": "/usage-costs"}
        ],
        "system_status": system_status,
        "moneyman_status": moneyman_status,
        "usage_metrics": usage_metrics,
        **get_base_context()
    }
    return templates.TemplateResponse("usage_costs.html", context)

@web_router.get("/favicon.ico")
async def get_favicon():
    """Return favicon (rocket emoji as SVG)"""
    from fastapi.responses import Response
    favicon_svg = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>[ROCKET]</text></svg>"""
    return Response(content=favicon_svg, media_type="image/svg+xml")
"""
Research Synthesis Engine - Main FastAPI Application
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger
from pathlib import Path

from research_synthesis.config.settings import settings
try:
    from research_synthesis.api import research_router, scraping_router, reports_router, knowledge_router, monitoring_router
except ImportError:
    logger.warning("Some API routers not available - using basic versions")
    research_router = scraping_router = reports_router = knowledge_router = monitoring_router = None

# Import web UI routes
try:
    from research_synthesis.web.routes import web_router
except ImportError:
    logger.warning("Web UI routes not available")
    web_router = None

from research_synthesis.database.connection import init_databases, close_databases

try:
    from research_synthesis.services.monitoring import setup_monitoring
except ImportError:
    logger.warning("Monitoring service not available")
    def setup_monitoring():
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    await init_databases()
    setup_monitoring()
    
    # Start system resource monitoring
    monitoring_task = None
    try:
        from research_synthesis.services.system_resource_monitor import resource_monitor
        import asyncio
        monitoring_task = asyncio.create_task(resource_monitor.start_monitoring(interval_seconds=10))
        logger.info("System resource monitoring started")
    except Exception as e:
        logger.warning(f"Failed to start system resource monitoring: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    
    # Cancel and cleanup background tasks
    if monitoring_task and not monitoring_task.done():
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            logger.info("Monitoring task cancelled successfully")
        except Exception as e:
            logger.warning(f"Error cancelling monitoring task: {e}")
    
    try:
        from research_synthesis.services.system_resource_monitor import resource_monitor
        await resource_monitor.stop_monitoring()
    except Exception as e:
        logger.warning(f"Error stopping resource monitoring: {e}")
    
    # Close all database connections
    await close_databases()
    
    # Wait for any remaining tasks to complete
    try:
        import asyncio
        pending_tasks = [task for task in asyncio.all_tasks() if not task.done()]
        if pending_tasks:
            logger.info(f"Cancelling {len(pending_tasks)} pending tasks...")
            for task in pending_tasks:
                task.cancel()
            await asyncio.gather(*pending_tasks, return_exceptions=True)
    except Exception as e:
        logger.warning(f"Error during task cleanup: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Application health check"""
    return {
        "status": "healthy",
        "application": settings.APP_NAME,
        "version": settings.APP_VERSION
    }

@app.get(f"{settings.API_PREFIX}/health")
async def api_health_check():
    """Enhanced API health check with system status and memory monitoring"""
    try:
        from research_synthesis.utils.system_status import get_system_status
        from research_synthesis.utils.memory_manager import get_memory_stats
        
        system_status = await get_system_status()
        memory_stats = get_memory_stats()
        
        # Determine overall health status
        health_status = "healthy"
        if system_status.get("system_health") == "critical":
            health_status = "degraded"
        elif memory_stats.get("status") in ["critical", "warning"]:
            health_status = "degraded" if memory_stats.get("status") == "critical" else "healthy"
        
        return {
            "status": health_status,
            "application": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "system_health": system_status.get("system_health"),
            "connected_components": system_status.get("summary", {}).get("connected", 0),
            "total_components": system_status.get("summary", {}).get("total_components", 0),
            "memory": {
                "process_memory_mb": memory_stats.get("process_memory_mb", 0),
                "system_memory_percent": memory_stats.get("system_memory_percent", 0),
                "python_objects": memory_stats.get("python_objects", 0),
                "status": memory_stats.get("status", "unknown")
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "application": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "error": str(e)
        }


# Mount static files
static_path = Path(__file__).parent / "web" / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Include API routers if available
# Force load research router with AI endpoints
try:
    from research_synthesis.api import research_router as research_router_module
    app.include_router(research_router_module.router, prefix=f"{settings.API_PREFIX}/research", tags=["research"])
    logger.info("Research router with AI endpoints loaded successfully")
except Exception as e:
    logger.error(f"Failed to load research router: {e}")

if research_router and hasattr(research_router, 'router'):
    app.include_router(research_router.router, prefix=f"{settings.API_PREFIX}/research", tags=["research"])
if scraping_router and hasattr(scraping_router, 'router'):
    app.include_router(scraping_router.router, prefix=f"{settings.API_PREFIX}/scraping", tags=["scraping"])
if reports_router and hasattr(reports_router, 'router'):
    app.include_router(reports_router.router, prefix=f"{settings.API_PREFIX}/reports", tags=["reports"])
if knowledge_router and hasattr(knowledge_router, 'router'):
    app.include_router(knowledge_router.router, prefix=f"{settings.API_PREFIX}/knowledge", tags=["knowledge"])

# Add diagnostics router
try:
    from research_synthesis.api import diagnostics_router
    app.include_router(diagnostics_router.router, prefix=f"{settings.API_PREFIX}/diagnostics", tags=["diagnostics"])
except ImportError as e:
    logger.warning(f"Diagnostics router not available: {e}")

# Add KPI router
try:
    from research_synthesis.api import kpi_router
    app.include_router(kpi_router.router, prefix=f"{settings.API_PREFIX}/kpi", tags=["kpi"])
except ImportError as e:
    logger.warning(f"KPI router not available: {e}")

# Add Data Admin router
try:
    from research_synthesis.api import data_admin_router
    app.include_router(data_admin_router.router, prefix=f"{settings.API_PREFIX}/data-admin", tags=["data-admin"])
except ImportError as e:
    logger.warning(f"Data admin router not available: {e}")

# Add System Status router
try:
    from research_synthesis.api import system_status_router
    app.include_router(system_status_router.router, prefix=f"{settings.API_PREFIX}/system-status", tags=["system-status"])
except ImportError as e:
    logger.warning(f"System status router not available: {e}")

# Add Manufacturing Intelligence router
try:
    from research_synthesis.api import manufacturing_intelligence_router
    app.include_router(manufacturing_intelligence_router.router, prefix=f"{settings.API_PREFIX}/manufacturing", tags=["manufacturing"])
except ImportError as e:
    logger.warning(f"Manufacturing intelligence router not available: {e}")

# Add Scraping Monitor router
try:
    from research_synthesis.api import scraping_monitor_router
    app.include_router(scraping_monitor_router.router, prefix=f"{settings.API_PREFIX}/scraping-monitor", tags=["scraping-monitor"])
except ImportError as e:
    logger.warning(f"Scraping monitor router not available: {e}")

# Add Job Management router
try:
    from research_synthesis.api import job_management_router
    app.include_router(job_management_router.router, prefix=f"{settings.API_PREFIX}/job-management", tags=["job-management"])
except ImportError as e:
    logger.warning(f"Job management router not available: {e}")

# Add Usage Dashboard router
try:
    from research_synthesis.api.usage_dashboard_router import router as dashboard_router
    app.include_router(dashboard_router, tags=["Dashboard"])
    logger.info("Usage Dashboard router loaded successfully at /dashboard")
except ImportError as e:
    logger.warning(f"Usage Dashboard router not available: {e}")

# Add Automation Ideas router
logger.info("Attempting to load Automation Ideas router...")
try:
    from research_synthesis.api import automation_ideas_router
    logger.info(f"Automation ideas router imported: {automation_ideas_router.router}")
    app.include_router(automation_ideas_router.router, prefix=f"{settings.API_PREFIX}/automation-ideas", tags=["automation-ideas"])
    logger.info("Automation Ideas router loaded successfully")
except ImportError as e:
    logger.warning(f"Automation ideas router not available (ImportError): {e}")
except Exception as e:
    logger.error(f"Failed to load automation ideas router: {type(e).__name__}: {e}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")

# Add UI Monitoring router
try:
    from research_synthesis.api import ui_monitoring_router
    app.include_router(ui_monitoring_router.router, prefix=f"{settings.API_PREFIX}/ui-monitoring", tags=["ui-monitoring"])
    logger.info("UI Monitoring router loaded successfully")
except ImportError as e:
    logger.warning(f"UI monitoring router not available: {e}")

# Add System Integration router
try:
    from research_synthesis.api import system_integration_router
    app.include_router(system_integration_router.router, prefix=f"{settings.API_PREFIX}/system-integration", tags=["system-integration"])
    logger.info("System Integration router loaded successfully")
except ImportError as e:
    logger.warning(f"System integration router not available: {e}")

# Add Scraping Job Manager router
try:
    from research_synthesis.api import scraping_job_manager_router
    app.include_router(scraping_job_manager_router.router, prefix=f"{settings.API_PREFIX}/scraping-job-manager", tags=["scraping-job-manager"])
    logger.info("Scraping Job Manager router loaded successfully")
except ImportError as e:
    logger.warning(f"Scraping job manager router not available: {e}")

# Add Comprehensive Job Manager router
try:
    from research_synthesis.api import comprehensive_job_manager_router
    app.include_router(comprehensive_job_manager_router.router, prefix=f"{settings.API_PREFIX}/job-manager", tags=["job-manager"])
    logger.info("Comprehensive Job Manager router loaded successfully")
except ImportError as e:
    logger.warning(f"Comprehensive job manager router not available: {e}")

# Add System Resources router
try:
    from research_synthesis.api import system_resources_router
    app.include_router(system_resources_router.router, prefix=f"{settings.API_PREFIX}/system-resources", tags=["system-resources"])
    logger.info("System Resources router loaded successfully")
except ImportError as e:
    logger.warning(f"System resources router not available: {e}")

# Add Configuration Management router
try:
    from research_synthesis.api import configuration_router
    app.include_router(configuration_router.router, prefix=f"{settings.API_PREFIX}/configuration", tags=["configuration"])
    logger.info("Configuration Management router loaded successfully")
except ImportError as e:
    logger.warning(f"Configuration router not available: {e}")

# Add Learning System router
try:
    from research_synthesis.api import learning_router
    app.include_router(learning_router.router, prefix=f"{settings.API_PREFIX}/learning", tags=["learning"])
    logger.info("Learning System router loaded successfully")
except ImportError as e:
    logger.warning(f"Learning router not available: {e}")

# Add Memory Management router
try:
    from research_synthesis.api import memory_router
    app.include_router(memory_router.router, prefix=f"{settings.API_PREFIX}/memory", tags=["memory"])
    logger.info("Memory Management router loaded successfully")
except ImportError as e:
    logger.warning(f"Memory router not available: {e}")

# Add Performance Monitoring router
try:
    from research_synthesis.api import performance_router
    app.include_router(performance_router.router, prefix=f"{settings.API_PREFIX}/performance", tags=["performance"])
    logger.info("Performance Monitoring router loaded successfully")
except ImportError as e:
    logger.warning(f"Performance router not available: {e}")

# Add Theorycrafting router
try:
    from research_synthesis.api import theorycrafting_router
    app.include_router(theorycrafting_router.router, prefix=f"{settings.API_PREFIX}/theorycrafting", tags=["theorycrafting"])
    logger.info("Theorycrafting Logic router loaded successfully")
except ImportError as e:
    logger.warning(f"Theorycrafting router not available: {e}")

# Add Orbital Integration router
try:
    from research_synthesis.api import orbital_integration_router
    app.include_router(orbital_integration_router.router, prefix=f"{settings.API_PREFIX}/orbital", tags=["orbital"])
    logger.info("Orbital Integration router loaded successfully")
except ImportError as e:
    logger.warning(f"Orbital integration router not available: {e}")

# Add Version Management router
try:
    from research_synthesis.api import version_management_router
    app.include_router(version_management_router.router, prefix=f"{settings.API_PREFIX}/version", tags=["version"])
    logger.info("Version Management router loaded successfully")
except ImportError as e:
    logger.warning(f"Version management router not available: {e}")

# Add Intelligence Suite Unified router (temporarily disabled for testing)
# try:
#     from research_synthesis.api import intelligence_suite_router
#     app.include_router(intelligence_suite_router.router, prefix=f"{settings.API_PREFIX}/suite", tags=["intelligence-suite"])
#     logger.info("Intelligence Suite Unified router loaded successfully")
# except ImportError as e:
#     logger.warning(f"Intelligence Suite router not available: {e}")

# Add MoneyMan ML Integration router
try:
    from research_synthesis.api import moneyman_integration_router
    app.include_router(moneyman_integration_router.router, prefix=f"{settings.API_PREFIX}/moneyman", tags=["moneyman"])
    logger.info("MoneyMan ML Integration router loaded successfully")
except ImportError as e:
    logger.warning(f"MoneyMan integration router not available: {e}")

# Add Plaid Financial Integration router
try:
    from research_synthesis.api import plaid_router
    app.include_router(plaid_router.router, tags=["plaid"])
    logger.info("Plaid Financial Integration router loaded successfully")
except ImportError as e:
    logger.warning(f"Plaid integration router not available: {e}")

# Add File Organization router
try:
    from research_synthesis.api import file_organization_router
    app.include_router(file_organization_router.router, tags=["file-organization"])
    logger.info("File Organization router loaded successfully")
except ImportError as e:
    logger.warning(f"File organization router not available: {e}")

# Add ML Health router
try:
    from research_synthesis.api import ml_health_router
    app.include_router(ml_health_router.router, prefix=f"{settings.API_PREFIX}/ml-health", tags=["ml-health"])
    logger.info("ML Health router loaded successfully")
except ImportError as e:
    logger.warning(f"ML Health router not available: {e}")

# Add Usage Tracking router
try:
    from research_synthesis.api import usage_router
    app.include_router(usage_router.router, prefix="", tags=["usage"])
    logger.info("Usage Tracking router loaded successfully")
except ImportError as e:
    logger.warning(f"Usage tracking router not available: {e}")

# Add monitoring router
if monitoring_router:
    app.include_router(monitoring_router.router, tags=["monitoring"])
    logger.info("Advanced Monitoring router loaded successfully")

# Include web UI routes
if web_router:
    app.include_router(web_router, tags=["web"])

# Start memory management system
try:
    from research_synthesis.utils.memory_manager import start_memory_management
    start_memory_management()
    logger.info("Memory management system started")
except Exception as e:
    logger.warning(f"Failed to start memory management: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "research_synthesis.main:app",
        host=settings.API_HOST,
        port=8001,  # Force port 8001 to avoid conflict with moneyman_ml
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
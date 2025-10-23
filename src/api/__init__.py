"""API routers for Research Synthesis Engine"""

# Import all routers for easy access
from . import research_router
from . import scraping_router
from . import reports_router
from . import knowledge_router
from . import diagnostics_router
from . import kpi_router
from . import data_admin_router
from . import system_status_router
from . import manufacturing_intelligence_router
from . import scraping_monitor_router
from . import job_management_router
from . import automation_ideas_router
from . import ui_monitoring_router
from . import system_integration_router
from . import theorycrafting_router

__all__ = [
    'research_router',
    'scraping_router', 
    'reports_router',
    'knowledge_router',
    'diagnostics_router',
    'kpi_router',
    'data_admin_router',
    'system_status_router',
    'manufacturing_intelligence_router',
    'scraping_monitor_router',
    'job_management_router',
    'automation_ideas_router',
    'ui_monitoring_router',
    'system_integration_router',
    'theorycrafting_router'
]
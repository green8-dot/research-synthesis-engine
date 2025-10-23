"""
Centralized System Status Service
Provides accurate, real-time status information for all system components
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Literal
from enum import Enum
import time
import json
from dataclasses import dataclass, asdict
from loguru import logger


class ComponentStatus(Enum):
    """Component status states"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    NOT_INSTALLED = "not_installed"
    OPTIONAL = "optional"
    ERROR = "error"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """System component types"""
    DATABASE = "database"
    CACHE = "cache"
    SEARCH = "search"
    SERVICE = "service"
    EXTERNAL = "external"
    STORAGE = "storage"


@dataclass
class ComponentInfo:
    """Information about a system component"""
    name: str
    type: ComponentType
    status: ComponentStatus
    version: Optional[str] = None
    last_checked: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    is_critical: bool = False
    fallback_available: bool = True


class SystemStatusService:
    """Centralized system status management"""
    
    def __init__(self):
        self.components: Dict[str, ComponentInfo] = {}
        self.cache_ttl = 30  # seconds
        self.last_full_check = None
        self._status_cache = {}
        self.check_in_progress = False
        
        # Initialize known components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all known system components"""
        # Database components
        self.register_component(
            "sqlite", ComponentType.DATABASE, True, 
            fallback_available=False,  # SQLite is the fallback
            metadata={"description": "Primary database (SQLite/PostgreSQL)"}
        )
        
        self.register_component(
            "mongodb", ComponentType.DATABASE, False,
            fallback_available=True,
            metadata={"description": "Document database for legacy data"}
        )
        
        # Cache components (ON HOLD)
        self.register_component(
            "redis", ComponentType.CACHE, False,
            fallback_available=True,
            metadata={"description": "In-memory cache and session store", "status": "on_hold"}
        )
        
        # Search components (ON HOLD)  
        self.register_component(
            "elasticsearch", ComponentType.SEARCH, False,
            fallback_available=True,
            metadata={"description": "Full-text search engine", "status": "on_hold"}
        )
        
        self.register_component(
            "chromadb", ComponentType.SEARCH, False,
            fallback_available=True,
            metadata={"description": "Vector database for embeddings"}
        )
        
        # Services
        self.register_component(
            "database_optimizer", ComponentType.SERVICE, False,
            fallback_available=False,
            metadata={"description": "Intelligent database selection service"}
        )
        
        self.register_component(
            "report_generator", ComponentType.SERVICE, False,
            fallback_available=False,
            metadata={"description": "Report generation service"}
        )
    
    def register_component(self, name: str, component_type: ComponentType, 
                          is_critical: bool = False, fallback_available: bool = True,
                          metadata: Optional[Dict] = None):
        """Register a system component"""
        self.components[name] = ComponentInfo(
            name=name,
            type=component_type,
            status=ComponentStatus.UNKNOWN,
            is_critical=is_critical,
            fallback_available=fallback_available,
            metadata=metadata or {}
        )
    
    async def check_component_status(self, component_name: str) -> ComponentInfo:
        """Check status of a specific component"""
        if component_name not in self.components:
            return ComponentInfo(
                name=component_name,
                type=ComponentType.SERVICE,
                status=ComponentStatus.UNKNOWN,
                error_message="Component not registered"
            )
        
        component = self.components[component_name]
        
        # Check if component is on hold
        if component.metadata and component.metadata.get("status") == "on_hold":
            component.status = ComponentStatus.OPTIONAL
            component.error_message = "Optional component - not currently needed"
            component.last_checked = datetime.now()
            return component
        
        try:
            if component_name == "sqlite":
                component.status, component.version, component.error_message = await self._check_sqlite()
            elif component_name == "mongodb":
                component.status, component.version, component.error_message = await self._check_mongodb()
            elif component_name == "redis":
                component.status, component.version, component.error_message = await self._check_redis()
            elif component_name == "elasticsearch":
                component.status, component.version, component.error_message = await self._check_elasticsearch()
            elif component_name == "chromadb":
                component.status, component.version, component.error_message = await self._check_chromadb()
            elif component_name == "database_optimizer":
                component.status, component.version, component.error_message = await self._check_database_optimizer()
            elif component_name == "report_generator":
                component.status, component.version, component.error_message = await self._check_report_generator()
            else:
                component.status = ComponentStatus.UNKNOWN
                component.error_message = "No check method implemented"
                
        except Exception as e:
            component.status = ComponentStatus.ERROR
            component.error_message = str(e)
            logger.error(f"Error checking {component_name}: {e}")
        
        component.last_checked = datetime.now()
        return component
    
    async def _check_sqlite(self) -> tuple[ComponentStatus, Optional[str], Optional[str]]:
        """Check SQLite/PostgreSQL database status"""
        try:
            from research_synthesis.database.connection import AsyncSessionLocal, init_postgres
            from research_synthesis.database.models import Report
            from sqlalchemy import select, text, func
            
            # Ensure database is initialized
            if AsyncSessionLocal is None:
                try:
                    await init_postgres()
                except Exception as init_e:
                    return ComponentStatus.DISCONNECTED, None, f"Failed to initialize: {init_e}"
            
            # Use the session factory directly
            if AsyncSessionLocal is None:
                return ComponentStatus.DISCONNECTED, None, "Database not initialized"
            
            async with AsyncSessionLocal() as db_session:
                try:
                    # Test basic query
                    result = await db_session.execute(select(func.count(Report.id)))
                    count = result.scalar()
                    
                    # Get database version
                    version_result = await db_session.execute(text("SELECT sqlite_version()"))
                    version = version_result.scalar()
                    
                    return ComponentStatus.CONNECTED, f"SQLite {version}", f"Reports: {count}"
                except Exception as inner_e:
                    return ComponentStatus.ERROR, None, str(inner_e)
                
        except Exception as e:
            return ComponentStatus.DISCONNECTED, None, str(e)
    
    async def _check_mongodb(self) -> tuple[ComponentStatus, Optional[str], Optional[str]]:
        """Check MongoDB status"""
        try:
            from research_synthesis.database.connection import mongodb_client, init_mongodb
            
            # Ensure MongoDB is initialized
            if mongodb_client is None:
                try:
                    await init_mongodb()
                except Exception as init_e:
                    return ComponentStatus.NOT_INSTALLED, None, f"Failed to initialize: {init_e}"
            
            if mongodb_client is None:
                return ComponentStatus.NOT_INSTALLED, None, "MongoDB client not available"
            
            # Test connection by trying to get server info
            try:
                server_info = await mongodb_client.server_info()
                version = server_info.get('version', 'unknown')
                return ComponentStatus.CONNECTED, f"MongoDB {version}", None
            except Exception as conn_error:
                if "connection" in str(conn_error).lower() or "refused" in str(conn_error).lower():
                    return ComponentStatus.DISCONNECTED, None, str(conn_error)
                return ComponentStatus.ERROR, None, str(conn_error)
            
        except Exception as e:
            return ComponentStatus.NOT_INSTALLED, None, str(e)
    
    async def _check_redis(self) -> tuple[ComponentStatus, Optional[str], Optional[str]]:
        """Check Redis status"""
        try:
            from research_synthesis.database.connection import get_redis
            redis_client = get_redis()
            
            if not redis_client:
                return ComponentStatus.NOT_INSTALLED, None, "Redis client not initialized"
            
            # Test ping with timeout
            try:
                pong = await asyncio.wait_for(redis_client.ping(), timeout=1.0)
                if pong:
                    # Get version info with timeout
                    info = await asyncio.wait_for(redis_client.info(), timeout=1.0)
                    version = info.get('redis_version', 'unknown')
                    return ComponentStatus.CONNECTED, f"Redis {version}", None
            except asyncio.TimeoutError:
                return ComponentStatus.DISCONNECTED, None, "Connection timeout"
            
            return ComponentStatus.DISCONNECTED, None, "Ping failed"
            
        except Exception as e:
            if "connection" in str(e).lower() or "refused" in str(e).lower():
                return ComponentStatus.DISCONNECTED, None, str(e)
            return ComponentStatus.NOT_INSTALLED, None, str(e)
    
    async def _check_elasticsearch(self) -> tuple[ComponentStatus, Optional[str], Optional[str]]:
        """Check Elasticsearch status"""
        try:
            from research_synthesis.database.connection import get_elasticsearch
            es_client = get_elasticsearch()
            
            if not es_client:
                return ComponentStatus.NOT_INSTALLED, None, "Elasticsearch client not initialized"
            
            # Test connection with timeout
            try:
                info = await asyncio.wait_for(es_client.info(), timeout=2.0)
                version = info['version']['number']
                return ComponentStatus.CONNECTED, f"Elasticsearch {version}", None
            except asyncio.TimeoutError:
                return ComponentStatus.DISCONNECTED, None, "Connection timeout"
            
        except Exception as e:
            if "connection" in str(e).lower() or "refused" in str(e).lower():
                return ComponentStatus.DISCONNECTED, None, str(e)
            return ComponentStatus.NOT_INSTALLED, None, str(e)
    
    async def _check_chromadb(self) -> tuple[ComponentStatus, Optional[str], Optional[str]]:
        """Check ChromaDB status"""
        try:
            from research_synthesis.database.connection import chroma_client, init_chromadb
            
            # Ensure ChromaDB is initialized
            if chroma_client is None:
                try:
                    await init_chromadb()
                except Exception as init_e:
                    return ComponentStatus.NOT_INSTALLED, None, f"Failed to initialize: {init_e}"
            
            if chroma_client is None:
                return ComponentStatus.NOT_INSTALLED, None, "ChromaDB client not available"
            
            # Test basic operation
            try:
                collections = chroma_client.list_collections()
                return ComponentStatus.CONNECTED, "ChromaDB", f"Collections: {len(collections)}"
            except Exception as op_error:
                return ComponentStatus.ERROR, None, str(op_error)
            
        except Exception as e:
            return ComponentStatus.NOT_INSTALLED, None, str(e)
    
    async def _check_database_optimizer(self) -> tuple[ComponentStatus, Optional[str], Optional[str]]:
        """Check database optimizer service"""
        try:
            from research_synthesis.utils.database_optimizer import database_optimizer
            
            # Test optimizer functionality
            from research_synthesis.utils.database_optimizer import OperationType
            optimal_db = database_optimizer.choose_optimal_database(
                OperationType.READ, 1.0, priority="cost"
            )
            
            return ComponentStatus.CONNECTED, "v1.0", None
            
        except Exception as e:
            return ComponentStatus.ERROR, None, str(e)
    
    async def _check_report_generator(self) -> tuple[ComponentStatus, Optional[str], Optional[str]]:
        """Check report generator service"""
        try:
            # Test report generation capability
            from research_synthesis.database.connection import AsyncSessionLocal, init_postgres
            from research_synthesis.database.models import Report
            from sqlalchemy import select, func
            
            # Ensure database is initialized
            if AsyncSessionLocal is None:
                try:
                    await init_postgres()
                except Exception as init_e:
                    return ComponentStatus.ERROR, None, f"Failed to initialize database: {init_e}"
            
            if AsyncSessionLocal is None:
                return ComponentStatus.ERROR, None, "Database not initialized"
            
            async with AsyncSessionLocal() as db_session:
                try:
                    # Check if we can access reports table
                    result = await db_session.execute(select(func.count(Report.id)))
                    count = result.scalar()
                    return ComponentStatus.CONNECTED, "v1.0", f"Reports available: {count}"
                except Exception as inner_e:
                    return ComponentStatus.ERROR, None, str(inner_e)
                
        except Exception as e:
            return ComponentStatus.ERROR, None, str(e)
    
    async def get_system_status(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get comprehensive system status"""
        now = datetime.now()
        
        # Check cache
        if (not force_refresh and 
            self.last_full_check and 
            (now - self.last_full_check).seconds < self.cache_ttl and
            self._status_cache):
            return self._status_cache
        
        # Prevent concurrent checks
        if self.check_in_progress:
            await asyncio.sleep(0.1)
            return self._status_cache if self._status_cache else {"status": "checking"}
        
        self.check_in_progress = True
        
        try:
            # Check all components
            status_tasks = [
                self.check_component_status(name) 
                for name in self.components.keys()
            ]
            
            component_results = await asyncio.gather(*status_tasks, return_exceptions=True)
            
            # Process results
            components_status = {}
            critical_down = 0
            total_connected = 0
            
            for i, result in enumerate(component_results):
                component_name = list(self.components.keys())[i]
                
                if isinstance(result, Exception):
                    component_info = ComponentInfo(
                        name=component_name,
                        type=self.components[component_name].type,
                        status=ComponentStatus.ERROR,
                        error_message=str(result)
                    )
                else:
                    component_info = result
                
                # Convert component info to dict with enum values as strings
                component_dict = asdict(component_info)
                component_dict["type"] = component_info.type.value
                component_dict["status"] = component_info.status.value
                components_status[component_name] = component_dict
                
                # Count statistics - treat OPTIONAL components as connected
                if component_info.status in (ComponentStatus.CONNECTED, ComponentStatus.OPTIONAL):
                    total_connected += 1
                elif component_info.is_critical and component_info.status not in (ComponentStatus.CONNECTED, ComponentStatus.OPTIONAL):
                    critical_down += 1
            
            # Determine overall system health
            total_components = len(self.components)
            health_percentage = (total_connected / total_components) * 100
            
            if critical_down > 0:
                system_health = "critical"
            elif health_percentage >= 80:
                system_health = "healthy"
            elif health_percentage >= 60:
                system_health = "degraded"
            else:
                system_health = "unhealthy"
            
            # Build status response
            status_response = {
                "timestamp": now.isoformat(),
                "system_health": system_health,
                "health_percentage": round(health_percentage, 1),
                "components": components_status,
                "summary": {
                    "total_components": total_components,
                    "connected": total_connected,
                    "critical_down": critical_down,
                    "has_fallbacks": sum(1 for c in self.components.values() if c.fallback_available)
                },
                "recommendations": self._get_recommendations(components_status)
            }
            
            # Cache results
            self._status_cache = status_response
            self.last_full_check = now
            
            return status_response
            
        finally:
            self.check_in_progress = False
    
    def _get_recommendations(self, components: Dict[str, Any]) -> List[str]:
        """Generate system recommendations based on component status"""
        recommendations = []
        
        for name, info in components.items():
            status = info['status']
            metadata = info.get('metadata', {})
            is_on_hold = metadata.get('status') == 'on_hold'
            
            if status == ComponentStatus.DISCONNECTED.value:
                if name == "mongodb":
                    recommendations.append("MongoDB server not running - using SQLite for all operations")
            
            elif status == ComponentStatus.NOT_INSTALLED.value:
                if name == "mongodb" and not is_on_hold:
                    recommendations.append("MongoDB available but not connected - check configuration")
        
        # Add note about components on hold
        on_hold_components = [name for name, info in components.items() 
                             if info.get('metadata', {}).get('status') == 'on_hold']
        
        if on_hold_components:
            recommendations.append(f"Components on hold: {', '.join(on_hold_components)} - not currently in use")
        
        if not recommendations or len(recommendations) == 1 and "on hold" in recommendations[0]:
            recommendations.append("System running optimally with intelligent fallbacks")
        
        return recommendations
    
    async def get_component_status(self, component_name: str) -> Dict[str, Any]:
        """Get status for a specific component"""
        component = await self.check_component_status(component_name)
        # Convert to dict with enum values as strings
        component_dict = asdict(component)
        component_dict["type"] = component.type.value
        component_dict["status"] = component.status.value
        return component_dict
    
    def get_status_badge_info(self, component_name: str, status: ComponentStatus) -> Dict[str, str]:
        """Get badge information for UI display"""
        badge_map = {
            ComponentStatus.CONNECTED: {"class": "bg-success", "text": "Connected"},
            ComponentStatus.DISCONNECTED: {"class": "bg-warning", "text": "Disconnected"},
            ComponentStatus.NOT_INSTALLED: {"class": "bg-secondary", "text": "Not Installed"},
            ComponentStatus.ERROR: {"class": "bg-danger", "text": "Error"},
            ComponentStatus.STARTING: {"class": "bg-info", "text": "Starting"},
            ComponentStatus.STOPPING: {"class": "bg-warning", "text": "Stopping"},
            ComponentStatus.UNKNOWN: {"class": "bg-light text-dark", "text": "Unknown"}
        }
        
        return badge_map.get(status, {"class": "bg-secondary", "text": "Unknown"})


# Global system status service instance
system_status_service = SystemStatusService()


# Convenience functions
async def get_system_status(force_refresh: bool = False) -> Dict[str, Any]:
    """Get complete system status"""
    return await system_status_service.get_system_status(force_refresh)


async def get_database_status() -> Dict[str, Any]:
    """Get database-specific status (for backward compatibility)"""
    full_status = await system_status_service.get_system_status()
    
    # Convert to legacy format for existing templates
    database_status = {}
    
    # Map component names to legacy names for template compatibility
    component_mapping = {
        "sqlite": "postgresql",  # Dashboard expects "postgresql" key for SQLite
        "mongodb": "mongodb",
        "redis": "redis",
        "elasticsearch": "elasticsearch"
    }
    
    for component_name, component_info in full_status["components"].items():
        if component_info["type"] in ["database", "cache", "search"]:
            status_map = {
                "connected": "connected",
                "disconnected": "disconnected", 
                "not_installed": "not_installed",
                "error": "disconnected"
            }
            
            # Use legacy name if available, otherwise use original name
            legacy_name = component_mapping.get(component_name, component_name)
            
            database_status[legacy_name] = {
                "status": status_map.get(component_info["status"], "unknown"),
                "type": component_info["type"],
                "error": component_info.get("error_message")
            }
    
    return database_status


async def check_component_health(component_name: str) -> bool:
    """Quick health check for a component"""
    component = await system_status_service.check_component_status(component_name)
    return component.status == ComponentStatus.CONNECTED
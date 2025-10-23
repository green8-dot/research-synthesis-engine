"""
Enhanced Database Health Checker with Fallback Mechanisms
"""
import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class DatabaseHealthChecker:
    """Enhanced database health monitoring with fallback mechanisms"""
    
    def __init__(self):
        self.health_status = {}
        self.fallback_connections = {}
        
    async def check_all_databases(self) -> Dict[str, Any]:
        """Check health of all database connections"""
        
        checks = {
            "mongodb": self._check_mongodb,
            "postgres": self._check_postgres,
            "elasticsearch": self._check_elasticsearch,
            "chromadb": self._check_chromadb
        }
        
        results = {}
        for db_name, check_func in checks.items():
            try:
                results[db_name] = await check_func()
            except Exception as e:
                results[db_name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
        # Calculate overall health
        healthy_count = sum(1 for r in results.values() if r.get("status") == "healthy")
        total_count = len(results)
        
        overall_status = {
            "overall_health": "healthy" if healthy_count >= total_count * 0.8 else "degraded",
            "healthy_connections": healthy_count,
            "total_connections": total_count,
            "individual_status": results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return overall_status
        
    async def _check_mongodb(self) -> Dict[str, Any]:
        """Check MongoDB connection health"""
        try:
            from research_synthesis.database.connection import mongodb_client
            if mongodb_client:
                # Test with a simple ping
                await mongodb_client.admin.command('ping')
                return {"status": "healthy", "type": "mongodb"}
            else:
                return {"status": "disconnected", "type": "mongodb"}
        except Exception as e:
            return {"status": "error", "type": "mongodb", "error": str(e)}
            
    async def _check_postgres(self) -> Dict[str, Any]:
        """Check PostgreSQL/SQLite connection health"""
        try:
            from research_synthesis.database.connection import get_postgres_session
            async for session in get_postgres_session():
                # Simple query test
                await session.execute("SELECT 1")
                return {"status": "healthy", "type": "postgres"}
            return {"status": "error", "type": "postgres", "error": "No session available"}
        except Exception as e:
            return {"status": "error", "type": "postgres", "error": str(e)}
            
    async def _check_elasticsearch(self) -> Dict[str, Any]:
        """Check Elasticsearch connection health"""
        try:
            from research_synthesis.database.connection import elasticsearch_client
            if elasticsearch_client:
                health = await elasticsearch_client.cluster.health()
                return {"status": "healthy", "type": "elasticsearch", "cluster_status": health.get("status")}
            else:
                return {"status": "disabled", "type": "elasticsearch", "message": "Not configured"}
        except Exception as e:
            return {"status": "error", "type": "elasticsearch", "error": str(e)}
            
    async def _check_chromadb(self) -> Dict[str, Any]:
        """Check ChromaDB connection health"""
        try:
            from research_synthesis.database.connection import chroma_client
            if chroma_client:
                # Test basic functionality
                collections = chroma_client.list_collections()
                return {"status": "healthy", "type": "chromadb", "collections": len(collections)}
            else:
                return {"status": "error", "type": "chromadb", "error": "Client not available"}
        except Exception as e:
            return {"status": "error", "type": "chromadb", "error": str(e)}

# Global instance
health_checker = DatabaseHealthChecker()

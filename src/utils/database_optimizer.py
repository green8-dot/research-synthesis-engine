"""
Database Usage Optimizer - Smart Database Selection Logic
Automatically chooses the most cost-effective database for each operation
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Literal
from enum import Enum
from dataclasses import dataclass
from loguru import logger

class DatabaseType(Enum):
    """Available database types with cost implications"""
    SQLITE = "sqlite"           # Free, local
    MONGODB = "mongodb"         # Costs money (Atlas), local free
    REDIS = "redis"             # Costs money (cloud), local free  
    ELASTICSEARCH = "elasticsearch"  # Costs money (cloud), local free
    CHROMADB = "chromadb"       # Free, local

class OperationType(Enum):
    """Types of database operations"""
    READ = "read"
    WRITE = "write" 
    SEARCH = "search"
    AGGREGATE = "aggregate"
    CACHE = "cache"

@dataclass
class DatabaseCapabilities:
    """Database capabilities and cost structure"""
    type: DatabaseType
    available: bool
    is_cloud: bool
    cost_per_operation: float  # Estimated cost in USD
    cost_per_gb_storage: float
    max_free_operations: int   # Operations per month before charges
    strengths: List[str]
    limitations: List[str]

class DatabaseOptimizer:
    """Intelligently selects databases to minimize costs while maximizing performance"""
    
    def __init__(self):
        self.operation_counts = {}  # Track usage to avoid overages
        self.monthly_reset = datetime.now().replace(day=1, hour=0, minute=0, second=0)
        self.database_config = self._initialize_database_config()
        
    def _initialize_database_config(self) -> Dict[DatabaseType, DatabaseCapabilities]:
        """Initialize database configuration with cost and capability data"""
        return {
            DatabaseType.SQLITE: DatabaseCapabilities(
                type=DatabaseType.SQLITE,
                available=True,
                is_cloud=False,
                cost_per_operation=0.0,  # Completely free
                cost_per_gb_storage=0.0,
                max_free_operations=-1,  # Unlimited
                strengths=["Free", "Fast for small datasets", "No network latency", "ACID compliance"],
                limitations=["Single-node only", "Limited concurrent writes", "Storage size limits"]
            ),
            
            DatabaseType.MONGODB: DatabaseCapabilities(
                type=DatabaseType.MONGODB,
                available=self._check_mongodb_available(),
                is_cloud=self._is_mongodb_cloud(),
                cost_per_operation=0.00001,  # ~$0.01 per 1000 operations
                cost_per_gb_storage=0.25,    # ~$0.25 per GB/month
                max_free_operations=1000000,  # 1M operations/month free tier
                strengths=["Document storage", "Flexible schema", "Horizontal scaling", "Rich queries"],
                limitations=["Cost for large datasets", "Memory intensive", "Complex for simple data"]
            ),
            
            DatabaseType.REDIS: DatabaseCapabilities(
                type=DatabaseType.REDIS,
                available=self._check_redis_available(),
                is_cloud=self._is_redis_cloud(),
                cost_per_operation=0.000001,  # Very fast, minimal cost
                cost_per_gb_storage=0.50,     # Higher storage cost (in-memory)
                max_free_operations=10000000, # 10M operations/month
                strengths=["Extremely fast", "In-memory", "Pub/Sub", "Data structures"],
                limitations=["Memory cost", "Data persistence concerns", "Size limits"]
            ),
            
            DatabaseType.ELASTICSEARCH: DatabaseCapabilities(
                type=DatabaseType.ELASTICSEARCH,
                available=self._check_elasticsearch_available(),
                is_cloud=self._is_elasticsearch_cloud(),
                cost_per_operation=0.00005,   # Higher cost for complex searches
                cost_per_gb_storage=0.30,     # Moderate storage cost
                max_free_operations=100000,   # Lower free tier
                strengths=["Full-text search", "Analytics", "Real-time", "Scalable"],
                limitations=["High cost for large datasets", "Complex setup", "Resource intensive"]
            ),
            
            DatabaseType.CHROMADB: DatabaseCapabilities(
                type=DatabaseType.CHROMADB,
                available=self._check_chromadb_available(),
                is_cloud=False,  # Currently local only
                cost_per_operation=0.0,
                cost_per_gb_storage=0.0,
                max_free_operations=-1,
                strengths=["Vector similarity", "AI/ML integration", "Free", "Easy setup"],
                limitations=["Specialized use case", "Limited query types", "New technology"]
            )
        }
    
    def choose_optimal_database(self, 
                              operation: OperationType, 
                              data_size_mb: float,
                              query_complexity: Literal["simple", "medium", "complex"] = "simple",
                              priority: Literal["cost", "performance", "balanced"] = "balanced") -> DatabaseType:
        """
        Choose the optimal database for a specific operation
        
        Args:
            operation: Type of database operation
            data_size_mb: Expected data size in MB
            query_complexity: Complexity of the query/operation
            priority: Optimization priority (cost vs performance)
            
        Returns:
            Recommended database type
        """
        
        # Check if monthly limits reset
        self._check_monthly_reset()
        
        # Get available databases
        available_dbs = [db for db in self.database_config.values() if db.available]
        
        if not available_dbs:
            logger.warning("No databases available - defaulting to SQLite")
            return DatabaseType.SQLITE
        
        # Score databases based on operation type and priorities
        scores = {}
        
        for db in available_dbs:
            score = self._calculate_database_score(db, operation, data_size_mb, query_complexity, priority)
            scores[db.type] = score
        
        # Choose highest scoring database
        optimal_db = max(scores, key=scores.get)
        
        # Log the decision
        logger.info(f"Database selection: {optimal_db.value} (score: {scores[optimal_db]:.2f}) for {operation.value} operation")
        
        # Track usage
        self._track_operation(optimal_db)
        
        return optimal_db
    
    def _calculate_database_score(self, 
                                 db: DatabaseCapabilities,
                                 operation: OperationType,
                                 data_size_mb: float,
                                 query_complexity: str,
                                 priority: str) -> float:
        """Calculate a score for a database given the operation requirements"""
        
        score = 0.0
        
        # Base availability score
        if not db.available:
            return -1000  # Heavily penalize unavailable databases
        
        # Cost scoring (higher is better for lower cost)
        if priority in ["cost", "balanced"]:
            cost_factor = 100 if priority == "cost" else 50
            
            if db.cost_per_operation == 0:
                score += cost_factor  # Bonus for free operations
            else:
                # Penalize based on estimated monthly cost
                estimated_ops = self._estimate_monthly_operations(operation)
                monthly_cost = estimated_ops * db.cost_per_operation
                storage_cost = (data_size_mb / 1024) * db.cost_per_gb_storage
                total_cost = monthly_cost + storage_cost
                
                # Score inversely related to cost (lower cost = higher score)
                score += cost_factor / (1 + total_cost)
        
        # Performance scoring
        if priority in ["performance", "balanced"]:
            perf_factor = 100 if priority == "performance" else 50
            
            # Operation-specific performance bonuses
            if operation == OperationType.CACHE and db.type == DatabaseType.REDIS:
                score += perf_factor * 2  # Redis excels at caching
            elif operation == OperationType.SEARCH and db.type == DatabaseType.ELASTICSEARCH:
                score += perf_factor * 2  # Elasticsearch excels at search
            elif operation in [OperationType.READ, OperationType.WRITE] and db.type == DatabaseType.SQLITE:
                if data_size_mb < 100:  # SQLite great for small data
                    score += perf_factor * 1.5
            elif operation == OperationType.AGGREGATE and db.type == DatabaseType.MONGODB:
                score += perf_factor * 1.5  # MongoDB good for aggregation
        
        # Data size considerations
        if data_size_mb > 1000:  # Large datasets (>1GB)
            if db.type == DatabaseType.SQLITE:
                score -= 50  # SQLite not ideal for large data
            elif db.type in [DatabaseType.MONGODB, DatabaseType.ELASTICSEARCH]:
                score += 20  # These handle large data better
        
        # Query complexity considerations
        complexity_multiplier = {"simple": 1.0, "medium": 1.2, "complex": 1.5}[query_complexity]
        
        if query_complexity == "complex":
            if db.type == DatabaseType.ELASTICSEARCH:
                score += 30  # Elasticsearch handles complex queries well
            elif db.type == DatabaseType.SQLITE:
                score -= 20  # SQLite limited for complex operations
        
        # Free tier usage considerations
        if db.max_free_operations > 0:
            current_usage = self.operation_counts.get(db.type, 0)
            remaining_free = db.max_free_operations - current_usage
            
            if remaining_free <= 0:
                score -= 200  # Heavy penalty for exceeding free tier
            elif remaining_free < 10000:  # Close to limit
                score -= 50   # Moderate penalty
        
        # Local vs cloud considerations
        if db.is_cloud and priority == "cost":
            score -= 30  # Prefer local for cost optimization
        
        return score
    
    def _estimate_monthly_operations(self, operation: OperationType) -> int:
        """Estimate monthly operations based on operation type"""
        # Base estimates (can be refined with actual usage data)
        estimates = {
            OperationType.READ: 50000,      # Frequent reads
            OperationType.WRITE: 10000,     # Less frequent writes
            OperationType.SEARCH: 5000,     # Occasional searches
            OperationType.AGGREGATE: 1000,  # Rare aggregations
            OperationType.CACHE: 100000     # Very frequent cache operations
        }
        return estimates.get(operation, 10000)
    
    def _track_operation(self, db_type: DatabaseType):
        """Track database operations for cost monitoring"""
        if db_type not in self.operation_counts:
            self.operation_counts[db_type] = 0
        self.operation_counts[db_type] += 1
    
    def _check_monthly_reset(self):
        """Check if we need to reset monthly counters"""
        now = datetime.now()
        if now > self.monthly_reset + timedelta(days=32):  # Next month
            self.operation_counts = {}
            self.monthly_reset = now.replace(day=1, hour=0, minute=0, second=0)
            logger.info("Monthly database usage counters reset")
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get current database usage and cost estimates"""
        summary = {
            "month": self.monthly_reset.strftime("%Y-%m"),
            "databases": {},
            "total_estimated_cost": 0.0,
            "warnings": []
        }
        
        for db_type, count in self.operation_counts.items():
            db_config = self.database_config[db_type]
            estimated_cost = count * db_config.cost_per_operation
            
            db_summary = {
                "operations": count,
                "estimated_cost": estimated_cost,
                "free_operations_remaining": max(0, db_config.max_free_operations - count) if db_config.max_free_operations > 0 else "unlimited",
                "is_over_limit": count > db_config.max_free_operations if db_config.max_free_operations > 0 else False
            }
            
            summary["databases"][db_type.value] = db_summary
            summary["total_estimated_cost"] += estimated_cost
            
            # Add warnings for databases approaching limits
            if db_config.max_free_operations > 0:
                remaining = db_config.max_free_operations - count
                if remaining <= 0:
                    summary["warnings"].append(f"{db_type.value}: Exceeded free tier limit!")
                elif remaining < 10000:
                    summary["warnings"].append(f"{db_type.value}: Approaching free tier limit ({remaining:,} operations remaining)")
        
        return summary
    
    def get_cost_optimization_recommendations(self) -> List[str]:
        """Get recommendations for reducing database costs"""
        recommendations = []
        usage = self.get_usage_summary()
        
        # Check for over-usage of expensive databases
        for db_name, stats in usage["databases"].items():
            db_type = DatabaseType(db_name)
            db_config = self.database_config[db_type]
            
            if stats["is_over_limit"]:
                recommendations.append(f"Consider migrating some {db_name} operations to SQLite to avoid overage charges")
            
            if stats["estimated_cost"] > 5.0:  # More than $5/month
                if db_type == DatabaseType.ELASTICSEARCH:
                    recommendations.append("High Elasticsearch usage detected. Consider using SQLite FTS for simpler searches")
                elif db_type == DatabaseType.MONGODB:
                    recommendations.append("High MongoDB usage detected. Consider using SQLite for structured data")
        
        # General recommendations
        if usage["total_estimated_cost"] > 20.0:  # More than $20/month
            recommendations.append("Consider setting up local database instances to reduce cloud costs")
            
        if not recommendations:
            recommendations.append("Your current database usage is cost-optimized!")
            
        return recommendations
    
    # Helper methods to check database availability
    def _check_mongodb_available(self) -> bool:
        try:
            from research_synthesis.database.connection import get_mongodb
            return get_mongodb() is not None
        except:
            return False
    
    def _check_redis_available(self) -> bool:
        try:
            from research_synthesis.database.connection import get_redis
            return get_redis() is not None
        except:
            return False
    
    def _check_elasticsearch_available(self) -> bool:
        try:
            from research_synthesis.database.connection import get_elasticsearch
            return get_elasticsearch() is not None
        except:
            return False
            
    def _check_chromadb_available(self) -> bool:
        try:
            import chromadb
            return True
        except:
            return False
    
    def _is_mongodb_cloud(self) -> bool:
        """Check if MongoDB is running on cloud (Atlas)"""
        try:
            from research_synthesis.config.settings import settings
            return "mongodb.net" in settings.MONGODB_URI or "atlas" in settings.MONGODB_URI.lower()
        except:
            return False
    
    def _is_redis_cloud(self) -> bool:
        """Check if Redis is running on cloud"""
        try:
            from research_synthesis.config.settings import settings
            return "redis" in settings.redis_url and ("amazonaws.com" in settings.redis_url or "redis.com" in settings.redis_url)
        except:
            return False
    
    def _is_elasticsearch_cloud(self) -> bool:
        """Check if Elasticsearch is running on cloud"""
        try:
            from research_synthesis.config.settings import settings
            return "cloud.elastic.co" in getattr(settings, 'ELASTICSEARCH_URL', '') or "amazonaws.com" in getattr(settings, 'ELASTICSEARCH_URL', '')
        except:
            return False

# Global optimizer instance
database_optimizer = DatabaseOptimizer()

# Convenience functions for common operations
def get_optimal_database_for_caching() -> DatabaseType:
    """Get optimal database for caching operations"""
    return database_optimizer.choose_optimal_database(
        OperationType.CACHE,
        data_size_mb=1.0,  # Small cache data
        priority="cost"
    )

def get_optimal_database_for_search(data_size_mb: float = 10.0) -> DatabaseType:
    """Get optimal database for search operations"""
    return database_optimizer.choose_optimal_database(
        OperationType.SEARCH,
        data_size_mb=data_size_mb,
        query_complexity="medium",
        priority="balanced"
    )

def get_optimal_database_for_storage(data_size_mb: float) -> DatabaseType:
    """Get optimal database for data storage"""
    if data_size_mb < 100:  # Small data - prefer free SQLite
        priority = "cost"
    else:  # Large data - balance cost and performance
        priority = "balanced"
        
    return database_optimizer.choose_optimal_database(
        OperationType.WRITE,
        data_size_mb=data_size_mb,
        priority=priority
    )
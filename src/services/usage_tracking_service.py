"""
Usage Tracking Service
Tracks MongoDB operations, Claude API usage, and calculates costs in real-time
"""
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import os
from pathlib import Path

from loguru import logger
from research_synthesis.database.connection import get_mongodb, get_redis
from research_synthesis.config.settings import settings


@dataclass
class UsageRecord:
    """Individual usage record"""
    timestamp: datetime
    service: str  # 'mongodb', 'claude', 'openai'
    operation: str  # 'read', 'write', 'query', 'api_call'
    resource_type: str  # 'collection', 'document', 'tokens', 'requests'
    quantity: int  # Number of operations/tokens/documents
    cost_usd: float = 0.0
    metadata: Dict[str, Any] = None


class UsageTracker:
    """Real-time usage tracking and cost calculation"""
    
    # Claude API Pricing (as of 2024)
    CLAUDE_PRICING = {
        'claude-3-sonnet': {
            'input_tokens': 0.000003,   # $3 per 1M input tokens
            'output_tokens': 0.000015,  # $15 per 1M output tokens
        },
        'claude-3-haiku': {
            'input_tokens': 0.00000025, # $0.25 per 1M input tokens  
            'output_tokens': 0.00000125, # $1.25 per 1M output tokens
        },
        'claude-3-opus': {
            'input_tokens': 0.000015,   # $15 per 1M input tokens
            'output_tokens': 0.000075,  # $75 per 1M output tokens
        }
    }
    
    # MongoDB Atlas Pricing (estimated)
    MONGODB_PRICING = {
        'read_operation': 0.0000001,    # $0.1 per 1M reads
        'write_operation': 0.0000005,   # $0.5 per 1M writes
        'storage_gb_hour': 0.00034,     # $0.25 per GB per month / 730 hours
        'bandwidth_gb': 0.10            # $0.10 per GB transfer
    }
    
    def __init__(self):
        self.usage_file = Path("data/usage_tracking.json")
        self.usage_file.parent.mkdir(exist_ok=True)
        self.daily_usage = defaultdict(list)
        self.monthly_totals = defaultdict(float)
        self._load_usage_history()
    
    def _load_usage_history(self):
        """Load existing usage history"""
        if self.usage_file.exists():
            try:
                with open(self.usage_file, 'r') as f:
                    data = json.load(f)
                    for date_str, records in data.get('daily_usage', {}).items():
                        self.daily_usage[date_str] = [
                            UsageRecord(**record) for record in records
                        ]
                    self.monthly_totals = data.get('monthly_totals', {})
            except Exception as e:
                logger.warning(f"Failed to load usage history: {e}")
    
    def _save_usage_history(self):
        """Save usage history to file"""
        try:
            data = {
                'daily_usage': {
                    date_str: [asdict(record) for record in records]
                    for date_str, records in self.daily_usage.items()
                },
                'monthly_totals': dict(self.monthly_totals),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.usage_file, 'w') as f:
                json.dump(data, f, default=str, indent=2)
        except Exception as e:
            logger.error(f"Failed to save usage history: {e}")
    
    async def track_mongodb_operation(self, operation: str, collection: str, 
                                    document_count: int = 1, data_size_kb: float = 0):
        """Track MongoDB operation and calculate cost"""
        try:
            cost = 0.0
            if operation in ['read', 'find', 'query']:
                cost = document_count * self.MONGODB_PRICING['read_operation']
            elif operation in ['write', 'insert', 'update', 'delete']:
                cost = document_count * self.MONGODB_PRICING['write_operation']
            
            # Add storage cost estimate
            if data_size_kb > 0:
                storage_cost = (data_size_kb / 1024 / 1024) * self.MONGODB_PRICING['storage_gb_hour']
                cost += storage_cost
            
            record = UsageRecord(
                timestamp=datetime.now(),
                service='mongodb',
                operation=operation,
                resource_type='document',
                quantity=document_count,
                cost_usd=cost,
                metadata={
                    'collection': collection,
                    'data_size_kb': data_size_kb
                }
            )
            
            await self._record_usage(record)
            logger.debug(f"MongoDB usage tracked: {operation} on {collection}, cost: ${cost:.6f}")
            
        except Exception as e:
            logger.error(f"Failed to track MongoDB usage: {e}")
    
    async def track_claude_api_usage(self, model: str, input_tokens: int, 
                                   output_tokens: int, operation: str = 'api_call'):
        """Track Claude API usage and calculate cost"""
        try:
            model_pricing = self.CLAUDE_PRICING.get(model, self.CLAUDE_PRICING['claude-3-sonnet'])
            
            input_cost = input_tokens * model_pricing['input_tokens']
            output_cost = output_tokens * model_pricing['output_tokens']
            total_cost = input_cost + output_cost
            
            record = UsageRecord(
                timestamp=datetime.now(),
                service='claude',
                operation=operation,
                resource_type='tokens',
                quantity=input_tokens + output_tokens,
                cost_usd=total_cost,
                metadata={
                    'model': model,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'input_cost': input_cost,
                    'output_cost': output_cost
                }
            )
            
            await self._record_usage(record)
            logger.debug(f"Claude usage tracked: {model}, tokens: {input_tokens + output_tokens}, cost: ${total_cost:.6f}")
            
        except Exception as e:
            logger.error(f"Failed to track Claude usage: {e}")
    
    async def _record_usage(self, record: UsageRecord):
        """Record usage in memory and persistent storage"""
        date_key = record.timestamp.strftime('%Y-%m-%d')
        month_key = record.timestamp.strftime('%Y-%m')
        
        # Add to daily usage
        self.daily_usage[date_key].append(record)
        
        # Update monthly totals
        service_key = f"{month_key}_{record.service}"
        self.monthly_totals[service_key] += record.cost_usd
        
        # Save to Redis for real-time access
        redis_client = get_redis()
        if redis_client:
            try:
                await redis_client.lpush(
                    f"usage_tracking:recent", 
                    json.dumps(asdict(record), default=str)
                )
                # Keep only last 1000 records
                await redis_client.ltrim("usage_tracking:recent", 0, 999)
                
                # Update daily totals
                await redis_client.incrbyfloat(f"usage_tracking:daily:{date_key}", record.cost_usd)
                await redis_client.expire(f"usage_tracking:daily:{date_key}", 86400 * 30)  # 30 days
                
                # Update monthly totals
                await redis_client.incrbyfloat(f"usage_tracking:monthly:{service_key}", record.cost_usd)
                await redis_client.expire(f"usage_tracking:monthly:{service_key}", 86400 * 365)  # 1 year
                
            except Exception as e:
                logger.warning(f"Failed to cache usage data in Redis: {e}")
        
        # Periodic save to file
        if len(self.daily_usage[date_key]) % 10 == 0:
            self._save_usage_history()
    
    async def get_usage_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        total_cost = 0.0
        service_costs = defaultdict(float)
        operation_counts = defaultdict(int)
        daily_costs = {}
        
        # Analyze usage records
        for days_back in range(days):
            date = end_date - timedelta(days=days_back)
            date_key = date.strftime('%Y-%m-%d')
            daily_cost = 0.0
            
            for record in self.daily_usage.get(date_key, []):
                total_cost += record.cost_usd
                service_costs[record.service] += record.cost_usd
                operation_counts[record.operation] += 1
                daily_cost += record.cost_usd
            
            daily_costs[date_key] = daily_cost
        
        # Get current month costs
        current_month = datetime.now().strftime('%Y-%m')
        mongodb_monthly = self.monthly_totals.get(f"{current_month}_mongodb", 0.0)
        claude_monthly = self.monthly_totals.get(f"{current_month}_claude", 0.0)
        
        # Get database storage info
        storage_stats = await self._get_storage_stats()
        
        return {
            'period': f"{start_date} to {end_date}",
            'total_cost_usd': round(total_cost, 6),
            'service_breakdown': {
                'mongodb': {
                    'cost_usd': round(service_costs['mongodb'], 6),
                    'monthly_cost_usd': round(mongodb_monthly, 6),
                    'operations': operation_counts.get('read', 0) + operation_counts.get('write', 0)
                },
                'claude': {
                    'cost_usd': round(service_costs['claude'], 6),
                    'monthly_cost_usd': round(claude_monthly, 6),
                    'api_calls': operation_counts.get('api_call', 0)
                }
            },
            'daily_breakdown': {
                date: round(cost, 6) for date, cost in daily_costs.items()
            },
            'storage_stats': storage_stats,
            'projected_monthly': {
                'total_usd': round((mongodb_monthly + claude_monthly) * (30 / datetime.now().day), 2),
                'mongodb_usd': round(mongodb_monthly * (30 / datetime.now().day), 2),
                'claude_usd': round(claude_monthly * (30 / datetime.now().day), 2)
            }
        }
    
    async def _get_storage_stats(self) -> Dict[str, Any]:
        """Get current storage statistics from MongoDB"""
        try:
            mongodb = get_mongodb()
            if not mongodb:
                return {'error': 'MongoDB not available'}
            
            # Get database stats
            stats = await mongodb.command('dbStats')
            collections = await mongodb.list_collection_names()
            
            collection_stats = {}
            for collection_name in collections:
                try:
                    collection = mongodb[collection_name]
                    count = await collection.count_documents({})
                    # Estimate size (rough calculation)
                    if count > 0:
                        sample = await collection.find_one()
                        estimated_doc_size = len(str(sample)) if sample else 1000
                        estimated_size_mb = (count * estimated_doc_size) / (1024 * 1024)
                    else:
                        estimated_size_mb = 0
                    
                    collection_stats[collection_name] = {
                        'documents': count,
                        'estimated_size_mb': round(estimated_size_mb, 2)
                    }
                except Exception as e:
                    logger.warning(f"Failed to get stats for collection {collection_name}: {e}")
            
            return {
                'database_size_mb': round(stats.get('dataSize', 0) / (1024 * 1024), 2),
                'storage_size_mb': round(stats.get('storageSize', 0) / (1024 * 1024), 2),
                'collections': collection_stats,
                'estimated_monthly_storage_cost_usd': round(
                    (stats.get('storageSize', 0) / (1024 * 1024 * 1024)) * 
                    self.MONGODB_PRICING['storage_gb_hour'] * 730, 4
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {'error': str(e)}


# Global instance
usage_tracker = UsageTracker()


# Decorator to automatically track MongoDB operations
def track_mongodb(operation: str, collection: str):
    """Decorator to automatically track MongoDB operations"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = datetime.now()
            result = await func(*args, **kwargs)
            
            # Estimate document count and size
            doc_count = 1
            data_size = 0
            
            if hasattr(result, '__len__'):
                try:
                    doc_count = len(result)
                except:
                    doc_count = 1
            
            if isinstance(result, (list, dict)):
                data_size = len(str(result)) / 1024  # KB
            
            await usage_tracker.track_mongodb_operation(
                operation=operation,
                collection=collection,
                document_count=doc_count,
                data_size_kb=data_size
            )
            
            return result
        return wrapper
    return decorator


# Decorator to track Claude API calls
def track_claude_usage(model: str = 'claude-3-sonnet'):
    """Decorator to track Claude API usage"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Extract token usage from result or estimate
            input_tokens = getattr(result, 'input_tokens', 0)
            output_tokens = getattr(result, 'output_tokens', 0)
            
            # Estimate if not available
            if not input_tokens and len(args) > 0:
                input_text = str(args[0]) if args else ""
                input_tokens = len(input_text.split()) * 1.3  # Rough estimation
            
            if not output_tokens and hasattr(result, 'content'):
                output_text = str(result.content)
                output_tokens = len(output_text.split()) * 1.3  # Rough estimation
            
            await usage_tracker.track_claude_api_usage(
                model=model,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens)
            )
            
            return result
        return wrapper
    return decorator
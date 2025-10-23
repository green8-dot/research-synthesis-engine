"""
Advanced Cache Management System
Provides disk-based persistent caching with intelligent invalidation
"""

import diskcache
import aiocache
import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Advanced cache management with disk persistence and in-memory acceleration
    """
    
    def __init__(self, cache_dir: str = "cache", max_memory_cache_size: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Disk cache for persistence
        self.disk_cache = diskcache.Cache(str(self.cache_dir))
        
        # Memory cache for speed
        self.memory_cache = {}
        self.max_memory_cache_size = max_memory_cache_size
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "disk_hits": 0,
            "memory_hits": 0,
            "writes": 0,
            "evictions": 0
        }
        
        logger.info(f"Cache manager initialized with disk cache at {self.cache_dir}")
        
    def _generate_key(self, key: str, namespace: str = "default") -> str:
        """Generate a namespaced cache key"""
        return f"{namespace}:{key}"
        
    def _hash_key(self, data: Any) -> str:
        """Generate a hash key for complex data"""
        if isinstance(data, (str, int, float, bool)):
            return str(data)
        elif isinstance(data, dict):
            # Sort dict keys for consistent hashing
            sorted_data = json.dumps(data, sort_keys=True)
            return hashlib.md5(sorted_data.encode()).hexdigest()
        else:
            return hashlib.md5(str(data).encode()).hexdigest()
    
    def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get value from cache (memory first, then disk)"""
        full_key = self._generate_key(key, namespace)
        
        # Check memory cache first
        if full_key in self.memory_cache:
            self.stats["hits"] += 1
            self.stats["memory_hits"] += 1
            return self.memory_cache[full_key]["value"]
            
        # Check disk cache
        try:
            value = self.disk_cache.get(full_key)
            if value is not None:
                self.stats["hits"] += 1
                self.stats["disk_hits"] += 1
                
                # Promote to memory cache
                self._set_memory_cache(full_key, value)
                return value
        except Exception as e:
            logger.error(f"Error reading from disk cache: {e}")
            
        self.stats["misses"] += 1
        return None
        
    def set(self, key: str, value: Any, ttl: int = 3600, namespace: str = "default"):
        """Set value in both memory and disk cache"""
        full_key = self._generate_key(key, namespace)
        
        try:
            # Set in disk cache with TTL
            self.disk_cache.set(full_key, value, expire=ttl)
            
            # Set in memory cache
            self._set_memory_cache(full_key, value, ttl)
            
            self.stats["writes"] += 1
            logger.debug(f"Cached {full_key} with TTL {ttl}s")
            
        except Exception as e:
            logger.error(f"Error writing to cache: {e}")
            
    def _set_memory_cache(self, key: str, value: Any, ttl: int = 3600):
        """Set value in memory cache with eviction"""
        expiry = datetime.now() + timedelta(seconds=ttl)
        
        # Evict expired entries
        self._evict_expired_memory_cache()
        
        # Evict LRU if at capacity
        if len(self.memory_cache) >= self.max_memory_cache_size:
            self._evict_lru_memory_cache()
            
        self.memory_cache[key] = {
            "value": value,
            "expiry": expiry,
            "access_count": 1,
            "last_access": datetime.now()
        }
        
    def _evict_expired_memory_cache(self):
        """Remove expired entries from memory cache"""
        now = datetime.now()
        expired_keys = [
            key for key, data in self.memory_cache.items()
            if data["expiry"] < now
        ]
        
        for key in expired_keys:
            del self.memory_cache[key]
            self.stats["evictions"] += 1
            
    def _evict_lru_memory_cache(self):
        """Evict least recently used entry from memory cache"""
        if not self.memory_cache:
            return
            
        lru_key = min(
            self.memory_cache.keys(),
            key=lambda k: self.memory_cache[k]["last_access"]
        )
        
        del self.memory_cache[lru_key]
        self.stats["evictions"] += 1
        
    def delete(self, key: str, namespace: str = "default"):
        """Delete key from both caches"""
        full_key = self._generate_key(key, namespace)
        
        # Delete from memory cache
        if full_key in self.memory_cache:
            del self.memory_cache[full_key]
            
        # Delete from disk cache
        try:
            self.disk_cache.delete(full_key)
        except Exception as e:
            logger.error(f"Error deleting from disk cache: {e}")
            
    def clear_namespace(self, namespace: str = "default"):
        """Clear all keys in a namespace"""
        prefix = f"{namespace}:"
        
        # Clear from memory cache
        memory_keys_to_delete = [
            key for key in self.memory_cache.keys()
            if key.startswith(prefix)
        ]
        for key in memory_keys_to_delete:
            del self.memory_cache[key]
            
        # Clear from disk cache
        try:
            disk_keys_to_delete = [
                key for key in self.disk_cache.iterkeys()
                if key.startswith(prefix)
            ]
            for key in disk_keys_to_delete:
                self.disk_cache.delete(key)
        except Exception as e:
            logger.error(f"Error clearing namespace from disk cache: {e}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2),
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_size": len(self.disk_cache),
            "cache_dir": str(self.cache_dir),
            "timestamp": datetime.now().isoformat()
        }
        
    def get_cache_info(self, namespace: str = None) -> Dict[str, Any]:
        """Get detailed cache information"""
        info = {
            "memory_cache": {
                "size": len(self.memory_cache),
                "max_size": self.max_memory_cache_size,
                "keys": list(self.memory_cache.keys()) if namespace is None 
                       else [k for k in self.memory_cache.keys() if k.startswith(f"{namespace}:")]
            },
            "disk_cache": {
                "size": len(self.disk_cache),
                "volume_info": self.disk_cache.volume(),
                "keys": list(self.disk_cache.iterkeys()) if namespace is None
                       else [k for k in self.disk_cache.iterkeys() if k.startswith(f"{namespace}:")]
            },
            "statistics": self.get_stats()
        }
        
        return info
        
    def optimize(self):
        """Optimize cache by cleaning expired entries and defragmenting"""
        logger.info("Starting cache optimization...")
        
        # Clean expired memory cache entries
        initial_memory_size = len(self.memory_cache)
        self._evict_expired_memory_cache()
        memory_cleaned = initial_memory_size - len(self.memory_cache)
        
        # Clean expired disk cache entries
        initial_disk_size = len(self.disk_cache)
        try:
            self.disk_cache.cull()  # Remove expired entries
        except Exception as e:
            logger.error(f"Error culling disk cache: {e}")
            
        final_disk_size = len(self.disk_cache)
        disk_cleaned = initial_disk_size - final_disk_size
        
        logger.info(f"Cache optimization complete: cleaned {memory_cleaned} memory entries, "
                   f"{disk_cleaned} disk entries")
                   
        return {
            "memory_entries_cleaned": memory_cleaned,
            "disk_entries_cleaned": disk_cleaned,
            "final_memory_size": len(self.memory_cache),
            "final_disk_size": final_disk_size
        }
        
    def close(self):
        """Close cache connections"""
        try:
            self.disk_cache.close()
            self.memory_cache.clear()
            logger.info("Cache manager closed successfully")
        except Exception as e:
            logger.error(f"Error closing cache manager: {e}")

# Cache decorators for easy usage
class cached:
    """Decorator for caching function results"""
    
    def __init__(self, ttl: int = 3600, namespace: str = "functions", key_func=None):
        self.ttl = ttl
        self.namespace = namespace
        self.key_func = key_func
        
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if self.key_func:
                cache_key = self.key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{cache_manager._hash_key((args, kwargs))}"
                
            # Try to get from cache
            result = cache_manager.get(cache_key, self.namespace)
            if result is not None:
                return result
                
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, self.ttl, self.namespace)
            
            return result
            
        wrapper._cache_info = lambda: cache_manager.get_cache_info(self.namespace)
        wrapper._cache_clear = lambda: cache_manager.clear_namespace(self.namespace)
        
        return wrapper

# Global cache manager instance
cache_manager = CacheManager()

# Convenience functions
def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics"""
    return cache_manager.get_stats()

def clear_cache(namespace: str = None):
    """Clear cache namespace or all cache"""
    if namespace:
        cache_manager.clear_namespace(namespace)
    else:
        cache_manager.memory_cache.clear()
        cache_manager.disk_cache.clear()

def optimize_cache():
    """Optimize global cache"""
    return cache_manager.optimize()

def get_cached(key: str, namespace: str = "default") -> Any:
    """Get value from cache"""
    return cache_manager.get(key, namespace)

def set_cached(key: str, value: Any, ttl: int = 3600, namespace: str = "default"):
    """Set value in cache"""
    cache_manager.set(key, value, ttl, namespace)
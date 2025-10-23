"""
ML Intelligent Caching System
Provides smart caching for ML predictions, model results, and expensive computations
"""

import asyncio
import hashlib
import json
import pickle
import time
import threading
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List, Callable, Union
from functools import wraps
from pathlib import Path

from loguru import logger
import psutil


class MLCacheEntry:
    """Represents a cached ML result with metadata"""
    def __init__(self, result: Any, ttl: int, cache_time: datetime, 
                 computation_time: float, memory_size: int):
        self.result = result
        self.ttl = ttl
        self.cache_time = cache_time
        self.expires_at = cache_time + timedelta(seconds=ttl)
        self.computation_time = computation_time  # Original computation time in ms
        self.memory_size = memory_size  # Approximate memory size in bytes
        self.access_count = 1
        self.last_accessed = cache_time
        
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return datetime.now() > self.expires_at
    
    def access(self):
        """Mark entry as accessed (for LRU tracking)"""
        self.access_count += 1
        self.last_accessed = datetime.now()


class MLCache:
    """Intelligent caching system for ML operations"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 1800, 
                 max_memory_mb: int = 500, enable_persistence: bool = True):
        self.max_size = max_size
        self.default_ttl = default_ttl  # 30 minutes default
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.enable_persistence = enable_persistence
        
        # Cache storage
        self.cache: OrderedDict[str, MLCacheEntry] = OrderedDict()
        self.current_memory_usage = 0
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0,
            'total_computation_time_saved': 0.0
        }
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Persistence
        self.cache_file = Path("data/ml_cache.pkl") if enable_persistence else None
        if self.enable_persistence:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._load_cache_from_disk()
        
        # Background cleanup task
        self._cleanup_task = None
        self._start_background_cleanup()
        
        logger.info(f"ML Cache initialized: max_size={max_size}, default_ttl={default_ttl}s, max_memory={max_memory_mb}MB")
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate deterministic cache key from function signature"""
        try:
            # Create a deterministic representation
            key_data = {
                'function': func_name,
                'args': str(args),
                'kwargs': json.dumps(kwargs, sort_keys=True, default=str)
            }
            
            key_string = json.dumps(key_data, sort_keys=True)
            return hashlib.md5(key_string.encode()).hexdigest()
            
        except Exception as e:
            # Fallback to simple string representation
            key_data = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
            return hashlib.md5(key_data.encode()).hexdigest()
    
    def _estimate_memory_size(self, obj: Any) -> int:
        """Estimate memory size of an object"""
        try:
            # Use pickle to estimate size
            return len(pickle.dumps(obj))
        except:
            # Fallback estimation
            if isinstance(obj, (str, int, float, bool)):
                return 64  # Small objects
            elif isinstance(obj, (list, dict)):
                return 1024 * len(obj) if len(obj) < 1000 else 1024 * 1000  # Estimate
            else:
                return 4096  # Default size
    
    def _cleanup_expired_entries(self):
        """Remove expired entries from cache"""
        with self.lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                logger.debug(f"ML Cache: Removed {len(expired_keys)} expired entries")
    
    def _enforce_size_limit(self):
        """Enforce cache size and memory limits using LRU eviction"""
        with self.lock:
            evicted_count = 0
            
            # Remove entries until we're within limits
            while (len(self.cache) > self.max_size or 
                   self.current_memory_usage > self.max_memory_bytes):
                
                if not self.cache:
                    break
                
                # Remove least recently used entry (first item in OrderedDict)
                oldest_key = next(iter(self.cache))
                self._remove_entry(oldest_key)
                evicted_count += 1
            
            if evicted_count > 0:
                self.stats['evictions'] += evicted_count
                logger.debug(f"ML Cache: Evicted {evicted_count} entries (LRU)")
    
    def _remove_entry(self, key: str):
        """Remove a single entry from cache"""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_memory_usage -= entry.memory_size
    
    def _move_to_end(self, key: str):
        """Move entry to end of OrderedDict (most recently used)"""
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key].access()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            self.stats['total_requests'] += 1
            
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if expired
                if entry.is_expired():
                    self._remove_entry(key)
                    self.stats['misses'] += 1
                    return None
                
                # Cache hit
                self._move_to_end(key)
                self.stats['hits'] += 1
                self.stats['total_computation_time_saved'] += entry.computation_time
                
                logger.debug(f"ML Cache HIT: {key[:8]}... (saved {entry.computation_time:.1f}ms)")
                return entry.result
            
            # Cache miss
            self.stats['misses'] += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None, 
            computation_time: float = 0.0):
        """Put item in cache"""
        with self.lock:
            # Calculate memory size
            memory_size = self._estimate_memory_size(value)
            
            # Skip caching if item is too large
            if memory_size > self.max_memory_bytes // 4:  # Don't cache items >25% of total memory
                logger.warning(f"ML Cache: Skipping large object (size: {memory_size / 1024 / 1024:.1f}MB)")
                return
            
            # Create cache entry
            entry = MLCacheEntry(
                result=value,
                ttl=ttl or self.default_ttl,
                cache_time=datetime.now(),
                computation_time=computation_time,
                memory_size=memory_size
            )
            
            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)
            
            # Add new entry
            self.cache[key] = entry
            self.current_memory_usage += memory_size
            
            # Enforce limits
            self._enforce_size_limit()
            
            logger.debug(f"ML Cache PUT: {key[:8]}... (ttl={entry.ttl}s, size={memory_size}B)")
    
    def cached(self, ttl: Optional[int] = None, cache_key_func: Optional[Callable] = None):
        """Decorator for caching ML function results"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                if cache_key_func:
                    cache_key = cache_key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                # Check cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and measure time
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    computation_time = (time.time() - start_time) * 1000  # Convert to ms
                    
                    # Cache result
                    self.put(cache_key, result, ttl, computation_time)
                    
                    logger.debug(f"ML Cache MISS: {func.__name__} (computed in {computation_time:.1f}ms)")
                    return result
                    
                except Exception as e:
                    logger.error(f"ML Cache: Error in {func.__name__}: {e}")
                    raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For synchronous functions
                if cache_key_func:
                    cache_key = cache_key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    computation_time = (time.time() - start_time) * 1000
                    
                    self.put(cache_key, result, ttl, computation_time)
                    return result
                    
                except Exception as e:
                    logger.error(f"ML Cache: Error in {func.__name__}: {e}")
                    raise
            
            # Return appropriate wrapper
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    def invalidate(self, pattern: Optional[str] = None):
        """Invalidate cache entries matching pattern"""
        with self.lock:
            if pattern is None:
                # Clear all cache
                self.cache.clear()
                self.current_memory_usage = 0
                logger.info("ML Cache: All entries invalidated")
            else:
                # Remove entries matching pattern
                keys_to_remove = [key for key in self.cache.keys() if pattern in key]
                for key in keys_to_remove:
                    self._remove_entry(key)
                logger.info(f"ML Cache: Invalidated {len(keys_to_remove)} entries matching '{pattern}'")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            hit_rate = (self.stats['hits'] / max(self.stats['total_requests'], 1)) * 100
            
            return {
                'total_requests': self.stats['total_requests'],
                'cache_hits': self.stats['hits'],
                'cache_misses': self.stats['misses'],
                'hit_rate_percent': round(hit_rate, 2),
                'evictions': self.stats['evictions'],
                'current_size': len(self.cache),
                'max_size': self.max_size,
                'memory_usage_mb': round(self.current_memory_usage / 1024 / 1024, 2),
                'max_memory_mb': round(self.max_memory_bytes / 1024 / 1024, 2),
                'time_saved_seconds': round(self.stats['total_computation_time_saved'] / 1000, 2),
                'average_entry_size_kb': round(
                    (self.current_memory_usage / max(len(self.cache), 1)) / 1024, 2
                )
            }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information"""
        with self.lock:
            entries_by_age = []
            for key, entry in self.cache.items():
                age_seconds = (datetime.now() - entry.cache_time).total_seconds()
                entries_by_age.append({
                    'key': key[:16] + '...',  # Truncated key
                    'age_seconds': round(age_seconds, 1),
                    'access_count': entry.access_count,
                    'size_kb': round(entry.memory_size / 1024, 2),
                    'computation_time_ms': entry.computation_time
                })
            
            # Sort by access count (most accessed first)
            entries_by_age.sort(key=lambda x: x['access_count'], reverse=True)
            
            return {
                'cache_entries': entries_by_age[:20],  # Top 20 entries
                'statistics': self.get_stats(),
                'system_memory': {
                    'system_memory_percent': psutil.virtual_memory().percent,
                    'cache_memory_percent': (self.current_memory_usage / self.max_memory_bytes) * 100
                }
            }
    
    def preload_common_queries(self, preload_data: List[Dict[str, Any]]):
        """Preload cache with common queries for warming up"""
        logger.info(f"ML Cache: Preloading {len(preload_data)} common queries")
        
        for item in preload_data:
            key = item['key']
            value = item['value']
            ttl = item.get('ttl', self.default_ttl)
            
            self.put(key, value, ttl)
        
        logger.info("ML Cache: Preloading complete")
    
    def _start_background_cleanup(self):
        """Start background task for periodic cleanup"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # Run every 5 minutes
                    self._cleanup_expired_entries()
                    
                    # Save cache to disk periodically
                    if self.enable_persistence:
                        self._save_cache_to_disk()
                        
                except Exception as e:
                    logger.error(f"ML Cache cleanup error: {e}")
        
        if asyncio.get_event_loop().is_running():
            self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    def _save_cache_to_disk(self):
        """Save cache to disk for persistence"""
        if not self.enable_persistence or not self.cache_file:
            return
        
        try:
            with self.lock:
                cache_data = {
                    'cache': dict(self.cache),
                    'stats': self.stats,
                    'current_memory_usage': self.current_memory_usage
                }
                
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                    
                logger.debug(f"ML Cache: Saved {len(self.cache)} entries to disk")
                
        except Exception as e:
            logger.warning(f"ML Cache: Failed to save to disk: {e}")
    
    def _load_cache_from_disk(self):
        """Load cache from disk on startup"""
        if not self.enable_persistence or not self.cache_file or not self.cache_file.exists():
            return
        
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
                self.cache = OrderedDict(cache_data.get('cache', {}))
                self.stats = cache_data.get('stats', self.stats)
                self.current_memory_usage = cache_data.get('current_memory_usage', 0)
                
                # Remove expired entries
                self._cleanup_expired_entries()
                
                logger.info(f"ML Cache: Loaded {len(self.cache)} entries from disk")
                
        except Exception as e:
            logger.warning(f"ML Cache: Failed to load from disk: {e}")
            self.cache.clear()
            self.stats = {'hits': 0, 'misses': 0, 'evictions': 0, 'total_requests': 0, 'total_computation_time_saved': 0.0}
            self.current_memory_usage = 0
    
    def shutdown(self):
        """Clean shutdown of cache system"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        if self.enable_persistence:
            self._save_cache_to_disk()
        
        logger.info("ML Cache: Shutdown complete")


# Global cache instance
ml_cache = MLCache(
    max_size=1000,
    default_ttl=1800,  # 30 minutes
    max_memory_mb=500,  # 500MB max cache memory
    enable_persistence=True
)
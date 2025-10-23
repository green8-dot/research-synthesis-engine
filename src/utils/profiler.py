"""
Performance profiling utilities for the Research Synthesis Engine
Provides decorators and tools for monitoring function performance
"""

import time
import functools
import logging
from typing import Callable, Any, Dict
from datetime import datetime
import psutil
import gc

logger = logging.getLogger(__name__)

class PerformanceProfiler:
    """Performance profiling utility class"""
    
    def __init__(self):
        self.profiles: Dict[str, Dict[str, Any]] = {}
        
    def profile_function(self, func_name: str = None):
        """Decorator to profile function performance"""
        def decorator(func: Callable) -> Callable:
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self._profile_execution(name, func, args, kwargs)
            
            @functools.wraps(func)  
            async def async_wrapper(*args, **kwargs):
                return await self._profile_async_execution(name, func, args, kwargs)
            
            # Return appropriate wrapper based on function type
            if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
                return async_wrapper
            else:
                return sync_wrapper
                
        return decorator
    
    def _profile_execution(self, name: str, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Profile synchronous function execution"""
        # Get memory before execution
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        objects_before = len(gc.get_objects())
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # Calculate metrics
            execution_time = time.time() - start_time
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            objects_after = len(gc.get_objects())
            
            # Store profile data
            self._record_profile(name, {
                'execution_time_ms': round(execution_time * 1000, 2),
                'memory_used_mb': round(memory_after - memory_before, 2),
                'objects_created': objects_after - objects_before,
                'success': True,
                'timestamp': datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            self._record_profile(name, {
                'execution_time_ms': round(execution_time * 1000, 2),
                'memory_used_mb': 0,
                'objects_created': 0,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
            raise
    
    async def _profile_async_execution(self, name: str, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Profile asynchronous function execution"""
        # Get memory before execution
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        objects_before = len(gc.get_objects())
        
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            
            # Calculate metrics
            execution_time = time.time() - start_time
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            objects_after = len(gc.get_objects())
            
            # Store profile data
            self._record_profile(name, {
                'execution_time_ms': round(execution_time * 1000, 2),
                'memory_used_mb': round(memory_after - memory_before, 2),
                'objects_created': objects_after - objects_before,
                'success': True,
                'timestamp': datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            self._record_profile(name, {
                'execution_time_ms': round(execution_time * 1000, 2),
                'memory_used_mb': 0,
                'objects_created': 0,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
            raise
    
    def _record_profile(self, name: str, metrics: Dict[str, Any]):
        """Record profile metrics for a function"""
        if name not in self.profiles:
            self.profiles[name] = {
                'call_count': 0,
                'total_time_ms': 0,
                'avg_time_ms': 0,
                'total_memory_mb': 0,
                'avg_memory_mb': 0,
                'success_count': 0,
                'error_count': 0,
                'last_metrics': {}
            }
        
        profile = self.profiles[name]
        profile['call_count'] += 1
        profile['last_metrics'] = metrics
        
        if metrics['success']:
            profile['success_count'] += 1
            profile['total_time_ms'] += metrics['execution_time_ms']
            profile['avg_time_ms'] = profile['total_time_ms'] / profile['success_count']
            
            profile['total_memory_mb'] += metrics['memory_used_mb']
            profile['avg_memory_mb'] = profile['total_memory_mb'] / profile['success_count']
        else:
            profile['error_count'] += 1
        
        # Log slow functions (>1000ms)
        if metrics['execution_time_ms'] > 1000:
            logger.warning(f"Slow function detected: {name} took {metrics['execution_time_ms']:.2f}ms")
        
        # Log memory-intensive functions (>50MB)
        if metrics.get('memory_used_mb', 0) > 50:
            logger.warning(f"Memory-intensive function: {name} used {metrics['memory_used_mb']:.2f}MB")
    
    def get_profile_stats(self, function_name: str = None) -> Dict[str, Any]:
        """Get profiling statistics for all functions or a specific function"""
        if function_name:
            return self.profiles.get(function_name, {})
        return self.profiles
    
    def reset_profiles(self):
        """Clear all profiling data"""
        self.profiles.clear()
        logger.info("Performance profiles reset")

# Global profiler instance
profiler = PerformanceProfiler()

# Convenience decorator
def profile_performance(func_name: str = None):
    """Decorator to profile function performance"""
    return profiler.profile_function(func_name)

# Convenience functions
def get_performance_stats(function_name: str = None) -> Dict[str, Any]:
    """Get performance statistics"""
    return profiler.get_profile_stats(function_name)

def reset_performance_stats():
    """Reset all performance statistics"""
    profiler.reset_profiles()
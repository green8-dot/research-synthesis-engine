"""
Performance Monitoring API Router
Provides endpoints for monitoring application performance metrics
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/")
async def get_performance_overview():
    """Get performance monitoring overview"""
    try:
        from research_synthesis.utils.profiler import get_performance_stats
        from research_synthesis.utils.memory_manager import get_memory_stats
        from research_synthesis.utils.cache_manager import get_cache_stats
        
        performance_stats = get_performance_stats()
        memory_stats = get_memory_stats()
        cache_stats = get_cache_stats()
        
        # Calculate performance summary
        total_functions = len(performance_stats)
        slow_functions = sum(1 for stats in performance_stats.values() 
                           if stats.get('avg_time_ms', 0) > 1000)
        memory_intensive_functions = sum(1 for stats in performance_stats.values() 
                                       if stats.get('avg_memory_mb', 0) > 50)
        
        return {
            "performance_summary": {
                "total_profiled_functions": total_functions,
                "slow_functions": slow_functions,
                "memory_intensive_functions": memory_intensive_functions,
                "cache_hit_rate": cache_stats.get("hit_rate_percent", 0)
            },
            "memory_status": memory_stats,
            "cache_status": cache_stats,
            "function_profiles": performance_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting performance overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/functions")
async def get_function_profiles():
    """Get detailed performance profiles for all monitored functions"""
    try:
        from research_synthesis.utils.profiler import get_performance_stats
        return get_performance_stats()
    except Exception as e:
        logger.error(f"Error getting function profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/functions/{function_name}")
async def get_function_profile(function_name: str):
    """Get performance profile for a specific function"""
    try:
        from research_synthesis.utils.profiler import get_performance_stats
        profile = get_performance_stats(function_name)
        
        if not profile:
            raise HTTPException(status_code=404, detail=f"No profile data found for function: {function_name}")
        
        return {
            "function_name": function_name,
            "profile": profile
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting function profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/slow-functions")
async def get_slow_functions(threshold_ms: int = 1000):
    """Get list of functions that exceed the specified execution time threshold"""
    try:
        from research_synthesis.utils.profiler import get_performance_stats
        all_profiles = get_performance_stats()
        
        slow_functions = {
            name: stats for name, stats in all_profiles.items()
            if stats.get('avg_time_ms', 0) > threshold_ms
        }
        
        # Sort by average execution time (descending)
        sorted_slow = dict(sorted(
            slow_functions.items(), 
            key=lambda x: x[1].get('avg_time_ms', 0), 
            reverse=True
        ))
        
        return {
            "threshold_ms": threshold_ms,
            "slow_functions_count": len(sorted_slow),
            "slow_functions": sorted_slow
        }
        
    except Exception as e:
        logger.error(f"Error getting slow functions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/memory-intensive")
async def get_memory_intensive_functions(threshold_mb: float = 50.0):
    """Get list of functions that exceed the specified memory usage threshold"""
    try:
        from research_synthesis.utils.profiler import get_performance_stats
        all_profiles = get_performance_stats()
        
        memory_intensive = {
            name: stats for name, stats in all_profiles.items()
            if stats.get('avg_memory_mb', 0) > threshold_mb
        }
        
        # Sort by average memory usage (descending)
        sorted_memory = dict(sorted(
            memory_intensive.items(), 
            key=lambda x: x[1].get('avg_memory_mb', 0), 
            reverse=True
        ))
        
        return {
            "threshold_mb": threshold_mb,
            "memory_intensive_count": len(sorted_memory),
            "memory_intensive_functions": sorted_memory
        }
        
    except Exception as e:
        logger.error(f"Error getting memory intensive functions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache-stats")
async def get_cache_performance():
    """Get detailed cache performance statistics"""
    try:
        from research_synthesis.utils.cache_manager import get_cache_stats
        return get_cache_stats()
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset")
async def reset_performance_stats():
    """Reset all performance monitoring statistics"""
    try:
        from research_synthesis.utils.profiler import reset_performance_stats
        reset_performance_stats()
        
        return {
            "message": "Performance statistics reset successfully",
            "timestamp": "2025-08-24T21:30:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error resetting performance stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/benchmarks")
async def get_performance_benchmarks():
    """Get performance benchmarks and recommendations"""
    try:
        from research_synthesis.utils.profiler import get_performance_stats
        from research_synthesis.utils.memory_manager import get_memory_stats
        
        performance_stats = get_performance_stats()
        memory_stats = get_memory_stats()
        
        # Generate recommendations
        recommendations = []
        
        # Check for slow functions
        for name, stats in performance_stats.items():
            avg_time = stats.get('avg_time_ms', 0)
            if avg_time > 2000:
                recommendations.append({
                    "type": "critical",
                    "function": name,
                    "issue": f"Very slow execution: {avg_time:.2f}ms average",
                    "recommendation": "Consider caching, optimization, or async processing"
                })
            elif avg_time > 1000:
                recommendations.append({
                    "type": "warning", 
                    "function": name,
                    "issue": f"Slow execution: {avg_time:.2f}ms average",
                    "recommendation": "Consider performance optimization"
                })
        
        # Check memory usage
        memory_percent = memory_stats.get('system_memory_percent', 0)
        if memory_percent > 90:
            recommendations.append({
                "type": "critical",
                "function": "system",
                "issue": f"High memory usage: {memory_percent:.1f}%",
                "recommendation": "Immediate memory optimization required"
            })
        elif memory_percent > 80:
            recommendations.append({
                "type": "warning",
                "function": "system", 
                "issue": f"Elevated memory usage: {memory_percent:.1f}%",
                "recommendation": "Monitor memory usage closely"
            })
        
        return {
            "performance_grade": _calculate_performance_grade(performance_stats, memory_stats),
            "total_recommendations": len(recommendations),
            "recommendations": recommendations,
            "metrics_summary": {
                "average_response_time_ms": _calculate_average_response_time(performance_stats),
                "memory_efficiency_score": _calculate_memory_efficiency(memory_stats),
                "cache_efficiency_score": memory_stats.get('status', 'unknown')
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting performance benchmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _calculate_performance_grade(performance_stats: Dict, memory_stats: Dict) -> str:
    """Calculate overall performance grade"""
    slow_functions = sum(1 for stats in performance_stats.values() 
                        if stats.get('avg_time_ms', 0) > 1000)
    total_functions = max(len(performance_stats), 1)
    slow_ratio = slow_functions / total_functions
    
    memory_percent = memory_stats.get('system_memory_percent', 0)
    
    if slow_ratio < 0.1 and memory_percent < 70:
        return "A"
    elif slow_ratio < 0.2 and memory_percent < 80:
        return "B"  
    elif slow_ratio < 0.3 and memory_percent < 90:
        return "C"
    else:
        return "D"

def _calculate_average_response_time(performance_stats: Dict) -> float:
    """Calculate average response time across all functions"""
    if not performance_stats:
        return 0.0
    
    total_time = sum(stats.get('avg_time_ms', 0) for stats in performance_stats.values())
    return round(total_time / len(performance_stats), 2)

def _calculate_memory_efficiency(memory_stats: Dict) -> int:
    """Calculate memory efficiency score (0-100)"""
    memory_percent = memory_stats.get('system_memory_percent', 0)
    
    if memory_percent < 50:
        return 100
    elif memory_percent < 70:
        return 80
    elif memory_percent < 85:
        return 60
    elif memory_percent < 95:
        return 40
    else:
        return 20
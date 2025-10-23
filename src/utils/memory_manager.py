"""
Memory Management System for Research Synthesis Engine
Provides periodic garbage collection and memory monitoring
"""

import gc
import psutil
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Memory statistics container"""
    process_memory_mb: float
    system_memory_percent: float
    python_objects_count: int
    gc_collections: Dict[str, int]
    timestamp: datetime
    
class MemoryManager:
    """
    Handles periodic garbage collection and memory monitoring
    """
    
    def __init__(self):
        self.is_running = False
        self.cleanup_interval = 300  # 5 minutes
        self.monitoring_interval = 60  # 1 minute
        self.cleanup_thread: Optional[threading.Thread] = None
        self.monitoring_thread: Optional[threading.Thread] = None
        self.memory_history = []
        self.max_history_size = 100
        
        # Memory thresholds
        self.memory_warning_threshold = 80.0  # %
        self.memory_critical_threshold = 90.0  # %
        
        # Process info
        self.process = psutil.Process()
        
    def start(self):
        """Start memory management services"""
        if self.is_running:
            logger.warning("Memory manager already running")
            return
            
        self.is_running = True
        logger.info("Starting memory management system...")
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_worker, 
            daemon=True,
            name="MemoryCleanup"
        )
        self.cleanup_thread.start()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True, 
            name="MemoryMonitor"
        )
        self.monitoring_thread.start()
        
        logger.info("Memory management system started")
        
    def stop(self):
        """Stop memory management services"""
        self.is_running = False
        logger.info("Memory management system stopped")
        
    def _cleanup_worker(self):
        """Background worker for periodic garbage collection"""
        logger.info(f"Garbage collection worker started (interval: {self.cleanup_interval}s)")
        
        while self.is_running:
            try:
                # Perform garbage collection
                collected = self._perform_gc()
                
                # Log results if significant
                if collected > 0:
                    logger.info(f"Garbage collection freed {collected} objects")
                    
            except Exception as e:
                logger.error(f"Error in garbage collection worker: {e}")
                
            # Wait for next cleanup cycle
            for _ in range(self.cleanup_interval):
                if not self.is_running:
                    break
                time.sleep(1)
                
        logger.info("Garbage collection worker stopped")
        
    def _monitoring_worker(self):
        """Background worker for memory monitoring"""
        logger.info(f"Memory monitoring worker started (interval: {self.monitoring_interval}s)")
        
        while self.is_running:
            try:
                # Collect memory stats
                stats = self._collect_memory_stats()
                
                # Add to history
                self.memory_history.append(stats)
                if len(self.memory_history) > self.max_history_size:
                    self.memory_history.pop(0)
                    
                # Check thresholds
                self._check_memory_thresholds(stats)
                    
            except Exception as e:
                logger.error(f"Error in memory monitoring worker: {e}")
                
            # Wait for next monitoring cycle
            for _ in range(self.monitoring_interval):
                if not self.is_running:
                    break
                time.sleep(1)
                
        logger.info("Memory monitoring worker stopped")
        
    def _perform_gc(self) -> int:
        """Perform garbage collection and return objects collected"""
        try:
            # Force garbage collection for all generations
            collected = 0
            for generation in range(3):
                collected += gc.collect(generation)
                
            # Additional cleanup
            gc.collect()  # Final collection
            
            return collected
            
        except Exception as e:
            logger.error(f"Error during garbage collection: {e}")
            return 0
            
    def _collect_memory_stats(self) -> MemoryStats:
        """Collect current memory statistics"""
        try:
            # Process memory info
            memory_info = self.process.memory_info()
            process_memory_mb = memory_info.rss / (1024 * 1024)
            
            # System memory info
            system_memory = psutil.virtual_memory()
            system_memory_percent = system_memory.percent
            
            # Python objects count
            python_objects_count = len(gc.get_objects())
            
            # GC statistics
            gc_stats = gc.get_stats()
            gc_collections = {
                f"gen_{i}": stats.get("collections", 0) 
                for i, stats in enumerate(gc_stats)
            }
            
            return MemoryStats(
                process_memory_mb=process_memory_mb,
                system_memory_percent=system_memory_percent,
                python_objects_count=python_objects_count,
                gc_collections=gc_collections,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error collecting memory stats: {e}")
            return MemoryStats(
                process_memory_mb=0.0,
                system_memory_percent=0.0,
                python_objects_count=0,
                gc_collections={},
                timestamp=datetime.now()
            )
            
    def _check_memory_thresholds(self, stats: MemoryStats):
        """Check memory usage against thresholds and log warnings"""
        system_memory = stats.system_memory_percent
        
        if system_memory >= self.memory_critical_threshold:
            logger.critical(
                f"CRITICAL: System memory usage at {system_memory:.1f}% "
                f"(threshold: {self.memory_critical_threshold}%)"
            )
        elif system_memory >= self.memory_warning_threshold:
            logger.warning(
                f"WARNING: System memory usage at {system_memory:.1f}% "
                f"(threshold: {self.memory_warning_threshold}%)"
            )
            
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current memory statistics for health checks"""
        try:
            stats = self._collect_memory_stats()
            
            return {
                "process_memory_mb": round(stats.process_memory_mb, 2),
                "system_memory_percent": round(stats.system_memory_percent, 1),
                "python_objects": stats.python_objects_count,
                "gc_collections": stats.gc_collections,
                "status": self._get_memory_status(stats),
                "last_updated": stats.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting current memory stats: {e}")
            return {
                "error": str(e),
                "status": "error",
                "last_updated": datetime.now().isoformat()
            }
            
    def _get_memory_status(self, stats: MemoryStats) -> str:
        """Determine memory status based on usage"""
        system_memory = stats.system_memory_percent
        
        if system_memory >= self.memory_critical_threshold:
            return "critical"
        elif system_memory >= self.memory_warning_threshold:
            return "warning"
        else:
            return "healthy"
            
    def get_memory_trends(self, minutes: int = 30) -> Dict[str, Any]:
        """Get memory usage trends for the specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            recent_stats = [
                stats for stats in self.memory_history
                if stats.timestamp >= cutoff_time
            ]
            
            if not recent_stats:
                return {"error": "No recent memory data available"}
                
            # Calculate trends
            memory_values = [stats.system_memory_percent for stats in recent_stats]
            object_counts = [stats.python_objects_count for stats in recent_stats]
            
            return {
                "time_period_minutes": minutes,
                "data_points": len(recent_stats),
                "memory_usage": {
                    "current": round(memory_values[-1], 1),
                    "average": round(sum(memory_values) / len(memory_values), 1),
                    "min": round(min(memory_values), 1),
                    "max": round(max(memory_values), 1)
                },
                "python_objects": {
                    "current": object_counts[-1],
                    "average": int(sum(object_counts) / len(object_counts)),
                    "min": min(object_counts),
                    "max": max(object_counts)
                },
                "trend": "increasing" if memory_values[-1] > memory_values[0] else "stable"
            }
            
        except Exception as e:
            logger.error(f"Error getting memory trends: {e}")
            return {"error": str(e)}
            
    def force_cleanup(self) -> Dict[str, Any]:
        """Force immediate garbage collection and return results"""
        try:
            logger.info("Performing forced memory cleanup...")
            
            # Collect before stats
            before_stats = self._collect_memory_stats()
            
            # Perform cleanup
            collected = self._perform_gc()
            
            # Collect after stats  
            after_stats = self._collect_memory_stats()
            
            # Calculate improvements
            memory_freed_mb = before_stats.process_memory_mb - after_stats.process_memory_mb
            objects_freed = before_stats.python_objects_count - after_stats.python_objects_count
            
            result = {
                "objects_collected": collected,
                "objects_freed": objects_freed,
                "memory_freed_mb": round(memory_freed_mb, 2),
                "before": {
                    "memory_mb": round(before_stats.process_memory_mb, 2),
                    "objects": before_stats.python_objects_count
                },
                "after": {
                    "memory_mb": round(after_stats.process_memory_mb, 2), 
                    "objects": after_stats.python_objects_count
                },
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Forced cleanup completed: {collected} objects collected, "
                       f"{memory_freed_mb:.2f}MB freed")
                       
            return result
            
        except Exception as e:
            logger.error(f"Error during forced cleanup: {e}")
            return {"error": str(e)}

# Global memory manager instance
memory_manager = MemoryManager()

def start_memory_management():
    """Start the global memory manager"""
    memory_manager.start()
    
def stop_memory_management():
    """Stop the global memory manager"""
    memory_manager.stop()
    
def get_memory_stats() -> Dict[str, Any]:
    """Get current memory statistics"""
    return memory_manager.get_current_stats()
    
def get_memory_trends(minutes: int = 30) -> Dict[str, Any]:
    """Get memory trends"""
    return memory_manager.get_memory_trends(minutes)
    
def force_memory_cleanup() -> Dict[str, Any]:
    """Force immediate memory cleanup"""
    return memory_manager.force_cleanup()
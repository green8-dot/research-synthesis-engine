"""
System Resource Monitor
Provides real-time system resource information including CPU, memory, disk usage
"""

import psutil
import asyncio
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger


@dataclass
class ResourceSnapshot:
    """Single resource measurement snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    python_memory_mb: float


class SystemResourceMonitor:
    """Real-time system resource monitoring service"""
    
    def __init__(self, history_minutes: int = 60):
        self.history_minutes = history_minutes
        self.snapshots: List[ResourceSnapshot] = []
        self.monitoring_active = False
        self._last_network_io = None
        self._monitoring_task = None
        
    async def start_monitoring(self, interval_seconds: int = 5):
        """Start background resource monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop(interval_seconds))
        logger.info(f"System resource monitoring started (interval: {interval_seconds}s)")
        
    async def stop_monitoring(self):
        """Stop background resource monitoring"""
        self.monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("System resource monitoring stopped")
        
    async def _monitoring_loop(self, interval_seconds: int):
        """Background monitoring loop"""
        try:
            while self.monitoring_active:
                snapshot = await self.get_current_snapshot()
                self.snapshots.append(snapshot)
                
                # Clean old snapshots
                cutoff_time = datetime.now() - timedelta(minutes=self.history_minutes)
                self.snapshots = [s for s in self.snapshots if s.timestamp > cutoff_time]
                
                await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            logger.info("Resource monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in resource monitoring loop: {e}")
            
    async def get_current_snapshot(self) -> ResourceSnapshot:
        """Get current system resource snapshot"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            if hasattr(psutil.disk_usage, '__call__'):
                try:
                    # Windows - use C: drive
                    disk = psutil.disk_usage('C:')
                except:
                    # Fallback to current directory
                    disk = psutil.disk_usage('.')
            
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            # Python process memory
            current_process = psutil.Process()
            python_memory_mb = current_process.memory_info().rss / (1024**2)
            
            return ResourceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=round(memory_used_gb, 2),
                memory_total_gb=round(memory_total_gb, 2),
                disk_percent=disk.percent,
                disk_used_gb=round(disk_used_gb, 1),
                disk_total_gb=round(disk_total_gb, 1),
                network_bytes_sent=net_io.bytes_sent,
                network_bytes_recv=net_io.bytes_recv,
                process_count=process_count,
                python_memory_mb=round(python_memory_mb, 1)
            )
            
        except Exception as e:
            logger.error(f"Error getting system snapshot: {e}")
            # Return basic snapshot with minimal data
            return ResourceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_gb=0.0,
                memory_total_gb=0.0,
                disk_percent=0.0,
                disk_used_gb=0.0,
                disk_total_gb=0.0,
                network_bytes_sent=0,
                network_bytes_recv=0,
                process_count=0,
                python_memory_mb=0.0
            )
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get current resource summary with trends"""
        if not self.snapshots:
            return {"error": "No resource data available"}
            
        current = self.snapshots[-1]
        
        # Calculate trends if we have enough data
        trends = {}
        if len(self.snapshots) >= 2:
            prev = self.snapshots[-2]
            trends = {
                "cpu_trend": "up" if current.cpu_percent > prev.cpu_percent else "down" if current.cpu_percent < prev.cpu_percent else "stable",
                "memory_trend": "up" if current.memory_percent > prev.memory_percent else "down" if current.memory_percent < prev.memory_percent else "stable",
                "disk_trend": "up" if current.disk_percent > prev.disk_percent else "down" if current.disk_percent < prev.disk_percent else "stable"
            }
            
        # Calculate averages over last 10 minutes
        ten_min_ago = datetime.now() - timedelta(minutes=10)
        recent_snapshots = [s for s in self.snapshots if s.timestamp > ten_min_ago]
        
        averages = {}
        if recent_snapshots:
            averages = {
                "avg_cpu_10min": round(sum(s.cpu_percent for s in recent_snapshots) / len(recent_snapshots), 1),
                "avg_memory_10min": round(sum(s.memory_percent for s in recent_snapshots) / len(recent_snapshots), 1),
                "max_cpu_10min": round(max(s.cpu_percent for s in recent_snapshots), 1),
                "max_memory_10min": round(max(s.memory_percent for s in recent_snapshots), 1)
            }
            
        return {
            "timestamp": current.timestamp.isoformat(),
            "current": {
                "cpu_percent": current.cpu_percent,
                "memory_percent": current.memory_percent,
                "memory_used_gb": current.memory_used_gb,
                "memory_total_gb": current.memory_total_gb,
                "disk_percent": current.disk_percent,
                "disk_used_gb": current.disk_used_gb,
                "disk_total_gb": current.disk_total_gb,
                "process_count": current.process_count,
                "python_memory_mb": current.python_memory_mb
            },
            "trends": trends,
            "averages": averages,
            "monitoring_status": {
                "active": self.monitoring_active,
                "snapshots_count": len(self.snapshots),
                "history_minutes": self.history_minutes
            },
            "health_indicators": self._get_health_indicators(current)
        }
        
    def _get_health_indicators(self, snapshot: ResourceSnapshot) -> Dict[str, str]:
        """Get health status indicators based on resource usage"""
        indicators = {}
        
        # CPU health
        if snapshot.cpu_percent < 50:
            indicators["cpu"] = "healthy"
        elif snapshot.cpu_percent < 80:
            indicators["cpu"] = "warning"
        else:
            indicators["cpu"] = "critical"
            
        # Memory health
        if snapshot.memory_percent < 70:
            indicators["memory"] = "healthy"
        elif snapshot.memory_percent < 90:
            indicators["memory"] = "warning"
        else:
            indicators["memory"] = "critical"
            
        # Disk health
        if snapshot.disk_percent < 80:
            indicators["disk"] = "healthy"
        elif snapshot.disk_percent < 95:
            indicators["disk"] = "warning"
        else:
            indicators["disk"] = "critical"
            
        # Python process health
        if snapshot.python_memory_mb < 500:
            indicators["python"] = "healthy"
        elif snapshot.python_memory_mb < 1000:
            indicators["python"] = "warning"
        else:
            indicators["python"] = "critical"
            
        return indicators
        
    def get_history_data(self, minutes: int = None) -> List[Dict[str, Any]]:
        """Get historical resource data"""
        if minutes:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            snapshots = [s for s in self.snapshots if s.timestamp > cutoff_time]
        else:
            snapshots = self.snapshots
            
        return [
            {
                "timestamp": s.timestamp.isoformat(),
                "cpu_percent": s.cpu_percent,
                "memory_percent": s.memory_percent,
                "disk_percent": s.disk_percent,
                "python_memory_mb": s.python_memory_mb
            }
            for s in snapshots
        ]


# Global resource monitor instance
resource_monitor = SystemResourceMonitor()
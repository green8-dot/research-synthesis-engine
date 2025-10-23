"""
Comprehensive Job Management System

Advanced job management with:
- Real-time job monitoring
- Automatic cleanup and recovery
- Job lifecycle management
- Performance tracking
- Resource management
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from enum import Enum
import uuid
from dataclasses import dataclass
import time

from loguru import logger
from research_synthesis.database.connection import get_postgres_session
from research_synthesis.database.models import ScrapingJob
from sqlalchemy import select, update, delete
from sqlalchemy.exc import SQLAlchemyError


class JobPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class JobType(Enum):
    REDDIT_SCRAPING = "reddit_scraping"
    AUTOMATION_DISCOVERY = "automation_discovery"
    DATA_PROCESSING = "data_processing"
    REPORT_GENERATION = "report_generation"
    SYSTEM_MAINTENANCE = "system_maintenance"


@dataclass
class JobMetrics:
    """Job performance metrics"""
    job_id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0
    cpu_usage: float = 0
    memory_usage_mb: float = 0
    network_requests: int = 0
    success_rate: float = 0
    error_count: int = 0


class ComprehensiveJobManager:
    """Advanced job management with comprehensive monitoring and cleanup"""
    
    def __init__(self):
        # Configuration
        self.max_concurrent_jobs = 3
        self.job_timeout_minutes = 30
        self.cleanup_interval_seconds = 300  # 5 minutes
        self.health_check_interval_seconds = 60  # 1 minute
        self.max_retries = 3
        
        # Runtime state
        self.active_jobs: Set[int] = set()
        self.job_metrics: Dict[int, JobMetrics] = {}
        self.job_queue: List[Dict[str, Any]] = []
        self.cleanup_running = False
        self.health_monitor_running = False
        
        # Performance tracking
        self.total_jobs_processed = 0
        self.total_jobs_failed = 0
        self.average_job_duration = 0
        self.system_health_score = 100.0
        
        # Background services will be started when first accessed
        self._services_started = False
    
    async def _ensure_services_started(self):
        """Ensure background services are started"""
        if not self._services_started:
            await self._start_background_services()
            self._services_started = True
    
    async def _start_background_services(self):
        """Start background monitoring and cleanup services"""
        try:
            asyncio.create_task(self._continuous_cleanup())
            asyncio.create_task(self._health_monitor())
            asyncio.create_task(self._performance_tracker())
            logger.info("Comprehensive Job Manager background services started")
        except Exception as e:
            logger.error(f"Failed to start background services: {e}")
    
    async def _continuous_cleanup(self):
        """Continuous cleanup of hanging and failed jobs"""
        while True:
            try:
                if not self.cleanup_running:
                    self.cleanup_running = True
                    await self._comprehensive_cleanup()
                    self.cleanup_running = False
                
                await asyncio.sleep(self.cleanup_interval_seconds)
                
            except Exception as e:
                logger.error(f"Continuous cleanup error: {e}")
                self.cleanup_running = False
                await asyncio.sleep(30)  # Wait before retry
    
    async def _health_monitor(self):
        """Monitor system health and job performance"""
        while True:
            try:
                if not self.health_monitor_running:
                    self.health_monitor_running = True
                    await self._check_system_health()
                    self.health_monitor_running = False
                
                await asyncio.sleep(self.health_check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                self.health_monitor_running = False
                await asyncio.sleep(30)
    
    async def _performance_tracker(self):
        """Track and update performance metrics"""
        while True:
            try:
                await self._update_performance_metrics()
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Performance tracker error: {e}")
                await asyncio.sleep(30)
    
    async def _comprehensive_cleanup(self) -> Dict[str, Any]:
        """Comprehensive cleanup of all problematic jobs"""
        cleanup_results = {
            "hanging_jobs_cleaned": 0,
            "failed_jobs_cleaned": 0,
            "stuck_jobs_recovered": 0,
            "orphaned_jobs_removed": 0,
            "errors": [],
            "cleanup_time": datetime.utcnow().isoformat()
        }
        
        try:
            async for db_session in get_postgres_session():
                # 1. Clean hanging jobs (running too long)
                hanging_cleaned = await self._clean_hanging_jobs(db_session)
                cleanup_results["hanging_jobs_cleaned"] = len(hanging_cleaned)
                
                # 2. Clean permanently failed jobs
                failed_cleaned = await self._clean_failed_jobs(db_session)
                cleanup_results["failed_jobs_cleaned"] = len(failed_cleaned)
                
                # 3. Recover stuck jobs
                stuck_recovered = await self._recover_stuck_jobs(db_session)
                cleanup_results["stuck_jobs_recovered"] = len(stuck_recovered)
                
                # 4. Remove orphaned jobs
                orphaned_removed = await self._remove_orphaned_jobs(db_session)
                cleanup_results["orphaned_jobs_removed"] = len(orphaned_removed)
                
                # Update active jobs tracking
                await self._update_active_jobs_tracking(db_session)
                
                if any([hanging_cleaned, failed_cleaned, stuck_recovered, orphaned_removed]):
                    logger.info(f"Cleanup completed: {cleanup_results}")
                
                break
                
        except Exception as e:
            error_msg = f"Comprehensive cleanup failed: {e}"
            logger.error(error_msg)
            cleanup_results["errors"].append(error_msg)
        
        return cleanup_results
    
    async def _clean_hanging_jobs(self, db_session) -> List[int]:
        """Clean jobs that have been running too long"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(minutes=self.job_timeout_minutes)
            
            result = await db_session.execute(
                select(ScrapingJob).where(
                    ScrapingJob.status == "running",
                    ScrapingJob.started_at < cutoff_time
                )
            )
            hanging_jobs = result.scalars().all()
            
            cleaned_ids = []
            for job in hanging_jobs:
                try:
                    duration = datetime.utcnow() - (job.started_at or datetime.utcnow())
                    
                    job.status = "failed"
                    job.completed_at = datetime.utcnow()
                    job.duration = duration.total_seconds()
                    
                    error_msg = f"Job timeout after {duration.total_seconds()/3600:.1f} hours"
                    if isinstance(job.error_messages, list):
                        job.error_messages.append(error_msg)
                    else:
                        job.error_messages = [error_msg]
                    
                    cleaned_ids.append(job.id)
                    
                    # Remove from active tracking
                    self.active_jobs.discard(job.id)
                    
                except Exception as e:
                    logger.error(f"Failed to clean hanging job {job.id}: {e}")
            
            if cleaned_ids:
                await db_session.commit()
                logger.warning(f"Cleaned {len(cleaned_ids)} hanging jobs: {cleaned_ids}")
            
            return cleaned_ids
            
        except Exception as e:
            logger.error(f"Failed to clean hanging jobs: {e}")
            return []
    
    async def _clean_failed_jobs(self, db_session) -> List[int]:
        """Clean up old failed jobs (keep only recent failures for analysis)"""
        try:
            # Keep failed jobs for 24 hours for analysis
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            result = await db_session.execute(
                select(ScrapingJob).where(
                    ScrapingJob.status.in_(["failed", "pipeline_unavailable"]),
                    ScrapingJob.completed_at < cutoff_time
                )
            )
            old_failed_jobs = result.scalars().all()
            
            cleaned_ids = []
            for job in old_failed_jobs:
                try:
                    cleaned_ids.append(job.id)
                    await db_session.delete(job)
                except Exception as e:
                    logger.error(f"Failed to delete old failed job {job.id}: {e}")
            
            if cleaned_ids:
                await db_session.commit()
                logger.info(f"Cleaned {len(cleaned_ids)} old failed jobs")
            
            return cleaned_ids
            
        except Exception as e:
            logger.error(f"Failed to clean old failed jobs: {e}")
            return []
    
    async def _recover_stuck_jobs(self, db_session) -> List[int]:
        """Recover jobs that appear to be stuck in pending state"""
        try:
            # Jobs pending for more than 10 minutes might be stuck
            cutoff_time = datetime.utcnow() - timedelta(minutes=10)
            
            result = await db_session.execute(
                select(ScrapingJob).where(
                    ScrapingJob.status == "pending",
                    ScrapingJob.created_at < cutoff_time
                )
            )
            stuck_jobs = result.scalars().all()
            
            recovered_ids = []
            for job in stuck_jobs:
                try:
                    # Check if job can be recovered or should be marked as failed
                    if self._can_recover_job(job):
                        # Reset job to pending with updated timestamp
                        job.created_at = datetime.utcnow()
                        recovered_ids.append(job.id)
                    else:
                        # Mark as failed
                        job.status = "failed"
                        job.completed_at = datetime.utcnow()
                        error_msg = "Job stuck in pending state - marked as failed"
                        job.error_messages = [error_msg] if not job.error_messages else job.error_messages + [error_msg]
                        recovered_ids.append(job.id)
                
                except Exception as e:
                    logger.error(f"Failed to recover stuck job {job.id}: {e}")
            
            if recovered_ids:
                await db_session.commit()
                logger.info(f"Recovered {len(recovered_ids)} stuck jobs")
            
            return recovered_ids
            
        except Exception as e:
            logger.error(f"Failed to recover stuck jobs: {e}")
            return []
    
    async def _remove_orphaned_jobs(self, db_session) -> List[int]:
        """Remove jobs that are orphaned (no longer have valid references)"""
        try:
            # Find jobs that have been completed for more than 7 days
            cutoff_time = datetime.utcnow() - timedelta(days=7)
            
            result = await db_session.execute(
                select(ScrapingJob).where(
                    ScrapingJob.status == "completed",
                    ScrapingJob.completed_at < cutoff_time
                )
            )
            old_completed_jobs = result.scalars().all()
            
            removed_ids = []
            for job in old_completed_jobs:
                try:
                    removed_ids.append(job.id)
                    await db_session.delete(job)
                except Exception as e:
                    logger.error(f"Failed to remove old completed job {job.id}: {e}")
            
            if removed_ids:
                await db_session.commit()
                logger.info(f"Removed {len(removed_ids)} old completed jobs")
            
            return removed_ids
            
        except Exception as e:
            logger.error(f"Failed to remove orphaned jobs: {e}")
            return []
    
    def _can_recover_job(self, job) -> bool:
        """Determine if a stuck job can be recovered"""
        # Simple recovery logic - can be enhanced based on job type
        if not job.job_type:
            return False
        
        # Don't recover jobs that have been retried too many times
        retry_count = 0
        if job.error_messages:
            retry_count = sum(1 for msg in job.error_messages if "retry" in str(msg).lower())
        
        return retry_count < self.max_retries
    
    async def _update_active_jobs_tracking(self, db_session):
        """Update internal tracking of active jobs"""
        try:
            result = await db_session.execute(
                select(ScrapingJob.id).where(ScrapingJob.status == "running")
            )
            current_active_ids = {row[0] for row in result.fetchall()}
            
            # Update internal tracking
            self.active_jobs = current_active_ids
            
        except Exception as e:
            logger.error(f"Failed to update active jobs tracking: {e}")
    
    async def _check_system_health(self):
        """Check overall system health"""
        try:
            health_score = 100.0
            issues = []
            
            # Check job failure rate
            if self.total_jobs_processed > 0:
                failure_rate = (self.total_jobs_failed / self.total_jobs_processed) * 100
                if failure_rate > 20:
                    health_score -= 30
                    issues.append(f"High failure rate: {failure_rate:.1f}%")
                elif failure_rate > 10:
                    health_score -= 15
                    issues.append(f"Elevated failure rate: {failure_rate:.1f}%")
            
            # Check active jobs vs capacity
            if len(self.active_jobs) >= self.max_concurrent_jobs:
                health_score -= 20
                issues.append("At maximum concurrent job capacity")
            
            # Check average job duration
            if self.average_job_duration > 1800:  # 30 minutes
                health_score -= 15
                issues.append(f"Long average job duration: {self.average_job_duration/60:.1f} minutes")
            
            self.system_health_score = max(0, health_score)
            
            if issues:
                logger.warning(f"System health issues detected: {issues}")
            
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            self.system_health_score = max(0, self.system_health_score - 10)
    
    async def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        try:
            async for db_session in get_postgres_session():
                # Get job statistics
                all_jobs_result = await db_session.execute(select(ScrapingJob))
                all_jobs = all_jobs_result.scalars().all()
                
                self.total_jobs_processed = len(all_jobs)
                self.total_jobs_failed = len([j for j in all_jobs if j.status in ["failed", "pipeline_unavailable"]])
                
                # Calculate average duration for completed jobs
                completed_jobs = [j for j in all_jobs if j.status == "completed" and j.duration]
                if completed_jobs:
                    self.average_job_duration = sum(j.duration for j in completed_jobs) / len(completed_jobs)
                
                break
                
        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            await self._ensure_services_started()
            async for db_session in get_postgres_session():
                # Get current jobs
                result = await db_session.execute(select(ScrapingJob))
                all_jobs = result.scalars().all()
                
                # Count by status
                status_counts = {}
                for job in all_jobs:
                    status = job.status
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                # Recent activity (last hour)
                recent_cutoff = datetime.utcnow() - timedelta(hours=1)
                recent_jobs = [j for j in all_jobs if j.created_at and j.created_at > recent_cutoff]
                
                return {
                    "system_health": {
                        "score": self.system_health_score,
                        "status": "healthy" if self.system_health_score >= 80 else "degraded" if self.system_health_score >= 50 else "critical"
                    },
                    "job_statistics": {
                        "total_jobs": len(all_jobs),
                        "status_breakdown": status_counts,
                        "active_jobs": len(self.active_jobs),
                        "max_concurrent": self.max_concurrent_jobs,
                        "recent_activity": len(recent_jobs)
                    },
                    "performance_metrics": {
                        "total_processed": self.total_jobs_processed,
                        "total_failed": self.total_jobs_failed,
                        "success_rate": ((self.total_jobs_processed - self.total_jobs_failed) / max(1, self.total_jobs_processed)) * 100,
                        "average_duration_minutes": self.average_job_duration / 60 if self.average_job_duration else 0
                    },
                    "configuration": {
                        "job_timeout_minutes": self.job_timeout_minutes,
                        "cleanup_interval_seconds": self.cleanup_interval_seconds,
                        "max_concurrent_jobs": self.max_concurrent_jobs,
                        "max_retries": self.max_retries
                    },
                    "background_services": {
                        "cleanup_running": self.cleanup_running,
                        "health_monitor_running": self.health_monitor_running
                    }
                }
                break
                
        except Exception as e:
            logger.error(f"Failed to get comprehensive status: {e}")
            return {"error": str(e), "system_health": {"score": 0, "status": "error"}}
    
    async def force_cleanup_all(self) -> Dict[str, Any]:
        """Force cleanup of ALL jobs (emergency function)"""
        logger.warning("FORCE CLEANUP ALL JOBS - This will terminate all running jobs!")
        
        try:
            await self._ensure_services_started()
            async for db_session in get_postgres_session():
                # Get all non-completed jobs
                result = await db_session.execute(
                    select(ScrapingJob).where(
                        ScrapingJob.status.in_(["running", "pending"])
                    )
                )
                jobs_to_clean = result.scalars().all()
                
                cleaned_jobs = []
                for job in jobs_to_clean:
                    try:
                        job.status = "cancelled"
                        job.completed_at = datetime.utcnow()
                        
                        if job.started_at:
                            duration = datetime.utcnow() - job.started_at
                            job.duration = duration.total_seconds()
                        
                        error_msg = "Job forcefully cancelled during emergency cleanup"
                        if isinstance(job.error_messages, list):
                            job.error_messages.append(error_msg)
                        else:
                            job.error_messages = [error_msg]
                        
                        cleaned_jobs.append(job.id)
                        
                    except Exception as e:
                        logger.error(f"Failed to clean job {job.id}: {e}")
                
                if cleaned_jobs:
                    await db_session.commit()
                
                # Clear internal tracking
                self.active_jobs.clear()
                self.job_metrics.clear()
                
                return {
                    "status": "success",
                    "jobs_cleaned": len(cleaned_jobs),
                    "job_ids": cleaned_jobs,
                    "cleanup_time": datetime.utcnow().isoformat()
                }
                
                break
                
        except Exception as e:
            error_msg = f"Force cleanup failed: {e}"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}
    
    async def optimize_job_performance(self) -> Dict[str, Any]:
        """Optimize job performance based on historical data"""
        optimizations = []
        
        try:
            # Analyze historical performance
            if self.average_job_duration > 600:  # 10 minutes
                optimizations.append("Consider reducing job timeout for better responsiveness")
            
            if len(self.active_jobs) < self.max_concurrent_jobs / 2:
                optimizations.append("System can handle more concurrent jobs - consider increasing limit")
            
            if self.total_jobs_failed > self.total_jobs_processed * 0.1:  # > 10% failure rate
                optimizations.append("High failure rate detected - investigate common failure causes")
            
            return {
                "status": "success",
                "optimizations": optimizations,
                "current_performance": {
                    "health_score": self.system_health_score,
                    "average_duration_minutes": self.average_job_duration / 60,
                    "failure_rate_percent": (self.total_jobs_failed / max(1, self.total_jobs_processed)) * 100
                }
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Global comprehensive job manager instance
job_manager = ComprehensiveJobManager()
comprehensive_job_manager = ComprehensiveJobManager()
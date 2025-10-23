"""
Scraping Job Manager

Manages scraping jobs, handles cleanup of hanging jobs, and prevents jobs 
from starting when prerequisites are not met.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum

from research_synthesis.database.connection import get_postgres_session
from research_synthesis.database.models import ScrapingJob
from sqlalchemy import select, update


class JobStatus(Enum):
    """Scraping job statuses"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PIPELINE_UNAVAILABLE = "pipeline_unavailable"


class ScrapingJobManager:
    """Manages scraping job lifecycle and cleanup"""
    
    def __init__(self):
        self.max_job_duration_minutes = 30  # Max time before job is considered hanging
        self.cleanup_interval_minutes = 15  # How often to run cleanup
        self.pipeline_check_cache_seconds = 60  # Cache pipeline availability check
        self._last_pipeline_check = None
        self._pipeline_available = None
    
    async def check_pipeline_availability(self) -> bool:
        """Check if the data processing pipeline is available"""
        # Cache the result for a minute to avoid repeated checks
        if (self._last_pipeline_check and 
            time.time() - self._last_pipeline_check < self.pipeline_check_cache_seconds):
            return self._pipeline_available
        
        try:
            # Try to import the pipeline
            from research_synthesis.pipeline.data_pipeline import pipeline
            self._pipeline_available = pipeline is not None
        except ImportError:
            self._pipeline_available = False
        
        self._last_pipeline_check = time.time()
        return self._pipeline_available
    
    async def cleanup_hanging_jobs(self) -> Dict[str, Any]:
        """Clean up jobs that have been running too long"""
        cleaned_jobs = []
        errors = []
        
        try:
            cutoff_time = datetime.now() - timedelta(minutes=self.max_job_duration_minutes)
            
            async for db_session in get_postgres_session():
                # Find hanging jobs
                result = await db_session.execute(
                    select(ScrapingJob).where(
                        ScrapingJob.status == JobStatus.RUNNING.value,
                        ScrapingJob.started_at < cutoff_time
                    )
                )
                hanging_jobs = result.scalars().all()
                
                for job in hanging_jobs:
                    try:
                        # Calculate how long the job has been running
                        duration = datetime.now() - job.started_at if job.started_at else timedelta(0)
                        
                        # Update job status to failed with appropriate message
                        job.status = JobStatus.FAILED.value
                        job.completed_at = datetime.now()
                        job.duration = duration.total_seconds()
                        
                        # Set error message based on pipeline availability
                        pipeline_available = await self.check_pipeline_availability()
                        if not pipeline_available:
                            job.status = JobStatus.PIPELINE_UNAVAILABLE.value
                            error_msg = "Job failed: Data processing pipeline not available"
                        else:
                            error_msg = f"Job failed: Timeout after {duration}"
                        
                        if job.error_messages:
                            if isinstance(job.error_messages, list):
                                job.error_messages.append(error_msg)
                            else:
                                job.error_messages = [str(job.error_messages), error_msg]
                        else:
                            job.error_messages = [error_msg]
                        
                        await db_session.commit()
                        
                        cleaned_jobs.append({
                            "job_id": job.id,
                            "started_at": job.started_at.isoformat() if job.started_at else None,
                            "duration_minutes": duration.total_seconds() / 60,
                            "reason": "pipeline_unavailable" if not pipeline_available else "timeout"
                        })
                        
                    except Exception as e:
                        errors.append(f"Failed to clean job {job.id}: {str(e)}")
                
                break
                
        except Exception as e:
            errors.append(f"Cleanup operation failed: {str(e)}")
        
        return {
            "cleaned_jobs": cleaned_jobs,
            "total_cleaned": len(cleaned_jobs),
            "errors": errors,
            "cleanup_time": datetime.now().isoformat()
        }
    
    async def validate_job_prerequisites(self) -> Dict[str, Any]:
        """Check if prerequisites are met for starting new scraping jobs"""
        validation_result = {
            "can_start_jobs": True,
            "blocking_issues": [],
            "warnings": [],
            "pipeline_available": False,
            "estimated_job_time": "2-5 minutes"
        }
        
        # Check pipeline availability
        pipeline_available = await self.check_pipeline_availability()
        validation_result["pipeline_available"] = pipeline_available
        
        if not pipeline_available:
            validation_result["can_start_jobs"] = False
            validation_result["blocking_issues"].append({
                "issue": "Data processing pipeline not available",
                "description": "The pipeline module is not installed or configured properly",
                "impact": "Scraping jobs will start but fail during data processing",
                "recommendation": "Install and configure the data processing pipeline before starting jobs"
            })
        
        # Check for existing running jobs (prevent too many concurrent jobs)
        try:
            async for db_session in get_postgres_session():
                result = await db_session.execute(
                    select(ScrapingJob).where(ScrapingJob.status == JobStatus.RUNNING.value)
                )
                running_jobs = result.scalars().all()
                
                if len(running_jobs) >= 3:  # Limit concurrent jobs
                    validation_result["warnings"].append({
                        "issue": f"{len(running_jobs)} jobs already running",
                        "recommendation": "Wait for current jobs to complete before starting new ones"
                    })
                
                # Check for recent failed jobs due to pipeline
                recent_pipeline_failures = await db_session.execute(
                    select(ScrapingJob).where(
                        ScrapingJob.status == JobStatus.PIPELINE_UNAVAILABLE.value,
                        ScrapingJob.created_at > datetime.now() - timedelta(hours=1)
                    )
                )
                pipeline_failures = recent_pipeline_failures.scalars().all()
                
                if len(pipeline_failures) > 0:
                    validation_result["warnings"].append({
                        "issue": f"{len(pipeline_failures)} recent jobs failed due to pipeline issues",
                        "recommendation": "Resolve pipeline issues before starting new jobs"
                    })
                
                break
                
        except Exception as e:
            validation_result["warnings"].append({
                "issue": "Could not check existing jobs",
                "description": str(e)
            })
        
        return validation_result
    
    async def get_job_statistics(self) -> Dict[str, Any]:
        """Get statistics about scraping jobs"""
        stats = {
            "total_jobs": 0,
            "running_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "pipeline_unavailable_jobs": 0,
            "average_duration_minutes": 0,
            "success_rate": 0,
            "last_24h": {
                "jobs_started": 0,
                "jobs_completed": 0,
                "jobs_failed": 0
            }
        }
        
        try:
            async for db_session in get_postgres_session():
                # Get all jobs
                all_jobs_result = await db_session.execute(select(ScrapingJob))
                all_jobs = all_jobs_result.scalars().all()
                
                stats["total_jobs"] = len(all_jobs)
                
                # Count by status
                for job in all_jobs:
                    if job.status == JobStatus.RUNNING.value:
                        stats["running_jobs"] += 1
                    elif job.status == JobStatus.COMPLETED.value:
                        stats["completed_jobs"] += 1
                    elif job.status == JobStatus.FAILED.value:
                        stats["failed_jobs"] += 1
                    elif job.status == JobStatus.PIPELINE_UNAVAILABLE.value:
                        stats["pipeline_unavailable_jobs"] += 1
                
                # Calculate success rate
                total_finished = stats["completed_jobs"] + stats["failed_jobs"] + stats["pipeline_unavailable_jobs"]
                if total_finished > 0:
                    stats["success_rate"] = (stats["completed_jobs"] / total_finished) * 100
                
                # Calculate average duration for completed jobs
                completed_jobs = [j for j in all_jobs if j.status == JobStatus.COMPLETED.value and j.duration]
                if completed_jobs:
                    total_duration = sum(job.duration for job in completed_jobs)
                    stats["average_duration_minutes"] = (total_duration / len(completed_jobs)) / 60
                
                # Last 24 hours stats
                cutoff_24h = datetime.now() - timedelta(hours=24)
                recent_jobs = [j for j in all_jobs if j.created_at and j.created_at > cutoff_24h]
                
                stats["last_24h"]["jobs_started"] = len(recent_jobs)
                stats["last_24h"]["jobs_completed"] = len([j for j in recent_jobs if j.status == JobStatus.COMPLETED.value])
                stats["last_24h"]["jobs_failed"] = len([j for j in recent_jobs if j.status in [JobStatus.FAILED.value, JobStatus.PIPELINE_UNAVAILABLE.value]])
                
                break
                
        except Exception as e:
            stats["error"] = str(e)
        
        return stats
    
    async def start_background_cleanup(self):
        """Start background task for periodic cleanup"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval_minutes * 60)
                cleanup_result = await self.cleanup_hanging_jobs()
                
                if cleanup_result["total_cleaned"] > 0:
                    print(f"[ScrapingJobManager] Cleaned up {cleanup_result['total_cleaned']} hanging jobs")
                    
            except Exception as e:
                print(f"[ScrapingJobManager] Background cleanup error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying


# Global job manager instance
scraping_job_manager = ScrapingJobManager()
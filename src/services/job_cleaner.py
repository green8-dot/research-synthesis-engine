"""
Job Cleaner Service - Clean up stuck scraping jobs and manage job lifecycle
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any
from loguru import logger

try:
    from research_synthesis.database.connection import get_postgres_session
    from research_synthesis.database.models import ScrapingJob
    from sqlalchemy import select, update, and_, delete
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

try:
    from research_synthesis.api.scraping_router import active_jobs
    JOBS_AVAILABLE = True
except ImportError:
    JOBS_AVAILABLE = False

async def cleanup_stuck_jobs(max_age_hours: int = 1) -> Dict[str, Any]:
    """
    Clean up jobs that have been stuck in 'running' state
    
    Args:
        max_age_hours: Jobs older than this are considered stuck
        
    Returns:
        Dictionary with cleanup results
    """
    if not DATABASE_AVAILABLE:
        return {"error": "Database not available"}
    
    cleaned_jobs = []
    stuck_jobs = []
    
    try:
        async for db_session in get_postgres_session():
            # Find stuck jobs
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            stuck_query = await db_session.execute(
                select(ScrapingJob)
                .filter(
                    and_(
                        ScrapingJob.status == "running",
                        ScrapingJob.started_at < cutoff_time
                    )
                )
            )
            stuck_jobs_db = stuck_query.scalars().all()
            
            logger.info(f"Found {len(stuck_jobs_db)} stuck jobs in database")
            
            # Update database records
            for job in stuck_jobs_db:
                job.status = "failed"
                job.completed_at = datetime.now()
                
                if job.started_at:
                    job.duration = (job.completed_at - job.started_at).total_seconds()
                
                if not job.error_messages:
                    job.error_messages = []
                job.error_messages.append(f"Job terminated - stuck for over {max_age_hours} hours")
                
                cleaned_jobs.append({
                    "id": job.id,
                    "source": job.source_id,
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "stuck_duration_hours": (datetime.now() - job.started_at).total_seconds() / 3600 if job.started_at else 0
                })
                
                stuck_jobs.append(f"job_{job.id}")
            
            await db_session.commit()
            logger.info(f"Updated {len(cleaned_jobs)} stuck jobs in database")
            break
            
    except Exception as e:
        logger.error(f"Error cleaning database jobs: {e}")
        return {"error": f"Database cleanup failed: {str(e)}"}
    
    # Clean up in-memory active jobs
    if JOBS_AVAILABLE:
        try:
            cleaned_memory_jobs = []
            jobs_to_remove = []
            
            for job_id, job_data in active_jobs.items():
                started_at = job_data.get("started_at")
                if not started_at:
                    continue
                    
                # Parse the started_at timestamp
                try:
                    if isinstance(started_at, str):
                        started_time = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                    else:
                        started_time = started_at
                        
                    age_hours = (datetime.now() - started_time.replace(tzinfo=None)).total_seconds() / 3600
                    
                    if age_hours > max_age_hours and job_data.get("status") == "running":
                        jobs_to_remove.append(job_id)
                        cleaned_memory_jobs.append({
                            "id": job_id,
                            "age_hours": age_hours,
                            "sources": job_data.get("sources", [])
                        })
                except Exception as e:
                    logger.error(f"Error parsing job {job_id} timestamp: {e}")
                    continue
            
            # Remove stuck jobs from active_jobs
            for job_id in jobs_to_remove:
                active_jobs[job_id]["status"] = "failed"
                active_jobs[job_id]["completed_at"] = datetime.now().isoformat()
                active_jobs[job_id]["errors_count"] = active_jobs[job_id].get("errors_count", 0) + 1
                
            logger.info(f"Cleaned {len(cleaned_memory_jobs)} stuck jobs from memory")
            
        except Exception as e:
            logger.error(f"Error cleaning in-memory jobs: {e}")
    
    return {
        "status": "completed",
        "database_jobs_cleaned": len(cleaned_jobs),
        "memory_jobs_cleaned": len(cleaned_memory_jobs) if JOBS_AVAILABLE else 0,
        "cleaned_job_ids": stuck_jobs,
        "details": cleaned_jobs,
        "message": f"Successfully cleaned {len(cleaned_jobs)} stuck jobs"
    }

async def force_stop_job(job_id: str) -> Dict[str, Any]:
    """
    Force stop a specific job
    
    Args:
        job_id: Job ID to stop (e.g., "job_1")
        
    Returns:
        Operation result
    """
    # Extract numeric ID if provided as "job_X"
    if job_id.startswith("job_"):
        numeric_id = int(job_id.replace("job_", ""))
    else:
        numeric_id = int(job_id)
    
    if not DATABASE_AVAILABLE:
        return {"error": "Database not available"}
    
    try:
        async for db_session in get_postgres_session():
            # Find the job
            job_query = await db_session.execute(
                select(ScrapingJob).filter(ScrapingJob.id == numeric_id)
            )
            job = job_query.scalar_one_or_none()
            
            if not job:
                return {"error": f"Job {job_id} not found"}
            
            # Update job status
            old_status = job.status
            job.status = "stopped"
            job.completed_at = datetime.now()
            
            if job.started_at:
                job.duration = (job.completed_at - job.started_at).total_seconds()
            
            if not job.error_messages:
                job.error_messages = []
            job.error_messages.append("Job manually stopped")
            
            await db_session.commit()
            
            # Update in-memory job if exists
            if JOBS_AVAILABLE and f"job_{numeric_id}" in active_jobs:
                active_jobs[f"job_{numeric_id}"]["status"] = "stopped"
                active_jobs[f"job_{numeric_id}"]["completed_at"] = datetime.now().isoformat()
            
            return {
                "status": "success",
                "job_id": job_id,
                "old_status": old_status,
                "new_status": "stopped",
                "message": f"Successfully stopped job {job_id}"
            }
            
    except Exception as e:
        logger.error(f"Error stopping job {job_id}: {e}")
        return {"error": f"Failed to stop job: {str(e)}"}

async def clear_completed_jobs(max_age_hours: int = 24) -> Dict[str, Any]:
    """
    Clear completed jobs from in-memory storage and optionally from database
    
    Args:
        max_age_hours: Clear jobs completed more than this many hours ago
        
    Returns:
        Dictionary with clear results
    """
    cleared_memory_jobs = []
    cleared_db_jobs = []
    
    # Clear from in-memory storage
    if JOBS_AVAILABLE:
        try:
            jobs_to_remove = []
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            for job_id, job_data in active_jobs.items():
                status = job_data.get("status")
                completed_at = job_data.get("completed_at")
                
                # Remove completed jobs (failed, stopped, completed)
                if status in ["failed", "stopped", "completed"]:
                    # If no max_age_hours restriction, remove immediately
                    if max_age_hours <= 0:
                        jobs_to_remove.append(job_id)
                        cleared_memory_jobs.append(job_id)
                    # Otherwise check completion time
                    elif completed_at:
                        try:
                            completed_time = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
                            if completed_time.replace(tzinfo=None) < cutoff_time:
                                jobs_to_remove.append(job_id)
                                cleared_memory_jobs.append(job_id)
                        except Exception as e:
                            logger.error(f"Error parsing completed_at for {job_id}: {e}")
                            # If we can't parse the time, remove it anyway to clean up
                            jobs_to_remove.append(job_id)
                            cleared_memory_jobs.append(job_id)
            
            # Remove jobs from active_jobs
            for job_id in jobs_to_remove:
                del active_jobs[job_id]
                
            logger.info(f"Cleared {len(cleared_memory_jobs)} completed jobs from memory")
            
        except Exception as e:
            logger.error(f"Error clearing in-memory jobs: {e}")
    
    # Optional: Clear very old jobs from database
    if DATABASE_AVAILABLE and max_age_hours > 24:  # Only clear from DB if jobs are very old
        try:
            async for db_session in get_postgres_session():
                cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
                
                # Find old completed jobs
                old_jobs_query = await db_session.execute(
                    select(ScrapingJob)
                    .filter(
                        and_(
                            ScrapingJob.status.in_(["failed", "stopped", "completed"]),
                            ScrapingJob.completed_at < cutoff_time
                        )
                    )
                )
                old_jobs = old_jobs_query.scalars().all()
                
                # Delete old jobs
                for job in old_jobs:
                    cleared_db_jobs.append(f"job_{job.id}")
                    await db_session.delete(job)
                
                await db_session.commit()
                logger.info(f"Deleted {len(cleared_db_jobs)} old jobs from database")
                break
                
        except Exception as e:
            logger.error(f"Error clearing database jobs: {e}")
    
    return {
        "status": "completed",
        "memory_jobs_cleared": len(cleared_memory_jobs),
        "database_jobs_cleared": len(cleared_db_jobs),
        "cleared_memory_jobs": cleared_memory_jobs,
        "cleared_db_jobs": cleared_db_jobs,
        "message": f"Cleared {len(cleared_memory_jobs)} jobs from memory, {len(cleared_db_jobs)} from database"
    }
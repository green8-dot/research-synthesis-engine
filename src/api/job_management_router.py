"""
Job Management API Router - Simple job management without circular imports
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any
from loguru import logger
from datetime import datetime, timedelta

try:
    from research_synthesis.database.connection import get_postgres_session
    from research_synthesis.database.models import ScrapingJob
    from sqlalchemy import select, delete, and_
    DATABASE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Database not available: {e}")
    DATABASE_AVAILABLE = False

router = APIRouter()

@router.get("/")
async def get_job_management_status():
    """Get job management service status"""
    return {
        "service": "Job Management API",
        "status": "active" if DATABASE_AVAILABLE else "unavailable",
        "endpoints": [
            "/clear-completed - Clear completed jobs from database",
            "/clear-job/{job_id} - Clear specific job from database",
            "/clear-old - Clear jobs older than specified hours"
        ]
    }

@router.post("/clear-completed")
async def clear_completed_jobs():
    """Clear all completed jobs (failed, stopped, completed, cancelled) from database"""
    if not DATABASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        cleared_jobs = []
        
        async for db_session in get_postgres_session():
            # Find completed jobs (including cancelled)
            completed_query = await db_session.execute(
                select(ScrapingJob).filter(
                    ScrapingJob.status.in_(['failed', 'stopped', 'completed', 'cancelled'])
                )
            )
            completed_jobs = completed_query.scalars().all()
            
            logger.info(f"Found {len(completed_jobs)} completed jobs to clear")
            
            # Delete completed jobs
            for job in completed_jobs:
                cleared_jobs.append({
                    "id": f"job_{job.id}",
                    "status": job.status,
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None
                })
                await db_session.delete(job)
            
            await db_session.commit()
            logger.info(f"Successfully cleared {len(cleared_jobs)} completed jobs")
            break
            
        return {
            "status": "success",
            "cleared_count": len(cleared_jobs),
            "cleared_jobs": cleared_jobs,
            "message": f"Successfully cleared {len(cleared_jobs)} completed jobs from database"
        }
        
    except Exception as e:
        logger.error(f"Error clearing completed jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")

@router.delete("/clear-job/{job_id}")
async def clear_specific_job(job_id: str):
    """Clear a specific job from database"""
    if not DATABASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Extract numeric ID
        if job_id.startswith("job_"):
            numeric_id = int(job_id.replace("job_", ""))
        else:
            numeric_id = int(job_id)
        
        async for db_session in get_postgres_session():
            # Find the job
            job_query = await db_session.execute(
                select(ScrapingJob).filter(ScrapingJob.id == numeric_id)
            )
            job = job_query.scalar_one_or_none()
            
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
            
            job_info = {
                "id": f"job_{job.id}",
                "status": job.status,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            }
            
            await db_session.delete(job)
            await db_session.commit()
            
            logger.info(f"Successfully cleared job {job_id}")
            break
            
        return {
            "status": "success",
            "cleared_job": job_info,
            "message": f"Successfully cleared job {job_id} from database"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid job ID format: {str(e)}")
    except Exception as e:
        logger.error(f"Error clearing job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")

@router.post("/clear-old")
async def clear_old_jobs(max_age_hours: int = 24):
    """Clear jobs older than specified hours"""
    if not DATABASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleared_jobs = []
        
        async for db_session in get_postgres_session():
            # Find old jobs
            old_jobs_query = await db_session.execute(
                select(ScrapingJob).filter(
                    and_(
                        ScrapingJob.status.in_(['failed', 'stopped', 'completed']),
                        ScrapingJob.completed_at < cutoff_time
                    )
                )
            )
            old_jobs = old_jobs_query.scalars().all()
            
            logger.info(f"Found {len(old_jobs)} jobs older than {max_age_hours} hours")
            
            # Delete old jobs
            for job in old_jobs:
                cleared_jobs.append({
                    "id": f"job_{job.id}",
                    "status": job.status,
                    "age_hours": (datetime.now() - job.completed_at).total_seconds() / 3600 if job.completed_at else None
                })
                await db_session.delete(job)
            
            await db_session.commit()
            logger.info(f"Successfully cleared {len(cleared_jobs)} old jobs")
            break
            
        return {
            "status": "success",
            "cleared_count": len(cleared_jobs),
            "max_age_hours": max_age_hours,
            "cleared_jobs": cleared_jobs,
            "message": f"Successfully cleared {len(cleared_jobs)} jobs older than {max_age_hours} hours"
        }
        
    except Exception as e:
        logger.error(f"Error clearing old jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")

@router.get("/job-counts")
async def get_job_counts():
    """Get counts of jobs by status"""
    if not DATABASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        counts = {
            "total": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "stopped": 0
        }
        
        async for db_session in get_postgres_session():
            # Get all jobs
            all_jobs_query = await db_session.execute(select(ScrapingJob))
            all_jobs = all_jobs_query.scalars().all()
            
            counts["total"] = len(all_jobs)
            
            for job in all_jobs:
                if job.status in counts:
                    counts[job.status] += 1
            
            break
            
        return {
            "job_counts": counts,
            "completed_jobs": counts["completed"] + counts["failed"] + counts["stopped"],
            "message": f"Found {counts['total']} total jobs in database"
        }
        
    except Exception as e:
        logger.error(f"Error getting job counts: {e}")
        raise HTTPException(status_code=500, detail=f"Count failed: {str(e)}")
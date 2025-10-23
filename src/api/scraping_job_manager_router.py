"""
Scraping Job Manager API Router
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
from loguru import logger

from research_synthesis.services.scraping_job_manager import scraping_job_manager

router = APIRouter()

@router.get("/status")
async def get_job_manager_status():
    """Get scraping job manager status and configuration"""
    return {
        "service": "Scraping Job Manager",
        "status": "active",
        "configuration": {
            "max_job_duration_minutes": scraping_job_manager.max_job_duration_minutes,
            "cleanup_interval_minutes": scraping_job_manager.cleanup_interval_minutes,
            "pipeline_check_cache_seconds": scraping_job_manager.pipeline_check_cache_seconds
        },
        "endpoints": [
            "/cleanup - Clean up hanging jobs manually",
            "/validate - Check job prerequisites", 
            "/statistics - Get job statistics",
            "/pipeline-status - Check pipeline availability"
        ]
    }

@router.post("/cleanup")
async def cleanup_hanging_jobs():
    """Manually trigger cleanup of hanging jobs"""
    try:
        logger.info("Manual cleanup of hanging jobs triggered")
        result = await scraping_job_manager.cleanup_hanging_jobs()
        
        if result["total_cleaned"] > 0:
            logger.info(f"Cleaned up {result['total_cleaned']} hanging jobs")
        
        return {
            "status": "success",
            "cleanup_result": result
        }
        
    except Exception as e:
        logger.error(f"Manual job cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@router.get("/validate")
async def validate_job_prerequisites():
    """Check if prerequisites are met for starting new scraping jobs"""
    try:
        validation_result = await scraping_job_manager.validate_job_prerequisites()
        return {
            "status": "success",
            "validation": validation_result
        }
        
    except Exception as e:
        logger.error(f"Job prerequisites validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.get("/statistics")
async def get_job_statistics():
    """Get comprehensive statistics about scraping jobs"""
    try:
        stats = await scraping_job_manager.get_job_statistics()
        return {
            "status": "success", 
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get job statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Statistics retrieval failed: {str(e)}")

@router.get("/pipeline-status")
async def check_pipeline_status():
    """Check if the data processing pipeline is available"""
    try:
        pipeline_available = await scraping_job_manager.check_pipeline_availability()
        return {
            "status": "success",
            "pipeline_available": pipeline_available,
            "cache_info": {
                "last_check": scraping_job_manager._last_pipeline_check,
                "cache_duration_seconds": scraping_job_manager.pipeline_check_cache_seconds
            }
        }
        
    except Exception as e:
        logger.error(f"Pipeline status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline check failed: {str(e)}")

@router.post("/background-cleanup/start")
async def start_background_cleanup(background_tasks: BackgroundTasks):
    """Start background cleanup task"""
    try:
        background_tasks.add_task(scraping_job_manager.start_background_cleanup)
        return {
            "status": "success",
            "message": "Background cleanup task started"
        }
        
    except Exception as e:
        logger.error(f"Failed to start background cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Background cleanup start failed: {str(e)}")
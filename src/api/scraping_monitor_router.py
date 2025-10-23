"""
Scraping Monitor API Router
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Any
from loguru import logger
import asyncio

try:
    from research_synthesis.services.scraping_monitor import run_scraping_health_check, fix_reddit_sources
    from research_synthesis.services.reddit_scraper import scrape_reddit_json
    from research_synthesis.services.job_cleaner import cleanup_stuck_jobs, force_stop_job, clear_completed_jobs
    MONITOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Scraping monitor not available: {e}")
    MONITOR_AVAILABLE = False

router = APIRouter()

@router.get("/")
async def get_monitor_status():
    """Get scraping monitor service status"""
    return {
        "service": "Scraping Monitor API",
        "status": "active" if MONITOR_AVAILABLE else "unavailable",
        "endpoints": [
            "/health-check - Run comprehensive scraping health check",
            "/fix-reddit - Fix Reddit source configuration issues", 
            "/test-reddit - Test Reddit JSON API scraping",
            "/auto-fix - Automatically fix detected issues",
            "/cleanup-stuck-jobs - Clean up jobs stuck in running state",
            "/stop-job/{job_id} - Force stop a specific job",
            "/clear-completed-jobs - Clear completed jobs from display"
        ]
    }

@router.get("/health-check")
async def run_health_check():
    """Run comprehensive scraping health check"""
    if not MONITOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Scraping monitor service unavailable")
    
    try:
        logger.info("Starting scraping health check...")
        report = await run_scraping_health_check()
        
        return {
            "status": "completed",
            "health_report": report,
            "summary": {
                "overall_status": report.get("status", "unknown"),
                "total_issues": report.get("issues_found", 0),
                "critical_issues": report.get("critical_issues", 0),
                "fixes_applied": report.get("auto_fixes_applied", 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.post("/fix-reddit")
async def fix_reddit_sources_endpoint():
    """Fix Reddit source configuration issues"""
    if not MONITOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Scraping monitor service unavailable")
    
    try:
        logger.info("Fixing Reddit source configurations...")
        result = await fix_reddit_sources()
        
        return {
            "status": "completed",
            "result": result,
            "message": f"Applied {result.get('fixes_applied', 0)} Reddit source fixes"
        }
        
    except Exception as e:
        logger.error(f"Reddit fix failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reddit fix failed: {str(e)}")

@router.get("/test-reddit")
async def test_reddit_scraping(subreddit: str = "space", limit: int = 5):
    """Test Reddit JSON API scraping"""
    if not MONITOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Scraping monitor service unavailable")
    
    try:
        # Construct Reddit JSON URL
        reddit_url = f"https://old.reddit.com/r/{subreddit}/.json"
        
        logger.info(f"Testing Reddit scraping: {reddit_url}")
        articles = await scrape_reddit_json(reddit_url, limit)
        
        return {
            "status": "completed",
            "test_url": reddit_url,
            "articles_found": len(articles),
            "articles": [
                {
                    "title": article.get("title", "")[:100],
                    "author": article.get("author", ""),
                    "score": article.get("metadata", {}).get("score", 0),
                    "comments": article.get("metadata", {}).get("num_comments", 0),
                    "url": article.get("url", "")
                }
                for article in articles[:limit]
            ],
            "sample_article": articles[0] if articles else None
        }
        
    except Exception as e:
        logger.error(f"Reddit test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reddit test failed: {str(e)}")

@router.post("/auto-fix")
async def auto_fix_issues():
    """Automatically fix detected scraping issues"""
    if not MONITOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Scraping monitor service unavailable")
    
    try:
        logger.info("Running auto-fix for scraping issues...")
        
        # First run health check to detect issues
        report = await run_scraping_health_check()
        
        return {
            "status": "completed", 
            "issues_detected": report.get("issues_found", 0),
            "fixes_applied": report.get("auto_fixes_applied", 0),
            "remaining_issues": report.get("issues_found", 0) - report.get("auto_fixes_applied", 0),
            "recommendations": report.get("recommendations", []),
            "details": {
                "overall_status": report.get("status", "unknown"),
                "critical_issues": report.get("critical_issues", 0),
                "issues_by_type": report.get("issues_by_type", {}),
                "issues_by_source": report.get("issues_by_source", {})
            }
        }
        
    except Exception as e:
        logger.error(f"Auto-fix failed: {e}")
        raise HTTPException(status_code=500, detail=f"Auto-fix failed: {str(e)}")

@router.get("/sources-status")
async def get_sources_status():
    """Get detailed status of all scraping sources"""
    try:
        from research_synthesis.api.scraping_router import sources_data
        
        sources_status = {}
        for source_name, source_config in sources_data.items():
            sources_status[source_name] = {
                "url": source_config.get("url", ""),
                "scraping_strategy": source_config.get("scraping_strategy", "default"),
                "last_scraped": source_config.get("last_scraped", "never"),
                "status": source_config.get("status", "unknown"),
                "articles_count": source_config.get("articles_count", 0),
                "is_reddit": "reddit.com" in source_config.get("url", "").lower(),
                "uses_json_api": source_config.get("url", "").endswith(".json")
            }
            
        return {
            "total_sources": len(sources_status),
            "sources": sources_status,
            "reddit_sources": {
                name: status for name, status in sources_status.items() 
                if status["is_reddit"]
            }
        }
        
    except ImportError:
        raise HTTPException(status_code=503, detail="Sources data not available")
    except Exception as e:
        logger.error(f"Error getting sources status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting sources status: {str(e)}")

@router.post("/run-background-monitor")
async def run_background_monitor(background_tasks: BackgroundTasks):
    """Start background monitoring of scraping sources"""
    if not MONITOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Scraping monitor service unavailable")
    
    async def background_monitoring():
        """Background task to monitor scraping"""
        try:
            logger.info("Starting background scraping monitoring...")
            
            # Run health check every 30 minutes
            while True:
                try:
                    report = await run_scraping_health_check()
                    logger.info(f"Background monitor: {report.get('status', 'unknown')} status, {report.get('issues_found', 0)} issues")
                    
                    # Log critical issues
                    if report.get("critical_issues", 0) > 0:
                        logger.error(f"CRITICAL: {report['critical_issues']} critical scraping issues detected!")
                        
                except Exception as e:
                    logger.error(f"Background monitor error: {e}")
                    
                # Wait 30 minutes
                await asyncio.sleep(1800)
                
        except Exception as e:
            logger.error(f"Background monitoring failed: {e}")
    
    background_tasks.add_task(background_monitoring)
    
    return {
        "status": "started",
        "message": "Background scraping monitoring started",
        "monitoring_interval": "30 minutes"
    }

@router.post("/cleanup-stuck-jobs")
async def cleanup_stuck_jobs_endpoint(max_age_hours: int = 1):
    """Clean up jobs stuck in running state"""
    if not MONITOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Scraping monitor service unavailable")
    
    try:
        logger.info(f"Cleaning up jobs stuck for more than {max_age_hours} hours...")
        result = await cleanup_stuck_jobs(max_age_hours)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Stuck job cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@router.post("/stop-job/{job_id}")
async def stop_job_endpoint(job_id: str):
    """Force stop a specific job"""
    if not MONITOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Scraping monitor service unavailable")
    
    try:
        logger.info(f"Force stopping job {job_id}...")
        result = await force_stop_job(job_id)
        
        if "error" in result:
            raise HTTPException(status_code=404 if "not found" in result["error"] else 500, detail=result["error"])
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid job ID: {str(e)}")
    except Exception as e:
        logger.error(f"Job stop failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stop failed: {str(e)}")

@router.get("/stuck-jobs")
async def get_stuck_jobs(max_age_hours: int = 1):
    """Get list of jobs that appear to be stuck"""
    try:
        from research_synthesis.api.scraping_router import active_jobs
        from datetime import datetime, timedelta
        
        stuck_jobs = []
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for job_id, job_data in active_jobs.items():
            if job_data.get("status") != "running":
                continue
                
            started_at = job_data.get("started_at")
            if not started_at:
                continue
                
            try:
                if isinstance(started_at, str):
                    started_time = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                else:
                    started_time = started_at
                    
                age_hours = (datetime.now() - started_time.replace(tzinfo=None)).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    stuck_jobs.append({
                        "id": job_id,
                        "started_at": started_at,
                        "age_hours": round(age_hours, 2),
                        "progress": job_data.get("progress", 0),
                        "sources": job_data.get("sources", []),
                        "status": job_data.get("status")
                    })
                    
            except Exception as e:
                logger.error(f"Error checking job {job_id}: {e}")
                continue
        
        return {
            "stuck_jobs_found": len(stuck_jobs),
            "max_age_hours": max_age_hours,
            "stuck_jobs": stuck_jobs,
            "recommendations": [
                f"Use /cleanup-stuck-jobs to clean all jobs older than {max_age_hours} hours",
                "Use /stop-job/{{job_id}} to stop individual jobs"
            ]
        }
        
    except ImportError:
        raise HTTPException(status_code=503, detail="Active jobs not available")
    except Exception as e:
        logger.error(f"Error getting stuck jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.post("/clear-completed-jobs")
async def clear_completed_jobs_endpoint(max_age_hours: int = 0):
    """Clear completed jobs from display (0 = clear all completed jobs immediately)"""
    if not MONITOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Scraping monitor service unavailable")
    
    try:
        logger.info(f"Clearing completed jobs (max_age_hours={max_age_hours})...")
        result = await clear_completed_jobs(max_age_hours)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Clear completed jobs failed: {e}")
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")
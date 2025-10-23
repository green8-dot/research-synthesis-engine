"""
File Organization and System Monitoring API Router
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional
import asyncio

from research_synthesis.utils.intelligent_file_organizer import file_organizer
from loguru import logger

router = APIRouter(prefix="/api/v1/file-organization", tags=["file-organization"])


@router.post("/run-organization")
async def run_file_organization():
    """Run intelligent file organization"""
    try:
        logger.info("Starting file organization...")
        results = await file_organizer.organize_files()
        
        return JSONResponse(content={
            "success": True,
            "message": "File organization completed",
            "results": results
        })
    except Exception as e:
        logger.error(f"File organization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system-health")
async def get_system_health():
    """Get current system health status"""
    try:
        alerts = await file_organizer.monitor_system_health()
        
        return JSONResponse(content={
            "success": True,
            "system_status": "healthy" if not any(a.severity == "critical" for a in alerts) else "critical",
            "alerts": [
                {
                    "type": alert.alert_type,
                    "severity": alert.severity,
                    "component": alert.component,
                    "description": alert.description,
                    "suggested_action": alert.suggested_action,
                    "auto_fixable": alert.auto_fixable,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in alerts
            ],
            "summary": {
                "total_alerts": len(alerts),
                "critical": len([a for a in alerts if a.severity == "critical"]),
                "warning": len([a for a in alerts if a.severity == "warning"]),
                "info": len([a for a in alerts if a.severity == "info"])
            }
        })
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auto-fix")
async def run_auto_fix():
    """Run automatic issue fixing"""
    try:
        alerts = await file_organizer.monitor_system_health()
        fix_results = await file_organizer.auto_fix_issues(alerts)
        
        return JSONResponse(content={
            "success": True,
            "message": "Auto-fix completed",
            "results": fix_results
        })
    except Exception as e:
        logger.error(f"Auto-fix failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health-report")
async def generate_health_report():
    """Generate comprehensive system health report"""
    try:
        report = await file_organizer.generate_health_report()
        
        return JSONResponse(content={
            "success": True,
            "report": report
        })
    except Exception as e:
        logger.error(f"Health report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/organization-status")
async def get_organization_status():
    """Get current file organization status"""
    try:
        # Quick file scan without moving files
        file_count = 0
        categories = {}
        
        import os
        from pathlib import Path
        
        root_path = Path(os.getcwd())
        for root, dirs, files in os.walk(root_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]
            
            for file in files:
                if not file.startswith('.'):
                    file_count += 1
                    file_path = Path(root) / file
                    category = file_organizer._categorize_file(file_path)
                    
                    if category not in categories:
                        categories[category] = 0
                    categories[category] += 1
        
        return JSONResponse(content={
            "success": True,
            "status": {
                "total_files": file_count,
                "categories": categories,
                "needs_organization": any(count > 50 for count in categories.values())
            }
        })
    except Exception as e:
        logger.error(f"Organization status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
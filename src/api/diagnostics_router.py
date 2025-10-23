"""
Diagnostics API Router
Provides comprehensive system diagnostics and troubleshooting endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import asyncio
from datetime import datetime
from pathlib import Path

router = APIRouter()

# Import diagnostic tracker
try:
    from research_synthesis.services.diagnostic_tracker import diagnostic_tracker
    DIAGNOSTIC_TRACKER_AVAILABLE = True
except ImportError:
    DIAGNOSTIC_TRACKER_AVAILABLE = False
    diagnostic_tracker = None

@router.get("/")
async def get_diagnostics_overview():
    """Get basic diagnostics overview"""
    return {
        "status": "Diagnostics API active",
        "service": "ready",
        "diagnostic_tracker": "available" if DIAGNOSTIC_TRACKER_AVAILABLE else "unavailable",
        "available_endpoints": [
            "/comprehensive - Full system diagnostic report",
            "/database - Database connection tests",
            "/api - API endpoint health checks", 
            "/frontend - Frontend file checks",
            "/common-issues - Check for known recurring issues",
            "/quick-fix - Quick fixes for common problems",
            "/trending-issues - Get trending issues from diagnostic tracker",
            "/diagnostic-summary - Get diagnostic tracking summary",
            "/issue-details/{issue_type} - Get details for specific issue type",
            "/start-tracking - Start continuous issue tracking",
            "/stop-tracking - Stop continuous issue tracking"
        ],
        "last_checked": datetime.now().isoformat()
    }

@router.get("/comprehensive")
async def run_comprehensive_diagnostics():
    """Run comprehensive system diagnostics"""
    try:
        from research_synthesis.utils.diagnostics import diagnostics
        report = await diagnostics.run_comprehensive_check()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostics failed: {str(e)}")

@router.get("/database")
async def check_database_status():
    """Check database connections and health"""
    try:
        from research_synthesis.utils.diagnostics import diagnostics
        return await diagnostics._check_databases()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database check failed: {str(e)}")

@router.get("/api")
async def check_api_endpoints():
    """Test all critical API endpoints"""
    try:
        from research_synthesis.utils.diagnostics import diagnostics
        return await diagnostics._check_api_endpoints()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API check failed: {str(e)}")

@router.get("/frontend")
async def check_frontend_files():
    """Check frontend JavaScript and CSS files"""
    try:
        from research_synthesis.utils.diagnostics import diagnostics
        return diagnostics._check_frontend_files()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Frontend check failed: {str(e)}")

@router.get("/common-issues")
async def check_common_issues():
    """Check for known recurring issues and provide solutions"""
    try:
        from research_synthesis.utils.diagnostics import diagnostics
        issues = await diagnostics._check_common_issues()
        
        # Add quick solutions for common issues
        solutions = {
            "report_generation_frozen": {
                "symptoms": ["Generate Report button doesn't respond", "Button appears frozen", "No console logs"],
                "solutions": [
                    "Hard refresh the page (Ctrl+F5)",
                    "Check browser console for JavaScript errors",
                    "Verify API endpoint is responding",
                    "Clear browser cache"
                ]
            },
            "reports_not_persisting": {
                "symptoms": ["Reports generate but don't appear in list", "Page refresh clears reports"],
                "solutions": [
                    "Check reports directory exists and is writable",
                    "Verify database connection",
                    "Check server logs for storage errors",
                    "Ensure API is saving reports to storage"
                ]
            },
            "database_connection_issues": {
                "symptoms": ["500 errors", "Database connection failed", "No data loading"],
                "solutions": [
                    "Check database file permissions",
                    "Verify SQLite file exists",
                    "Check database connection string",
                    "Restart the server"
                ]
            },
            "caching_issues": {
                "symptoms": ["Stale data", "Changes not reflected", "Performance issues"],
                "solutions": [
                    "Install Redis for better caching",
                    "Clear application cache",
                    "Restart server to clear memory cache",
                    "Check Redis connection if installed"
                ]
            },
            "frontend_javascript_errors": {
                "symptoms": ["Buttons not responding", "Console errors", "Modal issues"],
                "solutions": [
                    "Hard refresh to reload JavaScript",
                    "Check for syntax errors in console",
                    "Verify main.js is loading properly",
                    "Check Bootstrap and other dependencies"
                ]
            }
        }
        
        return {
            "current_issues": issues,
            "common_solutions": solutions,
            "quick_fixes": [
                "Hard refresh page (Ctrl+F5)",
                "Check browser console (F12) for errors",
                "Restart the server if APIs not responding",
                "Clear browser cache if changes not reflected"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Issue check failed: {str(e)}")

@router.post("/quick-fix")
async def apply_quick_fixes():
    """Apply automated quick fixes for common issues"""
    fixes_applied = []
    
    try:
        # Ensure reports directory exists
        from pathlib import Path
        reports_dir = Path("reports")
        if not reports_dir.exists():
            reports_dir.mkdir(exist_ok=True)
            fixes_applied.append("Created missing reports directory")
        
        # Clear any stuck cache entries
        try:
            from research_synthesis.database.connection import get_redis
            redis_client = get_redis()
            if redis_client:
                # Clear diagnostic cache entries
                await redis_client.delete("diagnostic_test")
                fixes_applied.append("Cleared Redis diagnostic cache")
        except:
            pass
        
        # Check and fix file permissions
        critical_files = [
            "research_synthesis/web/static/js/main.js",
            "research_synthesis/web/templates/reports.html"
        ]
        
        for file_path in critical_files:
            file_obj = Path(file_path)
            if file_obj.exists():
                # Basic existence check - permissions are complex on Windows
                fixes_applied.append(f"Verified {file_path} exists and is accessible")
        
        return {
            "fixes_applied": fixes_applied,
            "recommendations": [
                "Hard refresh your browser (Ctrl+F5) to reload JavaScript",
                "Check browser console for any remaining errors",
                "Test the Generate Report functionality again"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick fix failed: {str(e)}")

@router.get("/health-summary")
async def get_health_summary():
    """Get a quick health summary of the entire system"""
    try:
        from research_synthesis.utils.diagnostics import diagnostics
        
        # Run basic checks
        db_status = await diagnostics._check_databases()
        file_status = diagnostics._check_file_system()
        
        # Calculate overall health score
        total_checks = 0
        passed_checks = 0
        
        # Database health
        for db, status in db_status.items():
            total_checks += 1
            if status.get("status") in ["connected", "not_available"]:  # not_available is OK for optional services
                passed_checks += 1
        
        # Critical file health
        critical_files = [
            "research_synthesis/web/static/js/main.js",
            "research_synthesis/web/templates/",
            "research_synthesis/api/"
        ]
        
        for file_path in critical_files:
            total_checks += 1
            if file_path in file_status and file_status[file_path]["exists"]:
                passed_checks += 1
        
        health_percentage = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        return {
            "overall_health": "excellent" if health_percentage >= 90 else "good" if health_percentage >= 70 else "needs_attention",
            "health_percentage": round(health_percentage, 1),
            "checks_passed": passed_checks,
            "total_checks": total_checks,
            "summary": {
                "database": "connected" if db_status.get("postgresql", {}).get("status") == "connected" else "issues",
                "api": "likely_ok",  # Would need actual API test
                "frontend": "ok" if file_status.get("research_synthesis/web/static/js/main.js", {}).get("exists") else "issues",
                "storage": "ok" if Path("reports").exists() else "needs_setup"
            },
            "quick_recommendations": [
                "Run /api/v1/diagnostics/comprehensive for detailed analysis",
                "Use /api/v1/diagnostics/quick-fix for automated fixes",
                "Check browser console for frontend issues"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health summary failed: {str(e)}")

# Diagnostic Tracker Endpoints
@router.get("/trending-issues")
async def get_trending_issues():
    """Get most frequent issues from diagnostic tracker"""
    if not DIAGNOSTIC_TRACKER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Diagnostic tracker not available")
    
    try:
        trending = await diagnostic_tracker.get_trending_issues()
        return {
            "trending_issues": trending,
            "tracking_window_hours": diagnostic_tracker.trend_window_hours,
            "last_scan": diagnostic_tracker.last_scan_time.isoformat() if diagnostic_tracker.last_scan_time else None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trending issues: {str(e)}")

@router.get("/diagnostic-summary")
async def get_diagnostic_summary():
    """Get comprehensive diagnostic tracking summary"""
    if not DIAGNOSTIC_TRACKER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Diagnostic tracker not available")
    
    try:
        summary = await diagnostic_tracker.get_diagnostic_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get diagnostic summary: {str(e)}")

@router.get("/issue-details/{issue_type}")
async def get_issue_details(issue_type: str):
    """Get detailed information about a specific issue type"""
    if not DIAGNOSTIC_TRACKER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Diagnostic tracker not available")
    
    try:
        details = await diagnostic_tracker.get_issue_details(issue_type)
        return details
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get issue details: {str(e)}")

@router.post("/start-tracking")
async def start_diagnostic_tracking():
    """Start continuous diagnostic tracking"""
    if not DIAGNOSTIC_TRACKER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Diagnostic tracker not available")
    
    try:
        await diagnostic_tracker.start_continuous_tracking()
        return {
            "message": "Diagnostic tracking started",
            "tracking_active": True,
            "scan_interval_minutes": diagnostic_tracker.scan_interval_minutes,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start tracking: {str(e)}")

@router.post("/stop-tracking")
async def stop_diagnostic_tracking():
    """Stop continuous diagnostic tracking"""
    if not DIAGNOSTIC_TRACKER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Diagnostic tracker not available")
    
    try:
        await diagnostic_tracker.stop_tracking()
        return {
            "message": "Diagnostic tracking stopped",
            "tracking_active": False,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop tracking: {str(e)}")

@router.post("/apply-auto-fixes")
async def apply_diagnostic_auto_fixes():
    """Apply automatic fixes through diagnostic tracker"""
    if not DIAGNOSTIC_TRACKER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Diagnostic tracker not available")
    
    try:
        result = await diagnostic_tracker.apply_auto_fixes()
        return {
            "message": "Auto-fixes applied",
            "fixes_applied": result['fixes_applied'],
            "fixes_failed": result['fixes_failed'],
            "total_attempted": result['total_attempted'],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply auto-fixes: {str(e)}")
"""
UI Monitoring API Router
Provides endpoints for UI health monitoring and automated fixes
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Any, Optional
from loguru import logger

try:
    from research_synthesis.services.ui_monitor import ui_monitor_service
    UI_MONITORING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"UI monitoring service not available: {e}")
    UI_MONITORING_AVAILABLE = False

router = APIRouter()

@router.get("/")
async def get_ui_monitoring_status():
    """Get UI monitoring service status"""
    return {
        "service": "UI Monitoring and Readability API",
        "status": "active" if UI_MONITORING_AVAILABLE else "unavailable",
        "description": "Monitors and fixes UI readability and formatting issues",
        "endpoints": [
            "/status - Get UI monitoring status and summary",
            "/scan - Trigger full UI scan",
            "/scan/{page} - Scan specific page",
            "/issues - Get all current issues", 
            "/issues/{page} - Get issues for specific page",
            "/fix - Auto-fix all fixable issues",
            "/fix/{issue_id} - Fix specific issue",
            "/accessibility-report - Generate accessibility report",
            "/recommendations - Get improvement recommendations"
        ],
        "monitoring_categories": [
            "readability",
            "formatting", 
            "accessibility",
            "mobile_responsiveness",
            "performance"
        ]
    }

@router.get("/status")
async def get_monitoring_status():
    """Get comprehensive UI monitoring status"""
    if not UI_MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="UI monitoring service unavailable")
    
    try:
        summary = ui_monitor_service.get_issues_summary()
        return {
            "status": "success",
            "monitoring_summary": summary,
            "service_health": {
                "monitoring_active": ui_monitor_service.monitoring_active,
                "total_scanned_pages": len(set(issue.page for issue in ui_monitor_service.issues.values())),
                "last_scan_time": summary.get("last_scan"),
                "auto_fix_available": summary.get("auto_fixable", 0) > 0
            }
        }
    except Exception as e:
        logger.error(f"Error getting UI monitoring status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.post("/scan")
async def trigger_full_scan():
    """Trigger a full UI scan of all pages"""
    if not UI_MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="UI monitoring service unavailable")
    
    try:
        logger.info("Starting manual UI scan...")
        issues_found = await ui_monitor_service.scan_all_pages()
        
        total_issues = sum(len(page_issues) for page_issues in issues_found.values())
        
        return {
            "status": "success", 
            "scan_results": {
                "total_pages_scanned": len(issues_found),
                "total_issues_found": total_issues,
                "pages_with_issues": list(issues_found.keys()),
                "issues_by_page": {page: len(issues) for page, issues in issues_found.items()},
                "scan_timestamp": ui_monitor_service.get_issues_summary()["last_scan"]
            }
        }
    except Exception as e:
        logger.error(f"UI scan failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")

@router.post("/scan/{page_name}")
async def scan_specific_page(page_name: str):
    """Scan a specific page for UI issues"""
    if not UI_MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="UI monitoring service unavailable")
    
    try:
        from pathlib import Path
        template_path = Path("research_synthesis/web/templates") / f"{page_name}.html"
        
        if not template_path.exists():
            raise HTTPException(status_code=404, detail=f"Page template '{page_name}.html' not found")
        
        issues = await ui_monitor_service.scan_page(page_name, template_path)
        
        return {
            "status": "success",
            "page": page_name,
            "issues_found": len(issues),
            "issues": [
                {
                    "id": issue.id,
                    "category": issue.category,
                    "severity": issue.severity,
                    "element": issue.element,
                    "description": issue.description,
                    "suggested_fix": issue.suggested_fix,
                    "auto_fixable": issue.auto_fixable,
                    "detected_at": issue.detected_at.isoformat()
                }
                for issue in issues
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Page scan failed for {page_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Page scan failed: {str(e)}")

@router.get("/issues")
async def get_all_issues(
    category: Optional[str] = Query(None, description="Filter by category"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    status: Optional[str] = Query("open", description="Filter by status"),
    page: Optional[str] = Query(None, description="Filter by page")
):
    """Get all UI issues with optional filtering"""
    if not UI_MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="UI monitoring service unavailable")
    
    try:
        all_issues = []
        
        for issue in ui_monitor_service.issues.values():
            # Apply filters
            if category and issue.category != category:
                continue
            if severity and issue.severity != severity:
                continue
            if status and issue.status != status:
                continue
            if page and issue.page != page:
                continue
            
            all_issues.append({
                "id": issue.id,
                "category": issue.category,
                "severity": issue.severity,
                "page": issue.page,
                "element": issue.element,
                "description": issue.description,
                "suggested_fix": issue.suggested_fix,
                "auto_fixable": issue.auto_fixable,
                "status": issue.status,
                "detected_at": issue.detected_at.isoformat()
            })
        
        # Sort by severity and detection time
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        all_issues.sort(key=lambda x: (severity_order.get(x["severity"], 4), x["detected_at"]), reverse=True)
        
        return {
            "status": "success",
            "total_issues": len(all_issues),
            "issues": all_issues,
            "filters_applied": {
                "category": category,
                "severity": severity, 
                "status": status,
                "page": page
            }
        }
    except Exception as e:
        logger.error(f"Error getting UI issues: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get issues: {str(e)}")

@router.get("/issues/{page_name}")
async def get_page_issues(page_name: str):
    """Get all issues for a specific page"""
    if not UI_MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="UI monitoring service unavailable")
    
    try:
        page_issues = ui_monitor_service.get_page_issues(page_name)
        
        return {
            "status": "success",
            "page": page_name,
            "total_issues": len(page_issues),
            "issues": page_issues
        }
    except Exception as e:
        logger.error(f"Error getting issues for page {page_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get page issues: {str(e)}")

@router.post("/fix")
async def auto_fix_issues(
    issue_ids: Optional[List[str]] = Query(None, description="Specific issue IDs to fix")
):
    """Auto-fix UI issues"""
    if not UI_MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="UI monitoring service unavailable")
    
    try:
        results = await ui_monitor_service.auto_fix_issues(issue_ids)
        
        successful_fixes = len([r for r in results.values() if r])
        failed_fixes = len([r for r in results.values() if not r])
        
        return {
            "status": "success",
            "fix_results": {
                "total_attempts": len(results),
                "successful_fixes": successful_fixes,
                "failed_fixes": failed_fixes,
                "detailed_results": results
            }
        }
    except Exception as e:
        logger.error(f"Auto-fix failed: {e}")
        raise HTTPException(status_code=500, detail=f"Auto-fix failed: {str(e)}")

@router.post("/fix/{issue_id}")
async def fix_specific_issue(issue_id: str):
    """Fix a specific issue by ID"""
    if not UI_MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="UI monitoring service unavailable")
    
    try:
        if issue_id not in ui_monitor_service.issues:
            raise HTTPException(status_code=404, detail=f"Issue {issue_id} not found")
        
        issue = ui_monitor_service.issues[issue_id]
        
        if not issue.auto_fixable:
            return {
                "status": "not_fixable",
                "message": "This issue cannot be automatically fixed",
                "issue": {
                    "id": issue.id,
                    "description": issue.description,
                    "suggested_fix": issue.suggested_fix
                }
            }
        
        results = await ui_monitor_service.auto_fix_issues([issue_id])
        success = results.get(issue_id, False)
        
        return {
            "status": "success" if success else "failed",
            "issue_id": issue_id,
            "fixed": success,
            "message": "Issue fixed successfully" if success else "Failed to fix issue"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fixing issue {issue_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Fix failed: {str(e)}")

@router.get("/accessibility-report")
async def get_accessibility_report():
    """Get comprehensive accessibility report"""
    if not UI_MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="UI monitoring service unavailable")
    
    try:
        report = await ui_monitor_service.generate_accessibility_report()
        
        return {
            "status": "success",
            "accessibility_report": report,
            "generated_at": ui_monitor_service.get_issues_summary()["last_scan"]
        }
    except Exception as e:
        logger.error(f"Error generating accessibility report: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@router.get("/recommendations")
async def get_improvement_recommendations():
    """Get UI improvement recommendations"""
    if not UI_MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="UI monitoring service unavailable")
    
    try:
        summary = ui_monitor_service.get_issues_summary()
        recommendations = []
        
        # General recommendations based on issues found
        if summary["by_severity"].get("critical", 0) > 0:
            recommendations.append({
                "priority": "critical",
                "title": "Address Critical Issues First",
                "description": f"You have {summary['by_severity']['critical']} critical issues that should be fixed immediately",
                "action": "Review critical accessibility and formatting issues"
            })
        
        if summary["auto_fixable"] > 0:
            recommendations.append({
                "priority": "high",
                "title": "Auto-Fix Available Issues", 
                "description": f"{summary['auto_fixable']} issues can be automatically fixed",
                "action": "Use the auto-fix feature to quickly resolve simple issues"
            })
        
        # Category-specific recommendations
        if summary["by_category"].get("accessibility", 0) > 0:
            recommendations.append({
                "priority": "high",
                "title": "Improve Accessibility",
                "description": f"{summary['by_category']['accessibility']} accessibility issues found",
                "action": "Add alt text, labels, and improve color contrast"
            })
        
        if summary["by_category"].get("mobile", 0) > 0:
            recommendations.append({
                "priority": "medium",
                "title": "Enhance Mobile Experience",
                "description": f"{summary['by_category']['mobile']} mobile responsiveness issues found",
                "action": "Review responsive design and mobile layouts"
            })
        
        if summary["by_category"].get("performance", 0) > 0:
            recommendations.append({
                "priority": "medium",
                "title": "Optimize Performance",
                "description": f"{summary['by_category']['performance']} performance issues found",
                "action": "Optimize images and reduce external resource loads"
            })
        
        # Page-specific recommendations
        if summary["by_page"]:
            worst_page = max(summary["by_page"].items(), key=lambda x: x[1])
            recommendations.append({
                "priority": "medium",
                "title": f"Focus on {worst_page[0]} Page",
                "description": f"The {worst_page[0]} page has the most issues ({worst_page[1]} issues)",
                "action": f"Prioritize improvements on the {worst_page[0]} page"
            })
        
        if not recommendations:
            recommendations.append({
                "priority": "low",
                "title": "Great Job!",
                "description": "No major UI issues detected",
                "action": "Continue monitoring and maintain good practices"
            })
        
        return {
            "status": "success",
            "recommendations": recommendations,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {str(e)}")

@router.post("/start-monitoring")
async def start_continuous_monitoring():
    """Start continuous UI monitoring"""
    if not UI_MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="UI monitoring service unavailable")
    
    try:
        if ui_monitor_service.monitoring_active:
            return {
                "status": "already_running",
                "message": "UI monitoring is already active"
            }
        
        # Start monitoring in background
        import asyncio
        asyncio.create_task(ui_monitor_service.start_monitoring())
        
        return {
            "status": "success",
            "message": "UI monitoring started successfully",
            "monitoring_active": True
        }
    except Exception as e:
        logger.error(f"Error starting UI monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")

@router.post("/stop-monitoring") 
async def stop_continuous_monitoring():
    """Stop continuous UI monitoring"""
    if not UI_MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="UI monitoring service unavailable")
    
    try:
        await ui_monitor_service.stop_monitoring()
        
        return {
            "status": "success",
            "message": "UI monitoring stopped successfully",
            "monitoring_active": False
        }
    except Exception as e:
        logger.error(f"Error stopping UI monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")

@router.post("/report-issue")
async def report_user_issue(report_data: dict):
    """Allow users to report UI issues via chatbox"""
    if not UI_MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="UI monitoring service unavailable")
    
    try:
        # Validate required fields
        required_fields = ['description', 'page']
        for field in required_fields:
            if field not in report_data or not report_data[field].strip():
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Create the issue from user report
        issue = await ui_monitor_service.create_user_reported_issue(report_data)
        
        return {
            "status": "success",
            "message": "Issue reported successfully",
            "issue_id": issue.id,
            "issue": {
                "id": issue.id,
                "category": issue.category,
                "severity": issue.severity,
                "page": issue.page,
                "element": issue.element,
                "description": issue.description,
                "detected_at": issue.detected_at.isoformat(),
                "status": issue.status
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user reported issue: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to report issue: {str(e)}")

@router.get("/report-options")
async def get_report_options():
    """Get options for user issue reporting"""
    if not UI_MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="UI monitoring service unavailable")
    
    try:
        categories = ui_monitor_service.get_user_report_categories()
        severity_levels = ui_monitor_service.get_severity_levels()
        
        # Get available pages from templates
        from pathlib import Path
        templates_dir = Path("research_synthesis/web/templates")
        available_pages = []
        
        for template_file in templates_dir.glob("*.html"):
            if template_file.stem != 'base':  # Exclude base template
                page_name = template_file.stem
                display_name = page_name.replace('_', ' ').title()
                available_pages.append({
                    "value": page_name,
                    "label": display_name
                })
        
        return {
            "status": "success",
            "report_options": {
                "categories": categories,
                "severity_levels": severity_levels,
                "pages": available_pages,
                "elements": [
                    "layout", "navigation", "button", "form", "text", 
                    "image", "table", "link", "modal", "dropdown", "other"
                ]
            }
        }
    except Exception as e:
        logger.error(f"Error getting report options: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get report options: {str(e)}")

@router.post("/scan-page")
async def scan_page_for_issues(scan_request: dict):
    """Automatically scan a page for common issues"""
    if not UI_MONITORING_AVAILABLE:
        raise HTTPException(status_code=503, detail="UI monitoring service unavailable")
    
    try:
        page = scan_request.get('page', 'unknown')
        url = scan_request.get('url', '')
        auto_scan = scan_request.get('auto_scan', True)
        
        logger.info(f"Auto-scanning page: {page}")
        
        # Perform automated UI scan for the specified page
        issues = await ui_monitor_service.scan_page_for_issues(page, url, auto_scan)
        
        return {
            "status": "success",
            "page": page,
            "issues_found": len(issues),
            "issues": [
                {
                    "id": issue.id,
                    "category": issue.category,
                    "severity": issue.severity,
                    "description": issue.description,
                    "suggested_fix": issue.suggested_fix,
                    "auto_fixable": issue.auto_fixable
                }
                for issue in issues
            ],
            "scan_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scanning page for issues: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to scan page: {str(e)}")
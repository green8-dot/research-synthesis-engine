"""
System Tester API Router

Provides API endpoints for comprehensive system testing and issue monitoring.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from datetime import datetime

router = APIRouter()

# Import system tester
try:
    from research_synthesis.services.system_tester import system_tester
    SYSTEM_TESTER_AVAILABLE = True
except ImportError:
    SYSTEM_TESTER_AVAILABLE = False
    system_tester = None


@router.get("/")
async def get_system_tester_overview():
    """Get system tester overview"""
    if not SYSTEM_TESTER_AVAILABLE:
        raise HTTPException(status_code=503, detail="System tester not available")
    
    stats = system_tester.get_test_statistics()
    
    return {
        "status": "System Tester active",
        "service": "ready", 
        "available_endpoints": [
            "/comprehensive - Run comprehensive system tests",
            "/quick-check - Run quick health check",
            "/statistics - Get testing statistics",
            "/known-issues - List known issues from system analysis",
            "/test-categories - Get available test categories"
        ],
        "statistics": stats,
        "last_checked": datetime.now().isoformat()
    }


@router.post("/comprehensive")
async def run_comprehensive_tests():
    """Run comprehensive system tests"""
    if not SYSTEM_TESTER_AVAILABLE:
        raise HTTPException(status_code=503, detail="System tester not available")
    
    try:
        results = await system_tester.run_comprehensive_tests()
        return {
            "message": "Comprehensive system tests completed",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System tests failed: {str(e)}")


@router.get("/quick-check")
async def run_quick_health_check():
    """Run quick health check"""
    if not SYSTEM_TESTER_AVAILABLE:
        raise HTTPException(status_code=503, detail="System tester not available")
    
    try:
        results = await system_tester.run_quick_health_check()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick check failed: {str(e)}")


@router.get("/statistics")
async def get_test_statistics():
    """Get testing statistics and history"""
    if not SYSTEM_TESTER_AVAILABLE:
        raise HTTPException(status_code=503, detail="System tester not available")
    
    try:
        stats = system_tester.get_test_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.get("/known-issues")
async def get_known_issues():
    """Get list of known issues detected from system analysis"""
    if not SYSTEM_TESTER_AVAILABLE:
        raise HTTPException(status_code=503, detail="System tester not available")
    
    return {
        "known_issues": [
            {
                "issue": "Database retrieval failed: invalid literal for int() with base 10: 'undefined'",
                "category": "api",
                "severity": "medium", 
                "description": "API endpoints receiving 'undefined' as ID parameter",
                "impact": "Report download functionality affected",
                "auto_fixable": True,
                "recommendations": [
                    "Add input validation for ID parameters",
                    "Check frontend JavaScript ID passing",
                    "Implement better error handling"
                ]
            },
            {
                "issue": "Real pipeline unavailable, job will fail",
                "category": "integration",
                "severity": "high",
                "description": "Data scraping pipeline not available for processing jobs",
                "impact": "Scraping functionality completely broken",
                "auto_fixable": False,
                "recommendations": [
                    "Check pipeline service status",
                    "Implement fallback processing",
                    "Add user notification for pipeline status"
                ]
            },
            {
                "issue": "UI contrast and accessibility issues",
                "category": "ui",
                "severity": "medium",
                "description": "Tables without responsive classes, buttons missing aria-labels",
                "impact": "Poor mobile experience and accessibility",
                "auto_fixable": True,
                "recommendations": [
                    "Continue running UI health monitor",
                    "Verify auto-fixes are persistent",
                    "Regular accessibility testing"
                ]
            },
            {
                "issue": "System integration loading hanging", 
                "category": "performance",
                "severity": "medium",
                "description": "System integration element stuck on loading component status",
                "impact": "User interface showing loading indefinitely",
                "auto_fixable": False,
                "recommendations": [
                    "Add timeout handling",
                    "Implement loading state management",
                    "Check component status service performance"
                ]
            },
            {
                "issue": "Multiple concurrent report generation requests",
                "category": "performance", 
                "severity": "low",
                "description": "System handling multiple concurrent report generation",
                "impact": "Potential resource usage spikes",
                "auto_fixable": False,
                "recommendations": [
                    "Implement request queuing",
                    "Add rate limiting",
                    "Monitor resource usage during peaks"
                ]
            }
        ],
        "total_issues": 5,
        "auto_fixable_count": 2,
        "severity_breakdown": {
            "critical": 0,
            "high": 1,
            "medium": 3,
            "low": 1
        },
        "category_breakdown": {
            "api": 1,
            "integration": 1, 
            "ui": 1,
            "performance": 2
        },
        "last_updated": datetime.now().isoformat()
    }


@router.get("/test-categories")
async def get_test_categories():
    """Get available test categories and descriptions"""
    return {
        "categories": [
            {
                "category": "database",
                "name": "Database Tests",
                "description": "Test database connections, queries, and data integrity",
                "tests": [
                    "Database connectivity",
                    "Query performance",
                    "Data consistency", 
                    "Connection pooling"
                ]
            },
            {
                "category": "api", 
                "name": "API Tests",
                "description": "Test API endpoints, responses, and error handling",
                "tests": [
                    "Endpoint availability",
                    "Response validation",
                    "Error handling",
                    "Input validation"
                ]
            },
            {
                "category": "ui",
                "name": "UI Tests", 
                "description": "Test user interface responsiveness and accessibility",
                "tests": [
                    "Responsive design",
                    "Accessibility compliance",
                    "Color contrast",
                    "Mobile compatibility"
                ]
            },
            {
                "category": "performance",
                "name": "Performance Tests",
                "description": "Test system performance and resource usage",
                "tests": [
                    "Response times",
                    "Resource utilization", 
                    "Concurrent load handling",
                    "Memory usage"
                ]
            },
            {
                "category": "security",
                "name": "Security Tests",
                "description": "Test basic security configurations and practices", 
                "tests": [
                    "Input validation",
                    "Error disclosure",
                    "Authentication checks",
                    "HTTPS configuration"
                ]
            },
            {
                "category": "integration",
                "name": "Integration Tests",
                "description": "Test system integration and component interactions",
                "tests": [
                    "Service dependencies",
                    "Data flow integrity",
                    "Component communication",
                    "External service availability"
                ]
            },
            {
                "category": "functionality", 
                "name": "Functionality Tests",
                "description": "Test core system functionality and features",
                "tests": [
                    "Report generation",
                    "Data processing",
                    "Search functionality", 
                    "Export features"
                ]
            },
            {
                "category": "data_quality",
                "name": "Data Quality Tests",
                "description": "Test data integrity, consistency, and quality",
                "tests": [
                    "Data validation",
                    "Consistency checks",
                    "Completeness analysis",
                    "Quality metrics"
                ]
            }
        ],
        "total_categories": 8,
        "estimated_test_time": "30-60 seconds for comprehensive tests"
    }


@router.get("/health")
async def get_system_tester_health():
    """Get system tester health status"""
    if not SYSTEM_TESTER_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "System tester not installed or configured"
        }
    
    stats = system_tester.get_test_statistics()
    
    return {
        "status": "excellent",
        "service": "active",
        "total_tests_run": stats["total_tests_run"],
        "last_test": stats.get("last_full_test"),
        "test_coverage": stats.get("test_coverage", {}),
        "known_issues_tracked": len(system_tester.known_issues),
        "recommendations": [
            "Run comprehensive tests daily",
            "Monitor known issues for resolution",
            "Keep test coverage comprehensive"
        ],
        "timestamp": datetime.now().isoformat()
    }
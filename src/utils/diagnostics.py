"""
Comprehensive Diagnostics and Troubleshooting System
Provides detailed system health checks, error analysis, and debugging tools
"""
import asyncio
import json
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

class SystemDiagnostics:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.info = []
        
    async def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run all diagnostic checks and return comprehensive report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_health": {},
            "database_status": {},
            "api_status": {},
            "frontend_status": {},
            "common_issues": {},
            "recommendations": [],
            "debug_info": {}
        }
        
        # Database diagnostics
        report["database_status"] = await self._check_databases()
        
        # API diagnostics
        report["api_status"] = await self._check_api_endpoints()
        
        # File system diagnostics
        report["system_health"]["file_system"] = self._check_file_system()
        
        # Frontend diagnostics
        report["frontend_status"] = self._check_frontend_files()
        
        # Common issue patterns
        report["common_issues"] = await self._check_common_issues()
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations()
        
        # Debug information
        report["debug_info"] = self._collect_debug_info()
        
        return report
    
    async def _check_databases(self) -> Dict[str, Any]:
        """Comprehensive database connection testing"""
        db_status = {}
        
        # PostgreSQL/SQLite
        try:
            from research_synthesis.database.connection import get_postgres_session
            from research_synthesis.database.models import Entity
            from sqlalchemy import select, text
            
            async for db_session in get_postgres_session():
                # Test basic connection
                result = await db_session.execute(text("SELECT 1"))
                
                # Test table access
                entities_result = await db_session.execute(select(Entity).limit(1))
                
                db_status["postgresql"] = {
                    "status": "connected",
                    "type": "primary",
                    "connection_test": "passed",
                    "table_access": "passed",
                    "last_checked": datetime.now().isoformat()
                }
                break
        except Exception as e:
            db_status["postgresql"] = {
                "status": "error",
                "type": "primary",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "last_checked": datetime.now().isoformat()
            }
        
        # Redis
        try:
            from research_synthesis.database.connection import get_redis
            redis_client = get_redis()
            if redis_client:
                await redis_client.ping()
                # Test set/get
                await redis_client.set("diagnostic_test", "ok", ex=10)
                test_value = await redis_client.get("diagnostic_test")
                
                db_status["redis"] = {
                    "status": "connected",
                    "type": "cache",
                    "ping_test": "passed",
                    "read_write_test": "passed" if test_value == "ok" else "failed",
                    "last_checked": datetime.now().isoformat()
                }
            else:
                db_status["redis"] = {
                    "status": "not_available",
                    "type": "cache",
                    "reason": "client_not_initialized",
                    "last_checked": datetime.now().isoformat()
                }
        except Exception as e:
            db_status["redis"] = {
                "status": "error",
                "type": "cache", 
                "error": str(e),
                "traceback": traceback.format_exc(),
                "last_checked": datetime.now().isoformat()
            }
        
        return db_status
    
    async def _check_api_endpoints(self) -> Dict[str, Any]:
        """Test all critical API endpoints"""
        import aiohttp
        
        endpoints = {
            "/api/v1/health": "GET",
            "/api/v1/reports/": "GET",
            "/api/v1/knowledge/": "GET",
            "/api/v1/scraping/": "GET",
            "/api/v1/reports/generate": "POST"
        }
        
        api_status = {}
        
        async with aiohttp.ClientSession() as session:
            for endpoint, method in endpoints.items():
                try:
                    url = f"http://localhost:8001{endpoint}"
                    
                    if method == "GET":
                        async with session.get(url, timeout=5) as response:
                            api_status[endpoint] = {
                                "status": "ok" if response.status == 200 else "error",
                                "http_status": response.status,
                                "response_time_ms": None,  # Would need timing
                                "last_checked": datetime.now().isoformat()
                            }
                    elif method == "POST" and "generate" in endpoint:
                        # Test report generation
                        test_data = {
                            "report_type": "diagnostic_test",
                            "title": "Diagnostic Test Report",
                            "focus_area": "testing"
                        }
                        async with session.post(url, json=test_data, timeout=10) as response:
                            response_data = await response.json()
                            api_status[endpoint] = {
                                "status": "ok" if response.status == 200 else "error",
                                "http_status": response.status,
                                "response_data": response_data,
                                "last_checked": datetime.now().isoformat()
                            }
                            
                except Exception as e:
                    api_status[endpoint] = {
                        "status": "error",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "last_checked": datetime.now().isoformat()
                    }
        
        return api_status
    
    def _check_file_system(self) -> Dict[str, Any]:
        """Check critical files and directories"""
        critical_paths = [
            "research_synthesis/web/templates/",
            "research_synthesis/web/static/js/main.js",
            "research_synthesis/web/static/css/main.css",
            "research_synthesis/api/",
            "research_synthesis/database/",
            "reports/"
        ]
        
        file_status = {}
        
        for path in critical_paths:
            path_obj = Path(path)
            file_status[path] = {
                "exists": path_obj.exists(),
                "is_file": path_obj.is_file() if path_obj.exists() else None,
                "is_dir": path_obj.is_dir() if path_obj.exists() else None,
                "size_bytes": path_obj.stat().st_size if path_obj.exists() and path_obj.is_file() else None,
                "last_modified": datetime.fromtimestamp(path_obj.stat().st_mtime).isoformat() if path_obj.exists() else None
            }
        
        return file_status
    
    def _check_frontend_files(self) -> Dict[str, Any]:
        """Check frontend JavaScript and CSS for common issues"""
        frontend_status = {}
        
        # Check main.js
        main_js_path = Path("research_synthesis/web/static/js/main.js")
        if main_js_path.exists():
            content = main_js_path.read_text(encoding='utf-8')
            frontend_status["main_js"] = {
                "exists": True,
                "size": len(content),
                "has_rsapp_class": "class ResearchSynthesisApp" in content,
                "has_form_handlers": "handleFormSubmission" in content,
                "has_axios": "axios" in content,
                "syntax_check": self._check_js_syntax(content)
            }
        else:
            frontend_status["main_js"] = {"exists": False, "error": "File not found"}
        
        # Check reports.html
        reports_html_path = Path("research_synthesis/web/templates/reports.html")
        if reports_html_path.exists():
            content = reports_html_path.read_text(encoding='utf-8')
            frontend_status["reports_html"] = {
                "exists": True,
                "size": len(content),
                "has_modal": "generateReportModal" in content,
                "has_form": "generate-report-form" in content,
                "has_submit_handler": "addEventListener" in content,
                "has_debug_function": "testReportGeneration" in content
            }
        
        return frontend_status
    
    def _check_js_syntax(self, js_content: str) -> Dict[str, Any]:
        """Basic JavaScript syntax checking"""
        issues = []
        
        # Check for common syntax issues
        lines = js_content.split('/n')
        for i, line in enumerate(lines, 1):
            # Check for unmatched brackets
            if line.count('{') != line.count('}') and not line.strip().endswith('{') and not line.strip().endswith('}'):
                if '{' in line or '}' in line:
                    issues.append(f"Line {i}: Possible unmatched brackets")
            
            # Check for console.log statements (should be removed in production)
            if 'console.log' in line:
                issues.append(f"Line {i}: Console.log statement found")
            
            # Check for common typos
            if 'fucntion' in line or 'funciton' in line:
                issues.append(f"Line {i}: Possible function typo")
        
        return {
            "issues_found": len(issues),
            "issues": issues[:10],  # Limit to first 10 issues
            "total_lines": len(lines)
        }
    
    async def _check_common_issues(self) -> Dict[str, Any]:
        """Check for known recurring issues"""
        common_issues = {}
        
        # Check for report generation issues
        try:
            from research_synthesis.api.reports_router import stored_reports
            common_issues["report_storage"] = {
                "reports_in_memory": len(stored_reports),
                "status": "ok" if len(stored_reports) > 0 else "no_reports"
            }
            
            # Check reports directory
            reports_dir = Path("reports")
            if reports_dir.exists():
                report_files = list(reports_dir.glob("*.json"))
                common_issues["report_files"] = {
                    "directory_exists": True,
                    "file_count": len(report_files),
                    "recent_files": [f.name for f in sorted(report_files, key=lambda x: x.stat().st_mtime)[-5:]]
                }
            else:
                common_issues["report_files"] = {
                    "directory_exists": False,
                    "issue": "Reports directory missing"
                }
                
        except Exception as e:
            common_issues["report_storage"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check for form submission issues
        common_issues["form_issues"] = self._diagnose_form_issues()
        
        # Check for caching issues
        common_issues["caching"] = await self._diagnose_caching_issues()
        
        return common_issues
    
    def _diagnose_form_issues(self) -> Dict[str, Any]:
        """Diagnose common form submission issues"""
        issues = []
        
        # Check if reports template has proper form structure
        reports_template = Path("research_synthesis/web/templates/reports.html")
        if reports_template.exists():
            content = reports_template.read_text()
            
            if 'data-submit-api="/reports/generate"' not in content:
                issues.append("Form API endpoint might be incorrect")
            
            if 'form="generate-report-form"' not in content:
                issues.append("Submit button not properly linked to form")
                
            if 'addEventListener' not in content:
                issues.append("Missing JavaScript event handlers")
        
        return {
            "potential_issues": issues,
            "recommendations": [
                "Ensure form has correct data-submit-api attribute",
                "Verify submit button has proper form attribute",
                "Check JavaScript event handlers are attached"
            ]
        }
    
    async def _diagnose_caching_issues(self) -> Dict[str, Any]:
        """Diagnose caching-related issues"""
        caching_status = {}
        
        try:
            from research_synthesis.database.connection import get_redis
            redis_client = get_redis()
            
            if redis_client:
                # Test cache operations
                test_key = "diagnostic_cache_test"
                await redis_client.set(test_key, "test_value", ex=60)
                retrieved = await redis_client.get(test_key)
                
                caching_status = {
                    "redis_available": True,
                    "write_test": "passed",
                    "read_test": "passed" if retrieved == "test_value" else "failed",
                    "recommendation": "Redis caching is working properly"
                }
            else:
                caching_status = {
                    "redis_available": False,
                    "fallback_active": True,
                    "recommendation": "Using database fallback - consider installing Redis for better performance"
                }
                
        except Exception as e:
            caching_status = {
                "redis_available": False,
                "error": str(e),
                "fallback_active": True,
                "recommendation": "Caching errors detected - using database fallback"
            }
        
        return caching_status
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on diagnostics"""
        recommendations = []
        
        # Always include these general recommendations
        recommendations.extend([
            "Run 'pip install redis' to enable caching for better performance",
            "Use hard refresh (Ctrl+F5) when JavaScript changes aren't reflected",
            "Check browser console (F12) for JavaScript errors during debugging",
            "Monitor server logs for backend errors and performance issues"
        ])
        
        # Add specific recommendations based on detected issues
        if self.issues:
            recommendations.extend([
                "Address critical issues found in system diagnostics",
                "Consider running diagnostic checks regularly to prevent recurring issues"
            ])
        
        return recommendations
    
    def _collect_debug_info(self) -> Dict[str, Any]:
        """Collect general debug information"""
        import sys
        import platform
        
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "current_working_directory": str(Path.cwd()),
            "environment_variables": {
                "API_PORT": "8001",  # Assuming default
                "PYTHON_PATH": sys.path[0] if sys.path else None
            },
            "diagnostic_run_time": datetime.now().isoformat()
        }

# Global instance
diagnostics = SystemDiagnostics()
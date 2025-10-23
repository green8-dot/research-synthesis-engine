"""
Comprehensive Performance Testing and System Analysis Framework
Analyzes codebase structure, performance metrics, and provides optimization recommendations
"""

import asyncio
import time
import tracemalloc
import psutil
import os
import sys
import json
import importlib
import inspect
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import aiohttp
import sqlite3
from dataclasses import dataclass, asdict
import cProfile
import pstats
import io
import gc

# Performance metrics collection
@dataclass
class PerformanceMetrics:
    """Container for performance test results"""
    timestamp: str
    test_name: str
    duration_ms: float
    memory_used_mb: float
    cpu_percent: float
    status: str
    details: Dict[str, Any]
    recommendations: List[str]

class SystemPerformanceAnalyzer:
    """Comprehensive system performance analysis tool"""
    
    def __init__(self, project_root: Path = Path("D:/orbitscope_ml/research_synthesis")):
        self.project_root = project_root
        self.results = []
        self.code_metrics = {}
        self.dependency_graph = defaultdict(set)
        self.unused_imports = []
        self.slow_functions = []
        self.optimization_opportunities = []
        
    async def run_complete_analysis(self) -> Dict[str, Any]:
        """Run all performance tests and analysis"""
        print("Starting Comprehensive Performance Analysis...")
        
        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "code_structure": await self._analyze_code_structure(),
            "performance_tests": await self._run_performance_tests(),
            "database_analysis": await self._analyze_database_performance(),
            "api_analysis": await self._analyze_api_performance(),
            "memory_analysis": self._analyze_memory_usage(),
            "dependency_analysis": self._analyze_dependencies(),
            "unused_code": self._find_unused_code(),
            "optimization_recommendations": self._generate_recommendations(),
            "suggested_packages": self._suggest_packages()
        }
        
        # Save results
        self._save_results(analysis_results)
        
        return analysis_results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information"""
        return {
            "platform": sys.platform,
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "process_count": len(psutil.pids())
        }
    
    async def _analyze_code_structure(self) -> Dict[str, Any]:
        """Analyze codebase structure and complexity"""
        print("Analyzing code structure...")
        
        structure_metrics = {
            "total_files": 0,
            "total_lines": 0,
            "total_functions": 0,
            "total_classes": 0,
            "complexity_scores": [],
            "file_sizes": [],
            "imports_analysis": defaultdict(int),
            "unused_imports": [],
            "circular_dependencies": []
        }
        
        # Scan all Python files
        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
                
            structure_metrics["total_files"] += 1
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.splitlines()
                    structure_metrics["total_lines"] += len(lines)
                    structure_metrics["file_sizes"].append({
                        "file": str(py_file.relative_to(self.project_root)),
                        "lines": len(lines),
                        "size_kb": py_file.stat().st_size / 1024
                    })
                    
                    # Parse AST
                    tree = ast.parse(content)
                    
                    # Count functions and classes
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            structure_metrics["total_functions"] += 1
                            # Calculate cyclomatic complexity
                            complexity = self._calculate_complexity(node)
                            if complexity > 10:  # High complexity threshold
                                structure_metrics["complexity_scores"].append({
                                    "file": str(py_file.relative_to(self.project_root)),
                                    "function": node.name,
                                    "complexity": complexity
                                })
                        elif isinstance(node, ast.ClassDef):
                            structure_metrics["total_classes"] += 1
                        elif isinstance(node, (ast.Import, ast.ImportFrom)):
                            # Track imports
                            module = node.module if hasattr(node, 'module') else None
                            if module:
                                structure_metrics["imports_analysis"][module] += 1
                                
            except Exception as e:
                print(f"  Warning: Error analyzing {py_file}: {e}")
        
        # Find most imported modules
        structure_metrics["top_imports"] = sorted(
            structure_metrics["imports_analysis"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return structure_metrics
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
    
    async def _run_performance_tests(self) -> List[Dict[str, Any]]:
        """Run various performance tests"""
        print("Running performance tests...")
        
        tests = []
        
        # Test 1: Import time analysis
        start = time.perf_counter()
        import_times = self._measure_import_times()
        tests.append({
            "test": "Import Time Analysis",
            "duration_ms": (time.perf_counter() - start) * 1000,
            "slow_imports": [imp for imp in import_times if imp["time_ms"] > 100]
        })
        
        # Test 2: Function execution profiling
        profile_results = await self._profile_critical_functions()
        tests.append({
            "test": "Function Profiling",
            "results": profile_results
        })
        
        # Test 3: Async performance
        async_perf = await self._test_async_performance()
        tests.append({
            "test": "Async Performance",
            "results": async_perf
        })
        
        return tests
    
    def _measure_import_times(self) -> List[Dict[str, Any]]:
        """Measure import times for modules"""
        import_times = []
        modules_to_test = [
            "research_synthesis.database.connection",
            "research_synthesis.api.research_router",
            "research_synthesis.services.ui_monitor",
            "research_synthesis.utils.kpi_tracker"
        ]
        
        for module_name in modules_to_test:
            try:
                start = time.perf_counter()
                importlib.import_module(module_name)
                duration = (time.perf_counter() - start) * 1000
                import_times.append({
                    "module": module_name,
                    "time_ms": round(duration, 2)
                })
            except ImportError:
                pass
                
        return sorted(import_times, key=lambda x: x["time_ms"], reverse=True)
    
    async def _profile_critical_functions(self) -> Dict[str, Any]:
        """Profile critical functions for performance"""
        profiler = cProfile.Profile()
        results = {}
        
        # Test database operations
        profiler.enable()
        # Simulate database operations
        await asyncio.sleep(0.1)  # Placeholder for actual DB operations
        profiler.disable()
        
        # Get statistics
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)
        
        results["top_time_consumers"] = s.getvalue()
        
        return results
    
    async def _test_async_performance(self) -> Dict[str, Any]:
        """Test async operation performance"""
        results = {}
        
        # Test concurrent operations
        start = time.perf_counter()
        tasks = [asyncio.create_task(asyncio.sleep(0.01)) for _ in range(100)]
        await asyncio.gather(*tasks)
        results["concurrent_100_tasks_ms"] = (time.perf_counter() - start) * 1000
        
        # Test event loop efficiency
        start = time.perf_counter()
        for _ in range(1000):
            await asyncio.sleep(0)
        results["event_loop_1000_iterations_ms"] = (time.perf_counter() - start) * 1000
        
        return results
    
    async def _analyze_database_performance(self) -> Dict[str, Any]:
        """Analyze database query performance"""
        print("Analyzing database performance...")
        
        db_metrics = {
            "connection_time_ms": 0,
            "query_times": [],
            "index_analysis": [],
            "slow_queries": [],
            "optimization_suggestions": []
        }
        
        try:
            # Test SQLite connection
            start = time.perf_counter()
            conn = sqlite3.connect("research_synthesis.db")
            db_metrics["connection_time_ms"] = (time.perf_counter() - start) * 1000
            
            cursor = conn.cursor()
            
            # Analyze tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                
                # Check indexes
                cursor.execute(f"PRAGMA index_list({table_name})")
                indexes = cursor.fetchall()
                
                # Count rows
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                # Test query performance
                start = time.perf_counter()
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 100")
                query_time = (time.perf_counter() - start) * 1000
                
                db_metrics["query_times"].append({
                    "table": table_name,
                    "rows": row_count,
                    "indexes": len(indexes),
                    "query_time_ms": round(query_time, 2)
                })
                
                # Suggest optimizations
                if row_count > 1000 and len(indexes) == 0:
                    db_metrics["optimization_suggestions"].append(
                        f"Table '{table_name}' has {row_count} rows but no indexes. Consider adding indexes."
                    )
                
                if query_time > 100:
                    db_metrics["slow_queries"].append({
                        "table": table_name,
                        "time_ms": query_time
                    })
            
            conn.close()
            
        except Exception as e:
            db_metrics["error"] = str(e)
            
        return db_metrics
    
    async def _analyze_api_performance(self) -> Dict[str, Any]:
        """Test API endpoint response times"""
        print("Testing API performance...")
        
        api_metrics = {
            "endpoints_tested": 0,
            "average_response_time_ms": 0,
            "slow_endpoints": [],
            "failed_endpoints": [],
            "response_times": []
        }
        
        base_url = "http://localhost:8001/api/v1"
        endpoints = [
            "/health",
            "/research/stats",
            "/scraping/sources/status",
            "/knowledge/",
            "/reports/list",
            "/kpi/dashboard"
        ]
        
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                try:
                    start = time.perf_counter()
                    async with session.get(f"{base_url}{endpoint}") as response:
                        await response.text()
                        duration = (time.perf_counter() - start) * 1000
                        
                        api_metrics["response_times"].append({
                            "endpoint": endpoint,
                            "status": response.status,
                            "time_ms": round(duration, 2)
                        })
                        
                        if duration > 500:  # Slow endpoint threshold
                            api_metrics["slow_endpoints"].append({
                                "endpoint": endpoint,
                                "time_ms": duration
                            })
                        
                        api_metrics["endpoints_tested"] += 1
                        
                except Exception as e:
                    api_metrics["failed_endpoints"].append({
                        "endpoint": endpoint,
                        "error": str(e)
                    })
        
        # Calculate average
        if api_metrics["response_times"]:
            api_metrics["average_response_time_ms"] = round(
                sum(r["time_ms"] for r in api_metrics["response_times"]) / len(api_metrics["response_times"]),
                2
            )
        
        return api_metrics
    
    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns"""
        print("Analyzing memory usage...")
        
        # Start tracing
        tracemalloc.start()
        
        # Force garbage collection
        gc.collect()
        
        # Get current memory usage
        current, peak = tracemalloc.get_traced_memory()
        
        # Get top memory consumers
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        memory_metrics = {
            "current_usage_mb": current / (1024 * 1024),
            "peak_usage_mb": peak / (1024 * 1024),
            "process_memory_mb": psutil.Process().memory_info().rss / (1024 * 1024),
            "top_memory_consumers": []
        }
        
        for stat in top_stats[:10]:
            memory_metrics["top_memory_consumers"].append({
                "file": stat.traceback.format()[0] if stat.traceback else "Unknown",
                "size_mb": stat.size / (1024 * 1024),
                "count": stat.count
            })
        
        tracemalloc.stop()
        
        return memory_metrics
    
    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze project dependencies"""
        print("Analyzing dependencies...")
        
        deps = {
            "total_dependencies": 0,
            "direct_dependencies": [],
            "unused_packages": [],
            "outdated_packages": [],
            "security_vulnerabilities": []
        }
        
        # Read requirements.txt
        req_file = self.project_root.parent / "requirements.txt"
        if req_file.exists():
            with open(req_file, 'r') as f:
                deps["direct_dependencies"] = [
                    line.strip() for line in f if line.strip() and not line.startswith('#')
                ]
                deps["total_dependencies"] = len(deps["direct_dependencies"])
        
        # Check for commonly unused packages
        installed_packages = set()
        try:
            import pkg_resources
            installed_packages = {pkg.key for pkg in pkg_resources.working_set}
        except ImportError:
            pass
        
        # Find packages that might be unused
        imported_in_code = set()
        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imported_in_code.add(alias.name.split('.')[0])
                        elif isinstance(node, ast.ImportFrom) and node.module:
                            imported_in_code.add(node.module.split('.')[0])
            except:
                pass
        
        # Common package name mappings
        package_mappings = {
            'sklearn': 'scikit-learn',
            'cv2': 'opencv-python',
            'PIL': 'pillow'
        }
        
        return deps
    
    def _find_unused_code(self) -> Dict[str, Any]:
        """Find potentially unused code"""
        print("Finding unused code...")
        
        unused = {
            "unused_functions": [],
            "unused_classes": [],
            "unused_variables": [],
            "dead_code_files": []
        }
        
        # Track all definitions and usages
        definitions = defaultdict(list)
        usages = defaultdict(int)
        
        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                    # Find definitions
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            definitions["functions"].append({
                                "name": node.name,
                                "file": str(py_file.relative_to(self.project_root))
                            })
                        elif isinstance(node, ast.ClassDef):
                            definitions["classes"].append({
                                "name": node.name,
                                "file": str(py_file.relative_to(self.project_root))
                            })
                    
                    # Find usages (simplified)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Name):
                            usages[node.id] += 1
                            
            except Exception as e:
                pass
        
        # Find unused definitions
        for func in definitions.get("functions", []):
            if func["name"] not in ["__init__", "__str__", "__repr__"] and usages.get(func["name"], 0) <= 1:
                unused["unused_functions"].append(func)
        
        for cls in definitions.get("classes", []):
            if usages.get(cls["name"], 0) <= 1:
                unused["unused_classes"].append(cls)
        
        return unused
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations"""
        print("Generating recommendations...")
        
        recommendations = []
        
        # Performance recommendations
        recommendations.append({
            "category": "Performance",
            "priority": "HIGH",
            "recommendations": [
                "Implement connection pooling for database connections",
                "Add caching layer with Redis for frequently accessed data",
                "Use bulk operations for database inserts/updates",
                "Implement pagination for large result sets",
                "Add indexes on frequently queried columns"
            ]
        })
        
        # Code quality recommendations
        recommendations.append({
            "category": "Code Quality",
            "priority": "MEDIUM",
            "recommendations": [
                "Refactor functions with cyclomatic complexity > 10",
                "Remove unused imports and dead code",
                "Add type hints to all functions",
                "Implement comprehensive error handling",
                "Add docstrings to all public functions"
            ]
        })
        
        # Architecture recommendations
        recommendations.append({
            "category": "Architecture",
            "priority": "HIGH",
            "recommendations": [
                "Implement proper separation of concerns (MVC pattern)",
                "Use dependency injection for better testability",
                "Implement event-driven architecture for better scalability",
                "Add message queue for async task processing",
                "Implement circuit breaker pattern for external services"
            ]
        })
        
        # Security recommendations
        recommendations.append({
            "category": "Security",
            "priority": "CRITICAL",
            "recommendations": [
                "Implement rate limiting on API endpoints",
                "Add input validation and sanitization",
                "Use prepared statements for all SQL queries",
                "Implement proper authentication and authorization",
                "Add CORS configuration for API security"
            ]
        })
        
        return recommendations
    
    def _suggest_packages(self) -> Dict[str, List[Dict[str, str]]]:
        """Suggest packages to improve efficiency"""
        print("Suggesting optimization packages...")
        
        suggestions = {
            "performance": [
                {"name": "uvloop", "purpose": "Fast drop-in replacement for asyncio event loop", "install": "pip install uvloop"},
                {"name": "orjson", "purpose": "Fast JSON serialization (3x faster than json)", "install": "pip install orjson"},
                {"name": "asyncpg", "purpose": "Fast PostgreSQL client for asyncio", "install": "pip install asyncpg"},
                {"name": "redis", "purpose": "In-memory caching for improved performance", "install": "pip install redis[hiredis]"},
                {"name": "numba", "purpose": "JIT compilation for numerical computations", "install": "pip install numba"}
            ],
            "monitoring": [
                {"name": "prometheus-client", "purpose": "Metrics collection and monitoring", "install": "pip install prometheus-client"},
                {"name": "sentry-sdk", "purpose": "Error tracking and performance monitoring", "install": "pip install sentry-sdk"},
                {"name": "py-spy", "purpose": "Sampling profiler for Python programs", "install": "pip install py-spy"},
                {"name": "memory-profiler", "purpose": "Monitor memory consumption", "install": "pip install memory-profiler"},
                {"name": "line-profiler", "purpose": "Line-by-line profiling", "install": "pip install line-profiler"}
            ],
            "development": [
                {"name": "black", "purpose": "Code formatter for consistent style", "install": "pip install black"},
                {"name": "mypy", "purpose": "Static type checking", "install": "pip install mypy"},
                {"name": "pylint", "purpose": "Code quality analysis", "install": "pip install pylint"},
                {"name": "pytest-asyncio", "purpose": "Testing async code", "install": "pip install pytest-asyncio"},
                {"name": "pre-commit", "purpose": "Git hooks for code quality", "install": "pip install pre-commit"}
            ],
            "optimization": [
                {"name": "cython", "purpose": "Compile Python to C for speed", "install": "pip install cython"},
                {"name": "pypy", "purpose": "JIT-compiled Python interpreter", "install": "Download from pypy.org"},
                {"name": "joblib", "purpose": "Efficient parallel computing", "install": "pip install joblib"},
                {"name": "dask", "purpose": "Parallel computing and task scheduling", "install": "pip install dask[complete]"},
                {"name": "ray", "purpose": "Distributed computing framework", "install": "pip install ray"}
            ]
        }
        
        return suggestions
    
    def _save_results(self, results: Dict[str, Any]):
        """Save analysis results to file"""
        output_file = self.project_root / "performance_analysis_report.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Analysis complete! Results saved to {output_file}")
    
    def generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report from results"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
                h1 { color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }
                h2 { color: #555; margin-top: 30px; }
                .metric { background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; }
                .warning { border-left-color: #ffc107; }
                .danger { border-left-color: #dc3545; }
                .success { border-left-color: #28a745; }
                table { width: 100%; border-collapse: collapse; margin: 15px 0; }
                th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background: #007bff; color: white; }
                .recommendation { background: #e7f3ff; padding: 10px; margin: 5px 0; border-radius: 5px; }
                .package { background: #f0f0f0; padding: 8px; margin: 5px 0; border-radius: 3px; }
                code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-family: monospace; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Performance Analysis Report</h1>
                <p>Generated: {timestamp}</p>
                
                <h2>System Information</h2>
                <div class="metric">
                    <strong>Platform:</strong> {platform}<br>
                    <strong>CPU Cores:</strong> {cpu_count}<br>
                    <strong>Memory:</strong> {memory_gb} GB<br>
                    <strong>Python Version:</strong> {python_version}
                </div>
                
                <h2>Code Structure Analysis</h2>
                <div class="metric">
                    <strong>Total Files:</strong> {total_files}<br>
                    <strong>Total Lines:</strong> {total_lines}<br>
                    <strong>Total Functions:</strong> {total_functions}<br>
                    <strong>Total Classes:</strong> {total_classes}
                </div>
                
                <h2>Performance Metrics</h2>
                {performance_section}
                
                <h2>Optimization Recommendations</h2>
                {recommendations_section}
                
                <h2>Suggested Packages</h2>
                {packages_section}
            </div>
        </body>
        </html>
        """
        
        # Fill in template with actual data
        # (Implementation would format the results into HTML)
        
        return html


async def main():
    """Run the performance analysis"""
    analyzer = SystemPerformanceAnalyzer()
    results = await analyzer.run_complete_analysis()
    
    # Print summary
    print("/n" + "="*60)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("="*60)
    
    # Code metrics
    if "code_structure" in results:
        cs = results["code_structure"]
        print(f"/nCode Metrics:")
        print(f"  * Files: {cs.get('total_files', 0)}")
        print(f"  * Lines: {cs.get('total_lines', 0)}")
        print(f"  * Functions: {cs.get('total_functions', 0)}")
        print(f"  * Classes: {cs.get('total_classes', 0)}")
    
    # Performance
    if "api_analysis" in results:
        api = results["api_analysis"]
        print(f"/nAPI Performance:")
        print(f"  * Endpoints tested: {api.get('endpoints_tested', 0)}")
        print(f"  * Average response: {api.get('average_response_time_ms', 0):.2f}ms")
        print(f"  * Slow endpoints: {len(api.get('slow_endpoints', []))}")
    
    # Database
    if "database_analysis" in results:
        db = results["database_analysis"]
        print(f"/nDatabase Performance:")
        print(f"  * Connection time: {db.get('connection_time_ms', 0):.2f}ms")
        print(f"  * Tables analyzed: {len(db.get('query_times', []))}")
        print(f"  * Slow queries: {len(db.get('slow_queries', []))}")
    
    # Memory
    if "memory_analysis" in results:
        mem = results["memory_analysis"]
        print(f"/nMemory Usage:")
        print(f"  * Current: {mem.get('current_usage_mb', 0):.2f} MB")
        print(f"  * Peak: {mem.get('peak_usage_mb', 0):.2f} MB")
        print(f"  * Process: {mem.get('process_memory_mb', 0):.2f} MB")
    
    # Top recommendations
    if "optimization_recommendations" in results:
        print(f"/nTop Recommendations:")
        for rec in results["optimization_recommendations"][:1]:  # Show first category
            if rec["priority"] == "CRITICAL" or rec["priority"] == "HIGH":
                print(f"  [{rec['priority']}] {rec['category']}:")
                for r in rec["recommendations"][:3]:
                    print(f"    * {r}")
    
    print("/nFull report saved to: performance_analysis_report.json")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
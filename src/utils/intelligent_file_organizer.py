"""
Intelligent File Organization and Monitoring System
Manages failing elements, monitors problems before they occur, and auto-organizes files
"""

import asyncio
import os
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
import hashlib
import subprocess
import psutil

from loguru import logger
from research_synthesis.database.connection import get_postgres_session
from research_synthesis.database.models import Base
from sqlalchemy import Column, Integer, String, DateTime, JSON, Boolean, Text, Float
from sqlalchemy.ext.declarative import declarative_base


@dataclass
class FileHealth:
    """File health assessment"""
    path: str
    health_score: float  # 0-100
    issues: List[str]
    last_modified: datetime
    size: int
    dependencies: List[str]
    usage_frequency: int
    critical_level: str  # low, medium, high, critical


@dataclass
class SystemAlert:
    """System monitoring alert"""
    alert_type: str
    severity: str  # info, warning, error, critical
    component: str
    description: str
    suggested_action: str
    timestamp: datetime
    auto_fixable: bool


class FileOrganization(Base):
    """Database model for file organization tracking"""
    __tablename__ = 'file_organization'
    
    id = Column(Integer, primary_key=True)
    file_path = Column(String(500), unique=True)
    original_location = Column(String(500))
    current_location = Column(String(500))
    category = Column(String(100))
    health_score = Column(Float)
    last_health_check = Column(DateTime, default=datetime.now)
    move_count = Column(Integer, default=0)
    dependencies = Column(JSON)
    usage_stats = Column(JSON)
    issues_detected = Column(JSON)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now)


class SystemMonitoring(Base):
    """Database model for system monitoring"""
    __tablename__ = 'system_monitoring'
    
    id = Column(Integer, primary_key=True)
    component_type = Column(String(100))
    component_name = Column(String(200))
    metric_name = Column(String(100))
    metric_value = Column(Float)
    threshold_warning = Column(Float)
    threshold_critical = Column(Float)
    status = Column(String(50))  # healthy, warning, critical, failed
    last_check = Column(DateTime, default=datetime.now)
    consecutive_failures = Column(Integer, default=0)
    alert_sent = Column(Boolean, default=False)
    meta_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.now)


class IntelligentFileOrganizer:
    """Advanced file organization and system monitoring"""
    
    def __init__(self, root_path: str = None):
        self.root_path = Path(root_path or os.getcwd())
        self.organization_rules = self._load_organization_rules()
        self.monitoring_config = self._load_monitoring_config()
        self.file_cache = {}
        self.system_health = {}
        
    def _load_organization_rules(self) -> Dict[str, Any]:
        """Load file organization rules"""
        return {
            "categories": {
                "core": {
                    "patterns": ["main.py", "app.py", "__init__.py", "config.py", "settings.py"],
                    "priority": "critical",
                    "max_files": 50,
                    "location": "core/"
                },
                "services": {
                    "patterns": ["*_service.py", "*_router.py", "*Service.py"],
                    "priority": "high",
                    "max_files": 100,
                    "location": "services/"
                },
                "utils": {
                    "patterns": ["*_utils.py", "*_helper.py", "*_tool.py"],
                    "priority": "medium",
                    "max_files": 200,
                    "location": "utils/"
                },
                "templates": {
                    "patterns": ["*.html", "*.jinja2", "*.j2"],
                    "priority": "medium",
                    "max_files": 100,
                    "location": "templates/"
                },
                "static": {
                    "patterns": ["*.css", "*.js", "*.ico", "*.png", "*.jpg", "*.gif"],
                    "priority": "low",
                    "max_files": 500,
                    "location": "static/"
                },
                "data": {
                    "patterns": ["*.json", "*.csv", "*.xlsx", "*.db", "*.sqlite"],
                    "priority": "high",
                    "max_files": 50,
                    "location": "data/"
                },
                "docs": {
                    "patterns": ["*.md", "*.rst", "*.txt", "README*", "*.pdf"],
                    "priority": "low",
                    "max_files": 100,
                    "location": "docs/"
                },
                "tests": {
                    "patterns": ["test_*.py", "*_test.py", "tests.py"],
                    "priority": "medium",
                    "max_files": 200,
                    "location": "tests/"
                },
                "temporary": {
                    "patterns": ["*.tmp", "*.temp", "*.log", "*.cache"],
                    "priority": "cleanup",
                    "max_age_days": 7,
                    "location": "temp/"
                },
                "backups": {
                    "patterns": ["*.bak", "*_backup*", "*.backup"],
                    "priority": "archive",
                    "max_age_days": 30,
                    "location": "backups/"
                }
            },
            "dependency_tracking": {
                "python": {
                    "import_patterns": [r"from/s+(/S+)", r"import/s+(/S+)"],
                    "config_files": ["requirements.txt", "setup.py", "pyproject.toml"]
                },
                "web": {
                    "import_patterns": [r"@import/s+['/"]([^'/"]+)", r"src=['/"]([^'/"]+)"],
                    "config_files": ["package.json", "webpack.config.js"]
                }
            }
        }
    
    def _load_monitoring_config(self) -> Dict[str, Any]:
        """Load system monitoring configuration"""
        return {
            "system_metrics": {
                "cpu_usage": {"warning": 80.0, "critical": 95.0},
                "memory_usage": {"warning": 85.0, "critical": 95.0},
                "disk_usage": {"warning": 85.0, "critical": 95.0},
                "disk_io": {"warning": 80.0, "critical": 95.0}
            },
            "application_metrics": {
                "response_time": {"warning": 2.0, "critical": 5.0},
                "error_rate": {"warning": 5.0, "critical": 10.0},
                "memory_leaks": {"warning": 100.0, "critical": 500.0},
                "connection_pool": {"warning": 80.0, "critical": 95.0}
            },
            "file_metrics": {
                "modification_frequency": {"warning": 100, "critical": 500},
                "size_growth": {"warning": 1000000, "critical": 10000000},
                "dependency_violations": {"warning": 5, "critical": 10},
                "circular_dependencies": {"warning": 1, "critical": 3}
            },
            "check_intervals": {
                "system": 60,  # seconds
                "application": 30,
                "files": 300
            }
        }
    
    async def analyze_file_health(self, file_path: Path) -> FileHealth:
        """Comprehensive file health analysis"""
        try:
            stat = file_path.stat()
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            issues = []
            health_score = 100.0
            
            # Size analysis
            if stat.st_size > 1000000:  # > 1MB
                issues.append(f"Large file size: {stat.st_size / 1024 / 1024:.1f}MB")
                health_score -= 10
            
            # Complexity analysis for Python files
            if file_path.suffix == '.py':
                lines = content.split('/n')
                
                # Line count analysis
                if len(lines) > 1000:
                    issues.append(f"Very long file: {len(lines)} lines")
                    health_score -= 15
                
                # Function complexity
                function_count = content.count('def ')
                if function_count > 50:
                    issues.append(f"Too many functions: {function_count}")
                    health_score -= 10
                
                # Import analysis
                import_count = len([line for line in lines if line.strip().startswith(('import ', 'from '))])
                if import_count > 30:
                    issues.append(f"Too many imports: {import_count}")
                    health_score -= 5
                
                # Syntax validation
                try:
                    compile(content, str(file_path), 'exec')
                except SyntaxError as e:
                    issues.append(f"Syntax error: {e}")
                    health_score -= 30
            
            # Dependency analysis
            dependencies = self._extract_dependencies(file_path, content)
            
            # Usage frequency (mock - would track actual usage)
            usage_frequency = self._calculate_usage_frequency(file_path)
            
            # Critical level assessment
            critical_level = self._assess_critical_level(file_path, content)
            
            return FileHealth(
                path=str(file_path),
                health_score=max(0, health_score),
                issues=issues,
                last_modified=datetime.fromtimestamp(stat.st_mtime),
                size=stat.st_size,
                dependencies=dependencies,
                usage_frequency=usage_frequency,
                critical_level=critical_level
            )
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return FileHealth(
                path=str(file_path),
                health_score=0,
                issues=[f"Analysis failed: {e}"],
                last_modified=datetime.now(),
                size=0,
                dependencies=[],
                usage_frequency=0,
                critical_level="unknown"
            )
    
    def _extract_dependencies(self, file_path: Path, content: str) -> List[str]:
        """Extract file dependencies"""
        dependencies = []
        
        if file_path.suffix == '.py':
            import re
            
            # Extract imports
            import_patterns = [
                r'from/s+([^/s]+)/s+import',
                r'import/s+([^/s,]+)',
            ]
            
            for pattern in import_patterns:
                matches = re.findall(pattern, content)
                dependencies.extend(matches)
        
        elif file_path.suffix in ['.html', '.j2', '.jinja2']:
            import re
            
            # Extract template dependencies
            extends_pattern = r'{%/s*extends/s+["/']([^"/']+)["/']'
            include_pattern = r'{%/s*include/s+["/']([^"/']+)["/']'
            
            dependencies.extend(re.findall(extends_pattern, content))
            dependencies.extend(re.findall(include_pattern, content))
        
        return list(set(dependencies))
    
    def _calculate_usage_frequency(self, file_path: Path) -> int:
        """Calculate how frequently a file is used (mock implementation)"""
        # In a real implementation, this would track:
        # - Git commit frequency
        # - Import frequency by other files
        # - Access patterns
        # - Execution frequency
        
        # Mock calculation based on file type and location
        if 'main' in file_path.name.lower():
            return 100
        elif file_path.suffix == '.py':
            return 50
        elif file_path.suffix in ['.html', '.css', '.js']:
            return 25
        else:
            return 10
    
    def _assess_critical_level(self, file_path: Path, content: str) -> str:
        """Assess how critical a file is to the system"""
        critical_indicators = [
            'main.py', 'app.py', '__init__.py', 'config.py', 'settings.py',
            'database', 'auth', 'security', 'api', 'server'
        ]
        
        path_lower = str(file_path).lower()
        content_lower = content.lower()
        
        critical_count = sum(1 for indicator in critical_indicators 
                           if indicator in path_lower or indicator in content_lower)
        
        if critical_count >= 3:
            return "critical"
        elif critical_count >= 2:
            return "high"
        elif critical_count >= 1:
            return "medium"
        else:
            return "low"
    
    async def organize_files(self) -> Dict[str, Any]:
        """Intelligent file organization"""
        organization_report = {
            "files_organized": 0,
            "files_moved": 0,
            "issues_fixed": 0,
            "categories": {},
            "alerts": []
        }
        
        try:
            # Scan all files
            all_files = []
            for root, dirs, files in os.walk(self.root_path):
                # Skip hidden directories and common ignore patterns
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]
                
                for file in files:
                    if not file.startswith('.'):
                        file_path = Path(root) / file
                        all_files.append(file_path)
            
            logger.info(f"Found {len(all_files)} files to analyze")
            
            # Analyze each file
            for file_path in all_files:
                try:
                    health = await self.analyze_file_health(file_path)
                    
                    # Determine category
                    category = self._categorize_file(file_path)
                    
                    if category not in organization_report["categories"]:
                        organization_report["categories"][category] = {
                            "files": [],
                            "avg_health": 0,
                            "issues": []
                        }
                    
                    organization_report["categories"][category]["files"].append({
                        "path": str(file_path),
                        "health": health.health_score,
                        "issues": health.issues,
                        "critical_level": health.critical_level
                    })
                    
                    # Check if file needs to be moved
                    if await self._should_reorganize_file(file_path, category, health):
                        await self._move_file_to_category(file_path, category)
                        organization_report["files_moved"] += 1
                    
                    organization_report["files_organized"] += 1
                    
                    # Generate alerts for problematic files
                    if health.health_score < 50:
                        alert = SystemAlert(
                            alert_type="file_health",
                            severity="warning" if health.health_score > 25 else "critical",
                            component=str(file_path),
                            description=f"File health score: {health.health_score:.1f}%, Issues: {', '.join(health.issues)}",
                            suggested_action=self._get_suggested_action(health),
                            timestamp=datetime.now(),
                            auto_fixable=health.health_score > 25
                        )
                        organization_report["alerts"].append(alert)
                        
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
            
            # Calculate category averages
            for category_data in organization_report["categories"].values():
                if category_data["files"]:
                    category_data["avg_health"] = sum(f["health"] for f in category_data["files"]) / len(category_data["files"])
                    category_data["issues"] = [issue for f in category_data["files"] for issue in f["issues"]]
            
            logger.info(f"File organization complete: {organization_report['files_organized']} files processed, {organization_report['files_moved']} moved")
            return organization_report
            
        except Exception as e:
            logger.error(f"Error during file organization: {e}")
            organization_report["error"] = str(e)
            return organization_report
    
    def _categorize_file(self, file_path: Path) -> str:
        """Categorize file based on patterns and content"""
        for category, config in self.organization_rules["categories"].items():
            for pattern in config["patterns"]:
                if file_path.match(pattern):
                    return category
        
        # Default categorization based on location
        parts = file_path.parts
        if 'test' in parts:
            return 'tests'
        elif 'template' in parts:
            return 'templates'
        elif 'static' in parts:
            return 'static'
        elif 'service' in parts:
            return 'services'
        elif 'util' in parts:
            return 'utils'
        else:
            return 'miscellaneous'
    
    async def _should_reorganize_file(self, file_path: Path, category: str, health: FileHealth) -> bool:
        """Determine if a file should be moved to better organize the project"""
        # Don't move critical files or files that are already well-placed
        if health.critical_level == "critical":
            return False
        
        # Check if file is in correct category location
        category_config = self.organization_rules["categories"].get(category, {})
        expected_location = category_config.get("location", "")
        
        if expected_location and expected_location not in str(file_path):
            return True
        
        return False
    
    async def _move_file_to_category(self, file_path: Path, category: str):
        """Move file to appropriate category directory"""
        try:
            category_config = self.organization_rules["categories"].get(category, {})
            target_location = category_config.get("location", f"{category}/")
            
            target_dir = self.root_path / target_location
            target_dir.mkdir(parents=True, exist_ok=True)
            
            target_path = target_dir / file_path.name
            
            # Avoid overwriting existing files
            counter = 1
            original_target = target_path
            while target_path.exists():
                stem = original_target.stem
                suffix = original_target.suffix
                target_path = target_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            shutil.move(str(file_path), str(target_path))
            logger.info(f"Moved {file_path} -> {target_path}")
            
        except Exception as e:
            logger.error(f"Error moving file {file_path}: {e}")
    
    def _get_suggested_action(self, health: FileHealth) -> str:
        """Get suggested action based on file health issues"""
        if "Syntax error" in str(health.issues):
            return "Fix syntax errors in the file"
        elif "Very long file" in str(health.issues):
            return "Consider breaking this file into smaller modules"
        elif "Too many functions" in str(health.issues):
            return "Refactor to reduce function count or split into multiple files"
        elif "Large file size" in str(health.issues):
            return "Review file content and optimize or split if necessary"
        elif "Too many imports" in str(health.issues):
            return "Review and consolidate imports, consider dependency injection"
        else:
            return "Review file and address reported issues"
    
    async def monitor_system_health(self) -> List[SystemAlert]:
        """Proactive system health monitoring"""
        alerts = []
        
        try:
            # System resource monitoring
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # CPU monitoring
            if cpu_percent > self.monitoring_config["system_metrics"]["cpu_usage"]["critical"]:
                alerts.append(SystemAlert(
                    alert_type="system_resource",
                    severity="critical",
                    component="CPU",
                    description=f"CPU usage at {cpu_percent:.1f}%",
                    suggested_action="Investigate high CPU processes, consider scaling",
                    timestamp=datetime.now(),
                    auto_fixable=False
                ))
            elif cpu_percent > self.monitoring_config["system_metrics"]["cpu_usage"]["warning"]:
                alerts.append(SystemAlert(
                    alert_type="system_resource",
                    severity="warning",
                    component="CPU",
                    description=f"CPU usage at {cpu_percent:.1f}%",
                    suggested_action="Monitor CPU usage trends",
                    timestamp=datetime.now(),
                    auto_fixable=False
                ))
            
            # Memory monitoring
            if memory.percent > self.monitoring_config["system_metrics"]["memory_usage"]["critical"]:
                alerts.append(SystemAlert(
                    alert_type="system_resource",
                    severity="critical",
                    component="Memory",
                    description=f"Memory usage at {memory.percent:.1f}%",
                    suggested_action="Investigate memory leaks, restart services if needed",
                    timestamp=datetime.now(),
                    auto_fixable=True
                ))
            
            # Disk monitoring
            if disk.percent > self.monitoring_config["system_metrics"]["disk_usage"]["warning"]:
                alerts.append(SystemAlert(
                    alert_type="system_resource",
                    severity="warning" if disk.percent < 95 else "critical",
                    component="Disk",
                    description=f"Disk usage at {disk.percent:.1f}%",
                    suggested_action="Clean temporary files, archive old data",
                    timestamp=datetime.now(),
                    auto_fixable=True
                ))
            
            # Application-specific monitoring
            alerts.extend(await self._monitor_application_health())
            
        except Exception as e:
            logger.error(f"Error monitoring system health: {e}")
            alerts.append(SystemAlert(
                alert_type="monitoring_error",
                severity="error",
                component="Health Monitor",
                description=f"Monitoring system error: {e}",
                suggested_action="Check monitoring system configuration",
                timestamp=datetime.now(),
                auto_fixable=False
            ))
        
        return alerts
    
    async def _monitor_application_health(self) -> List[SystemAlert]:
        """Monitor application-specific health metrics"""
        alerts = []
        
        try:
            # Check database connections
            try:
                # Test database connectivity
                async for session in get_postgres_session():
                    await session.execute("SELECT 1")
                    break
            except Exception as e:
                alerts.append(SystemAlert(
                    alert_type="database",
                    severity="critical",
                    component="PostgreSQL",
                    description=f"Database connection failed: {e}",
                    suggested_action="Check database service status and connection parameters",
                    timestamp=datetime.now(),
                    auto_fixable=False
                ))
            
            # Check for common application issues
            temp_files = list(self.root_path.glob("**/*.tmp"))
            if len(temp_files) > 100:
                alerts.append(SystemAlert(
                    alert_type="file_cleanup",
                    severity="warning",
                    component="File System",
                    description=f"Found {len(temp_files)} temporary files",
                    suggested_action="Clean up temporary files to free space",
                    timestamp=datetime.now(),
                    auto_fixable=True
                ))
            
            # Check log file sizes
            log_files = list(self.root_path.glob("**/*.log"))
            large_logs = [f for f in log_files if f.stat().st_size > 100_000_000]  # 100MB
            if large_logs:
                alerts.append(SystemAlert(
                    alert_type="log_management",
                    severity="warning",
                    component="Logging",
                    description=f"Found {len(large_logs)} large log files",
                    suggested_action="Rotate or archive large log files",
                    timestamp=datetime.now(),
                    auto_fixable=True
                ))
            
        except Exception as e:
            logger.error(f"Error monitoring application health: {e}")
        
        return alerts
    
    async def auto_fix_issues(self, alerts: List[SystemAlert]) -> Dict[str, Any]:
        """Automatically fix issues where possible"""
        fix_results = {
            "attempted_fixes": 0,
            "successful_fixes": 0,
            "failed_fixes": 0,
            "details": []
        }
        
        for alert in alerts:
            if not alert.auto_fixable:
                continue
            
            fix_results["attempted_fixes"] += 1
            
            try:
                success = False
                
                if alert.alert_type == "file_cleanup" and "temporary files" in alert.description:
                    # Clean temporary files
                    temp_files = list(self.root_path.glob("**/*.tmp"))
                    for temp_file in temp_files:
                        try:
                            temp_file.unlink()
                        except:
                            pass
                    success = True
                    fix_results["details"].append(f"Cleaned {len(temp_files)} temporary files")
                
                elif alert.alert_type == "log_management":
                    # Rotate large log files
                    log_files = list(self.root_path.glob("**/*.log"))
                    for log_file in log_files:
                        if log_file.stat().st_size > 100_000_000:
                            backup_name = f"{log_file}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
                            shutil.move(str(log_file), backup_name)
                            log_file.touch()  # Create new empty log file
                    success = True
                    fix_results["details"].append("Rotated large log files")
                
                elif alert.alert_type == "system_resource" and alert.component == "Memory":
                    # Memory cleanup (simplified)
                    import gc
                    gc.collect()
                    success = True
                    fix_results["details"].append("Performed memory garbage collection")
                
                if success:
                    fix_results["successful_fixes"] += 1
                else:
                    fix_results["failed_fixes"] += 1
                    
            except Exception as e:
                fix_results["failed_fixes"] += 1
                fix_results["details"].append(f"Failed to fix {alert.alert_type}: {e}")
                logger.error(f"Auto-fix failed for {alert.alert_type}: {e}")
        
        return fix_results
    
    async def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report"""
        logger.info("Generating comprehensive system health report...")
        
        # Run all monitoring and organization tasks
        organization_results = await self.organize_files()
        system_alerts = await self.monitor_system_health()
        auto_fix_results = await self.auto_fix_issues(system_alerts)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_status": "healthy" if len([a for a in system_alerts if a.severity == "critical"]) == 0 else "issues_detected",
            "file_organization": organization_results,
            "system_monitoring": {
                "total_alerts": len(system_alerts),
                "critical_alerts": len([a for a in system_alerts if a.severity == "critical"]),
                "warning_alerts": len([a for a in system_alerts if a.severity == "warning"]),
                "alerts": [
                    {
                        "type": alert.alert_type,
                        "severity": alert.severity,
                        "component": alert.component,
                        "description": alert.description,
                        "suggested_action": alert.suggested_action,
                        "auto_fixable": alert.auto_fixable
                    }
                    for alert in system_alerts
                ]
            },
            "auto_fixes": auto_fix_results,
            "recommendations": self._generate_recommendations(organization_results, system_alerts)
        }
        
        logger.info(f"Health report generated: {report['system_status']}")
        return report
    
    def _generate_recommendations(self, organization_results: Dict, alerts: List[SystemAlert]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # File organization recommendations
        if organization_results.get("files_moved", 0) > 0:
            recommendations.append(f"Successfully reorganized {organization_results['files_moved']} files for better structure")
        
        categories = organization_results.get("categories", {})
        unhealthy_categories = [cat for cat, data in categories.items() if data.get("avg_health", 100) < 70]
        if unhealthy_categories:
            recommendations.append(f"Review and refactor files in categories: {', '.join(unhealthy_categories)}")
        
        # System monitoring recommendations
        critical_alerts = [a for a in alerts if a.severity == "critical"]
        if critical_alerts:
            recommendations.append(f"Address {len(critical_alerts)} critical system issues immediately")
        
        warning_alerts = [a for a in alerts if a.severity == "warning"]
        if warning_alerts:
            recommendations.append(f"Monitor {len(warning_alerts)} warning conditions that may escalate")
        
        # Proactive recommendations
        recommendations.extend([
            "Set up automated monitoring alerts for critical system metrics",
            "Implement regular file cleanup and organization schedules",
            "Consider implementing automated testing for critical components",
            "Review and update dependency management regularly"
        ])
        
        return recommendations


# Global organizer instance
file_organizer = IntelligentFileOrganizer()
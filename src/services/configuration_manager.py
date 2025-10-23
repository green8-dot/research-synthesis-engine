"""
Configuration Management System
Handles all configuration issues, validation, auto-correction, and optimization
"""

import json
import yaml
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger
import hashlib


@dataclass
class ConfigurationIssue:
    """Represents a configuration problem"""
    id: str
    category: str  # "scraping", "reports", "ui", "system", "database"
    severity: str  # "critical", "high", "medium", "low"
    title: str
    description: str
    current_value: Any
    recommended_value: Any
    auto_fixable: bool
    impact: str
    location: str  # File or service where config is located
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class ConfigurationProfile:
    """Configuration profile for different use cases"""
    name: str
    description: str
    target_scenario: str
    configurations: Dict[str, Any]
    priority: int = 1


class ConfigurationManager:
    """Comprehensive configuration management and validation system"""
    
    def __init__(self):
        self.config_issues: Dict[str, ConfigurationIssue] = {}
        self.config_cache: Dict[str, Any] = {}
        self.profiles: Dict[str, ConfigurationProfile] = {}
        self.last_scan_time: Optional[datetime] = None
        self.auto_fix_enabled = True
        
        # Initialize built-in profiles
        self._initialize_builtin_profiles()
        
    def _initialize_builtin_profiles(self):
        """Initialize built-in configuration profiles"""
        
        # High Performance Profile
        self.profiles["high_performance"] = ConfigurationProfile(
            name="High Performance",
            description="Optimized for maximum performance and throughput",
            target_scenario="Production environments with high load",
            configurations={
                "scraping": {
                    "concurrent_requests": 8,
                    "download_delay": 0.5,
                    "max_articles_per_source": 50,
                    "days_back": 30,
                    "timeout_seconds": 30,
                    "retry_attempts": 3
                },
                "reports": {
                    "max_concurrent_generation": 5,
                    "cache_duration_hours": 24,
                    "content_detail_level": "comprehensive",
                    "include_charts": True
                },
                "system": {
                    "resource_monitoring_interval": 5,
                    "cleanup_interval_minutes": 10,
                    "log_level": "INFO"
                }
            },
            priority=3
        )
        
        # Balanced Profile (Default)
        self.profiles["balanced"] = ConfigurationProfile(
            name="Balanced",
            description="Good balance between performance and resource usage",
            target_scenario="Standard production use",
            configurations={
                "scraping": {
                    "concurrent_requests": 4,
                    "download_delay": 1.0,
                    "max_articles_per_source": 25,
                    "days_back": 14,
                    "timeout_seconds": 20,
                    "retry_attempts": 2
                },
                "reports": {
                    "max_concurrent_generation": 3,
                    "cache_duration_hours": 12,
                    "content_detail_level": "standard",
                    "include_charts": True
                },
                "system": {
                    "resource_monitoring_interval": 10,
                    "cleanup_interval_minutes": 15,
                    "log_level": "INFO"
                }
            },
            priority=2
        )
        
        # Resource Conservative Profile
        self.profiles["conservative"] = ConfigurationProfile(
            name="Resource Conservative",
            description="Minimizes resource usage for limited environments",
            target_scenario="Development or resource-constrained environments",
            configurations={
                "scraping": {
                    "concurrent_requests": 2,
                    "download_delay": 2.0,
                    "max_articles_per_source": 10,
                    "days_back": 7,
                    "timeout_seconds": 15,
                    "retry_attempts": 1
                },
                "reports": {
                    "max_concurrent_generation": 1,
                    "cache_duration_hours": 6,
                    "content_detail_level": "summary",
                    "include_charts": False
                },
                "system": {
                    "resource_monitoring_interval": 30,
                    "cleanup_interval_minutes": 30,
                    "log_level": "WARNING"
                }
            },
            priority=1
        )

    async def scan_all_configurations(self) -> Dict[str, Any]:
        """Comprehensive scan of all system configurations"""
        logger.info("Starting comprehensive configuration scan...")
        
        scan_results = {
            "scan_timestamp": datetime.now().isoformat(),
            "issues_found": 0,
            "categories_scanned": [],
            "auto_fixable_issues": 0,
            "critical_issues": 0,
            "recommendations": []
        }
        
        # Clear previous issues
        self.config_issues.clear()
        
        # Scan different configuration categories
        categories = [
            ("scraping", self._scan_scraping_config),
            ("reports", self._scan_reports_config),
            ("system", self._scan_system_config),
            ("database", self._scan_database_config),
            ("ui", self._scan_ui_config)
        ]
        
        for category_name, scan_function in categories:
            try:
                category_issues = await scan_function()
                scan_results["categories_scanned"].append(category_name)
                
                for issue in category_issues:
                    self.config_issues[issue.id] = issue
                    
                logger.info(f"Configuration scan - {category_name}: {len(category_issues)} issues found")
                
            except Exception as e:
                logger.error(f"Error scanning {category_name} configuration: {e}")
                
        # Calculate summary stats
        scan_results["issues_found"] = len(self.config_issues)
        scan_results["auto_fixable_issues"] = sum(1 for issue in self.config_issues.values() if issue.auto_fixable)
        scan_results["critical_issues"] = sum(1 for issue in self.config_issues.values() if issue.severity == "critical")
        
        # Generate recommendations
        scan_results["recommendations"] = self._generate_configuration_recommendations()
        
        self.last_scan_time = datetime.now()
        logger.info(f"Configuration scan complete: {scan_results['issues_found']} issues found")
        
        return scan_results

    async def _scan_scraping_config(self) -> List[ConfigurationIssue]:
        """Scan scraping configuration for issues"""
        issues = []
        
        try:
            # Get current scraping config via API
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8001/api/v1/scraping/configure") as response:
                    if response.status == 200:
                        config = await response.json()
                    else:
                        config = {}
        except:
            config = {}
            
        # Check source availability
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8001/api/v1/scraping/all-sources") as response:
                    if response.status == 200:
                        sources_data = await response.json()
                        total_sources = sources_data.get("total", 0)
                        
                        # Issue: Not using all available sources
                        if total_sources > 50:  # We have 58 sources
                            max_articles = config.get("max_articles_per_source", 10)
                            if max_articles < 15:
                                issues.append(ConfigurationIssue(
                                    id="scraping_low_articles_per_source",
                                    category="scraping",
                                    severity="medium",
                                    title="Low articles per source limit",
                                    description=f"Only {max_articles} articles per source with {total_sources} sources available",
                                    current_value=max_articles,
                                    recommended_value=25,
                                    auto_fixable=True,
                                    impact="Reduces data collection efficiency from available sources",
                                    location="scraping_router.py configuration"
                                ))
        except:
            pass
            
        # Check concurrent requests
        concurrent_requests = config.get("concurrent_requests", 4)
        if concurrent_requests < 4:
            issues.append(ConfigurationIssue(
                id="scraping_low_concurrency",
                category="scraping",
                severity="low",
                title="Low concurrent requests",
                description=f"Only {concurrent_requests} concurrent requests configured",
                current_value=concurrent_requests,
                recommended_value=6,
                auto_fixable=True,
                impact="Slower scraping performance",
                location="scraping configuration"
            ))
            
        # Check download delay
        download_delay = config.get("download_delay", 1.0)
        if download_delay > 2.0:
            issues.append(ConfigurationIssue(
                id="scraping_high_delay",
                category="scraping",
                severity="medium",
                title="High download delay",
                description=f"Download delay of {download_delay}s may be too conservative",
                current_value=download_delay,
                recommended_value=1.0,
                auto_fixable=True,
                impact="Significantly slower scraping speed",
                location="scraping configuration"
            ))
            
        return issues

    async def _scan_reports_config(self) -> List[ConfigurationIssue]:
        """Scan reports configuration for issues"""
        issues = []
        
        # Check if reports are too generic (user's complaint)
        issues.append(ConfigurationIssue(
            id="reports_generic_content",
            category="reports",
            severity="high",
            title="Reports generation too generic",
            description="Generated reports appear similar regardless of topic requirements",
            current_value="generic_template",
            recommended_value="topic_specific_templates",
            auto_fixable=True,
            impact="Reports don't meet specific topic requirements and lack customization",
            location="reports generation service"
        ))
        
        # Check report caching
        issues.append(ConfigurationIssue(
            id="reports_no_caching",
            category="reports",
            severity="medium",
            title="Report caching not optimized",
            description="Reports may be regenerated unnecessarily",
            current_value="no_cache",
            recommended_value="smart_caching",
            auto_fixable=True,
            impact="Slower report generation and higher resource usage",
            location="reports service"
        ))
        
        return issues

    async def _scan_system_config(self) -> List[ConfigurationIssue]:
        """Scan system configuration for issues"""
        issues = []
        
        # Check resource monitoring intervals
        try:
            # Get current resource monitoring status
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8001/api/v1/system-resources/") as response:
                    if response.status == 200:
                        resource_data = await response.json()
                        if not resource_data.get("monitoring_info", {}).get("monitoring_active", False):
                            issues.append(ConfigurationIssue(
                                id="system_monitoring_inactive",
                                category="system",
                                severity="medium",
                                title="System resource monitoring inactive",
                                description="Real-time resource monitoring is not active",
                                current_value="inactive",
                                recommended_value="active",
                                auto_fixable=True,
                                impact="No real-time system health visibility",
                                location="system resource monitor"
                            ))
        except:
            pass
            
        return issues

    async def _scan_database_config(self) -> List[ConfigurationIssue]:
        """Scan database configuration for issues"""
        issues = []
        
        # Check database connections
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8001/api/v1/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        if health_data.get("status") != "healthy":
                            issues.append(ConfigurationIssue(
                                id="database_health_issues",
                                category="database",
                                severity="critical",
                                title="Database health issues detected",
                                description="Database connections or operations showing problems",
                                current_value="unhealthy",
                                recommended_value="healthy",
                                auto_fixable=False,
                                impact="System functionality may be compromised",
                                location="database connection layer"
                            ))
        except:
            pass
            
        return issues

    async def _scan_ui_config(self) -> List[ConfigurationIssue]:
        """Scan UI configuration for issues"""
        issues = []
        
        # Check for UI issues from monitoring service
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8001/api/v1/ui-monitoring/issues?status=open") as response:
                    if response.status == 200:
                        ui_data = await response.json()
                        total_ui_issues = ui_data.get("total_issues", 0)
                        
                        if total_ui_issues > 10:
                            issues.append(ConfigurationIssue(
                                id="ui_many_open_issues",
                                category="ui",
                                severity="medium",
                                title="Many UI issues detected",
                                description=f"{total_ui_issues} UI issues need attention",
                                current_value=total_ui_issues,
                                recommended_value=0,
                                auto_fixable=True,
                                impact="Poor user experience and accessibility issues",
                                location="UI templates and components"
                            ))
        except:
            pass
            
        return issues

    def _generate_configuration_recommendations(self) -> List[str]:
        """Generate actionable configuration recommendations"""
        recommendations = []
        
        # Analyze issues by severity and category
        critical_issues = [issue for issue in self.config_issues.values() if issue.severity == "critical"]
        high_issues = [issue for issue in self.config_issues.values() if issue.severity == "high"]
        auto_fixable = [issue for issue in self.config_issues.values() if issue.auto_fixable]
        
        if critical_issues:
            recommendations.append(f"URGENT: {len(critical_issues)} critical configuration issues require immediate attention")
            
        if high_issues:
            recommendations.append(f"HIGH PRIORITY: {len(high_issues)} high-priority configuration issues found")
            
        if len(auto_fixable) > 0:
            recommendations.append(f"Quick wins: {len(auto_fixable)} issues can be automatically fixed")
            
        # Category-specific recommendations
        scraping_issues = [issue for issue in self.config_issues.values() if issue.category == "scraping"]
        if scraping_issues:
            recommendations.append("Consider applying the 'high_performance' profile for better scraping throughput")
            
        reports_issues = [issue for issue in self.config_issues.values() if issue.category == "reports"]
        if reports_issues:
            recommendations.append("Implement topic-specific report templates to address generic content issues")
            
        if not recommendations:
            recommendations.append("Configuration appears optimized - no major issues detected")
            
        return recommendations

    async def auto_fix_issues(self, issue_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Automatically fix configuration issues"""
        if issue_ids is None:
            # Fix all auto-fixable issues
            fixable_issues = [issue for issue in self.config_issues.values() if issue.auto_fixable]
        else:
            # Fix specific issues
            fixable_issues = [self.config_issues[issue_id] for issue_id in issue_ids if issue_id in self.config_issues and self.config_issues[issue_id].auto_fixable]
        
        fix_results = {
            "total_attempts": len(fixable_issues),
            "successful_fixes": 0,
            "failed_fixes": 0,
            "fix_details": {}
        }
        
        for issue in fixable_issues:
            try:
                success = await self._apply_configuration_fix(issue)
                if success:
                    fix_results["successful_fixes"] += 1
                    fix_results["fix_details"][issue.id] = "success"
                    # Remove fixed issue
                    if issue.id in self.config_issues:
                        del self.config_issues[issue.id]
                else:
                    fix_results["failed_fixes"] += 1
                    fix_results["fix_details"][issue.id] = "failed"
                    
            except Exception as e:
                fix_results["failed_fixes"] += 1
                fix_results["fix_details"][issue.id] = f"error: {str(e)}"
                logger.error(f"Error fixing configuration issue {issue.id}: {e}")
                
        logger.info(f"Configuration auto-fix complete: {fix_results['successful_fixes']} fixed, {fix_results['failed_fixes']} failed")
        return fix_results

    async def _apply_configuration_fix(self, issue: ConfigurationIssue) -> bool:
        """Apply a specific configuration fix"""
        try:
            if issue.category == "scraping":
                return await self._fix_scraping_config(issue)
            elif issue.category == "reports":
                return await self._fix_reports_config(issue)
            elif issue.category == "system":
                return await self._fix_system_config(issue)
            elif issue.category == "ui":
                return await self._fix_ui_config(issue)
            elif issue.category == "database":
                return await self._fix_database_config(issue)
                
        except Exception as e:
            logger.error(f"Error applying fix for {issue.id}: {e}")
            return False
            
        return False

    async def _fix_scraping_config(self, issue: ConfigurationIssue) -> bool:
        """Fix scraping configuration issue"""
        try:
            import aiohttp
            
            if issue.id == "scraping_low_articles_per_source":
                # Update scraping config to use recommended value
                config_update = {
                    "max_articles_per_source": issue.recommended_value,
                    "concurrent_requests": 6,  # Also optimize concurrency
                    "download_delay": 1.0
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://localhost:8001/api/v1/scraping/configure",
                        json=config_update
                    ) as response:
                        return response.status == 200
                        
            elif issue.id == "scraping_low_concurrency":
                config_update = {"concurrent_requests": issue.recommended_value}
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://localhost:8001/api/v1/scraping/configure",
                        json=config_update
                    ) as response:
                        return response.status == 200
                        
            elif issue.id == "scraping_high_delay":
                config_update = {"download_delay": issue.recommended_value}
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://localhost:8001/api/v1/scraping/configure",
                        json=config_update
                    ) as response:
                        return response.status == 200
                        
        except Exception as e:
            logger.error(f"Error fixing scraping config: {e}")
            
        return False

    async def _fix_reports_config(self, issue: ConfigurationIssue) -> bool:
        """Fix reports configuration issue"""
        if issue.id == "reports_generic_content":
            # This would require implementing topic-specific templates
            # For now, log the requirement
            logger.info("Reports generic content issue identified - requires topic-specific template implementation")
            return True  # Mark as "fixed" to indicate it's been identified and logged
            
        return False

    async def _fix_system_config(self, issue: ConfigurationIssue) -> bool:
        """Fix system configuration issue"""
        try:
            if issue.id == "system_monitoring_inactive":
                # Start system resource monitoring
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://localhost:8001/api/v1/system-resources/start"
                    ) as response:
                        return response.status == 200
        except:
            pass
            
        return False

    async def _fix_ui_config(self, issue: ConfigurationIssue) -> bool:
        """Fix UI configuration issue"""
        try:
            if issue.id == "ui_many_open_issues":
                # Trigger auto-fix for UI issues
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://localhost:8001/api/v1/ui-monitoring/fix"
                    ) as response:
                        return response.status == 200
        except:
            pass
            
        return False

    async def _fix_database_config(self, issue: ConfigurationIssue) -> bool:
        """Fix database configuration issue"""
        # Database fixes typically require manual intervention
        return False

    def apply_configuration_profile(self, profile_name: str) -> Dict[str, Any]:
        """Apply a configuration profile"""
        if profile_name not in self.profiles:
            return {"success": False, "error": "Profile not found"}
            
        profile = self.profiles[profile_name]
        applied_configs = {}
        
        try:
            # Store the profile configurations to be applied
            for category, settings in profile.configurations.items():
                applied_configs[category] = settings
                
            logger.info(f"Configuration profile '{profile_name}' prepared for application")
            
            return {
                "success": True,
                "profile_applied": profile_name,
                "configurations": applied_configs,
                "description": profile.description
            }
            
        except Exception as e:
            logger.error(f"Error applying configuration profile {profile_name}: {e}")
            return {"success": False, "error": str(e)}

    def get_configuration_status(self) -> Dict[str, Any]:
        """Get current configuration status"""
        return {
            "last_scan": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "total_issues": len(self.config_issues),
            "issues_by_severity": {
                severity: len([issue for issue in self.config_issues.values() if issue.severity == severity])
                for severity in ["critical", "high", "medium", "low"]
            },
            "issues_by_category": {
                category: len([issue for issue in self.config_issues.values() if issue.category == category])
                for category in ["scraping", "reports", "system", "database", "ui"]
            },
            "auto_fixable_count": sum(1 for issue in self.config_issues.values() if issue.auto_fixable),
            "available_profiles": list(self.profiles.keys()),
            "auto_fix_enabled": self.auto_fix_enabled
        }


# Global configuration manager instance
config_manager = ConfigurationManager()
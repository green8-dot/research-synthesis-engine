"""
Scraping Monitoring and Auto-Fix Service
"""
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
from dataclasses import dataclass
import re

try:
    from research_synthesis.database.connection import get_postgres_session
    from research_synthesis.database.models import ScrapingJob, Article
    from sqlalchemy import select, desc, and_, func
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

try:
    from research_synthesis.api.scraping_router import sources_data
    SOURCES_AVAILABLE = True
except ImportError:
    SOURCES_AVAILABLE = False

@dataclass
class ScrapingIssue:
    """Represents a scraping issue found during monitoring"""
    source_name: str
    issue_type: str
    description: str
    severity: str  # "critical", "high", "medium", "low"
    fix_attempted: bool = False
    fix_successful: bool = False
    error_details: Optional[str] = None
    detected_at: datetime = None

class ScrapingMonitor:
    """Monitor and fix scraping issues automatically"""
    
    def __init__(self):
        self.session = None
        self.issues: List[ScrapingIssue] = []
        self.reddit_json_endpoints = {
            "Reddit r/space": "https://old.reddit.com/r/space/.json",
            "Reddit r/SpaceX": "https://old.reddit.com/r/SpaceX/.json",
            "Reddit r/BlueOrigin": "https://old.reddit.com/r/BlueOrigin/.json"
        }
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Research Synthesis Bot 1.0 (Space Industry Analysis)'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def check_all_sources(self) -> List[ScrapingIssue]:
        """Check all configured sources for issues"""
        logger.info("Starting comprehensive scraping source check...")
        
        if not SOURCES_AVAILABLE:
            logger.warning("Sources data not available")
            return []
            
        issues = []
        
        # Check each source
        for source_name, source_config in sources_data.items():
            source_issues = await self._check_source(source_name, source_config)
            issues.extend(source_issues)
            
        self.issues = issues
        
        # Log summary
        critical_issues = [i for i in issues if i.severity == "critical"]
        high_issues = [i for i in issues if i.severity == "high"]
        
        logger.info(f"Found {len(issues)} total issues: {len(critical_issues)} critical, {len(high_issues)} high priority")
        
        return issues

    async def _check_source(self, source_name: str, source_config: dict) -> List[ScrapingIssue]:
        """Check a single source for issues"""
        issues = []
        
        # Check if source is accessible
        url = source_config.get("url", "")
        if not url:
            issues.append(ScrapingIssue(
                source_name=source_name,
                issue_type="missing_url",
                description="Source has no URL configured",
                severity="critical",
                detected_at=datetime.now()
            ))
            return issues
            
        # Check URL accessibility
        accessibility_issue = await self._check_url_accessibility(source_name, url)
        if accessibility_issue:
            issues.append(accessibility_issue)
            
        # Check Reddit-specific issues
        if "reddit.com" in url.lower():
            reddit_issues = await self._check_reddit_source(source_name, url)
            issues.extend(reddit_issues)
            
        # Check recent scraping performance
        performance_issues = await self._check_scraping_performance(source_name)
        issues.extend(performance_issues)
        
        return issues

    async def _check_url_accessibility(self, source_name: str, url: str) -> Optional[ScrapingIssue]:
        """Check if URL is accessible"""
        try:
            async with self.session.head(url) as response:
                if response.status >= 400:
                    return ScrapingIssue(
                        source_name=source_name,
                        issue_type="url_inaccessible",
                        description=f"URL returns HTTP {response.status}",
                        severity="high" if response.status >= 500 else "medium",
                        error_details=f"HTTP {response.status}: {response.reason}",
                        detected_at=datetime.now()
                    )
        except Exception as e:
            return ScrapingIssue(
                source_name=source_name,
                issue_type="url_connection_error",
                description=f"Cannot connect to URL: {str(e)}",
                severity="critical",
                error_details=str(e),
                detected_at=datetime.now()
            )
        return None

    async def _check_reddit_source(self, source_name: str, url: str) -> List[ScrapingIssue]:
        """Check Reddit-specific issues"""
        issues = []
        
        # Check if using HTML scraping instead of JSON API
        if "reddit.com" in url and not url.endswith(".json"):
            issues.append(ScrapingIssue(
                source_name=source_name,
                issue_type="reddit_wrong_format",
                description="Reddit source using HTML scraping instead of JSON API",
                severity="high",
                error_details=f"Current URL: {url}, Should use: {url.rstrip('/')}.json",
                detected_at=datetime.now()
            ))
            
        # Test Reddit JSON API accessibility
        if "reddit.com" in url:
            json_url = url.rstrip('/') + '.json' if not url.endswith('.json') else url
            try:
                async with self.session.get(json_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if not data.get('data', {}).get('children'):
                            issues.append(ScrapingIssue(
                                source_name=source_name,
                                issue_type="reddit_no_content",
                                description="Reddit API returns no posts",
                                severity="medium",
                                detected_at=datetime.now()
                            ))
                    else:
                        issues.append(ScrapingIssue(
                            source_name=source_name,
                            issue_type="reddit_api_error",
                            description=f"Reddit API returns HTTP {response.status}",
                            severity="high",
                            detected_at=datetime.now()
                        ))
            except Exception as e:
                issues.append(ScrapingIssue(
                    source_name=source_name,
                    issue_type="reddit_api_connection_error",
                    description=f"Cannot connect to Reddit API: {str(e)}",
                    severity="high",
                    error_details=str(e),
                    detected_at=datetime.now()
                ))
                
        return issues

    async def _check_scraping_performance(self, source_name: str) -> List[ScrapingIssue]:
        """Check recent scraping performance for a source"""
        issues = []
        
        if not DATABASE_AVAILABLE:
            return issues
            
        try:
            async for db_session in get_postgres_session():
                # Get recent jobs for this source
                recent_jobs = await db_session.execute(
                    select(ScrapingJob)
                    .filter(
                        and_(
                            ScrapingJob.source_id == source_name,
                            ScrapingJob.created_at >= datetime.now() - timedelta(hours=24)
                        )
                    )
                    .order_by(desc(ScrapingJob.created_at))
                    .limit(10)
                )
                jobs = recent_jobs.scalars().all()
                
                if not jobs:
                    issues.append(ScrapingIssue(
                        source_name=source_name,
                        issue_type="no_recent_scraping",
                        description="No scraping jobs in the last 24 hours",
                        severity="medium",
                        detected_at=datetime.now()
                    ))
                    break
                    
                # Check for high error rates
                failed_jobs = [j for j in jobs if j.status == "failed"]
                if len(failed_jobs) / len(jobs) > 0.5:
                    issues.append(ScrapingIssue(
                        source_name=source_name,
                        issue_type="high_failure_rate",
                        description=f"High failure rate: {len(failed_jobs)}/{len(jobs)} jobs failed",
                        severity="high",
                        detected_at=datetime.now()
                    ))
                    
                # Check for jobs returning no articles
                zero_article_jobs = [j for j in jobs if j.articles_found == 0]
                if len(zero_article_jobs) > 2:
                    issues.append(ScrapingIssue(
                        source_name=source_name,
                        issue_type="no_articles_found",
                        description=f"{len(zero_article_jobs)} recent jobs found no articles",
                        severity="medium",
                        detected_at=datetime.now()
                    ))
                
                break
                
        except Exception as e:
            logger.error(f"Error checking scraping performance for {source_name}: {e}")
            
        return issues

    async def auto_fix_issues(self) -> int:
        """Automatically fix detected issues where possible"""
        fixed_count = 0
        
        for issue in self.issues:
            if issue.fix_attempted:
                continue
                
            logger.info(f"Attempting to fix issue: {issue.description}")
            issue.fix_attempted = True
            
            try:
                if issue.issue_type == "reddit_wrong_format":
                    success = await self._fix_reddit_format(issue)
                elif issue.issue_type == "url_inaccessible":
                    success = await self._fix_url_accessibility(issue)
                else:
                    logger.info(f"No auto-fix available for issue type: {issue.issue_type}")
                    continue
                    
                if success:
                    issue.fix_successful = True
                    fixed_count += 1
                    logger.info(f"Successfully fixed issue for {issue.source_name}")
                else:
                    logger.warning(f"Failed to fix issue for {issue.source_name}")
                    
            except Exception as e:
                logger.error(f"Error fixing issue for {issue.source_name}: {e}")
                
        return fixed_count

    async def _fix_reddit_format(self, issue: ScrapingIssue) -> bool:
        """Fix Reddit sources to use JSON API"""
        if not SOURCES_AVAILABLE:
            return False
            
        try:
            source_config = sources_data.get(issue.source_name)
            if not source_config:
                return False
                
            current_url = source_config.get("url", "")
            if not current_url or ".json" in current_url:
                return False
                
            # Update URL to use JSON endpoint
            new_url = current_url.rstrip('/') + '.json'
            sources_data[issue.source_name]["url"] = new_url
            
            # Update scraping strategy
            sources_data[issue.source_name]["scraping_strategy"] = "reddit_json"
            
            logger.info(f"Updated {issue.source_name} URL from {current_url} to {new_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to fix Reddit format for {issue.source_name}: {e}")
            return False

    async def _fix_url_accessibility(self, issue: ScrapingIssue) -> bool:
        """Attempt to fix URL accessibility issues"""
        # For now, just log the issue - manual intervention usually required
        logger.info(f"URL accessibility issue for {issue.source_name} requires manual review")
        return False

    def generate_health_report(self) -> dict:
        """Generate a comprehensive health report"""
        if not self.issues:
            return {
                "status": "healthy",
                "total_sources": len(sources_data) if SOURCES_AVAILABLE else 0,
                "issues_found": 0,
                "critical_issues": 0,
                "auto_fixes_applied": 0,
                "report_generated_at": datetime.now().isoformat()
            }
            
        critical_issues = [i for i in self.issues if i.severity == "critical"]
        high_issues = [i for i in self.issues if i.severity == "high"]
        fixed_issues = [i for i in self.issues if i.fix_successful]
        
        # Determine overall status
        if critical_issues:
            status = "critical"
        elif high_issues:
            status = "degraded"
        elif self.issues:
            status = "minor_issues"
        else:
            status = "healthy"
            
        return {
            "status": status,
            "total_sources": len(sources_data) if SOURCES_AVAILABLE else 0,
            "issues_found": len(self.issues),
            "critical_issues": len(critical_issues),
            "high_issues": len(high_issues),
            "auto_fixes_applied": len(fixed_issues),
            "issues_by_type": self._group_issues_by_type(),
            "issues_by_source": self._group_issues_by_source(),
            "recommendations": self._generate_recommendations(),
            "report_generated_at": datetime.now().isoformat()
        }

    def _group_issues_by_type(self) -> dict:
        """Group issues by type"""
        types = {}
        for issue in self.issues:
            if issue.issue_type not in types:
                types[issue.issue_type] = []
            types[issue.issue_type].append({
                "source": issue.source_name,
                "severity": issue.severity,
                "fixed": issue.fix_successful
            })
        return types

    def _group_issues_by_source(self) -> dict:
        """Group issues by source"""
        sources = {}
        for issue in self.issues:
            if issue.source_name not in sources:
                sources[issue.source_name] = []
            sources[issue.source_name].append({
                "type": issue.issue_type,
                "severity": issue.severity,
                "description": issue.description,
                "fixed": issue.fix_successful
            })
        return sources

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on found issues"""
        recommendations = []
        
        critical_issues = [i for i in self.issues if i.severity == "critical"]
        reddit_issues = [i for i in self.issues if "reddit" in i.issue_type]
        performance_issues = [i for i in self.issues if "performance" in i.issue_type or "failure" in i.issue_type]
        
        if critical_issues:
            recommendations.append(f"Immediate attention required for {len(critical_issues)} critical issues")
            
        if reddit_issues:
            recommendations.append("Update Reddit sources to use JSON API endpoints instead of HTML scraping")
            
        if performance_issues:
            recommendations.append("Review scraping frequency and error handling for failing sources")
            
        # Check for patterns
        failing_sources = set(i.source_name for i in self.issues if i.severity in ["critical", "high"])
        if len(failing_sources) > 3:
            recommendations.append("Consider implementing circuit breaker pattern for frequently failing sources")
            
        return recommendations

# Global monitor instance
scraping_monitor = ScrapingMonitor()

async def run_scraping_health_check() -> dict:
    """Run a complete scraping health check"""
    async with scraping_monitor as monitor:
        await monitor.check_all_sources()
        fixes_applied = await monitor.auto_fix_issues()
        
        report = monitor.generate_health_report()
        report["auto_fixes_applied"] = fixes_applied
        
        return report

async def fix_reddit_sources() -> dict:
    """Specifically fix Reddit source issues"""
    if not SOURCES_AVAILABLE:
        return {"error": "Sources not available"}
        
    fixes_applied = []
    
    for source_name, source_config in sources_data.items():
        if "reddit.com" in source_config.get("url", "").lower():
            current_url = source_config["url"]
            
            # Fix URL format
            if not current_url.endswith(".json"):
                new_url = current_url.rstrip('/') + '.json'
                sources_data[source_name]["url"] = new_url
                sources_data[source_name]["scraping_strategy"] = "reddit_json"
                
                fixes_applied.append({
                    "source": source_name,
                    "old_url": current_url,
                    "new_url": new_url,
                    "fix": "Updated to JSON API endpoint"
                })
                
                logger.info(f"Fixed Reddit source {source_name}: {current_url} -> {new_url}")
    
    return {
        "fixes_applied": len(fixes_applied),
        "details": fixes_applied,
        "message": f"Applied {len(fixes_applied)} Reddit source fixes"
    }
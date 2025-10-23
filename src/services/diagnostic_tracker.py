"""
Diagnostic Tracker Service

Continuously tracks common issues and provides trending analysis for diagnostics.
Integrates with UI health monitor to provide comprehensive issue tracking.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from pathlib import Path
import hashlib

from research_synthesis.utils.system_status import system_status_service

# Import UI monitor with error handling
try:
    from research_synthesis.services.ui_monitor import UIMonitor
    ui_monitor = UIMonitor()
    UI_MONITOR_AVAILABLE = True
except ImportError:
    ui_monitor = None
    UI_MONITOR_AVAILABLE = False


class DiagnosticTracker:
    """Tracks diagnostic issues over time and provides trend analysis"""
    
    def __init__(self):
        self.issue_history = defaultdict(deque)  # Rolling window of issues
        self.issue_counts = defaultdict(int)
        self.last_scan_time = None
        self.trend_window_hours = 24
        self.max_history_items = 1000
        self.tracking_active = False
        self.scan_interval_minutes = 15
        
        # Common issue patterns with their solutions
        self.issue_catalog = {
            'database_disconnected': {
                'title': 'Database Connection Failed',
                'description': 'Database connection is unavailable or unstable',
                'category': 'database',
                'severity': 'high',
                'solutions': [
                    'Check database service is running',
                    'Verify connection credentials and settings',
                    'Check network connectivity to database server',
                    'Review database error logs',
                    'Test database connection manually'
                ],
                'auto_fixable': False
            },
            'api_endpoint_error': {
                'title': 'API Endpoint Error',
                'description': 'API endpoints returning errors or not responding',
                'category': 'api',
                'severity': 'high',
                'solutions': [
                    'Restart the API server',
                    'Check server error logs',
                    'Verify endpoint routing configuration',
                    'Test individual endpoints manually',
                    'Check for dependency service issues'
                ],
                'auto_fixable': False
            },
            'ui_color_contrast': {
                'title': 'UI Color Contrast Issues',
                'description': 'Text readability problems due to poor color contrast',
                'category': 'ui',
                'severity': 'medium',
                'solutions': [
                    'Run UI health monitor auto-fix',
                    'Update CSS color schemes manually',
                    'Check for dark/light theme conflicts',
                    'Verify Bootstrap class usage'
                ],
                'auto_fixable': True
            },
            'report_generation_failure': {
                'title': 'Report Generation Failed',
                'description': 'Report generation process encounters errors',
                'category': 'reports',
                'severity': 'high',
                'solutions': [
                    'Check report templates exist and are valid',
                    'Verify data source availability',
                    'Clear report cache and temp files',
                    'Check file system permissions',
                    'Review report generation logs'
                ],
                'auto_fixable': False
            },
            'system_resource_high': {
                'title': 'High System Resource Usage',
                'description': 'CPU, memory, or disk usage is abnormally high',
                'category': 'system',
                'severity': 'medium',
                'solutions': [
                    'Identify resource-intensive processes',
                    'Clear temporary files and caches',
                    'Restart services consuming high resources',
                    'Review system monitoring logs',
                    'Consider scaling resources if needed'
                ],
                'auto_fixable': False
            },
            'scraping_failures': {
                'title': 'Web Scraping Failures',
                'description': 'Web scraping jobs failing or timing out',
                'category': 'scraping',
                'severity': 'medium',
                'solutions': [
                    'Check target website availability',
                    'Review scraping rate limits',
                    'Update scraping selectors/patterns',
                    'Clear scraping cache',
                    'Verify proxy/VPN settings'
                ],
                'auto_fixable': False
            }
        }
    
    async def start_continuous_tracking(self):
        """Start continuous diagnostic tracking in background"""
        if self.tracking_active:
            return
            
        self.tracking_active = True
        asyncio.create_task(self._tracking_loop())
    
    async def stop_tracking(self):
        """Stop continuous tracking"""
        self.tracking_active = False
    
    async def _tracking_loop(self):
        """Main tracking loop that runs continuously"""
        while self.tracking_active:
            try:
                await self._perform_diagnostic_scan()
                await asyncio.sleep(self.scan_interval_minutes * 60)  # Convert to seconds
            except Exception as e:
                print(f"Diagnostic tracking error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _perform_diagnostic_scan(self):
        """Perform a single diagnostic scan and record results"""
        scan_time = datetime.now()
        
        # Get system status
        system_status = await system_status_service.get_system_status()
        
        # Get UI health status
        ui_health = {}
        if UI_MONITOR_AVAILABLE and ui_monitor:
            try:
                ui_health = await ui_monitor.get_health_status()
            except Exception as e:
                ui_health = {}
        
        # Analyze issues from both sources
        issues_found = []
        
        # Check system components
        for component_name, component_info in system_status.get('components', {}).items():
            status = component_info.get('status', 'unknown')
            if status in ['disconnected', 'error', 'failed']:
                issue_type = self._classify_system_issue(component_name, status)
                if issue_type:
                    issues_found.append({
                        'type': issue_type,
                        'component': component_name,
                        'status': status,
                        'timestamp': scan_time,
                        'source': 'system_status'
                    })
        
        # Check UI health issues
        ui_issues = ui_health.get('issues', [])
        for ui_issue in ui_issues:
            issue_type = self._classify_ui_issue(ui_issue)
            if issue_type:
                issues_found.append({
                    'type': issue_type,
                    'component': ui_issue.get('page', 'unknown'),
                    'details': ui_issue.get('description', ''),
                    'timestamp': scan_time,
                    'source': 'ui_monitor'
                })
        
        # Record issues in history
        for issue in issues_found:
            issue_id = self._generate_issue_id(issue)
            self.issue_history[issue_id].append(issue)
            self.issue_counts[issue['type']] += 1
            
            # Maintain rolling window
            while len(self.issue_history[issue_id]) > self.max_history_items:
                self.issue_history[issue_id].popleft()
        
        self.last_scan_time = scan_time
        
        # Clean old entries
        await self._cleanup_old_entries()
    
    def _classify_system_issue(self, component_name: str, status: str) -> Optional[str]:
        """Classify system component issues"""
        if component_name.lower() in ['sqlite', 'mongodb', 'redis', 'elasticsearch', 'chromadb']:
            return 'database_disconnected'
        elif 'api' in component_name.lower() or 'endpoint' in component_name.lower():
            return 'api_endpoint_error'
        elif component_name.lower() == 'report_generator' and status == 'error':
            return 'report_generation_failure'
        return None
    
    def _classify_ui_issue(self, ui_issue: Dict) -> Optional[str]:
        """Classify UI issues"""
        issue_type = ui_issue.get('type', '')
        if 'color' in issue_type or 'contrast' in issue_type:
            return 'ui_color_contrast'
        return None
    
    def _generate_issue_id(self, issue: Dict) -> str:
        """Generate unique ID for issue tracking"""
        key_parts = [
            issue['type'],
            issue['component'],
            issue.get('details', '')[:50]  # First 50 chars of details
        ]
        key_string = '|'.join(str(part) for part in key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()[:12]
    
    async def _cleanup_old_entries(self):
        """Remove entries older than the tracking window"""
        cutoff_time = datetime.now() - timedelta(hours=self.trend_window_hours)
        
        for issue_id in list(self.issue_history.keys()):
            # Filter out old entries
            self.issue_history[issue_id] = deque([
                entry for entry in self.issue_history[issue_id]
                if entry['timestamp'] > cutoff_time
            ])
            
            # Remove empty queues
            if not self.issue_history[issue_id]:
                del self.issue_history[issue_id]
    
    async def get_trending_issues(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequent issues in the tracking window"""
        issue_frequency = defaultdict(int)
        recent_examples = {}
        
        cutoff_time = datetime.now() - timedelta(hours=self.trend_window_hours)
        
        # Count recent occurrences
        for issue_id, entries in self.issue_history.items():
            recent_entries = [e for e in entries if e['timestamp'] > cutoff_time]
            if recent_entries:
                issue_type = recent_entries[0]['type']
                issue_frequency[issue_type] += len(recent_entries)
                recent_examples[issue_type] = recent_entries[-1]  # Most recent example
        
        # Sort by frequency and prepare results
        trending = []
        for issue_type, frequency in sorted(issue_frequency.items(), 
                                          key=lambda x: x[1], reverse=True)[:limit]:
            
            issue_info = self.issue_catalog.get(issue_type, {
                'title': f'Unknown Issue: {issue_type}',
                'description': 'Issue not in catalog',
                'category': 'unknown',
                'severity': 'medium',
                'solutions': ['Contact system administrator'],
                'auto_fixable': False
            })
            
            trending.append({
                'type': issue_type,
                'frequency': frequency,
                'last_seen': recent_examples[issue_type]['timestamp'].isoformat(),
                'title': issue_info['title'],
                'description': issue_info['description'],
                'category': issue_info['category'],
                'severity': issue_info['severity'],
                'solutions': issue_info['solutions'],
                'auto_fixable': issue_info['auto_fixable'],
                'component': recent_examples[issue_type]['component']
            })
        
        return trending
    
    async def get_issue_details(self, issue_type: str) -> Dict[str, Any]:
        """Get detailed information about a specific issue type"""
        issue_info = self.issue_catalog.get(issue_type, {
            'title': f'Unknown Issue: {issue_type}',
            'description': 'Issue not in catalog',
            'category': 'unknown',
            'severity': 'medium',
            'solutions': ['Contact system administrator'],
            'auto_fixable': False
        })
        
        # Get recent occurrences
        recent_occurrences = []
        cutoff_time = datetime.now() - timedelta(hours=self.trend_window_hours)
        
        for issue_id, entries in self.issue_history.items():
            for entry in entries:
                if entry['type'] == issue_type and entry['timestamp'] > cutoff_time:
                    recent_occurrences.append({
                        'timestamp': entry['timestamp'].isoformat(),
                        'component': entry['component'],
                        'source': entry['source'],
                        'details': entry.get('details', '')
                    })
        
        # Sort by timestamp (most recent first)
        recent_occurrences.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return {
            **issue_info,
            'type': issue_type,
            'recent_occurrences': recent_occurrences[:20],  # Last 20 occurrences
            'total_occurrences_24h': len(recent_occurrences)
        }
    
    async def get_diagnostic_summary(self) -> Dict[str, Any]:
        """Get summary of diagnostic tracking status"""
        trending_issues = await self.get_trending_issues(5)
        
        # Calculate overall stats
        total_issues_24h = sum(issue['frequency'] for issue in trending_issues)
        high_severity_count = len([i for i in trending_issues if i['severity'] == 'high'])
        auto_fixable_count = len([i for i in trending_issues if i['auto_fixable']])
        
        return {
            'tracking_active': self.tracking_active,
            'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'scan_interval_minutes': self.scan_interval_minutes,
            'trend_window_hours': self.trend_window_hours,
            'total_issues_24h': total_issues_24h,
            'unique_issue_types': len(trending_issues),
            'high_severity_issues': high_severity_count,
            'auto_fixable_issues': auto_fixable_count,
            'top_issues': trending_issues
        }
    
    async def apply_auto_fixes(self) -> Dict[str, Any]:
        """Apply automatic fixes for issues that support it"""
        fixes_applied = []
        fixes_failed = []
        
        trending_issues = await self.get_trending_issues()
        
        for issue in trending_issues:
            if issue['auto_fixable']:
                try:
                    if issue['type'] == 'ui_color_contrast' and UI_MONITOR_AVAILABLE and ui_monitor:
                        # Apply UI fixes through UI monitor
                        result = await ui_monitor.apply_auto_fixes()
                        if result.get('fixes_applied', 0) > 0:
                            fixes_applied.append(f"Fixed {result['fixes_applied']} UI color contrast issues")
                        else:
                            fixes_failed.append(f"UI color contrast: {result.get('error', 'No fixes needed')}")
                    elif issue['type'] == 'ui_color_contrast':
                        fixes_failed.append(f"UI color contrast: UI Monitor not available")
                    
                    # Add more auto-fix implementations here for other issue types
                    
                except Exception as e:
                    fixes_failed.append(f"{issue['title']}: {str(e)}")
        
        return {
            'fixes_applied': fixes_applied,
            'fixes_failed': fixes_failed,
            'total_attempted': len([i for i in trending_issues if i['auto_fixable']])
        }


# Global diagnostic tracker instance
diagnostic_tracker = DiagnosticTracker()
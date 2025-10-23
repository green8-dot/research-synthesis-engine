"""
Backend Error Monitor with Root Cause Analysis
Comprehensive error tracking, pattern recognition, and ML-powered root cause analysis
"""

import asyncio
import traceback
import sys
import os
import json
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, Counter, deque
import threading
import psutil
import socket
import platform
from pathlib import Path

from loguru import logger
import structlog
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from prometheus_client import Counter as PrometheusCounter, Histogram, Gauge, generate_latest
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

from research_synthesis.database.connection import get_mongodb, init_mongodb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


class BackendErrorMonitor:
    """Advanced backend error monitoring with ML-powered root cause analysis"""
    
    def __init__(self, app_name: str = "research_synthesis"):
        self.app_name = app_name
        self.errors = deque(maxlen=10000)  # Keep last 10000 errors
        self.error_patterns = defaultdict(list)
        self.error_signatures = {}
        self.root_cause_cache = {}
        self.performance_metrics = defaultdict(list)
        self.system_metrics = deque(maxlen=1000)
        self.dependency_graph = defaultdict(set)
        self.error_clusters = []
        self.ml_models = {}
        
        # Metrics
        try:
            self.error_counter = PrometheusCounter('backend_errors_total', 'Total backend errors', ['type', 'severity'])
            self.response_time_histogram = Histogram('response_time_seconds', 'Response time distribution')
            self.active_requests_gauge = Gauge('active_requests', 'Number of active requests')
            self.memory_usage_gauge = Gauge('memory_usage_bytes', 'Current memory usage')
            self.cpu_usage_gauge = Gauge('cpu_usage_percent', 'Current CPU usage')
        except Exception as e:
            logger.warning(f"Failed to initialize Prometheus metrics: {e}")
            # Create dummy metrics if Prometheus fails
            self.error_counter = None
            self.response_time_histogram = None
            self.active_requests_gauge = None
            self.memory_usage_gauge = None
            self.cpu_usage_gauge = None
        
        # Initialize monitoring components
        self._init_structured_logging()
        self._init_tracing()
        self._init_sentry()
        self._start_system_monitoring()
        self._init_ml_components()
    
    def _init_structured_logging(self):
        """Initialize structured logging with context"""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                self._error_processor,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        self.structured_logger = structlog.get_logger()
    
    def _error_processor(self, logger, method_name, event_dict):
        """Custom processor to capture and analyze errors"""
        if 'error' in event_dict or 'exception' in event_dict or method_name in ['error', 'critical']:
            self.capture_error(event_dict)
        return event_dict
    
    def _init_tracing(self):
        """Initialize OpenTelemetry tracing"""
        resource = Resource.create({
            "service.name": self.app_name,
            "service.version": "1.0.0",
            "deployment.environment": os.getenv("ENVIRONMENT", "development")
        })
        
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
        
        self.tracer = trace.get_tracer(__name__)
        
        # Auto-instrument libraries - temporarily disabled due to version compatibility issues
        try:
            # TODO: Fix OpenTelemetry instrumentation when library versions are compatible
            # FastAPIInstrumentor().instrument()
            # SQLAlchemyInstrumentor().instrument() 
            # RedisInstrumentor().instrument()
            logger.info("OpenTelemetry tracing initialized (auto-instrumentation disabled)")
        except Exception as e:
            logger.warning(f"OpenTelemetry setup failed: {e}")
            self.tracer = None
    
    def _init_sentry(self):
        """Initialize Sentry error tracking"""
        sentry_dsn = os.getenv("SENTRY_DSN")
        
        if sentry_dsn:
            sentry_sdk.init(
                dsn=sentry_dsn,
                integrations=[
                    FastApiIntegration(transaction_style="endpoint"),
                    SqlalchemyIntegration(),
                ],
                traces_sample_rate=0.1,
                environment=os.getenv("ENVIRONMENT", "development"),
                before_send=self._before_send_sentry
            )
            logger.info("Sentry error tracking initialized")
        else:
            logger.info("Sentry DSN not configured - using local error tracking only")
    
    def _before_send_sentry(self, event, hint):
        """Process errors before sending to Sentry"""
        # Enrich with additional context
        if 'exc_info' in hint:
            exc_type, exc_value, tb = hint['exc_info']
            
            # Perform root cause analysis
            root_cause = self.analyze_root_cause(exc_type, exc_value, tb)
            event['extra']['root_cause_analysis'] = root_cause
            
            # Add ML insights
            ml_insights = self.get_ml_insights(str(exc_value))
            event['extra']['ml_insights'] = ml_insights
        
        return event
    
    def _start_system_monitoring(self):
        """Start continuous system monitoring"""
        def monitor_loop():
            while True:
                try:
                    metrics = self.collect_system_metrics()
                    self.system_metrics.append(metrics)
                    
                    # Update Prometheus metrics
                    if self.memory_usage_gauge:
                        self.memory_usage_gauge.set(metrics['memory']['used'])
                    if self.cpu_usage_gauge:
                        self.cpu_usage_gauge.set(metrics['cpu']['percent'])
                    
                    # Check for anomalies
                    self.detect_system_anomalies(metrics)
                    
                    asyncio.run(self.store_metrics(metrics))
                    
                except Exception as e:
                    logger.error(f"System monitoring error: {e}")
                
                threading.Event().wait(10)  # Collect every 10 seconds
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def _init_ml_components(self):
        """Initialize ML components for pattern recognition"""
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.error_texts = []
        self.clustering_model = None
        
        # Load pre-trained models if available
        self._load_ml_models()
    
    def capture_error(self, error_data: Dict[str, Any]):
        """Capture and process an error"""
        error_entry = {
            'id': self.generate_error_id(error_data),
            'timestamp': datetime.now().isoformat(),
            'type': error_data.get('error_type', 'unknown'),
            'message': str(error_data.get('event', '')),
            'stack_trace': error_data.get('stack_trace', ''),
            'context': self.extract_context(error_data),
            'system_state': self.get_current_system_state(),
            'signature': self.generate_error_signature(error_data),
            'severity': self.calculate_severity(error_data)
        }
        
        # Store error
        self.errors.append(error_entry)
        
        # Update patterns
        self.update_error_patterns(error_entry)
        
        # Increment metrics
        if self.error_counter:
            self.error_counter.labels(
                type=error_entry['type'],
                severity=error_entry['severity']
            ).inc()
        
        # Perform immediate analysis for critical errors
        if error_entry['severity'] == 'critical':
            asyncio.create_task(self.perform_immediate_analysis(error_entry))
        
        # Queue for ML analysis
        self.queue_for_ml_analysis(error_entry)
        
        return error_entry
    
    def generate_error_id(self, error_data: Dict) -> str:
        """Generate unique error ID"""
        content = f"{error_data.get('event', '')}{error_data.get('stack_trace', '')}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def extract_context(self, error_data: Dict) -> Dict:
        """Extract relevant context from error data"""
        context = {}
        
        # Extract request context if available
        if 'request' in error_data:
            request = error_data['request']
            context['request'] = {
                'method': getattr(request, 'method', None),
                'path': getattr(request, 'url', {}).get('path'),
                'headers': dict(getattr(request, 'headers', {})),
                'client': getattr(request, 'client', {})
            }
        
        # Extract user context
        context['user'] = error_data.get('user', {})
        
        # Extract application context
        context['app'] = {
            'version': os.getenv('APP_VERSION', 'unknown'),
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'host': socket.gethostname()
        }
        
        return context
    
    def get_current_system_state(self) -> Dict:
        """Get current system state"""
        process = psutil.Process()
        
        return {
            'cpu_percent': process.cpu_percent(),
            'memory_info': process.memory_info()._asdict(),
            'num_threads': process.num_threads(),
            'open_files': len(process.open_files()),
            'connections': len(process.connections()),
            'io_counters': process.io_counters()._asdict() if sys.platform != 'darwin' else {}
        }
    
    def generate_error_signature(self, error_data: Dict) -> str:
        """Generate error signature for pattern matching"""
        # Extract key components
        error_type = error_data.get('error_type', '')
        message = error_data.get('event', '')
        
        # Clean message
        message = re.sub(r'/d+', 'N', message)  # Replace numbers
        message = re.sub(r'0x[0-9a-fA-F]+', 'ADDR', message)  # Replace addresses
        message = re.sub(r'["/'].*?["/']', 'STR', message)  # Replace strings
        
        # Generate signature
        signature = f"{error_type}:{message[:100]}"
        return hashlib.md5(signature.encode()).hexdigest()[:8]
    
    def calculate_severity(self, error_data: Dict) -> str:
        """Calculate error severity"""
        # Check for critical indicators
        critical_keywords = ['database', 'connection', 'auth', 'security', 'crash']
        message = str(error_data.get('event', '')).lower()
        
        for keyword in critical_keywords:
            if keyword in message:
                return 'critical'
        
        # Check error type
        error_type = error_data.get('error_type', '')
        if error_type in ['SystemError', 'MemoryError', 'KeyboardInterrupt']:
            return 'critical'
        elif error_type in ['ValueError', 'TypeError', 'AttributeError']:
            return 'warning'
        
        return 'info'
    
    def update_error_patterns(self, error_entry: Dict):
        """Update error patterns for pattern recognition"""
        signature = error_entry['signature']
        
        # Group by signature
        self.error_patterns[signature].append(error_entry)
        
        # Update signature metadata
        if signature not in self.error_signatures:
            self.error_signatures[signature] = {
                'first_seen': error_entry['timestamp'],
                'count': 0,
                'contexts': set(),
                'affected_users': set()
            }
        
        self.error_signatures[signature]['count'] += 1
        self.error_signatures[signature]['last_seen'] = error_entry['timestamp']
        
        # Track affected contexts
        if 'request' in error_entry['context']:
            path = error_entry['context']['request'].get('path')
            if path:
                self.error_signatures[signature]['contexts'].add(path)
    
    async def perform_immediate_analysis(self, error_entry: Dict):
        """Perform immediate analysis for critical errors"""
        try:
            # Analyze root cause
            root_cause = await self.analyze_root_cause_async(error_entry)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(error_entry, root_cause)
            
            # Store analysis results
            await self.store_analysis_results({
                'error_id': error_entry['id'],
                'root_cause': root_cause,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            })
            
            # Trigger alerts if needed
            if root_cause.get('confidence', 0) > 0.8:
                await self.trigger_alert(error_entry, root_cause, recommendations)
            
        except Exception as e:
            logger.error(f"Immediate analysis failed: {e}")
    
    def analyze_root_cause(self, exc_type, exc_value, tb) -> Dict:
        """Analyze root cause of an exception"""
        root_cause = {
            'primary_cause': None,
            'contributing_factors': [],
            'affected_components': [],
            'confidence': 0.0
        }
        
        # Extract traceback information
        tb_info = traceback.extract_tb(tb)
        
        # Analyze call stack
        call_stack = []
        for frame in tb_info:
            call_stack.append({
                'file': frame.filename,
                'line': frame.lineno,
                'function': frame.name,
                'code': frame.line
            })
        
        # Identify primary cause location
        if call_stack:
            primary_frame = call_stack[-1]
            root_cause['primary_cause'] = {
                'location': f"{primary_frame['file']}:{primary_frame['line']}",
                'function': primary_frame['function'],
                'code': primary_frame['code']
            }
        
        # Analyze error type
        if exc_type.__name__ == 'ConnectionError':
            root_cause['contributing_factors'].append('Network connectivity issue')
            root_cause['affected_components'].append('External service connection')
        elif exc_type.__name__ == 'MemoryError':
            root_cause['contributing_factors'].append('Memory exhaustion')
            root_cause['affected_components'].append('Memory management')
        elif exc_type.__name__ == 'TimeoutError':
            root_cause['contributing_factors'].append('Operation timeout')
            root_cause['affected_components'].append('Async operations')
        
        # Check for common patterns
        error_msg = str(exc_value).lower()
        
        if 'database' in error_msg or 'sql' in error_msg:
            root_cause['affected_components'].append('Database')
            root_cause['contributing_factors'].append('Database operation failure')
        
        if 'permission' in error_msg or 'denied' in error_msg:
            root_cause['affected_components'].append('Authorization')
            root_cause['contributing_factors'].append('Permission issue')
        
        if 'file' in error_msg or 'path' in error_msg:
            root_cause['affected_components'].append('File system')
            root_cause['contributing_factors'].append('File operation error')
        
        # Calculate confidence based on analysis depth
        confidence_factors = [
            len(call_stack) > 0,
            len(root_cause['contributing_factors']) > 0,
            len(root_cause['affected_components']) > 0,
            root_cause['primary_cause'] is not None
        ]
        
        root_cause['confidence'] = sum(confidence_factors) / len(confidence_factors)
        
        return root_cause
    
    async def analyze_root_cause_async(self, error_entry: Dict) -> Dict:
        """Async version of root cause analysis with additional checks"""
        root_cause = {
            'primary_cause': None,
            'contributing_factors': [],
            'affected_components': [],
            'related_errors': [],
            'confidence': 0.0
        }
        
        # Check for related errors
        signature = error_entry['signature']
        related = self.error_patterns.get(signature, [])
        
        if len(related) > 1:
            # Analyze pattern
            timestamps = [datetime.fromisoformat(e['timestamp']) for e in related]
            time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                         for i in range(1, len(timestamps))]
            
            if time_diffs and min(time_diffs) < 60:  # Errors within 1 minute
                root_cause['contributing_factors'].append('Rapid error recurrence')
            
            # Extract common context
            paths = set()
            for e in related:
                if 'request' in e.get('context', {}):
                    path = e['context']['request'].get('path')
                    if path:
                        paths.add(path)
            
            if paths:
                root_cause['affected_components'].extend(list(paths))
        
        # Check system state
        system_state = error_entry.get('system_state', {})
        
        if system_state.get('cpu_percent', 0) > 80:
            root_cause['contributing_factors'].append('High CPU usage')
        
        if system_state.get('memory_info', {}).get('rss', 0) > 1e9:  # > 1GB
            root_cause['contributing_factors'].append('High memory usage')
        
        # ML-based analysis
        ml_insights = await self.get_ml_insights_async(error_entry)
        if ml_insights:
            root_cause['ml_analysis'] = ml_insights
            
            if ml_insights.get('cluster_id') is not None:
                # Find similar errors in cluster
                similar_errors = self.find_similar_errors(error_entry, ml_insights['cluster_id'])
                root_cause['related_errors'] = similar_errors[:5]
        
        # Update confidence
        confidence_factors = [
            len(root_cause['contributing_factors']) > 0,
            len(root_cause['affected_components']) > 0,
            len(root_cause['related_errors']) > 0,
            'ml_analysis' in root_cause
        ]
        
        root_cause['confidence'] = sum(confidence_factors) / len(confidence_factors)
        
        return root_cause
    
    def generate_recommendations(self, error_entry: Dict, root_cause: Dict) -> List[Dict]:
        """Generate recommendations based on error and root cause analysis"""
        recommendations = []
        
        # Based on error type
        error_type = error_entry.get('type', '')
        
        if 'Connection' in error_type:
            recommendations.append({
                'action': 'implement_retry',
                'description': 'Implement exponential backoff retry mechanism',
                'priority': 'high',
                'code_sample': '''
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def connection_with_retry():
    # Your connection code here
    pass
'''
            })
            
            recommendations.append({
                'action': 'add_circuit_breaker',
                'description': 'Add circuit breaker pattern for external services',
                'priority': 'medium'
            })
        
        elif 'Memory' in error_type:
            recommendations.append({
                'action': 'optimize_memory',
                'description': 'Implement memory optimization strategies',
                'priority': 'critical',
                'steps': [
                    'Use generators instead of lists for large datasets',
                    'Implement object pooling for frequently created objects',
                    'Add memory profiling to identify leaks'
                ]
            })
        
        elif 'Timeout' in error_type:
            recommendations.append({
                'action': 'increase_timeout',
                'description': 'Increase timeout values for long-running operations',
                'priority': 'medium'
            })
            
            recommendations.append({
                'action': 'add_async_processing',
                'description': 'Move long-running tasks to background workers',
                'priority': 'high'
            })
        
        # Based on contributing factors
        for factor in root_cause.get('contributing_factors', []):
            if 'High CPU' in factor:
                recommendations.append({
                    'action': 'optimize_algorithms',
                    'description': 'Profile and optimize CPU-intensive algorithms',
                    'priority': 'high'
                })
            
            elif 'Rapid error recurrence' in factor:
                recommendations.append({
                    'action': 'add_rate_limiting',
                    'description': 'Implement rate limiting to prevent error storms',
                    'priority': 'critical'
                })
        
        # Based on affected components
        for component in root_cause.get('affected_components', []):
            if 'database' in component.lower():
                recommendations.append({
                    'action': 'optimize_queries',
                    'description': 'Review and optimize database queries',
                    'priority': 'medium',
                    'tools': ['EXPLAIN ANALYZE', 'Query profiler']
                })
            
            elif 'auth' in component.lower():
                recommendations.append({
                    'action': 'review_permissions',
                    'description': 'Review and update permission configurations',
                    'priority': 'high'
                })
        
        return recommendations
    
    def collect_system_metrics(self) -> Dict:
        """Collect current system metrics"""
        cpu = psutil.cpu_percent(interval=1, percpu=True)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': psutil.cpu_percent(interval=1),
                'per_cpu': cpu,
                'count': psutil.cpu_count()
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used,
                'free': memory.free
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            },
            'network': {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv,
                'errin': network.errin,
                'errout': network.errout
            },
            'process': self.get_current_system_state()
        }
    
    def detect_system_anomalies(self, metrics: Dict):
        """Detect system anomalies"""
        anomalies = []
        
        # CPU anomaly
        if metrics['cpu']['percent'] > 90:
            anomalies.append({
                'type': 'high_cpu',
                'value': metrics['cpu']['percent'],
                'severity': 'warning' if metrics['cpu']['percent'] < 95 else 'critical'
            })
        
        # Memory anomaly
        if metrics['memory']['percent'] > 85:
            anomalies.append({
                'type': 'high_memory',
                'value': metrics['memory']['percent'],
                'severity': 'warning' if metrics['memory']['percent'] < 90 else 'critical'
            })
        
        # Disk anomaly
        if metrics['disk']['percent'] > 80:
            anomalies.append({
                'type': 'high_disk_usage',
                'value': metrics['disk']['percent'],
                'severity': 'warning' if metrics['disk']['percent'] < 90 else 'critical'
            })
        
        # Network errors
        if metrics['network']['errin'] > 100 or metrics['network']['errout'] > 100:
            anomalies.append({
                'type': 'network_errors',
                'in_errors': metrics['network']['errin'],
                'out_errors': metrics['network']['errout'],
                'severity': 'warning'
            })
        
        if anomalies:
            self.handle_anomalies(anomalies, metrics)
    
    def handle_anomalies(self, anomalies: List[Dict], metrics: Dict):
        """Handle detected anomalies"""
        for anomaly in anomalies:
            logger.warning(f"System anomaly detected: {anomaly}")
            
            # Store anomaly
            self.capture_error({
                'error_type': 'SystemAnomaly',
                'event': f"System anomaly: {anomaly['type']}",
                'anomaly': anomaly,
                'metrics': metrics
            })
    
    def queue_for_ml_analysis(self, error_entry: Dict):
        """Queue error for ML analysis"""
        # Add error text for vectorization
        error_text = f"{error_entry['type']} {error_entry['message']}"
        self.error_texts.append(error_text)
        
        # Trigger clustering if enough samples
        if len(self.error_texts) >= 50 and len(self.error_texts) % 10 == 0:
            self.perform_error_clustering()
    
    def perform_error_clustering(self):
        """Perform DBSCAN clustering on error messages"""
        try:
            # Vectorize error texts
            X = self.vectorizer.fit_transform(self.error_texts[-500:])  # Last 500 errors
            
            # Perform clustering
            clustering = DBSCAN(eps=0.3, min_samples=2)
            clusters = clustering.fit_predict(X)
            
            # Store clustering results
            self.error_clusters = clusters
            self.clustering_model = clustering
            
            # Analyze clusters
            cluster_info = defaultdict(list)
            for i, cluster_id in enumerate(clusters):
                if cluster_id != -1:  # Not noise
                    cluster_info[cluster_id].append(self.error_texts[-500:][i])
            
            # Log cluster information
            for cluster_id, errors in cluster_info.items():
                logger.info(f"Error cluster {cluster_id}: {len(errors)} errors")
                
                # Find common pattern
                if errors:
                    common_words = Counter()
                    for error in errors[:10]:  # Sample first 10
                        words = error.split()
                        common_words.update(words)
                    
                    most_common = common_words.most_common(5)
                    logger.info(f"  Common terms: {most_common}")
            
        except Exception as e:
            logger.error(f"Error clustering failed: {e}")
    
    async def get_ml_insights_async(self, error_entry: Dict) -> Dict:
        """Get ML insights for an error"""
        insights = {
            'cluster_id': None,
            'similar_errors': [],
            'predicted_cause': None,
            'anomaly_score': 0.0
        }
        
        try:
            # Vectorize error
            error_text = f"{error_entry['type']} {error_entry['message']}"
            
            if self.vectorizer and self.clustering_model:
                X = self.vectorizer.transform([error_text])
                
                # Predict cluster
                cluster = self.clustering_model.fit_predict(X)[0]
                insights['cluster_id'] = int(cluster) if cluster != -1 else None
                
                # Calculate anomaly score (distance to nearest cluster)
                if hasattr(self.clustering_model, 'core_sample_indices_'):
                    # Simple anomaly scoring based on clustering
                    insights['anomaly_score'] = 0.0 if cluster != -1 else 1.0
            
            # Find similar errors
            similar = self.find_similar_errors_by_text(error_text)
            insights['similar_errors'] = similar[:5]
            
            # Predict likely cause based on patterns
            if similar:
                # Analyze similar errors for common causes
                causes = defaultdict(int)
                for err in similar:
                    if 'root_cause' in err:
                        primary = err['root_cause'].get('primary_cause')
                        if primary:
                            causes[str(primary)] += 1
                
                if causes:
                    predicted = max(causes.items(), key=lambda x: x[1])
                    insights['predicted_cause'] = predicted[0]
            
        except Exception as e:
            logger.error(f"ML insights generation failed: {e}")
        
        return insights
    
    def get_ml_insights(self, error_message: str) -> Dict:
        """Synchronous version of ML insights"""
        return asyncio.run(self.get_ml_insights_async({'message': error_message, 'type': 'unknown'}))
    
    def find_similar_errors(self, error_entry: Dict, cluster_id: int) -> List[Dict]:
        """Find similar errors in the same cluster"""
        similar = []
        
        for error in self.errors:
            if error.get('cluster_id') == cluster_id and error['id'] != error_entry['id']:
                similar.append({
                    'id': error['id'],
                    'message': error['message'][:100],
                    'timestamp': error['timestamp'],
                    'similarity': 0.9  # High similarity since same cluster
                })
        
        return sorted(similar, key=lambda x: x['timestamp'], reverse=True)
    
    def find_similar_errors_by_text(self, error_text: str, threshold: float = 0.7) -> List[Dict]:
        """Find similar errors by text similarity"""
        similar = []
        
        try:
            # Use simple text similarity for now
            for error in list(self.errors)[-100:]:  # Check last 100 errors
                similarity = self.calculate_text_similarity(error_text, error.get('message', ''))
                
                if similarity > threshold:
                    similar.append({
                        'id': error['id'],
                        'message': error['message'][:100],
                        'timestamp': error['timestamp'],
                        'similarity': similarity
                    })
        except Exception as e:
            logger.error(f"Text similarity search failed: {e}")
        
        return sorted(similar, key=lambda x: x['similarity'], reverse=True)
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        # Simple Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def store_metrics(self, metrics: Dict):
        """Store metrics in database"""
        try:
            await init_mongodb()
            db = get_mongodb()
            
            if db:
                await db.system_metrics.insert_one({
                    **metrics,
                    'app_name': self.app_name
                })
                
                # Clean old metrics (keep last 7 days)
                cutoff = datetime.now() - timedelta(days=7)
                await db.system_metrics.delete_many({
                    'timestamp': {'$lt': cutoff.isoformat()}
                })
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
    
    async def store_analysis_results(self, analysis: Dict):
        """Store analysis results"""
        try:
            await init_mongodb()
            db = get_mongodb()
            
            if db:
                await db.error_analysis.insert_one({
                    **analysis,
                    'app_name': self.app_name
                })
        except Exception as e:
            logger.error(f"Failed to store analysis: {e}")
    
    async def trigger_alert(self, error_entry: Dict, root_cause: Dict, recommendations: List[Dict]):
        """Trigger alert for critical errors"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'error_id': error_entry['id'],
            'severity': error_entry['severity'],
            'message': error_entry['message'][:200],
            'root_cause': root_cause.get('primary_cause'),
            'recommendations': [r['description'] for r in recommendations[:3]],
            'affected_components': root_cause.get('affected_components', [])
        }
        
        # Log alert
        logger.critical(f"ALERT: {alert}")
        
        # Store alert
        try:
            await init_mongodb()
            db = get_mongodb()
            
            if db:
                await db.alerts.insert_one({
                    **alert,
                    'app_name': self.app_name,
                    'acknowledged': False
                })
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
    
    def _load_ml_models(self):
        """Load pre-trained ML models if available"""
        model_path = Path('models/error_analysis')
        
        if model_path.exists():
            # Load vectorizer
            vectorizer_path = model_path / 'vectorizer.pkl'
            if vectorizer_path.exists():
                import pickle
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                logger.info("Loaded pre-trained vectorizer")
            
            # Load clustering model
            clustering_path = model_path / 'clustering.pkl'
            if clustering_path.exists():
                import pickle
                with open(clustering_path, 'rb') as f:
                    self.clustering_model = pickle.load(f)
                logger.info("Loaded pre-trained clustering model")
    
    def save_ml_models(self):
        """Save ML models for future use"""
        model_path = Path('models/error_analysis')
        model_path.mkdir(parents=True, exist_ok=True)
        
        if self.vectorizer:
            import pickle
            with open(model_path / 'vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
        
        if self.clustering_model:
            import pickle
            with open(model_path / 'clustering.pkl', 'wb') as f:
                pickle.dump(self.clustering_model, f)
    
    def get_metrics_endpoint(self):
        """Get Prometheus metrics endpoint"""
        return generate_latest()
    
    def get_error_summary(self) -> Dict:
        """Get error summary for dashboard"""
        summary = {
            'total_errors': len(self.errors),
            'error_rate': self.calculate_error_rate(),
            'top_errors': self.get_top_errors(),
            'error_trends': self.get_error_trends(),
            'affected_components': self.get_affected_components(),
            'system_health': self.calculate_system_health()
        }
        
        return summary
    
    def calculate_error_rate(self) -> float:
        """Calculate current error rate (errors per minute)"""
        if not self.errors:
            return 0.0
        
        now = datetime.now()
        recent_errors = [e for e in self.errors 
                        if (now - datetime.fromisoformat(e['timestamp'])).total_seconds() < 60]
        
        return len(recent_errors)
    
    def get_top_errors(self, limit: int = 5) -> List[Dict]:
        """Get top occurring errors"""
        error_counts = Counter()
        
        for error in self.errors:
            signature = error.get('signature', 'unknown')
            error_counts[signature] += 1
        
        top_errors = []
        for signature, count in error_counts.most_common(limit):
            error_info = self.error_signatures.get(signature, {})
            top_errors.append({
                'signature': signature,
                'count': count,
                'first_seen': error_info.get('first_seen'),
                'last_seen': error_info.get('last_seen'),
                'contexts': list(error_info.get('contexts', []))[:3]
            })
        
        return top_errors
    
    def get_error_trends(self) -> Dict:
        """Get error trends over time"""
        if not self.errors:
            return {'hourly': [], 'daily': []}
        
        now = datetime.now()
        hourly = defaultdict(int)
        daily = defaultdict(int)
        
        for error in self.errors:
            timestamp = datetime.fromisoformat(error['timestamp'])
            
            # Hourly for last 24 hours
            if (now - timestamp).days < 1:
                hour_key = timestamp.strftime('%Y-%m-%d %H:00')
                hourly[hour_key] += 1
            
            # Daily for last 7 days
            if (now - timestamp).days < 7:
                day_key = timestamp.strftime('%Y-%m-%d')
                daily[day_key] += 1
        
        return {
            'hourly': [{'time': k, 'count': v} for k, v in sorted(hourly.items())],
            'daily': [{'date': k, 'count': v} for k, v in sorted(daily.items())]
        }
    
    def get_affected_components(self) -> List[Dict]:
        """Get list of affected components"""
        components = defaultdict(int)
        
        for signature_data in self.error_signatures.values():
            for context in signature_data.get('contexts', []):
                components[context] += signature_data['count']
        
        return [{'component': k, 'error_count': v} 
                for k, v in sorted(components.items(), key=lambda x: x[1], reverse=True)[:10]]
    
    def calculate_system_health(self) -> float:
        """Calculate overall system health score"""
        if not self.system_metrics:
            return 100.0
        
        # Get latest metrics
        latest_metrics = self.system_metrics[-1]
        
        health_score = 100.0
        
        # Deduct for high resource usage
        cpu_penalty = max(0, latest_metrics['cpu']['percent'] - 70) * 0.5
        memory_penalty = max(0, latest_metrics['memory']['percent'] - 80) * 0.5
        disk_penalty = max(0, latest_metrics['disk']['percent'] - 85) * 0.3
        
        # Deduct for error rate
        error_rate = self.calculate_error_rate()
        error_penalty = min(error_rate * 2, 30)  # Max 30 point penalty
        
        health_score -= (cpu_penalty + memory_penalty + disk_penalty + error_penalty)
        
        return max(0, health_score)


# Global instance
backend_error_monitor = BackendErrorMonitor()
#!/usr/bin/env python3
"""
SYSTEM HEALTH PREDICTOR - Phase 2 Implementation
AI-powered system health monitoring and failure prediction
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# ML and system monitoring imports
try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import psutil
    import GPUtil
    HEALTH_PREDICTION_AVAILABLE = True
except ImportError as e:
    logging.error(f"Health prediction dependencies not available: {e}")
    HEALTH_PREDICTION_AVAILABLE = False

from loguru import logger

class HealthStatus(Enum):
    """System health status levels"""
    EXCELLENT = "excellent"
    GOOD = "good"  
    WARNING = "warning"
    CRITICAL = "critical"
    FAILING = "failing"

class ComponentType(Enum):
    """System component types"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DATABASE = "database"
    APPLICATION = "application"
    SERVICES = "services"

@dataclass
class HealthMetrics:
    """System health metrics snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, float]
    process_count: int
    thread_count: int
    open_files: int
    database_connections: int
    response_time_ms: float
    error_rate: float
    temperature: Optional[float] = None
    gpu_utilization: Optional[float] = None

@dataclass
class HealthPrediction:
    """Health prediction result"""
    component: ComponentType
    current_status: HealthStatus
    predicted_status: HealthStatus
    time_to_failure: Optional[timedelta]
    confidence: float
    risk_factors: List[str]
    recommendations: List[str]
    anomaly_score: float

@dataclass
class SystemHealthReport:
    """Comprehensive system health report"""
    overall_health: HealthStatus
    component_predictions: List[HealthPrediction]
    capacity_forecast: Dict[str, float]  # Next 7 days
    critical_alerts: List[str]
    optimization_suggestions: List[str]
    generated_at: datetime
    prediction_horizon: timedelta

class SystemHealthPredictor:
    """AI-powered system health monitoring and prediction"""
    
    def __init__(self):
        self.metrics_history: List[HealthMetrics] = []
        self.anomaly_detector = None
        self.failure_predictor = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_training = None
        
        # Health thresholds
        self.thresholds = {
            ComponentType.CPU: {'warning': 70, 'critical': 90},
            ComponentType.MEMORY: {'warning': 80, 'critical': 95},
            ComponentType.DISK: {'warning': 85, 'critical': 95},
            ComponentType.DATABASE: {'warning': 100, 'critical': 200},  # connections
            ComponentType.APPLICATION: {'warning': 1000, 'critical': 5000}  # response time ms
        }
        
        if HEALTH_PREDICTION_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for health prediction"""
        try:
            logger.info("Initializing system health prediction models...")
            
            # Anomaly detection model
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            # Failure prediction model
            self.failure_predictor = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            logger.info("Health prediction models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing health prediction models: {e}")
    
    async def collect_current_metrics(self) -> HealthMetrics:
        """Collect current system metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            net_io = psutil.net_io_counters()
            
            # Process information
            process_count = len(psutil.pids())
            
            # Calculate thread count
            thread_count = 0
            for proc in psutil.process_iter(['num_threads']):
                try:
                    thread_count += proc.info['num_threads'] or 0
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Open files (approximate)
            open_files = 0
            try:
                for proc in psutil.process_iter(['open_files']):
                    try:
                        files = proc.info['open_files']
                        if files:
                            open_files += len(files)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except:
                open_files = 0
            
            # GPU metrics (if available)
            gpu_utilization = None
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_utilization = gpus[0].load * 100
            except:
                pass
            
            # Temperature (if available)
            temperature = None
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get CPU temperature if available
                    for name, entries in temps.items():
                        if entries:
                            temperature = entries[0].current
                            break
            except:
                pass
            
            metrics = HealthMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                network_io={
                    'bytes_sent': float(net_io.bytes_sent),
                    'bytes_recv': float(net_io.bytes_recv),
                    'packets_sent': float(net_io.packets_sent),
                    'packets_recv': float(net_io.packets_recv)
                },
                process_count=process_count,
                thread_count=thread_count,
                open_files=open_files,
                database_connections=0,  # Will be updated by database monitoring
                response_time_ms=0.0,    # Will be updated by application monitoring
                error_rate=0.0,          # Will be updated by error monitoring
                temperature=temperature,
                gpu_utilization=gpu_utilization
            )
            
            # Store in history
            self.metrics_history.append(metrics)
            
            # Keep only last 1000 entries
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return self._get_default_metrics()
    
    def _get_default_metrics(self) -> HealthMetrics:
        """Get default metrics when collection fails"""
        return HealthMetrics(
            timestamp=datetime.now(),
            cpu_percent=0.0,
            memory_percent=0.0,
            disk_percent=0.0,
            network_io={'bytes_sent': 0.0, 'bytes_recv': 0.0, 'packets_sent': 0.0, 'packets_recv': 0.0},
            process_count=0,
            thread_count=0,
            open_files=0,
            database_connections=0,
            response_time_ms=0.0,
            error_rate=0.0
        )
    
    async def predict_system_health(self, forecast_hours: int = 24) -> SystemHealthReport:
        """Generate comprehensive system health prediction"""
        if not HEALTH_PREDICTION_AVAILABLE:
            return self._fallback_health_report()
        
        current_metrics = await self.collect_current_metrics()
        
        # Train models if needed
        if not self.is_trained or self._should_retrain():
            await self._train_models()
        
        # Generate predictions for each component
        component_predictions = []
        
        for component_type in ComponentType:
            prediction = await self._predict_component_health(
                component_type, current_metrics, forecast_hours
            )
            component_predictions.append(prediction)
        
        # Determine overall health
        overall_health = self._calculate_overall_health(component_predictions)
        
        # Generate capacity forecast
        capacity_forecast = await self._forecast_capacity(forecast_hours)
        
        # Generate critical alerts
        critical_alerts = self._generate_critical_alerts(component_predictions)
        
        # Generate optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions(
            component_predictions, current_metrics
        )
        
        return SystemHealthReport(
            overall_health=overall_health,
            component_predictions=component_predictions,
            capacity_forecast=capacity_forecast,
            critical_alerts=critical_alerts,
            optimization_suggestions=optimization_suggestions,
            generated_at=datetime.now(),
            prediction_horizon=timedelta(hours=forecast_hours)
        )
    
    async def _predict_component_health(
        self, 
        component: ComponentType, 
        current_metrics: HealthMetrics,
        forecast_hours: int
    ) -> HealthPrediction:
        """Predict health for a specific component"""
        
        # Get current status
        current_status = self._assess_current_status(component, current_metrics)
        
        # Predict future status
        predicted_status = current_status  # Default to current
        time_to_failure = None
        confidence = 0.5
        anomaly_score = 0.0
        
        if len(self.metrics_history) >= 10 and self.is_trained:
            try:
                # Prepare feature vector
                features = self._extract_features(current_metrics, component)
                
                # Anomaly detection
                anomaly_score = self.anomaly_detector.decision_function([features])[0]
                
                # Predict time to failure (in hours)
                predicted_hours = self.failure_predictor.predict([features])[0]
                
                if predicted_hours > 0 and predicted_hours < forecast_hours:
                    time_to_failure = timedelta(hours=predicted_hours)
                    predicted_status = HealthStatus.CRITICAL
                    confidence = 0.8
                elif anomaly_score < -0.5:  # Anomaly threshold
                    predicted_status = HealthStatus.WARNING
                    confidence = 0.7
                else:
                    confidence = 0.9
                    
            except Exception as e:
                logger.warning(f"Prediction failed for {component}: {e}")
        
        # Generate risk factors and recommendations
        risk_factors = self._identify_risk_factors(component, current_metrics)
        recommendations = self._generate_recommendations(component, current_status, risk_factors)
        
        return HealthPrediction(
            component=component,
            current_status=current_status,
            predicted_status=predicted_status,
            time_to_failure=time_to_failure,
            confidence=confidence,
            risk_factors=risk_factors,
            recommendations=recommendations,
            anomaly_score=abs(anomaly_score)
        )
    
    def _assess_current_status(self, component: ComponentType, metrics: HealthMetrics) -> HealthStatus:
        """Assess current health status of a component"""
        thresholds = self.thresholds.get(component, {'warning': 80, 'critical': 95})
        
        value = 0
        if component == ComponentType.CPU:
            value = metrics.cpu_percent
        elif component == ComponentType.MEMORY:
            value = metrics.memory_percent
        elif component == ComponentType.DISK:
            value = metrics.disk_percent
        elif component == ComponentType.DATABASE:
            value = metrics.database_connections
        elif component == ComponentType.APPLICATION:
            value = metrics.response_time_ms
        elif component == ComponentType.NETWORK:
            # Network health based on error rates and throughput
            value = metrics.error_rate * 100
        
        if value >= thresholds['critical']:
            return HealthStatus.CRITICAL
        elif value >= thresholds['warning']:
            return HealthStatus.WARNING
        elif value > thresholds['warning'] * 0.5:
            return HealthStatus.GOOD
        else:
            return HealthStatus.EXCELLENT
    
    def _extract_features(self, metrics: HealthMetrics, component: ComponentType) -> List[float]:
        """Extract features for ML prediction"""
        base_features = [
            metrics.cpu_percent,
            metrics.memory_percent,
            metrics.disk_percent,
            metrics.process_count / 100.0,  # Normalize
            metrics.thread_count / 1000.0,  # Normalize
            metrics.open_files / 1000.0,    # Normalize
            metrics.response_time_ms / 1000.0,  # Normalize
            metrics.error_rate
        ]
        
        # Add component-specific features
        if component == ComponentType.CPU:
            base_features.extend([
                metrics.temperature or 0,
                metrics.gpu_utilization or 0
            ])
        elif component == ComponentType.NETWORK:
            base_features.extend([
                metrics.network_io['bytes_sent'] / 1e9,  # Normalize to GB
                metrics.network_io['bytes_recv'] / 1e9,
                metrics.network_io['packets_sent'] / 1e6,  # Normalize to millions
                metrics.network_io['packets_recv'] / 1e6
            ])
        
        return base_features
    
    async def _train_models(self):
        """Train ML models on historical data"""
        if len(self.metrics_history) < 20:
            logger.info("Insufficient data for training health prediction models")
            return
        
        try:
            logger.info("Training health prediction models...")
            
            # Prepare training data
            features = []
            labels = []
            
            for i, metrics in enumerate(self.metrics_history[:-1]):
                for component in ComponentType:
                    feature_vector = self._extract_features(metrics, component)
                    features.append(feature_vector)
                    
                    # Create synthetic labels (hours until next status change)
                    # In a real system, this would be based on actual failure data
                    current_status = self._assess_current_status(component, metrics)
                    if current_status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
                        labels.append(24.0)  # Assume failure in 24 hours if critical/warning
                    else:
                        labels.append(168.0)  # Assume stable for a week if healthy
            
            if len(features) < 10:
                logger.info("Insufficient feature data for training")
                return
            
            # Convert to numpy arrays
            X = np.array(features)
            y = np.array(labels)
            
            # Handle NaN values
            X = np.nan_to_num(X)
            y = np.nan_to_num(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train anomaly detector
            self.anomaly_detector.fit(X_scaled)
            
            # Train failure predictor
            if len(X_scaled) > 5:  # Need minimum samples for train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                self.failure_predictor.fit(X_train, y_train)
            else:
                self.failure_predictor.fit(X_scaled, y)
            
            self.is_trained = True
            self.last_training = datetime.now()
            
            logger.info("Health prediction models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training health prediction models: {e}")
    
    def _should_retrain(self) -> bool:
        """Check if models should be retrained"""
        if not self.last_training:
            return True
        
        # Retrain daily
        return datetime.now() - self.last_training > timedelta(days=1)
    
    def _calculate_overall_health(self, predictions: List[HealthPrediction]) -> HealthStatus:
        """Calculate overall system health from component predictions"""
        if not predictions:
            return HealthStatus.GOOD
        
        # Count status levels
        status_counts = {}
        for prediction in predictions:
            status = prediction.predicted_status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Determine overall health (worst case wins)
        if status_counts.get(HealthStatus.CRITICAL, 0) > 0:
            return HealthStatus.CRITICAL
        elif status_counts.get(HealthStatus.WARNING, 0) >= 2:  # Multiple warnings = critical
            return HealthStatus.CRITICAL
        elif status_counts.get(HealthStatus.WARNING, 0) > 0:
            return HealthStatus.WARNING
        elif status_counts.get(HealthStatus.GOOD, 0) >= len(predictions) // 2:
            return HealthStatus.GOOD
        else:
            return HealthStatus.EXCELLENT
    
    async def _forecast_capacity(self, hours: int) -> Dict[str, float]:
        """Forecast system capacity for next N hours"""
        if len(self.metrics_history) < 5:
            return {'cpu': 50.0, 'memory': 50.0, 'disk': 50.0}
        
        # Simple linear trend forecasting
        recent_metrics = self.metrics_history[-10:]
        
        # Calculate trends
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        disk_values = [m.disk_percent for m in recent_metrics]
        
        # Forecast using simple linear regression
        forecast = {}
        forecast['cpu'] = min(100.0, max(0.0, np.mean(cpu_values) + np.std(cpu_values)))
        forecast['memory'] = min(100.0, max(0.0, np.mean(memory_values) + np.std(memory_values)))
        forecast['disk'] = min(100.0, max(0.0, np.mean(disk_values) + np.std(disk_values)))
        
        return forecast
    
    def _generate_critical_alerts(self, predictions: List[HealthPrediction]) -> List[str]:
        """Generate critical alerts based on predictions"""
        alerts = []
        
        for prediction in predictions:
            if prediction.predicted_status == HealthStatus.CRITICAL:
                if prediction.time_to_failure:
                    hours = prediction.time_to_failure.total_seconds() / 3600
                    alerts.append(f"CRITICAL: {prediction.component.value} predicted to fail in {hours:.1f} hours")
                else:
                    alerts.append(f"CRITICAL: {prediction.component.value} in critical state")
            
            elif prediction.predicted_status == HealthStatus.WARNING and prediction.confidence > 0.8:
                alerts.append(f"WARNING: {prediction.component.value} showing warning signs")
        
        return alerts
    
    def _generate_optimization_suggestions(
        self, 
        predictions: List[HealthPrediction], 
        current_metrics: HealthMetrics
    ) -> List[str]:
        """Generate system optimization suggestions"""
        suggestions = []
        
        # CPU optimization
        if current_metrics.cpu_percent > 70:
            suggestions.append("Consider scaling CPU resources or optimizing high-CPU processes")
        
        # Memory optimization
        if current_metrics.memory_percent > 80:
            suggestions.append("Memory usage is high - consider increasing RAM or optimizing memory-heavy applications")
        
        # Disk optimization
        if current_metrics.disk_percent > 85:
            suggestions.append("Disk space is running low - clean up old files or add storage capacity")
        
        # Process optimization
        if current_metrics.process_count > 300:
            suggestions.append("High process count detected - review running services and applications")
        
        # Response time optimization
        if current_metrics.response_time_ms > 2000:
            suggestions.append("Application response times are slow - consider performance tuning or scaling")
        
        # Add prediction-based suggestions
        for prediction in predictions:
            if prediction.recommendations:
                suggestions.extend(prediction.recommendations)
        
        return list(set(suggestions))  # Remove duplicates
    
    def _identify_risk_factors(self, component: ComponentType, metrics: HealthMetrics) -> List[str]:
        """Identify risk factors for component failure"""
        risks = []
        
        if component == ComponentType.CPU:
            if metrics.cpu_percent > 80:
                risks.append("High CPU utilization")
            if metrics.temperature and metrics.temperature > 75:
                risks.append("High CPU temperature")
        
        elif component == ComponentType.MEMORY:
            if metrics.memory_percent > 85:
                risks.append("High memory usage")
            if metrics.process_count > 300:
                risks.append("High process count increasing memory pressure")
        
        elif component == ComponentType.DISK:
            if metrics.disk_percent > 90:
                risks.append("Very low disk space")
        
        elif component == ComponentType.APPLICATION:
            if metrics.response_time_ms > 2000:
                risks.append("Slow application response times")
            if metrics.error_rate > 0.05:
                risks.append("High application error rate")
        
        return risks
    
    def _generate_recommendations(
        self, 
        component: ComponentType, 
        status: HealthStatus, 
        risks: List[str]
    ) -> List[str]:
        """Generate recommendations for component health"""
        recommendations = []
        
        if status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
            if component == ComponentType.CPU:
                recommendations.extend([
                    "Monitor CPU-intensive processes",
                    "Consider CPU scaling or load balancing",
                    "Check for CPU thermal throttling"
                ])
            
            elif component == ComponentType.MEMORY:
                recommendations.extend([
                    "Identify memory-intensive applications",
                    "Consider adding RAM or memory optimization",
                    "Check for memory leaks in applications"
                ])
            
            elif component == ComponentType.DISK:
                recommendations.extend([
                    "Clean up temporary files and logs",
                    "Archive or delete old data",
                    "Add additional storage capacity"
                ])
            
            elif component == ComponentType.APPLICATION:
                recommendations.extend([
                    "Review application performance metrics",
                    "Optimize database queries",
                    "Consider application scaling"
                ])
        
        return recommendations
    
    def _fallback_health_report(self) -> SystemHealthReport:
        """Fallback health report when ML models aren't available"""
        try:
            current_metrics = psutil.cpu_percent(), psutil.virtual_memory().percent, psutil.disk_usage('/').percent
            
            # Simple threshold-based assessment
            cpu_status = HealthStatus.GOOD if current_metrics[0] < 70 else HealthStatus.WARNING
            memory_status = HealthStatus.GOOD if current_metrics[1] < 80 else HealthStatus.WARNING  
            disk_status = HealthStatus.GOOD if current_metrics[2] < 85 else HealthStatus.WARNING
            
            predictions = [
                HealthPrediction(
                    component=ComponentType.CPU,
                    current_status=cpu_status,
                    predicted_status=cpu_status,
                    time_to_failure=None,
                    confidence=0.6,
                    risk_factors=[],
                    recommendations=[],
                    anomaly_score=0.0
                ),
                HealthPrediction(
                    component=ComponentType.MEMORY,
                    current_status=memory_status,
                    predicted_status=memory_status,
                    time_to_failure=None,
                    confidence=0.6,
                    risk_factors=[],
                    recommendations=[],
                    anomaly_score=0.0
                ),
                HealthPrediction(
                    component=ComponentType.DISK,
                    current_status=disk_status,
                    predicted_status=disk_status,
                    time_to_failure=None,
                    confidence=0.6,
                    risk_factors=[],
                    recommendations=[],
                    anomaly_score=0.0
                )
            ]
            
        except:
            predictions = []
        
        return SystemHealthReport(
            overall_health=HealthStatus.GOOD,
            component_predictions=predictions,
            capacity_forecast={'cpu': 50.0, 'memory': 50.0, 'disk': 50.0},
            critical_alerts=[],
            optimization_suggestions=["Advanced health prediction not available - install required packages"],
            generated_at=datetime.now(),
            prediction_horizon=timedelta(hours=24)
        )

# Global instance
system_health_predictor = SystemHealthPredictor()

def get_system_health_predictor() -> SystemHealthPredictor:
    """Get the global system health predictor instance"""
    return system_health_predictor
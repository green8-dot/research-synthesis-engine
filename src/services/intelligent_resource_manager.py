#!/usr/bin/env python3
"""
INTELLIGENT RESOURCE MANAGER - Phase 2 Implementation
AI-driven resource allocation and predictive auto-scaling
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# ML and system imports
try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    import psutil
    RESOURCE_MANAGEMENT_AVAILABLE = True
except ImportError as e:
    logging.error(f"Resource management dependencies not available: {e}")
    RESOURCE_MANAGEMENT_AVAILABLE = False

from loguru import logger

class ResourceType(Enum):
    """Types of system resources"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    WORKERS = "workers"
    CONNECTIONS = "connections"

class ScalingAction(Enum):
    """Scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down" 
    MAINTAIN = "maintain"
    OPTIMIZE = "optimize"

class Priority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ResourceUsage:
    """Current resource usage snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_io_mbps: float
    network_io_mbps: float
    active_connections: int
    queue_length: int
    response_time_ms: float
    throughput_rps: float  # requests per second

@dataclass
class ResourcePrediction:
    """Resource usage prediction"""
    resource_type: ResourceType
    current_usage: float
    predicted_usage: float
    confidence: float
    recommendation: ScalingAction
    reasoning: str
    time_horizon: timedelta

@dataclass
class TaskInfo:
    """Information about a task/job"""
    task_id: str
    priority: Priority
    estimated_duration: timedelta
    resource_requirements: Dict[ResourceType, float]
    dependencies: List[str]
    deadline: Optional[datetime]
    created_at: datetime

@dataclass
class WorkerCapacity:
    """Worker/service capacity information"""
    worker_id: str
    max_cpu: float
    max_memory: float
    current_load: float
    active_tasks: int
    avg_task_duration: float
    availability_score: float

@dataclass
class OptimizationRecommendation:
    """Resource optimization recommendation"""
    resource_type: ResourceType
    action: ScalingAction
    current_allocation: float
    recommended_allocation: float
    expected_improvement: str
    impact_score: float
    implementation_steps: List[str]

class IntelligentResourceManager:
    """AI-driven resource allocation and auto-scaling system"""
    
    def __init__(self):
        self.usage_history: List[ResourceUsage] = []
        self.task_queue: List[TaskInfo] = []
        self.worker_pool: List[WorkerCapacity] = []
        
        # ML models
        self.usage_predictor = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Configuration
        self.scaling_thresholds = {
            ResourceType.CPU: {'scale_up': 75, 'scale_down': 25},
            ResourceType.MEMORY: {'scale_up': 80, 'scale_down': 30},
            ResourceType.WORKERS: {'scale_up': 80, 'scale_down': 20}
        }
        
        self.max_resources = {
            ResourceType.CPU: 100,  # percentage
            ResourceType.MEMORY: 100,  # percentage 
            ResourceType.WORKERS: 20,  # number of workers
            ResourceType.CONNECTIONS: 1000  # max connections
        }
        
        if RESOURCE_MANAGEMENT_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for resource prediction"""
        try:
            logger.info("Initializing intelligent resource management models...")
            
            # Resource usage predictor
            self.usage_predictor = RandomForestRegressor(
                n_estimators=50,
                random_state=42,
                max_depth=8
            )
            
            logger.info("Resource management models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing resource management models: {e}")
    
    async def collect_resource_metrics(self) -> ResourceUsage:
        """Collect current resource usage metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Disk I/O (simplified)
            disk_io = psutil.disk_io_counters()
            disk_io_mbps = (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024) if disk_io else 0
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_io_mbps = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024) if net_io else 0
            
            # Connection count (approximate)
            active_connections = len(psutil.net_connections())
            
            usage = ResourceUsage(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_io_mbps=disk_io_mbps,
                network_io_mbps=network_io_mbps,
                active_connections=active_connections,
                queue_length=len(self.task_queue),
                response_time_ms=100.0,  # Would be measured from application
                throughput_rps=10.0      # Would be measured from application
            )
            
            # Store in history
            self.usage_history.append(usage)
            
            # Keep only last 500 entries
            if len(self.usage_history) > 500:
                self.usage_history = self.usage_history[-500:]
            
            return usage
            
        except Exception as e:
            logger.error(f"Error collecting resource metrics: {e}")
            return self._get_default_usage()
    
    def _get_default_usage(self) -> ResourceUsage:
        """Get default resource usage when collection fails"""
        return ResourceUsage(
            timestamp=datetime.now(),
            cpu_percent=0.0,
            memory_percent=0.0,
            disk_io_mbps=0.0,
            network_io_mbps=0.0,
            active_connections=0,
            queue_length=0,
            response_time_ms=0.0,
            throughput_rps=0.0
        )
    
    async def predict_resource_needs(self, hours_ahead: int = 2) -> List[ResourcePrediction]:
        """Predict future resource needs and generate scaling recommendations"""
        if not RESOURCE_MANAGEMENT_AVAILABLE:
            return self._fallback_predictions()
        
        current_usage = await self.collect_resource_metrics()
        
        # Train model if needed
        if not self.is_trained and len(self.usage_history) >= 20:
            await self._train_prediction_model()
        
        predictions = []
        
        for resource_type in ResourceType:
            prediction = await self._predict_single_resource(
                resource_type, current_usage, hours_ahead
            )
            predictions.append(prediction)
        
        return predictions
    
    async def _predict_single_resource(
        self,
        resource_type: ResourceType,
        current_usage: ResourceUsage,
        hours_ahead: int
    ) -> ResourcePrediction:
        """Predict usage for a single resource type"""
        
        current_value = self._get_resource_value(resource_type, current_usage)
        predicted_value = current_value  # Default to current
        confidence = 0.5
        
        if self.is_trained and len(self.usage_history) >= 10:
            try:
                # Prepare features for prediction
                features = self._extract_features(current_usage)
                
                # Make prediction
                prediction = self.usage_predictor.predict([features])[0]
                predicted_value = max(0, min(100, prediction))  # Clamp to 0-100%
                confidence = 0.8
                
            except Exception as e:
                logger.warning(f"Prediction failed for {resource_type}: {e}")
        
        # Generate scaling recommendation
        recommendation, reasoning = self._generate_scaling_recommendation(
            resource_type, current_value, predicted_value
        )
        
        return ResourcePrediction(
            resource_type=resource_type,
            current_usage=current_value,
            predicted_usage=predicted_value,
            confidence=confidence,
            recommendation=recommendation,
            reasoning=reasoning,
            time_horizon=timedelta(hours=hours_ahead)
        )
    
    def _get_resource_value(self, resource_type: ResourceType, usage: ResourceUsage) -> float:
        """Extract specific resource value from usage data"""
        if resource_type == ResourceType.CPU:
            return usage.cpu_percent
        elif resource_type == ResourceType.MEMORY:
            return usage.memory_percent
        elif resource_type == ResourceType.DISK:
            return min(100, usage.disk_io_mbps * 10)  # Normalize disk I/O
        elif resource_type == ResourceType.NETWORK:
            return min(100, usage.network_io_mbps * 5)  # Normalize network I/O
        elif resource_type == ResourceType.CONNECTIONS:
            return min(100, usage.active_connections / 10)  # Normalize connections
        elif resource_type == ResourceType.WORKERS:
            return min(100, len(self.worker_pool) * 10)  # Normalize worker count
        
        return 0.0
    
    def _extract_features(self, usage: ResourceUsage) -> List[float]:
        """Extract features for ML prediction"""
        # Time-based features
        hour = usage.timestamp.hour
        day_of_week = usage.timestamp.weekday()
        
        # Usage features
        features = [
            usage.cpu_percent,
            usage.memory_percent,
            usage.disk_io_mbps,
            usage.network_io_mbps,
            usage.active_connections / 100.0,  # Normalize
            usage.queue_length / 10.0,         # Normalize
            usage.response_time_ms / 1000.0,   # Normalize
            usage.throughput_rps / 100.0,      # Normalize
            hour / 24.0,                       # Normalize hour
            day_of_week / 7.0                  # Normalize day
        ]
        
        # Historical trend features (if available)
        if len(self.usage_history) >= 5:
            recent_usage = self.usage_history[-5:]
            cpu_trend = np.mean([u.cpu_percent for u in recent_usage])
            memory_trend = np.mean([u.memory_percent for u in recent_usage])
            features.extend([cpu_trend, memory_trend])
        else:
            features.extend([usage.cpu_percent, usage.memory_percent])
        
        return features
    
    async def _train_prediction_model(self):
        """Train the resource usage prediction model"""
        if len(self.usage_history) < 20:
            return
        
        try:
            logger.info("Training resource usage prediction model...")
            
            # Prepare training data
            features = []
            targets = []
            
            for i in range(len(self.usage_history) - 1):
                current = self.usage_history[i]
                next_usage = self.usage_history[i + 1]
                
                feature_vector = self._extract_features(current)
                features.append(feature_vector)
                
                # Use CPU as primary target (can be extended for other resources)
                targets.append(next_usage.cpu_percent)
            
            X = np.array(features)
            y = np.array(targets)
            
            # Handle NaN values
            X = np.nan_to_num(X)
            y = np.nan_to_num(y)
            
            # Train the model
            self.usage_predictor.fit(X, y)
            self.is_trained = True
            
            logger.info("Resource usage prediction model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training resource usage model: {e}")
    
    def _generate_scaling_recommendation(
        self, 
        resource_type: ResourceType, 
        current: float, 
        predicted: float
    ) -> Tuple[ScalingAction, str]:
        """Generate scaling recommendation based on usage"""
        
        thresholds = self.scaling_thresholds.get(resource_type, {'scale_up': 75, 'scale_down': 25})
        
        if predicted >= thresholds['scale_up']:
            return ScalingAction.SCALE_UP, f"Predicted usage ({predicted:.1f}%) exceeds scale-up threshold ({thresholds['scale_up']}%)"
        
        elif predicted <= thresholds['scale_down'] and current > thresholds['scale_down']:
            return ScalingAction.SCALE_DOWN, f"Predicted usage ({predicted:.1f}%) below scale-down threshold ({thresholds['scale_down']}%)"
        
        elif abs(predicted - current) > 20:
            return ScalingAction.OPTIMIZE, f"Significant usage change predicted ({current:.1f}% -> {predicted:.1f}%)"
        
        else:
            return ScalingAction.MAINTAIN, f"Predicted usage ({predicted:.1f}%) within normal range"
    
    async def optimize_task_scheduling(self, new_tasks: List[TaskInfo] = None) -> Dict[str, Any]:
        """Optimize task scheduling across available resources"""
        
        if new_tasks:
            self.task_queue.extend(new_tasks)
        
        # Sort tasks by priority and deadline
        self.task_queue.sort(key=lambda t: (t.priority.value, t.deadline or datetime.max), reverse=True)
        
        # Get current resource state
        current_usage = await self.collect_resource_metrics()
        
        # Simulate scheduling
        scheduled_tasks = []
        resource_allocation = {
            ResourceType.CPU: current_usage.cpu_percent,
            ResourceType.MEMORY: current_usage.memory_percent
        }
        
        for task in self.task_queue[:10]:  # Process up to 10 tasks
            # Check if resources are available
            can_schedule = True
            for resource, requirement in task.resource_requirements.items():
                if resource_allocation.get(resource, 0) + requirement > 90:  # Leave 10% buffer
                    can_schedule = False
                    break
            
            if can_schedule:
                # Allocate resources
                for resource, requirement in task.resource_requirements.items():
                    resource_allocation[resource] = resource_allocation.get(resource, 0) + requirement
                
                scheduled_tasks.append(task)
                self.task_queue.remove(task)
        
        return {
            'scheduled_tasks': len(scheduled_tasks),
            'queued_tasks': len(self.task_queue),
            'resource_utilization': resource_allocation,
            'optimization_score': len(scheduled_tasks) / (len(scheduled_tasks) + len(self.task_queue)) if (len(scheduled_tasks) + len(self.task_queue)) > 0 else 0
        }
    
    async def generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate resource optimization recommendations"""
        
        recommendations = []
        current_usage = await self.collect_resource_metrics()
        predictions = await self.predict_resource_needs()
        
        for prediction in predictions:
            if prediction.recommendation in [ScalingAction.SCALE_UP, ScalingAction.SCALE_DOWN]:
                
                recommendation = OptimizationRecommendation(
                    resource_type=prediction.resource_type,
                    action=prediction.recommendation,
                    current_allocation=prediction.current_usage,
                    recommended_allocation=prediction.predicted_usage,
                    expected_improvement=self._calculate_improvement(prediction),
                    impact_score=self._calculate_impact_score(prediction),
                    implementation_steps=self._generate_implementation_steps(prediction)
                )
                
                recommendations.append(recommendation)
        
        return recommendations
    
    def _calculate_improvement(self, prediction: ResourcePrediction) -> str:
        """Calculate expected improvement from optimization"""
        if prediction.recommendation == ScalingAction.SCALE_UP:
            return f"Prevent performance degradation, maintain response times under 500ms"
        elif prediction.recommendation == ScalingAction.SCALE_DOWN:
            cost_saving = (prediction.current_usage - prediction.predicted_usage) * 0.01  # Simplified
            return f"Reduce costs by approximately {cost_saving:.1%} while maintaining performance"
        else:
            return "Optimize resource allocation for better efficiency"
    
    def _calculate_impact_score(self, prediction: ResourcePrediction) -> float:
        """Calculate impact score for recommendation (0-1)"""
        usage_diff = abs(prediction.predicted_usage - prediction.current_usage)
        confidence_factor = prediction.confidence
        urgency_factor = 1.0 if prediction.predicted_usage > 80 else 0.5
        
        return min(1.0, (usage_diff / 100.0) * confidence_factor * urgency_factor)
    
    def _generate_implementation_steps(self, prediction: ResourcePrediction) -> List[str]:
        """Generate implementation steps for recommendation"""
        steps = []
        
        if prediction.recommendation == ScalingAction.SCALE_UP:
            if prediction.resource_type == ResourceType.CPU:
                steps = [
                    "Monitor CPU utilization trends",
                    "Identify CPU-intensive processes",
                    "Consider vertical scaling (more CPU cores)",
                    "Implement horizontal scaling if applicable"
                ]
            elif prediction.resource_type == ResourceType.MEMORY:
                steps = [
                    "Analyze memory usage patterns", 
                    "Identify memory-intensive applications",
                    "Consider increasing memory allocation",
                    "Optimize memory usage in applications"
                ]
            elif prediction.resource_type == ResourceType.WORKERS:
                steps = [
                    "Increase worker pool size",
                    "Monitor task queue length",
                    "Optimize task distribution",
                    "Consider auto-scaling policies"
                ]
        
        elif prediction.recommendation == ScalingAction.SCALE_DOWN:
            steps = [
                f"Gradually reduce {prediction.resource_type.value} allocation",
                "Monitor performance metrics closely",
                "Implement graceful degradation",
                "Set up alerts for usage spikes"
            ]
        
        return steps
    
    async def auto_scale_resources(self, enable_auto_scaling: bool = True) -> Dict[str, Any]:
        """Perform automatic resource scaling based on predictions"""
        
        if not enable_auto_scaling:
            return {'status': 'auto-scaling disabled', 'actions': []}
        
        predictions = await self.predict_resource_needs()
        actions_taken = []
        
        for prediction in predictions:
            if prediction.confidence > 0.7 and prediction.recommendation != ScalingAction.MAINTAIN:
                
                # Simulate scaling action (in real system, this would trigger actual scaling)
                action = {
                    'resource': prediction.resource_type.value,
                    'action': prediction.recommendation.value,
                    'from': prediction.current_usage,
                    'to': prediction.predicted_usage,
                    'confidence': prediction.confidence,
                    'timestamp': datetime.now()
                }
                
                actions_taken.append(action)
                
                logger.info(f"Auto-scaling: {action}")
        
        return {
            'status': 'completed',
            'actions': actions_taken,
            'total_actions': len(actions_taken)
        }
    
    def _fallback_predictions(self) -> List[ResourcePrediction]:
        """Fallback predictions when ML models aren't available"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            predictions = [
                ResourcePrediction(
                    resource_type=ResourceType.CPU,
                    current_usage=cpu_percent,
                    predicted_usage=cpu_percent,
                    confidence=0.5,
                    recommendation=ScalingAction.MAINTAIN,
                    reasoning="Basic monitoring - ML models not available",
                    time_horizon=timedelta(hours=1)
                ),
                ResourcePrediction(
                    resource_type=ResourceType.MEMORY,
                    current_usage=memory_percent,
                    predicted_usage=memory_percent,
                    confidence=0.5,
                    recommendation=ScalingAction.MAINTAIN,
                    reasoning="Basic monitoring - ML models not available",
                    time_horizon=timedelta(hours=1)
                )
            ]
            
            return predictions
            
        except:
            return []
    
    def add_worker(self, worker_id: str, max_cpu: float = 100, max_memory: float = 100) -> bool:
        """Add a new worker to the pool"""
        worker = WorkerCapacity(
            worker_id=worker_id,
            max_cpu=max_cpu,
            max_memory=max_memory,
            current_load=0.0,
            active_tasks=0,
            avg_task_duration=0.0,
            availability_score=1.0
        )
        
        self.worker_pool.append(worker)
        logger.info(f"Added worker {worker_id} to resource pool")
        return True
    
    def remove_worker(self, worker_id: str) -> bool:
        """Remove a worker from the pool"""
        for worker in self.worker_pool:
            if worker.worker_id == worker_id:
                self.worker_pool.remove(worker)
                logger.info(f"Removed worker {worker_id} from resource pool")
                return True
        
        return False

# Global instance
intelligent_resource_manager = IntelligentResourceManager()

def get_intelligent_resource_manager() -> IntelligentResourceManager:
    """Get the global intelligent resource manager instance"""
    return intelligent_resource_manager
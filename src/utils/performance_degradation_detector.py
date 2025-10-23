"""
ML Model Performance Degradation Detection System
"""
import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    latency_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass 
class ModelPerformanceHistory:
    model_name: str
    metrics_history: List[PerformanceMetrics] = field(default_factory=list)
    baseline_metrics: Optional[PerformanceMetrics] = None
    
class PerformanceDegradationDetector:
    """Detect ML model performance degradation"""
    
    def __init__(self, history_file: str = "model_performance_history.json"):
        self.history_file = Path(history_file)
        self.model_histories: Dict[str, ModelPerformanceHistory] = {}
        self.degradation_thresholds = {
            "accuracy": 0.05,  # 5% drop
            "precision": 0.05,
            "recall": 0.05, 
            "f1_score": 0.05,
            "latency_ms": 0.20,  # 20% increase
            "memory_usage_mb": 0.30  # 30% increase
        }
        self.load_history()
    
    def load_history(self):
        """Load performance history from file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    
                for model_name, history_data in data.items():
                    metrics_list = []
                    for m in history_data.get("metrics_history", []):
                        metrics_list.append(PerformanceMetrics(**m))
                    
                    baseline = None
                    if history_data.get("baseline_metrics"):
                        baseline = PerformanceMetrics(**history_data["baseline_metrics"])
                    
                    self.model_histories[model_name] = ModelPerformanceHistory(
                        model_name=model_name,
                        metrics_history=metrics_list,
                        baseline_metrics=baseline
                    )
            except Exception as e:
                logger.warning(f"Failed to load performance history: {e}")
    
    def save_history(self):
        """Save performance history to file"""
        try:
            data = {}
            for model_name, history in self.model_histories.items():
                metrics_data = []
                for m in history.metrics_history:
                    metrics_data.append({
                        "accuracy": m.accuracy,
                        "precision": m.precision,
                        "recall": m.recall,
                        "f1_score": m.f1_score,
                        "latency_ms": m.latency_ms,
                        "memory_usage_mb": m.memory_usage_mb,
                        "timestamp": m.timestamp
                    })
                
                baseline_data = None
                if history.baseline_metrics:
                    baseline_data = {
                        "accuracy": history.baseline_metrics.accuracy,
                        "precision": history.baseline_metrics.precision,
                        "recall": history.baseline_metrics.recall,
                        "f1_score": history.baseline_metrics.f1_score,
                        "latency_ms": history.baseline_metrics.latency_ms,
                        "memory_usage_mb": history.baseline_metrics.memory_usage_mb,
                        "timestamp": history.baseline_metrics.timestamp
                    }
                
                data[model_name] = {
                    "metrics_history": metrics_data,
                    "baseline_metrics": baseline_data
                }
            
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save performance history: {e}")
    
    def record_performance(self, model_name: str, metrics: PerformanceMetrics):
        """Record new performance metrics"""
        if model_name not in self.model_histories:
            self.model_histories[model_name] = ModelPerformanceHistory(model_name=model_name)
        
        history = self.model_histories[model_name]
        history.metrics_history.append(metrics)
        
        # Set baseline if not set
        if history.baseline_metrics is None:
            history.baseline_metrics = metrics
            
        # Keep only last 100 records
        if len(history.metrics_history) > 100:
            history.metrics_history = history.metrics_history[-100:]
            
        self.save_history()
    
    def detect_degradation(self, model_name: str) -> Dict[str, Any]:
        """Detect performance degradation for a model"""
        if model_name not in self.model_histories:
            return {"status": "no_data", "message": "No performance history available"}
        
        history = self.model_histories[model_name]
        if not history.baseline_metrics or len(history.metrics_history) < 2:
            return {"status": "insufficient_data", "message": "Insufficient data for degradation detection"}
        
        recent_metrics = history.metrics_history[-5:]  # Last 5 measurements
        baseline = history.baseline_metrics
        
        degradation_detected = []
        
        for metric_name, threshold in self.degradation_thresholds.items():
            baseline_value = getattr(baseline, metric_name)
            if baseline_value is None:
                continue
                
            recent_values = [getattr(m, metric_name) for m in recent_metrics if getattr(m, metric_name) is not None]
            if not recent_values:
                continue
                
            avg_recent = np.mean(recent_values)
            
            # Check for degradation
            if metric_name in ["accuracy", "precision", "recall", "f1_score"]:
                # Lower is worse
                degradation_ratio = (baseline_value - avg_recent) / baseline_value
            else:
                # Higher is worse (latency, memory)
                degradation_ratio = (avg_recent - baseline_value) / baseline_value
            
            if degradation_ratio > threshold:
                degradation_detected.append({
                    "metric": metric_name,
                    "baseline_value": baseline_value,
                    "recent_average": avg_recent,
                    "degradation_ratio": degradation_ratio,
                    "threshold": threshold
                })
        
        if degradation_detected:
            return {
                "status": "degradation_detected",
                "model_name": model_name,
                "degraded_metrics": degradation_detected,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            return {
                "status": "healthy",
                "model_name": model_name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

# Global performance detector
performance_detector = PerformanceDegradationDetector()

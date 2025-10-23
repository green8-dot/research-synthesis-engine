"""
ML Model Validation Framework
Provides comprehensive model validation, performance monitoring, and drift detection
"""

import asyncio
import json
import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from functools import wraps

import joblib
import numpy as np
from loguru import logger

try:
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, 
        mean_squared_error, mean_absolute_error, r2_score,
        classification_report, confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available - some validation features disabled")
    SKLEARN_AVAILABLE = False


class ModelValidationMetrics:
    """Container for model validation metrics"""
    
    def __init__(self, model_type: str, validation_type: str):
        self.model_type = model_type
        self.validation_type = validation_type
        self.timestamp = datetime.now()
        self.metrics = {}
        self.meets_threshold = False
        self.recommendation = ""
        self.confidence = 0.0
        
    def add_metric(self, name: str, value: Union[float, int, str]):
        """Add a validation metric"""
        self.metrics[name] = value
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'model_type': self.model_type,
            'validation_type': self.validation_type,
            'timestamp': self.timestamp.isoformat(),
            'metrics': self.metrics,
            'meets_threshold': self.meets_threshold,
            'recommendation': self.recommendation,
            'confidence': self.confidence
        }


class ModelDriftDetector:
    """Detects model performance drift over time"""
    
    def __init__(self, window_size: int = 100, drift_threshold: float = 0.05):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.performance_history = deque(maxlen=window_size)
        self.baseline_performance = None
        
    def add_performance(self, performance_score: float):
        """Add new performance score to history"""
        self.performance_history.append({
            'score': performance_score,
            'timestamp': datetime.now()
        })
        
        if self.baseline_performance is None and len(self.performance_history) >= 10:
            # Set baseline after collecting initial data
            scores = [p['score'] for p in list(self.performance_history)[:10]]
            self.baseline_performance = sum(scores) / len(scores)
            
    def detect_drift(self) -> Dict[str, Any]:
        """Detect if model performance has drifted"""
        if not self.performance_history or self.baseline_performance is None:
            return {'drift_detected': False, 'reason': 'Insufficient data'}
            
        recent_scores = [p['score'] for p in list(self.performance_history)[-20:]]
        recent_average = sum(recent_scores) / len(recent_scores)
        
        drift_amount = self.baseline_performance - recent_average
        drift_detected = drift_amount > self.drift_threshold
        
        return {
            'drift_detected': drift_detected,
            'baseline_performance': self.baseline_performance,
            'recent_performance': recent_average,
            'drift_amount': drift_amount,
            'threshold': self.drift_threshold,
            'data_points': len(self.performance_history),
            'recommendation': self._get_drift_recommendation(drift_detected, drift_amount)
        }
        
    def _get_drift_recommendation(self, drift_detected: bool, drift_amount: float) -> str:
        """Get recommendation based on drift analysis"""
        if not drift_detected:
            return "Model performance stable - continue monitoring"
        elif drift_amount > self.drift_threshold * 2:
            return "Critical drift detected - immediate retraining required"
        else:
            return "Performance degradation detected - consider retraining"


class MLModelValidator:
    """Comprehensive ML model validation framework"""
    
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or Path("models/trained")
        self.validation_history = []
        self.drift_detectors = {}
        self.performance_thresholds = {
            'accuracy': 0.8,
            'precision': 0.75,
            'recall': 0.75,
            'f1_score': 0.75,
            'r2_score': 0.7,
            'mse': 0.1  # Lower is better
        }
        self.lock = threading.RLock()
        
        # Create model directory if needed
        if self.model_dir:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
    def validate_classification_model(self, model, X_test: np.ndarray, 
                                    y_test: np.ndarray, model_name: str = "unknown") -> ModelValidationMetrics:
        """Validate a classification model comprehensively"""
        
        metrics = ModelValidationMetrics(model_type="classification", validation_type="full_validation")
        
        try:
            if not SKLEARN_AVAILABLE:
                metrics.recommendation = "Scikit-learn not available - basic validation only"
                return metrics
                
            # Cross-validation
            cv_scores = cross_val_score(model, X_test, y_test, cv=min(5, len(X_test) // 10))
            metrics.add_metric('cv_mean', float(cv_scores.mean()))
            metrics.add_metric('cv_std', float(cv_scores.std()))
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Basic metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            metrics.add_metric('accuracy', float(accuracy))
            metrics.add_metric('precision', float(precision))
            metrics.add_metric('recall', float(recall))
            metrics.add_metric('f1_score', float(f1))
            
            # Detailed classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            metrics.add_metric('classification_report', class_report)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            metrics.add_metric('confusion_matrix', conf_matrix.tolist())
            
            # Determine if model meets quality thresholds
            meets_threshold = (
                accuracy >= self.performance_thresholds['accuracy'] and
                precision >= self.performance_thresholds['precision'] and
                recall >= self.performance_thresholds['recall'] and
                f1 >= self.performance_thresholds['f1_score']
            )
            
            metrics.meets_threshold = meets_threshold
            metrics.confidence = float(accuracy)
            metrics.recommendation = self._get_validation_recommendation(meets_threshold, accuracy)
            
            # Update drift detection
            self._update_drift_detection(model_name, accuracy)
            
            # Store validation history
            with self.lock:
                self.validation_history.append(metrics)
                
            logger.info(f"Classification model '{model_name}' validated - Accuracy: {accuracy:.3f}, Meets threshold: {meets_threshold}")
            
        except Exception as e:
            logger.error(f"Error validating classification model: {e}")
            metrics.recommendation = f"Validation failed: {str(e)}"
            
        return metrics
    
    def validate_regression_model(self, model, X_test: np.ndarray, 
                                y_test: np.ndarray, model_name: str = "unknown") -> ModelValidationMetrics:
        """Validate a regression model comprehensively"""
        
        metrics = ModelValidationMetrics(model_type="regression", validation_type="full_validation")
        
        try:
            if not SKLEARN_AVAILABLE:
                metrics.recommendation = "Scikit-learn not available - basic validation only"
                return metrics
                
            # Predictions
            y_pred = model.predict(X_test)
            
            # Regression metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            metrics.add_metric('mse', float(mse))
            metrics.add_metric('mae', float(mae))
            metrics.add_metric('r2_score', float(r2))
            metrics.add_metric('rmse', float(rmse))
            
            # Cross-validation for R[?]
            cv_r2_scores = cross_val_score(model, X_test, y_test, cv=min(5, len(X_test) // 10), scoring='r2')
            metrics.add_metric('cv_r2_mean', float(cv_r2_scores.mean()))
            metrics.add_metric('cv_r2_std', float(cv_r2_scores.std()))
            
            # Determine if model meets quality thresholds
            meets_threshold = (
                r2 >= self.performance_thresholds['r2_score'] and
                mse <= self.performance_thresholds['mse']
            )
            
            metrics.meets_threshold = meets_threshold
            metrics.confidence = float(max(0, r2))  # R[?] can be negative
            metrics.recommendation = self._get_validation_recommendation(meets_threshold, r2, "regression")
            
            # Update drift detection
            self._update_drift_detection(model_name, r2)
            
            # Store validation history
            with self.lock:
                self.validation_history.append(metrics)
                
            logger.info(f"Regression model '{model_name}' validated - R[?]: {r2:.3f}, MSE: {mse:.3f}, Meets threshold: {meets_threshold}")
            
        except Exception as e:
            logger.error(f"Error validating regression model: {e}")
            metrics.recommendation = f"Validation failed: {str(e)}"
            
        return metrics
    
    def validate_model_auto(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                           model_name: str = "unknown") -> ModelValidationMetrics:
        """Automatically detect model type and validate accordingly"""
        
        # Try to determine if it's classification or regression
        try:
            # Check if target values look like classes or continuous values
            unique_values = np.unique(y_test)
            
            # Heuristic: if few unique values and they're integers, likely classification
            if len(unique_values) <= 20 and np.all(unique_values == unique_values.astype(int)):
                return self.validate_classification_model(model, X_test, y_test, model_name)
            else:
                return self.validate_regression_model(model, X_test, y_test, model_name)
                
        except Exception as e:
            logger.warning(f"Could not auto-detect model type, defaulting to classification: {e}")
            return self.validate_classification_model(model, X_test, y_test, model_name)
    
    def _update_drift_detection(self, model_name: str, performance_score: float):
        """Update drift detection for a model"""
        if model_name not in self.drift_detectors:
            self.drift_detectors[model_name] = ModelDriftDetector()
            
        self.drift_detectors[model_name].add_performance(performance_score)
    
    def _get_validation_recommendation(self, meets_threshold: bool, score: float, model_type: str = "classification") -> str:
        """Get recommendation based on validation results"""
        if meets_threshold:
            if score > 0.95:
                return "Excellent model performance - ready for production"
            elif score > 0.85:
                return "Good model performance - suitable for production"
            else:
                return "Acceptable performance - monitor closely"
        else:
            if score < 0.5:
                return "Poor performance - requires significant improvement or different approach"
            elif score < 0.7:
                return "Below threshold - consider additional training data or feature engineering"
            else:
                return "Close to threshold - minor improvements needed"
    
    def check_model_drift(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Check for performance drift in a specific model"""
        if model_name not in self.drift_detectors:
            return None
            
        return self.drift_detectors[model_name].detect_drift()
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation activities"""
        with self.lock:
            if not self.validation_history:
                return {'status': 'no_validations', 'total_validations': 0}
                
            recent_validations = [v for v in self.validation_history 
                                if (datetime.now() - v.timestamp).days < 7]
            
            passed_validations = len([v for v in recent_validations if v.meets_threshold])
            
            summary = {
                'total_validations': len(self.validation_history),
                'recent_validations': len(recent_validations),
                'passed_validations': passed_validations,
                'success_rate': (passed_validations / max(len(recent_validations), 1)) * 100,
                'model_types': list(set(v.model_type for v in self.validation_history)),
                'drift_status': {}
            }
            
            # Add drift status for each model
            for model_name, detector in self.drift_detectors.items():
                drift_info = detector.detect_drift()
                summary['drift_status'][model_name] = drift_info
                
            return summary
    
    def save_validation_report(self, output_path: Optional[Path] = None) -> Path:
        """Save comprehensive validation report to file"""
        output_path = output_path or (self.model_dir / "validation_report.json")
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': self.get_validation_summary(),
            'validation_history': [v.to_dict() for v in self.validation_history],
            'thresholds': self.performance_thresholds
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Validation report saved to {output_path}")
        return output_path
    
    def model_validation_decorator(self, model_name: str, test_data_func: Optional[callable] = None):
        """Decorator to automatically validate model after training/loading"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Execute the function (train/load model)
                result = await func(*args, **kwargs)
                
                # Try to validate if test data is available
                if test_data_func:
                    try:
                        X_test, y_test = test_data_func()
                        if result is not None and X_test is not None and y_test is not None:
                            validation_result = self.validate_model_auto(result, X_test, y_test, model_name)
                            logger.info(f"Auto-validation completed for {model_name}: {validation_result.recommendation}")
                    except Exception as e:
                        logger.warning(f"Auto-validation failed for {model_name}: {e}")
                
                return result
                
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Execute the function (train/load model)
                result = func(*args, **kwargs)
                
                # Try to validate if test data is available
                if test_data_func:
                    try:
                        X_test, y_test = test_data_func()
                        if result is not None and X_test is not None and y_test is not None:
                            validation_result = self.validate_model_auto(result, X_test, y_test, model_name)
                            logger.info(f"Auto-validation completed for {model_name}: {validation_result.recommendation}")
                    except Exception as e:
                        logger.warning(f"Auto-validation failed for {model_name}: {e}")
                
                return result
            
            # Return appropriate wrapper
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator


# Global model validator instance
model_validator = MLModelValidator()


class ModelValidatorHealthCheck:
    """Health check utilities for model validation system"""
    
    @staticmethod
    def get_health_status() -> Dict[str, Any]:
        """Get health status of the model validation system"""
        try:
            summary = model_validator.get_validation_summary()
            
            health_score = 100
            issues = []
            
            # Check validation success rate
            success_rate = summary.get('success_rate', 0)
            if success_rate < 70:
                health_score -= 30
                issues.append(f"Low validation success rate: {success_rate:.1f}%")
            elif success_rate < 85:
                health_score -= 15
                issues.append(f"Moderate validation success rate: {success_rate:.1f}%")
            
            # Check for drift issues
            drift_issues = 0
            for model_name, drift_info in summary.get('drift_status', {}).items():
                if drift_info.get('drift_detected', False):
                    drift_issues += 1
                    issues.append(f"Performance drift detected in {model_name}")
            
            if drift_issues > 0:
                health_score -= drift_issues * 20
            
            # Check recent validation activity
            recent_validations = summary.get('recent_validations', 0)
            if recent_validations == 0:
                health_score -= 20
                issues.append("No recent model validations")
            
            status = "healthy" if health_score >= 80 else "degraded" if health_score >= 60 else "critical"
            
            return {
                'status': status,
                'health_score': max(0, health_score),
                'sklearn_available': SKLEARN_AVAILABLE,
                'total_validations': summary.get('total_validations', 0),
                'recent_validations': recent_validations,
                'success_rate': success_rate,
                'drift_issues': drift_issues,
                'issues': issues,
                'supported_features': [
                    'Classification validation',
                    'Regression validation',
                    'Auto model type detection',
                    'Performance drift detection',
                    'Cross-validation',
                    'Automated validation decorators'
                ] if SKLEARN_AVAILABLE else ['Basic validation only - install scikit-learn for full features']
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'health_score': 0,
                'error': str(e),
                'sklearn_available': SKLEARN_AVAILABLE
            }


# Export validator health checker
validator_health = ModelValidatorHealthCheck()
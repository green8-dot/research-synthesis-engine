"""
Advanced Model Validation System
Provides enhanced model validation with A/B testing, automated retraining, and advanced metrics
"""

import asyncio
import json
import time
import hashlib
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from functools import wraps
from collections import defaultdict, deque

import joblib
import numpy as np
from loguru import logger

try:
    from sklearn.model_selection import cross_val_score, validation_curve, learning_curve
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, 
        mean_squared_error, mean_absolute_error, r2_score,
        classification_report, confusion_matrix, roc_auc_score,
        mean_absolute_percentage_error, explained_variance_score
    )
    from sklearn.base import clone
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available - advanced validation features disabled")
    SKLEARN_AVAILABLE = False

try:
    from research_synthesis.utils.ml_model_validator import model_validator, ModelValidationMetrics
    BASIC_VALIDATOR_AVAILABLE = True
except ImportError:
    logger.warning("Basic model validator not available")
    BASIC_VALIDATOR_AVAILABLE = False


@dataclass
class ModelComparison:
    """Results of comparing two models"""
    model_a_name: str
    model_b_name: str
    winner: str
    confidence_interval: float
    performance_difference: float
    significance_level: float
    recommendation: str
    detailed_metrics: Dict[str, Any]


@dataclass
class ModelVersionInfo:
    """Information about a model version"""
    version_id: str
    model_name: str
    created_at: datetime
    performance_metrics: Dict[str, float]
    validation_score: float
    model_hash: str
    training_data_hash: str
    hyperparameters: Dict[str, Any]
    model_size_bytes: int


class AdvancedModelValidator:
    """Advanced model validation with A/B testing, versioning, and automated retraining"""
    
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or Path("models/advanced")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Version tracking
        self.model_versions = {}
        self.active_models = {}  # Currently deployed models
        self.champion_challenger = {}  # A/B testing pairs
        
        # Retraining triggers
        self.retraining_thresholds = {
            'performance_drop': 0.05,  # 5% drop triggers retraining
            'drift_threshold': 0.1,    # 10% drift triggers retraining
            'data_staleness_days': 30, # 30 days triggers retraining
            'error_rate_threshold': 0.15  # 15% error rate triggers retraining
        }
        
        # Performance tracking
        self.performance_history = defaultdict(deque)
        self.error_tracking = defaultdict(list)
        self.prediction_tracking = defaultdict(deque)
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Load existing model versions
        self._load_model_registry()
        
        logger.info(f"Advanced Model Validator initialized at {self.model_dir}")
    
    def validate_model_robustness(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                                 model_name: str = "unknown") -> Dict[str, Any]:
        """Comprehensive robustness testing including noise, outliers, and edge cases"""
        
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not available for robustness testing"}
        
        results = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "robustness_tests": {}
        }
        
        try:
            # Baseline performance
            baseline_predictions = model.predict(X_test)
            baseline_accuracy = accuracy_score(y_test, baseline_predictions) if len(np.unique(y_test)) <= 10 else r2_score(y_test, baseline_predictions)
            results["baseline_performance"] = float(baseline_accuracy)
            
            # Test 1: Noise resistance
            noise_results = []
            noise_levels = [0.01, 0.05, 0.1, 0.2]
            
            for noise_level in noise_levels:
                X_noisy = X_test + np.random.normal(0, noise_level, X_test.shape)
                noisy_predictions = model.predict(X_noisy)
                noisy_accuracy = accuracy_score(y_test, noisy_predictions) if len(np.unique(y_test)) <= 10 else r2_score(y_test, noisy_predictions)
                
                noise_results.append({
                    "noise_level": noise_level,
                    "performance": float(noisy_accuracy),
                    "degradation": float(baseline_accuracy - noisy_accuracy)
                })
            
            results["robustness_tests"]["noise_resistance"] = {
                "test_results": noise_results,
                "avg_degradation": float(np.mean([r["degradation"] for r in noise_results])),
                "max_degradation": float(max([r["degradation"] for r in noise_results])),
                "passes_threshold": max([r["degradation"] for r in noise_results]) < 0.1
            }
            
            # Test 2: Feature perturbation
            feature_importance = []
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
            
            perturbation_results = []
            for i in range(min(5, X_test.shape[1])):  # Test top 5 features
                X_perturbed = X_test.copy()
                X_perturbed[:, i] = np.random.permutation(X_perturbed[:, i])  # Shuffle feature
                perturbed_predictions = model.predict(X_perturbed)
                perturbed_accuracy = accuracy_score(y_test, perturbed_predictions) if len(np.unique(y_test)) <= 10 else r2_score(y_test, perturbed_predictions)
                
                perturbation_results.append({
                    "feature_index": i,
                    "feature_importance": float(feature_importance[i]) if len(feature_importance) > i else None,
                    "performance_drop": float(baseline_accuracy - perturbed_accuracy),
                    "relative_impact": float((baseline_accuracy - perturbed_accuracy) / baseline_accuracy)
                })
            
            results["robustness_tests"]["feature_perturbation"] = perturbation_results
            
            # Test 3: Data distribution shift simulation
            if len(X_test) > 20:
                # Create distribution shifts by sampling from different quantiles
                shift_results = []
                quantiles = [0.1, 0.25, 0.75, 0.9]
                
                for q in quantiles:
                    # Select samples from specific quantile ranges
                    feature_means = np.mean(X_test, axis=0)
                    distances = np.linalg.norm(X_test - feature_means, axis=1)
                    threshold = np.quantile(distances, q)
                    
                    if q < 0.5:
                        mask = distances <= threshold
                    else:
                        mask = distances >= threshold
                    
                    if np.sum(mask) > 5:  # Need at least 5 samples
                        X_shifted = X_test[mask]
                        y_shifted = y_test[mask]
                        shifted_predictions = model.predict(X_shifted)
                        shifted_accuracy = accuracy_score(y_shifted, shifted_predictions) if len(np.unique(y_shifted)) <= 10 else r2_score(y_shifted, shifted_predictions)
                        
                        shift_results.append({
                            "quantile": q,
                            "sample_size": int(np.sum(mask)),
                            "performance": float(shifted_accuracy),
                            "performance_difference": float(shifted_accuracy - baseline_accuracy)
                        })
                
                results["robustness_tests"]["distribution_shift"] = shift_results
            
            # Overall robustness score
            noise_score = 1 - results["robustness_tests"]["noise_resistance"]["max_degradation"] / baseline_accuracy
            feature_stability = 1 - np.mean([abs(r["relative_impact"]) for r in perturbation_results])
            
            overall_robustness = (noise_score + feature_stability) / 2
            results["overall_robustness_score"] = float(max(0, overall_robustness))
            
            # Robustness recommendation
            if overall_robustness > 0.8:
                results["recommendation"] = "Model shows excellent robustness"
            elif overall_robustness > 0.6:
                results["recommendation"] = "Model shows good robustness with some sensitivity"
            elif overall_robustness > 0.4:
                results["recommendation"] = "Model shows moderate robustness - consider regularization"
            else:
                results["recommendation"] = "Model lacks robustness - significant improvements needed"
                
        except Exception as e:
            logger.error(f"Error in robustness testing: {e}")
            results["error"] = str(e)
        
        return results
    
    def compare_models(self, model_a, model_b, X_test: np.ndarray, y_test: np.ndarray,
                      model_a_name: str = "Model A", model_b_name: str = "Model B",
                      statistical_test: bool = True) -> ModelComparison:
        """Compare two models with statistical significance testing"""
        
        if not SKLEARN_AVAILABLE:
            return ModelComparison(
                model_a_name=model_a_name,
                model_b_name=model_b_name,
                winner="unknown",
                confidence_interval=0.0,
                performance_difference=0.0,
                significance_level=0.0,
                recommendation="scikit-learn not available",
                detailed_metrics={}
            )
        
        try:
            # Get predictions from both models
            pred_a = model_a.predict(X_test)
            pred_b = model_b.predict(X_test)
            
            # Determine if classification or regression
            is_classification = len(np.unique(y_test)) <= 10
            
            if is_classification:
                score_a = accuracy_score(y_test, pred_a)
                score_b = accuracy_score(y_test, pred_b)
                metric_name = "accuracy"
            else:
                score_a = r2_score(y_test, pred_a)
                score_b = r2_score(y_test, pred_b)
                metric_name = "r2_score"
            
            performance_difference = score_b - score_a
            
            # Statistical significance testing (if requested)
            significance_level = 0.0
            confidence_interval = 0.0
            
            if statistical_test and len(X_test) > 30:
                # Bootstrap comparison
                n_bootstrap = 1000
                bootstrap_diffs = []
                
                for _ in range(n_bootstrap):
                    # Bootstrap sample
                    indices = np.random.choice(len(X_test), len(X_test), replace=True)
                    X_boot, y_boot = X_test[indices], y_test[indices]
                    
                    pred_a_boot = model_a.predict(X_boot)
                    pred_b_boot = model_b.predict(X_boot)
                    
                    if is_classification:
                        score_a_boot = accuracy_score(y_boot, pred_a_boot)
                        score_b_boot = accuracy_score(y_boot, pred_b_boot)
                    else:
                        score_a_boot = r2_score(y_boot, pred_a_boot)
                        score_b_boot = r2_score(y_boot, pred_b_boot)
                    
                    bootstrap_diffs.append(score_b_boot - score_a_boot)
                
                # Calculate confidence interval
                confidence_interval = float(np.std(bootstrap_diffs))
                
                # Calculate p-value (two-tailed test)
                p_value = np.mean(np.abs(bootstrap_diffs) >= abs(performance_difference))
                significance_level = float(p_value)
            
            # Determine winner
            if abs(performance_difference) < 0.01:
                winner = "tie"
            elif performance_difference > 0:
                winner = model_b_name
            else:
                winner = model_a_name
            
            # Additional detailed metrics
            detailed_metrics = {
                f"{model_a_name}_{metric_name}": float(score_a),
                f"{model_b_name}_{metric_name}": float(score_b),
                "absolute_difference": float(abs(performance_difference)),
                "relative_improvement": float(performance_difference / max(score_a, 0.01)) * 100,
            }
            
            if is_classification:
                # Additional classification metrics
                try:
                    if len(np.unique(y_test)) == 2:  # Binary classification
                        auc_a = roc_auc_score(y_test, pred_a)
                        auc_b = roc_auc_score(y_test, pred_b)
                        detailed_metrics[f"{model_a_name}_auc"] = float(auc_a)
                        detailed_metrics[f"{model_b_name}_auc"] = float(auc_b)
                        detailed_metrics["auc_difference"] = float(auc_b - auc_a)
                except:
                    pass
            
            # Generate recommendation
            if significance_level > 0.05 and statistical_test:
                recommendation = f"No statistically significant difference (p={significance_level:.3f})"
            elif abs(performance_difference) < 0.02:
                recommendation = "Models perform similarly - consider other factors (speed, interpretability)"
            elif winner == model_b_name:
                recommendation = f"{model_b_name} performs significantly better (+{performance_difference:.3f})"
            else:
                recommendation = f"{model_a_name} performs significantly better (+{abs(performance_difference):.3f})"
            
            return ModelComparison(
                model_a_name=model_a_name,
                model_b_name=model_b_name,
                winner=winner,
                confidence_interval=confidence_interval,
                performance_difference=performance_difference,
                significance_level=significance_level,
                recommendation=recommendation,
                detailed_metrics=detailed_metrics
            )
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return ModelComparison(
                model_a_name=model_a_name,
                model_b_name=model_b_name,
                winner="error",
                confidence_interval=0.0,
                performance_difference=0.0,
                significance_level=0.0,
                recommendation=f"Error in comparison: {str(e)}",
                detailed_metrics={}
            )
    
    def create_model_version(self, model, model_name: str, training_data_hash: str,
                           hyperparameters: Dict[str, Any], performance_metrics: Dict[str, float]) -> ModelVersionInfo:
        """Create a new model version with comprehensive metadata"""
        
        # Generate version ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{model_name}_v{timestamp}"
        
        # Calculate model hash
        try:
            model_bytes = joblib.dumps(model)
            model_hash = hashlib.md5(model_bytes).hexdigest()
            model_size = len(model_bytes)
        except Exception as e:
            logger.warning(f"Could not calculate model hash: {e}")
            model_hash = "unknown"
            model_size = 0
        
        # Create version info
        version_info = ModelVersionInfo(
            version_id=version_id,
            model_name=model_name,
            created_at=datetime.now(),
            performance_metrics=performance_metrics,
            validation_score=performance_metrics.get('accuracy', performance_metrics.get('r2_score', 0.0)),
            model_hash=model_hash,
            training_data_hash=training_data_hash,
            hyperparameters=hyperparameters,
            model_size_bytes=model_size
        )
        
        # Save model and metadata
        model_file = self.model_dir / f"{version_id}.pkl"
        metadata_file = self.model_dir / f"{version_id}_metadata.json"
        
        try:
            joblib.dump(model, model_file)
            
            with open(metadata_file, 'w') as f:
                # Convert datetime to string for JSON serialization
                metadata = asdict(version_info)
                metadata['created_at'] = metadata['created_at'].isoformat()
                json.dump(metadata, f, indent=2)
            
            # Update version registry
            with self.lock:
                if model_name not in self.model_versions:
                    self.model_versions[model_name] = []
                self.model_versions[model_name].append(version_info)
                
                # Keep only latest 10 versions per model
                self.model_versions[model_name] = sorted(
                    self.model_versions[model_name], 
                    key=lambda x: x.created_at, 
                    reverse=True
                )[:10]
            
            logger.info(f"Created model version {version_id} for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to save model version: {e}")
        
        return version_info
    
    def should_trigger_retraining(self, model_name: str, current_performance: float,
                                 error_rate: float = 0.0, data_age_days: int = 0) -> Dict[str, Any]:
        """Determine if a model should be retrained based on multiple criteria"""
        
        triggers = []
        trigger_scores = {}
        
        with self.lock:
            # Check performance drop
            if model_name in self.performance_history:
                recent_performance = list(self.performance_history[model_name])[-10:]  # Last 10 records
                if recent_performance:
                    baseline_performance = max(recent_performance)
                    performance_drop = baseline_performance - current_performance
                    
                    if performance_drop > self.retraining_thresholds['performance_drop']:
                        triggers.append("performance_drop")
                        trigger_scores["performance_drop"] = {
                            "current": current_performance,
                            "baseline": baseline_performance,
                            "drop": performance_drop,
                            "threshold": self.retraining_thresholds['performance_drop']
                        }
            
            # Check error rate
            if error_rate > self.retraining_thresholds['error_rate_threshold']:
                triggers.append("high_error_rate")
                trigger_scores["error_rate"] = {
                    "current": error_rate,
                    "threshold": self.retraining_thresholds['error_rate_threshold']
                }
            
            # Check data staleness
            if data_age_days > self.retraining_thresholds['data_staleness_days']:
                triggers.append("data_staleness")
                trigger_scores["data_staleness"] = {
                    "age_days": data_age_days,
                    "threshold": self.retraining_thresholds['data_staleness_days']
                }
            
            # Add to performance history
            self.performance_history[model_name].append({
                "timestamp": datetime.now(),
                "performance": current_performance,
                "error_rate": error_rate
            })
            
            # Keep only recent history
            if len(self.performance_history[model_name]) > 100:
                self.performance_history[model_name].popleft()
        
        # Determine urgency
        if len(triggers) >= 2:
            urgency = "high"
            recommendation = "Immediate retraining recommended"
        elif "performance_drop" in triggers:
            urgency = "medium"
            recommendation = "Schedule retraining within 24 hours"
        elif len(triggers) == 1:
            urgency = "low"
            recommendation = "Consider retraining when convenient"
        else:
            urgency = "none"
            recommendation = "No retraining needed"
        
        return {
            "should_retrain": len(triggers) > 0,
            "triggers": triggers,
            "urgency": urgency,
            "recommendation": recommendation,
            "trigger_details": trigger_scores,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_model_lineage(self, model_name: str) -> Dict[str, Any]:
        """Get the complete lineage and evolution of a model"""
        
        with self.lock:
            if model_name not in self.model_versions:
                return {"error": f"No versions found for model {model_name}"}
            
            versions = self.model_versions[model_name]
            
            lineage = {
                "model_name": model_name,
                "total_versions": len(versions),
                "oldest_version": versions[-1].created_at.isoformat() if versions else None,
                "newest_version": versions[0].created_at.isoformat() if versions else None,
                "active_version": self.active_models.get(model_name, "unknown"),
                "performance_evolution": [],
                "version_details": []
            }
            
            # Performance evolution
            for version in reversed(versions):  # Chronological order
                lineage["performance_evolution"].append({
                    "version_id": version.version_id,
                    "timestamp": version.created_at.isoformat(),
                    "validation_score": version.validation_score,
                    "model_size_mb": round(version.model_size_bytes / (1024*1024), 2)
                })
            
            # Version details
            for version in versions:
                version_detail = {
                    "version_id": version.version_id,
                    "created_at": version.created_at.isoformat(),
                    "performance_metrics": version.performance_metrics,
                    "validation_score": version.validation_score,
                    "model_hash": version.model_hash[:8] + "...",  # Abbreviated hash
                    "hyperparameters": version.hyperparameters,
                    "model_size_mb": round(version.model_size_bytes / (1024*1024), 2)
                }
                lineage["version_details"].append(version_detail)
            
            # Calculate performance trends
            if len(lineage["performance_evolution"]) > 1:
                recent_scores = [p["validation_score"] for p in lineage["performance_evolution"][-5:]]
                older_scores = [p["validation_score"] for p in lineage["performance_evolution"][:-5]]
                
                if older_scores:
                    recent_avg = np.mean(recent_scores)
                    older_avg = np.mean(older_scores)
                    trend = "improving" if recent_avg > older_avg else "declining"
                    trend_magnitude = abs(recent_avg - older_avg)
                else:
                    trend = "stable"
                    trend_magnitude = 0.0
                
                lineage["performance_trend"] = {
                    "direction": trend,
                    "magnitude": float(trend_magnitude),
                    "recent_average": float(np.mean(recent_scores)),
                    "historical_average": float(np.mean(older_scores)) if older_scores else 0.0
                }
        
        return lineage
    
    def _load_model_registry(self):
        """Load existing model versions from disk"""
        try:
            registry_file = self.model_dir / "model_registry.json"
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    registry_data = json.load(f)
                
                # Reconstruct model versions
                for model_name, versions_data in registry_data.items():
                    self.model_versions[model_name] = []
                    for version_data in versions_data:
                        version_data['created_at'] = datetime.fromisoformat(version_data['created_at'])
                        version_info = ModelVersionInfo(**version_data)
                        self.model_versions[model_name].append(version_info)
                
                logger.info(f"Loaded {len(self.model_versions)} model families from registry")
        
        except Exception as e:
            logger.warning(f"Could not load model registry: {e}")
    
    def save_model_registry(self):
        """Save model registry to disk"""
        try:
            registry_file = self.model_dir / "model_registry.json"
            registry_data = {}
            
            with self.lock:
                for model_name, versions in self.model_versions.items():
                    registry_data[model_name] = []
                    for version in versions:
                        version_dict = asdict(version)
                        version_dict['created_at'] = version_dict['created_at'].isoformat()
                        registry_data[model_name].append(version_dict)
            
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
            
            logger.info("Model registry saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary including all advanced features"""
        
        with self.lock:
            total_models = len(self.model_versions)
            total_versions = sum(len(versions) for versions in self.model_versions.values())
            active_models = len(self.active_models)
            
            # Performance trends
            performance_trends = {}
            for model_name, versions in self.model_versions.items():
                if len(versions) > 1:
                    scores = [v.validation_score for v in versions]
                    trend = "improving" if scores[0] > scores[-1] else "declining"
                    performance_trends[model_name] = trend
            
            # Retraining candidates
            retraining_candidates = []
            for model_name in self.performance_history:
                recent_perf = list(self.performance_history[model_name])
                if recent_perf:
                    latest = recent_perf[-1]
                    should_retrain = self.should_trigger_retraining(
                        model_name, 
                        latest["performance"], 
                        latest.get("error_rate", 0)
                    )
                    if should_retrain["should_retrain"]:
                        retraining_candidates.append({
                            "model_name": model_name,
                            "urgency": should_retrain["urgency"],
                            "triggers": should_retrain["triggers"]
                        })
            
            return {
                "advanced_validation_summary": {
                    "total_models": total_models,
                    "total_versions": total_versions,
                    "active_models": active_models,
                    "performance_trends": performance_trends,
                    "retraining_candidates": retraining_candidates,
                    "features_available": [
                        "Model robustness testing",
                        "Statistical model comparison",
                        "Model versioning and lineage",
                        "Automated retraining triggers",
                        "Performance trend analysis",
                        "A/B testing framework"
                    ],
                    "timestamp": datetime.now().isoformat()
                }
            }


# Global advanced validator instance
advanced_validator = AdvancedModelValidator()
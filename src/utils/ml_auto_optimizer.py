"""
ML Auto-Optimization System
Automatically tunes hyperparameters, selects optimal algorithms, and optimizes model performance
"""

import asyncio
import json
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from functools import wraps
from collections import defaultdict, deque
import threading

import numpy as np
from loguru import logger

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from sklearn.model_selection import (
        GridSearchCV, RandomizedSearchCV, cross_val_score,
        StratifiedKFold, KFold
    )
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        RandomForestRegressor, GradientBoostingRegressor,
        ExtraTreesClassifier, ExtraTreesRegressor
    )
    from sklearn.linear_model import (
        LogisticRegression, LinearRegression, Ridge, Lasso,
        ElasticNet
    )
    from sklearn.svm import SVC, SVR
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        mean_squared_error, r2_score, mean_absolute_error
    )
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.feature_selection import (
        SelectKBest, f_classif, f_regression, RFE,
        SelectFromModel, mutual_info_classif, mutual_info_regression
    )
    from sklearn.pipeline import Pipeline
    from sklearn.base import clone
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available - ML auto-optimization features disabled")
    SKLEARN_AVAILABLE = False

try:
    from research_synthesis.utils.ml_model_validator import model_validator
    VALIDATOR_AVAILABLE = True
except ImportError:
    logger.warning("Model validator not available for auto-optimization")
    VALIDATOR_AVAILABLE = False


@dataclass
class OptimizationResult:
    """Result of ML optimization process"""
    algorithm_name: str
    best_score: float
    best_params: Dict[str, Any]
    optimization_time: float
    trials_completed: int
    improvement_over_baseline: float
    final_model: Any
    validation_metrics: Dict[str, float]
    feature_importance: List[Tuple[str, float]]
    optimization_history: List[Dict[str, Any]]


@dataclass
class AutoMLConfig:
    """Configuration for AutoML optimization"""
    algorithms_to_try: List[str]
    max_optimization_time: int  # seconds
    cv_folds: int
    scoring_metric: str
    feature_selection: bool
    preprocessing: bool
    ensemble_methods: bool
    early_stopping: bool
    n_trials: int


class MLAutoOptimizer:
    """Automated ML optimization system for hyperparameter tuning and algorithm selection"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("models/optimization_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Algorithm configurations
        self.classification_algorithms = {
            'random_forest': {
                'model': RandomForestClassifier,
                'param_space': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'param_space': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'extra_trees': {
                'model': ExtraTreesClassifier,
                'param_space': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression,
                'param_space': {
                    'C': [0.01, 0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'svm': {
                'model': SVC,
                'param_space': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'knn': {
                'model': KNeighborsClassifier,
                'param_space': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            }
        }
        
        self.regression_algorithms = {
            'random_forest': {
                'model': RandomForestRegressor,
                'param_space': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor,
                'param_space': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'linear_regression': {
                'model': LinearRegression,
                'param_space': {}
            },
            'ridge': {
                'model': Ridge,
                'param_space': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            'lasso': {
                'model': Lasso,
                'param_space': {
                    'alpha': [0.01, 0.1, 1.0, 10.0]
                }
            },
            'elastic_net': {
                'model': ElasticNet,
                'param_space': {
                    'alpha': [0.01, 0.1, 1.0],
                    'l1_ratio': [0.1, 0.5, 0.9]
                }
            },
            'knn': {
                'model': KNeighborsRegressor,
                'param_space': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance']
                }
            }
        }
        
        # Preprocessing options
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        # Feature selection methods
        self.feature_selectors = {
            'k_best': SelectKBest,
            'rfe': RFE,
            'model_based': SelectFromModel
        }
        
        # Optimization history
        self.optimization_history = defaultdict(list)
        self.best_models = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"ML Auto-Optimizer initialized with cache at {self.cache_dir}")
    
    def optimize_model(self, X: np.ndarray, y: np.ndarray, 
                      problem_type: str = 'auto',
                      config: Optional[AutoMLConfig] = None,
                      model_name: str = "optimized_model") -> OptimizationResult:
        """
        Automatically optimize ML model for given dataset
        
        Args:
            X: Feature matrix
            y: Target vector
            problem_type: 'classification', 'regression', or 'auto'
            config: Optimization configuration
            model_name: Name for the optimized model
        """
        
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not available for ML optimization")
        
        start_time = time.time()
        
        # Auto-detect problem type
        if problem_type == 'auto':
            unique_targets = len(np.unique(y))
            if unique_targets <= 20 and np.all(y == y.astype(int)):
                problem_type = 'classification'
            else:
                problem_type = 'regression'
        
        logger.info(f"Starting ML optimization for {problem_type} problem with {X.shape[0]} samples, {X.shape[1]} features")
        
        # Set default config
        if config is None:
            config = self._get_default_config(problem_type)
        
        # Get algorithms to try
        algorithms = (self.classification_algorithms if problem_type == 'classification' 
                     else self.regression_algorithms)
        
        if config.algorithms_to_try:
            algorithms = {k: v for k, v in algorithms.items() if k in config.algorithms_to_try}
        
        # Initialize tracking
        optimization_history = []
        best_score = -np.inf if problem_type == 'classification' else np.inf
        best_model = None
        best_params = {}
        best_algorithm = ""
        trials_completed = 0
        
        # Preprocessing pipeline optimization
        preprocessors = self._optimize_preprocessing(X, y, problem_type, config)
        
        # Feature selection optimization
        feature_selectors = self._optimize_feature_selection(X, y, problem_type, config)
        
        # Algorithm optimization
        for algo_name, algo_config in algorithms.items():
            if time.time() - start_time > config.max_optimization_time:
                logger.warning("Optimization time limit reached")
                break
            
            logger.info(f"Optimizing {algo_name}...")
            
            try:
                # Create pipeline
                pipeline_steps = []
                
                # Add preprocessing
                if config.preprocessing and preprocessors:
                    pipeline_steps.append(('scaler', preprocessors[0]))
                
                # Add feature selection
                if config.feature_selection and feature_selectors:
                    pipeline_steps.append(('selector', feature_selectors[0]))
                
                # Add model
                pipeline_steps.append(('model', algo_config['model']()))
                
                pipeline = Pipeline(pipeline_steps)
                
                # Create parameter grid for pipeline
                param_grid = {}
                for param, values in algo_config['param_space'].items():
                    param_grid[f'model__{param}'] = values
                
                # Add preprocessing parameters
                if config.preprocessing and len(preprocessors) > 1:
                    param_grid['scaler'] = preprocessors
                
                # Cross-validation
                cv = (StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=42) 
                     if problem_type == 'classification'
                     else KFold(n_splits=config.cv_folds, shuffle=True, random_state=42))
                
                # Hyperparameter optimization
                if len(param_grid) > 0:
                    search = RandomizedSearchCV(
                        pipeline,
                        param_grid,
                        n_iter=min(config.n_trials, 50),
                        cv=cv,
                        scoring=config.scoring_metric,
                        n_jobs=-1,
                        random_state=42
                    )
                else:
                    # No hyperparameters to tune
                    search = pipeline
                    
                # Fit model
                if hasattr(search, 'fit'):
                    search.fit(X, y)
                    
                    if hasattr(search, 'best_score_'):
                        score = search.best_score_
                        model = search.best_estimator_
                        params = search.best_params_
                    else:
                        # Simple pipeline without search
                        scores = cross_val_score(search, X, y, cv=cv, scoring=config.scoring_metric)
                        score = scores.mean()
                        model = search
                        params = {}
                
                # Track trial
                trial_result = {
                    'algorithm': algo_name,
                    'score': score,
                    'params': params,
                    'timestamp': datetime.now().isoformat()
                }
                optimization_history.append(trial_result)
                trials_completed += 1
                
                # Check if this is the best model
                is_better = (score > best_score if problem_type == 'classification' 
                           else score > best_score)  # For regression, higher R[?] is better
                
                if is_better:
                    best_score = score
                    best_model = model
                    best_params = params
                    best_algorithm = algo_name
                
                logger.info(f"{algo_name} score: {score:.4f}")
                
            except Exception as e:
                logger.error(f"Error optimizing {algo_name}: {e}")
                continue
        
        # Calculate final metrics
        optimization_time = time.time() - start_time
        
        # Baseline comparison (simple model)
        if problem_type == 'classification':
            baseline_model = RandomForestClassifier(n_estimators=10, random_state=42)
        else:
            baseline_model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        baseline_scores = cross_val_score(baseline_model, X, y, 
                                        cv=KFold(n_splits=3, shuffle=True, random_state=42),
                                        scoring=config.scoring_metric)
        baseline_score = baseline_scores.mean()
        improvement = best_score - baseline_score
        
        # Get feature importance
        feature_importance = self._extract_feature_importance(best_model, X.shape[1])
        
        # Comprehensive validation
        validation_metrics = self._get_validation_metrics(best_model, X, y, problem_type)
        
        # Create result
        result = OptimizationResult(
            algorithm_name=best_algorithm,
            best_score=best_score,
            best_params=best_params,
            optimization_time=optimization_time,
            trials_completed=trials_completed,
            improvement_over_baseline=improvement,
            final_model=best_model,
            validation_metrics=validation_metrics,
            feature_importance=feature_importance,
            optimization_history=optimization_history
        )
        
        # Cache result
        self._cache_optimization_result(model_name, result)
        
        # Store in history
        with self.lock:
            self.optimization_history[model_name].append(result)
            self.best_models[model_name] = best_model
        
        logger.info(f"Optimization complete: {best_algorithm} with score {best_score:.4f} "
                   f"(+{improvement:.4f} over baseline) in {optimization_time:.1f}s")
        
        return result
    
    def auto_ensemble(self, X: np.ndarray, y: np.ndarray, 
                     base_models: List[Any] = None,
                     problem_type: str = 'auto',
                     ensemble_size: int = 5) -> Dict[str, Any]:
        """Create an optimized ensemble of models"""
        
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not available"}
        
        # Auto-detect problem type
        if problem_type == 'auto':
            unique_targets = len(np.unique(y))
            problem_type = 'classification' if unique_targets <= 20 else 'regression'
        
        logger.info(f"Creating auto-ensemble for {problem_type}")
        
        # Get diverse base models if not provided
        if base_models is None:
            if problem_type == 'classification':
                base_models = [
                    RandomForestClassifier(n_estimators=100, random_state=42),
                    GradientBoostingClassifier(n_estimators=100, random_state=42),
                    ExtraTreesClassifier(n_estimators=100, random_state=42),
                    LogisticRegression(random_state=42),
                    SVC(probability=True, random_state=42)
                ]
            else:
                base_models = [
                    RandomForestRegressor(n_estimators=100, random_state=42),
                    GradientBoostingRegressor(n_estimators=100, random_state=42),
                    ExtraTreesRegressor(n_estimators=100, random_state=42),
                    Ridge(),
                    Lasso()
                ]
        
        # Limit ensemble size
        base_models = base_models[:ensemble_size]
        
        # Train base models and get predictions
        model_predictions = []
        model_scores = []
        
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for i, model in enumerate(base_models):
            try:
                # Cross-validation predictions
                if problem_type == 'classification':
                    from sklearn.model_selection import cross_val_predict
                    pred = cross_val_predict(model, X, y, cv=cv, method='predict_proba')
                    if pred.shape[1] == 2:  # Binary classification
                        pred = pred[:, 1]  # Use positive class probability
                    else:
                        pred = np.argmax(pred, axis=1)  # Multi-class
                else:
                    pred = cross_val_predict(model, X, y, cv=cv)
                
                # Calculate score
                if problem_type == 'classification':
                    score = accuracy_score(y, np.round(pred) if pred.ndim == 1 else pred)
                else:
                    score = r2_score(y, pred)
                
                model_predictions.append(pred)
                model_scores.append(score)
                
                logger.info(f"Base model {i+1} score: {score:.4f}")
                
            except Exception as e:
                logger.warning(f"Error with base model {i+1}: {e}")
                continue
        
        if len(model_predictions) == 0:
            return {"error": "No base models could be trained"}
        
        # Simple ensemble: weighted average based on performance
        weights = np.array(model_scores)
        weights = weights / weights.sum()  # Normalize
        
        # Create ensemble predictions
        ensemble_pred = np.average(model_predictions, axis=0, weights=weights)
        
        # Calculate ensemble score
        if problem_type == 'classification':
            ensemble_score = accuracy_score(y, np.round(ensemble_pred) if ensemble_pred.ndim == 1 else ensemble_pred)
        else:
            ensemble_score = r2_score(y, ensemble_pred)
        
        return {
            "ensemble_score": float(ensemble_score),
            "base_model_scores": [float(s) for s in model_scores],
            "ensemble_weights": weights.tolist(),
            "improvement_over_best_base": float(ensemble_score - max(model_scores)),
            "number_of_models": len(base_models),
            "problem_type": problem_type
        }
    
    def _get_default_config(self, problem_type: str) -> AutoMLConfig:
        """Get default configuration for optimization"""
        return AutoMLConfig(
            algorithms_to_try=['random_forest', 'gradient_boosting', 'extra_trees'],
            max_optimization_time=300,  # 5 minutes
            cv_folds=5,
            scoring_metric='accuracy' if problem_type == 'classification' else 'r2',
            feature_selection=True,
            preprocessing=True,
            ensemble_methods=True,
            early_stopping=True,
            n_trials=50
        )
    
    def _optimize_preprocessing(self, X: np.ndarray, y: np.ndarray, 
                              problem_type: str, config: AutoMLConfig) -> List[Any]:
        """Optimize preprocessing pipeline"""
        if not config.preprocessing:
            return []
        
        # Test different scalers
        best_scaler = None
        best_score = -np.inf
        
        for name, scaler in self.scalers.items():
            try:
                X_scaled = scaler.fit_transform(X)
                
                # Quick model test
                if problem_type == 'classification':
                    model = RandomForestClassifier(n_estimators=10, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=10, random_state=42)
                
                scores = cross_val_score(model, X_scaled, y, cv=3, 
                                       scoring=config.scoring_metric)
                score = scores.mean()
                
                if score > best_score:
                    best_score = score
                    best_scaler = scaler
                    
            except Exception as e:
                logger.warning(f"Error testing scaler {name}: {e}")
                continue
        
        return [best_scaler] if best_scaler else []
    
    def _optimize_feature_selection(self, X: np.ndarray, y: np.ndarray,
                                  problem_type: str, config: AutoMLConfig) -> List[Any]:
        """Optimize feature selection"""
        if not config.feature_selection or X.shape[1] < 10:
            return []
        
        # Select top features using mutual information
        if problem_type == 'classification':
            selector = SelectKBest(mutual_info_classif, k=min(X.shape[1] // 2, 20))
        else:
            selector = SelectKBest(mutual_info_regression, k=min(X.shape[1] // 2, 20))
        
        try:
            selector.fit(X, y)
            return [selector]
        except Exception as e:
            logger.warning(f"Error in feature selection: {e}")
            return []
    
    def _extract_feature_importance(self, model: Any, n_features: int) -> List[Tuple[str, float]]:
        """Extract feature importance from model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).flatten()
            else:
                # Try to get from pipeline
                if hasattr(model, 'named_steps') and 'model' in model.named_steps:
                    final_model = model.named_steps['model']
                    if hasattr(final_model, 'feature_importances_'):
                        importances = final_model.feature_importances_
                    elif hasattr(final_model, 'coef_'):
                        importances = np.abs(final_model.coef_).flatten()
                    else:
                        return []
                else:
                    return []
            
            # Create feature importance list
            feature_names = [f"feature_{i}" for i in range(len(importances))]
            importance_pairs = list(zip(feature_names, importances))
            
            # Sort by importance
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return importance_pairs[:10]  # Top 10 features
            
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return []
    
    def _get_validation_metrics(self, model: Any, X: np.ndarray, y: np.ndarray, 
                              problem_type: str) -> Dict[str, float]:
        """Get comprehensive validation metrics"""
        try:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            metrics = {}
            
            if problem_type == 'classification':
                # Classification metrics
                for metric, scorer in [
                    ('accuracy', 'accuracy'),
                    ('f1', 'f1_weighted'),
                    ('precision', 'precision_weighted'),
                    ('recall', 'recall_weighted')
                ]:
                    scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
                    metrics[metric] = float(scores.mean())
                    metrics[f'{metric}_std'] = float(scores.std())
            else:
                # Regression metrics
                for metric, scorer in [
                    ('r2', 'r2'),
                    ('neg_mse', 'neg_mean_squared_error'),
                    ('neg_mae', 'neg_mean_absolute_error')
                ]:
                    scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
                    metrics[metric] = float(scores.mean())
                    metrics[f'{metric}_std'] = float(scores.std())
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Could not calculate validation metrics: {e}")
            return {}
    
    def _cache_optimization_result(self, model_name: str, result: OptimizationResult):
        """Cache optimization result to disk"""
        try:
            cache_file = self.cache_dir / f"{model_name}_optimization.json"
            
            # Convert result to JSON-serializable format
            result_dict = asdict(result)
            result_dict['final_model'] = None  # Can't serialize model object
            
            with open(cache_file, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
                
            logger.debug(f"Cached optimization result to {cache_file}")
            
        except Exception as e:
            logger.warning(f"Could not cache optimization result: {e}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization activities"""
        with self.lock:
            total_optimizations = sum(len(history) for history in self.optimization_history.values())
            
            if total_optimizations == 0:
                return {
                    "status": "no_optimizations",
                    "total_optimizations": 0
                }
            
            # Calculate average improvements
            all_improvements = []
            algorithm_performance = defaultdict(list)
            
            for model_name, history in self.optimization_history.items():
                for result in history:
                    all_improvements.append(result.improvement_over_baseline)
                    algorithm_performance[result.algorithm_name].append(result.best_score)
            
            # Best performing algorithms
            avg_algorithm_performance = {
                algo: np.mean(scores) 
                for algo, scores in algorithm_performance.items()
            }
            
            return {
                "total_optimizations": total_optimizations,
                "models_optimized": len(self.optimization_history),
                "average_improvement": float(np.mean(all_improvements)),
                "best_improvement": float(max(all_improvements)),
                "algorithm_performance": {k: float(v) for k, v in avg_algorithm_performance.items()},
                "best_algorithm": max(avg_algorithm_performance.items(), key=lambda x: x[1])[0] if avg_algorithm_performance else None,
                "features_available": [
                    "Automated hyperparameter tuning",
                    "Algorithm selection and comparison",
                    "Feature selection optimization",
                    "Preprocessing pipeline optimization",
                    "Ensemble model creation",
                    "Cross-validation and metrics",
                    "Optimization caching and history"
                ]
            }


# Global auto-optimizer instance
auto_optimizer = MLAutoOptimizer()


class AutoOptimizerHealthCheck:
    """Health check utilities for auto-optimizer"""
    
    @staticmethod
    def get_health_status() -> Dict[str, Any]:
        """Get health status of the auto-optimization system"""
        try:
            summary = auto_optimizer.get_optimization_summary()
            
            health_score = 100
            issues = []
            
            if not SKLEARN_AVAILABLE:
                health_score = 0
                issues.append("scikit-learn not available - auto-optimization disabled")
                status = "critical"
            else:
                # Check optimization activity
                total_optimizations = summary.get('total_optimizations', 0)
                if total_optimizations == 0:
                    health_score -= 20
                    issues.append("No optimization activities recorded")
                
                # Check algorithm performance
                best_algorithm = summary.get('best_algorithm')
                if not best_algorithm:
                    health_score -= 10
                    issues.append("No best algorithm identified")
                
                status = "healthy" if health_score >= 80 else "degraded" if health_score >= 60 else "critical"
            
            return {
                'status': status,
                'health_score': max(0, health_score),
                'sklearn_available': SKLEARN_AVAILABLE,
                'total_optimizations': summary.get('total_optimizations', 0),
                'models_optimized': summary.get('models_optimized', 0),
                'average_improvement': summary.get('average_improvement', 0.0),
                'best_algorithm': summary.get('best_algorithm', 'unknown'),
                'issues': issues,
                'capabilities': [
                    'Automated hyperparameter optimization',
                    'Algorithm selection and comparison',
                    'Feature selection optimization',
                    'Preprocessing pipeline optimization',
                    'Ensemble model creation',
                    'Performance tracking and caching'
                ] if SKLEARN_AVAILABLE else ['Limited - install scikit-learn for full features']
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'health_score': 0,
                'error': str(e),
                'sklearn_available': SKLEARN_AVAILABLE
            }


# Export health checker
optimizer_health = AutoOptimizerHealthCheck()
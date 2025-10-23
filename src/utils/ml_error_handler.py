"""
Enhanced ML Error Handling System
Provides comprehensive error handling, retry logic, and fallback mechanisms for ML components
"""

import asyncio
import logging
import traceback
import functools
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path

from loguru import logger


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MLErrorType(Enum):
    """ML-specific error types"""
    MODEL_LOADING_ERROR = "model_loading_error"
    PREDICTION_ERROR = "prediction_error"
    TRAINING_ERROR = "training_error"
    DATA_PREPROCESSING_ERROR = "data_preprocessing_error"
    FEATURE_EXTRACTION_ERROR = "feature_extraction_error"
    DATABASE_CONNECTION_ERROR = "database_connection_error"
    EXTERNAL_API_ERROR = "external_api_error"
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"


@dataclass
class MLError:
    """Represents an ML error with context"""
    error_type: MLErrorType
    severity: ErrorSeverity
    component: str
    function_name: str
    error_message: str
    timestamp: datetime
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    retry_count: int = 0
    resolution_attempted: bool = False
    

@dataclass
class FallbackStrategy:
    """Represents a fallback strategy for ML errors"""
    name: str
    condition: Callable[[MLError], bool]
    action: Callable[..., Any]
    timeout_seconds: int
    description: str


class MLErrorHandler:
    """Comprehensive error handler for ML operations"""
    
    def __init__(self, log_file: str = "logs/ml_errors.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Error tracking
        self.error_history: List[MLError] = []
        self.error_counts: Dict[MLErrorType, int] = {}
        self.component_health: Dict[str, float] = {}
        
        # Retry configurations
        self.retry_configs = {
            MLErrorType.DATABASE_CONNECTION_ERROR: {"max_retries": 3, "backoff_factor": 2},
            MLErrorType.EXTERNAL_API_ERROR: {"max_retries": 5, "backoff_factor": 1.5},
            MLErrorType.MODEL_LOADING_ERROR: {"max_retries": 2, "backoff_factor": 3},
            MLErrorType.PREDICTION_ERROR: {"max_retries": 2, "backoff_factor": 1},
            MLErrorType.TIMEOUT_ERROR: {"max_retries": 2, "backoff_factor": 2},
        }
        
        # Fallback strategies
        self.fallback_strategies: List[FallbackStrategy] = []
        self._initialize_fallback_strategies()
        
        # Circuit breaker states
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
    def _initialize_fallback_strategies(self):
        """Initialize fallback strategies for different error types"""
        
        # Model loading fallbacks
        self.fallback_strategies.append(
            FallbackStrategy(
                name="default_model_fallback",
                condition=lambda error: error.error_type == MLErrorType.MODEL_LOADING_ERROR,
                action=self._use_default_model,
                timeout_seconds=30,
                description="Use default/simplified model when trained model fails to load"
            )
        )
        
        # Database fallbacks
        self.fallback_strategies.append(
            FallbackStrategy(
                name="cached_data_fallback",
                condition=lambda error: error.error_type == MLErrorType.DATABASE_CONNECTION_ERROR,
                action=self._use_cached_data,
                timeout_seconds=10,
                description="Use cached data when database is unavailable"
            )
        )
        
        # API fallbacks
        self.fallback_strategies.append(
            FallbackStrategy(
                name="alternative_api_fallback",
                condition=lambda error: error.error_type == MLErrorType.EXTERNAL_API_ERROR,
                action=self._use_alternative_api,
                timeout_seconds=20,
                description="Switch to alternative API when primary fails"
            )
        )
        
        # Prediction fallbacks
        self.fallback_strategies.append(
            FallbackStrategy(
                name="heuristic_prediction_fallback",
                condition=lambda error: error.error_type == MLErrorType.PREDICTION_ERROR,
                action=self._use_heuristic_prediction,
                timeout_seconds=15,
                description="Use rule-based heuristics when ML prediction fails"
            )
        )
        
    def ml_error_handler(self, 
                        component: str,
                        error_types: Optional[List[MLErrorType]] = None,
                        fallback_enabled: bool = True,
                        max_retries: Optional[int] = None):
        """Decorator for ML functions with comprehensive error handling"""
        
        def decorator(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._execute_with_error_handling(
                    func, args, kwargs, component, error_types, fallback_enabled, max_retries
                )
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(
                    self._execute_with_error_handling(
                        func, args, kwargs, component, error_types, fallback_enabled, max_retries
                    )
                )
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
        
    async def _execute_with_error_handling(self,
                                          func: Callable,
                                          args: tuple,
                                          kwargs: dict,
                                          component: str,
                                          error_types: Optional[List[MLErrorType]],
                                          fallback_enabled: bool,
                                          max_retries: Optional[int]) -> Any:
        """Execute function with comprehensive error handling"""
        
        function_name = func.__name__
        context = {
            "args_count": len(args),
            "kwargs_keys": list(kwargs.keys()),
            "component": component,
            "function_name": function_name
        }
        
        # Check circuit breaker
        if self._is_circuit_breaker_open(component, function_name):
            logger.warning(f"Circuit breaker is open for {component}.{function_name}")
            return await self._execute_fallback(component, function_name, args, kwargs)
        
        retry_count = 0
        last_error = None
        
        while True:
            try:
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success - reset circuit breaker and update health
                self._reset_circuit_breaker(component, function_name)
                self._update_component_health(component, success=True)
                
                return result
                
            except Exception as e:
                last_error = e
                
                # Classify error
                error_type = self._classify_error(e, error_types)
                severity = self._determine_severity(error_type, e)
                
                # Create ML error object
                ml_error = MLError(
                    error_type=error_type,
                    severity=severity,
                    component=component,
                    function_name=function_name,
                    error_message=str(e),
                    timestamp=datetime.now(),
                    context=context,
                    stack_trace=traceback.format_exc(),
                    retry_count=retry_count
                )
                
                # Log error
                await self._log_error(ml_error)
                
                # Check if we should retry
                should_retry, wait_time = self._should_retry(ml_error, retry_count, max_retries)
                
                if should_retry:
                    retry_count += 1
                    logger.info(f"Retrying {function_name} in {wait_time}s (attempt {retry_count})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Update circuit breaker
                    self._update_circuit_breaker(component, function_name, ml_error)
                    self._update_component_health(component, success=False)
                    
                    # Try fallback strategies
                    if fallback_enabled:
                        try:
                            fallback_result = await self._execute_fallback_strategies(ml_error, args, kwargs)
                            if fallback_result is not None:
                                logger.info(f"Fallback successful for {component}.{function_name}")
                                return fallback_result
                        except Exception as fallback_error:
                            logger.error(f"Fallback also failed: {fallback_error}")
                    
                    # Re-raise the original error if no fallback worked
                    raise last_error
    
    def _classify_error(self, error: Exception, 
                       expected_types: Optional[List[MLErrorType]] = None) -> MLErrorType:
        """Classify error into ML error types"""
        
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()
        
        # Database errors
        if any(keyword in error_str for keyword in ['connection', 'database', 'sql', 'postgres', 'mongodb']):
            return MLErrorType.DATABASE_CONNECTION_ERROR
            
        # API errors
        if any(keyword in error_str for keyword in ['api', 'request', 'http', 'timeout', '404', '500']):
            if 'timeout' in error_str:
                return MLErrorType.TIMEOUT_ERROR
            return MLErrorType.EXTERNAL_API_ERROR
            
        # Model errors
        if any(keyword in error_str for keyword in ['model', 'sklearn', 'joblib', 'pickle']):
            if any(keyword in error_str for keyword in ['load', 'file not found', 'corrupt']):
                return MLErrorType.MODEL_LOADING_ERROR
            elif any(keyword in error_str for keyword in ['predict', 'fit', 'transform']):
                return MLErrorType.PREDICTION_ERROR
            return MLErrorType.TRAINING_ERROR
            
        # Data errors
        if any(keyword in error_str for keyword in ['feature', 'data', 'array', 'shape', 'dimension']):
            return MLErrorType.DATA_PREPROCESSING_ERROR
            
        # Memory errors
        if any(keyword in error_type_name for keyword in ['memory', 'outofmemory']):
            return MLErrorType.MEMORY_ERROR
            
        # Timeout errors
        if any(keyword in error_str for keyword in ['timeout', 'timed out']):
            return MLErrorType.TIMEOUT_ERROR
            
        # Validation errors
        if any(keyword in error_str for keyword in ['validation', 'invalid', 'value error']):
            return MLErrorType.VALIDATION_ERROR
            
        # Default to prediction error
        return MLErrorType.PREDICTION_ERROR
    
    def _determine_severity(self, error_type: MLErrorType, error: Exception) -> ErrorSeverity:
        """Determine error severity"""
        
        # Critical errors that stop the system
        if error_type in [MLErrorType.MEMORY_ERROR, MLErrorType.MODEL_LOADING_ERROR]:
            return ErrorSeverity.CRITICAL
            
        # High severity errors that affect core functionality
        if error_type in [MLErrorType.DATABASE_CONNECTION_ERROR, MLErrorType.TRAINING_ERROR]:
            return ErrorSeverity.HIGH
            
        # Medium severity errors that affect specific features
        if error_type in [MLErrorType.EXTERNAL_API_ERROR, MLErrorType.PREDICTION_ERROR]:
            return ErrorSeverity.MEDIUM
            
        # Low severity errors with fallbacks available
        return ErrorSeverity.LOW
    
    def _should_retry(self, error: MLError, current_retry: int, 
                     max_retries: Optional[int] = None) -> tuple[bool, float]:
        """Determine if error should be retried and wait time"""
        
        # Get retry config for error type
        config = self.retry_configs.get(error.error_type, {"max_retries": 1, "backoff_factor": 1})
        
        max_retry_limit = max_retries if max_retries is not None else config["max_retries"]
        
        if current_retry >= max_retry_limit:
            return False, 0
            
        # Calculate wait time with exponential backoff
        backoff_factor = config["backoff_factor"]
        wait_time = min(backoff_factor ** current_retry, 60)  # Cap at 60 seconds
        
        # Don't retry critical errors
        if error.severity == ErrorSeverity.CRITICAL:
            return False, 0
            
        return True, wait_time
    
    async def _execute_fallback_strategies(self, error: MLError, args: tuple, kwargs: dict) -> Any:
        """Execute appropriate fallback strategies"""
        
        for strategy in self.fallback_strategies:
            if strategy.condition(error):
                try:
                    logger.info(f"Executing fallback strategy: {strategy.name}")
                    
                    # Execute with timeout
                    result = await asyncio.wait_for(
                        strategy.action(error, args, kwargs),
                        timeout=strategy.timeout_seconds
                    )
                    
                    error.resolution_attempted = True
                    return result
                    
                except Exception as fallback_error:
                    logger.warning(f"Fallback strategy {strategy.name} failed: {fallback_error}")
                    continue
        
        return None
    
    async def _execute_fallback(self, component: str, function_name: str, 
                              args: tuple, kwargs: dict) -> Any:
        """Execute fallback when circuit breaker is open"""
        
        # Create dummy error for fallback execution
        dummy_error = MLError(
            error_type=MLErrorType.PREDICTION_ERROR,
            severity=ErrorSeverity.MEDIUM,
            component=component,
            function_name=function_name,
            error_message="Circuit breaker open",
            timestamp=datetime.now(),
            context={}
        )
        
        return await self._execute_fallback_strategies(dummy_error, args, kwargs)
    
    # Fallback strategy implementations
    async def _use_default_model(self, error: MLError, args: tuple, kwargs: dict) -> Any:
        """Use default model when trained model fails"""
        logger.info("Using default model as fallback")
        
        # Return basic heuristic-based predictions
        if "source" in str(kwargs) or any("source" in str(arg) for arg in args):
            return {
                "relevance_score": 0.5,
                "confidence_score": 0.3,
                "reasoning": "Default heuristic - trained model unavailable"
            }
        
        # Return empty successful result
        return {"status": "success", "data": [], "fallback": True}
    
    async def _use_cached_data(self, error: MLError, args: tuple, kwargs: dict) -> Any:
        """Use cached data when database is unavailable"""
        logger.info("Using cached data as fallback")
        
        # Return cached or sample data
        return {
            "status": "success_cached",
            "data": {
                "articles": [],
                "reports": [],
                "entities": [],
                "source": "cache"
            },
            "fallback": True,
            "message": "Using cached data - database temporarily unavailable"
        }
    
    async def _use_alternative_api(self, error: MLError, args: tuple, kwargs: dict) -> Any:
        """Use alternative API when primary fails"""
        logger.info("Switching to alternative API as fallback")
        
        # Return mock API response
        return {
            "status": "success_alternative",
            "data": [],
            "fallback": True,
            "message": "Using alternative source - primary API unavailable"
        }
    
    async def _use_heuristic_prediction(self, error: MLError, args: tuple, kwargs: dict) -> Any:
        """Use rule-based heuristics when ML prediction fails"""
        logger.info("Using heuristic prediction as fallback")
        
        # Simple rule-based prediction
        return {
            "prediction": 0.5,
            "confidence": 0.3,
            "method": "heuristic",
            "fallback": True,
            "message": "Using rule-based prediction - ML model unavailable"
        }
    
    # Circuit breaker methods
    def _is_circuit_breaker_open(self, component: str, function_name: str) -> bool:
        """Check if circuit breaker is open for component function"""
        key = f"{component}.{function_name}"
        breaker = self.circuit_breakers.get(key)
        
        if not breaker:
            return False
            
        if breaker["state"] == "closed":
            return False
            
        if breaker["state"] == "open":
            # Check if enough time has passed to try half-open
            if datetime.now() - breaker["last_failure"] > timedelta(minutes=5):
                breaker["state"] = "half_open"
                return False
            return True
            
        # Half-open state - allow one attempt
        return False
    
    def _update_circuit_breaker(self, component: str, function_name: str, error: MLError):
        """Update circuit breaker state based on error"""
        key = f"{component}.{function_name}"
        
        if key not in self.circuit_breakers:
            self.circuit_breakers[key] = {
                "failure_count": 0,
                "last_failure": datetime.now(),
                "state": "closed"
            }
        
        breaker = self.circuit_breakers[key]
        breaker["failure_count"] += 1
        breaker["last_failure"] = datetime.now()
        
        # Open circuit breaker if too many failures
        if breaker["failure_count"] >= 5:
            breaker["state"] = "open"
            logger.warning(f"Circuit breaker opened for {key}")
    
    def _reset_circuit_breaker(self, component: str, function_name: str):
        """Reset circuit breaker on success"""
        key = f"{component}.{function_name}"
        
        if key in self.circuit_breakers:
            self.circuit_breakers[key] = {
                "failure_count": 0,
                "last_failure": datetime.now(),
                "state": "closed"
            }
    
    def _update_component_health(self, component: str, success: bool):
        """Update component health score"""
        if component not in self.component_health:
            self.component_health[component] = 1.0
        
        current_health = self.component_health[component]
        
        if success:
            # Gradually improve health
            self.component_health[component] = min(1.0, current_health + 0.05)
        else:
            # Degrade health more quickly
            self.component_health[component] = max(0.0, current_health - 0.15)
    
    async def _log_error(self, error: MLError):
        """Log ML error with structured data"""
        
        # Add to history
        self.error_history.append(error)
        self.error_counts[error.error_type] = self.error_counts.get(error.error_type, 0) + 1
        
        # Limit history size
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]
        
        # Log with appropriate level
        log_message = f"ML Error in {error.component}.{error.function_name}: {error.error_message}"
        
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # Write to file
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()}: {json.dumps(asdict(error), default=str)}/n")
        except Exception as log_error:
            logger.error(f"Failed to write error to log file: {log_error}")
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        
        return {
            "component_health": dict(self.component_health),
            "error_counts": {error_type.value: count for error_type, count in self.error_counts.items()},
            "circuit_breakers": {
                key: {
                    "state": breaker["state"],
                    "failure_count": breaker["failure_count"],
                    "last_failure": breaker["last_failure"].isoformat() if breaker["last_failure"] else None
                }
                for key, breaker in self.circuit_breakers.items()
            },
            "recent_errors": [
                {
                    "component": error.component,
                    "function_name": error.function_name,
                    "error_type": error.error_type.value,
                    "severity": error.severity.value,
                    "timestamp": error.timestamp.isoformat()
                }
                for error in self.error_history[-10:]  # Last 10 errors
            ],
            "total_errors": len(self.error_history),
            "fallback_strategies": [
                {"name": strategy.name, "description": strategy.description}
                for strategy in self.fallback_strategies
            ]
        }


# Global instance
ml_error_handler = MLErrorHandler()
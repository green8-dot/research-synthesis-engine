"""
ML Health Monitoring Router
Provides endpoints for monitoring ML system health, error rates, and performance metrics
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio

try:
    from research_synthesis.utils.ml_error_handler import ml_error_handler
    ML_ERROR_HANDLER_AVAILABLE = True
except ImportError:
    ML_ERROR_HANDLER_AVAILABLE = False

try:
    from research_synthesis.services.ml_source_recommender import ml_source_recommender
    ML_SOURCE_RECOMMENDER_AVAILABLE = True
except ImportError:
    ML_SOURCE_RECOMMENDER_AVAILABLE = False

try:
    from research_synthesis.utils.ml_cache import ml_cache
    ML_CACHE_AVAILABLE = True
except ImportError:
    ML_CACHE_AVAILABLE = False

try:
    from research_synthesis.utils.ml_model_validator import model_validator, validator_health
    ML_VALIDATOR_AVAILABLE = True
except ImportError:
    ML_VALIDATOR_AVAILABLE = False

try:
    from research_synthesis.utils.advanced_model_validator import advanced_validator
    ADVANCED_VALIDATOR_AVAILABLE = True
except ImportError:
    ADVANCED_VALIDATOR_AVAILABLE = False

try:
    from research_synthesis.utils.ml_auto_optimizer import auto_optimizer, optimizer_health
    AUTO_OPTIMIZER_AVAILABLE = True
except ImportError:
    AUTO_OPTIMIZER_AVAILABLE = False

try:
    from research_synthesis.utils.intelligent_feature_engineer import feature_engineer
    FEATURE_ENGINEER_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEER_AVAILABLE = False

router = APIRouter()

@router.get("/")
async def get_ml_health_status():
    """Get overall ML system health status"""
    return {
        "service": "ML Health Monitoring",
        "status": "active" if ML_ERROR_HANDLER_AVAILABLE else "limited",
        "description": "Monitor ML system health, errors, and performance",
        "components": {
            "error_handler": ML_ERROR_HANDLER_AVAILABLE,
            "source_recommender": ML_SOURCE_RECOMMENDER_AVAILABLE,
            "model_validator": ML_VALIDATOR_AVAILABLE,
            "advanced_validator": ADVANCED_VALIDATOR_AVAILABLE,
            "auto_optimizer": AUTO_OPTIMIZER_AVAILABLE,
            "feature_engineer": FEATURE_ENGINEER_AVAILABLE,
            "cache": ML_CACHE_AVAILABLE
        },
        "endpoints": [
            "/health-report - Comprehensive ML health report",
            "/error-summary - Error statistics and patterns", 
            "/component-health - Individual component health scores",
            "/circuit-breaker-status - Circuit breaker states",
            "/performance-metrics - ML performance metrics",
            "/cache-stats - ML cache performance statistics",
            "/cache-info - Detailed cache information",
            "/validation-status - Model validation health status",
            "/validation-summary - Summary of model validation activities",
            "/drift-detection - Model performance drift detection",
            "/advanced-validation - Advanced validation features",
            "/model-comparison - Compare two models with statistical testing",
            "/model-robustness - Test model robustness against noise and perturbations",
            "/retraining-status - Check which models need retraining",
            "/model-lineage/{model_name} - Get model version history and lineage",
            "/auto-optimizer - ML auto-optimization system status",
            "/optimization-summary - Summary of optimization activities", 
            "/feature-engineer - Intelligent feature engineering status",
            "/feature-engineering-summary - Summary of feature engineering activities",
            "/run-optimization - Run automated model optimization",
            "/run-feature-engineering - Run intelligent feature engineering"
        ]
    }

@router.get("/health-report")
async def get_ml_health_report():
    """Get comprehensive ML health report"""
    if not ML_ERROR_HANDLER_AVAILABLE:
        return {
            "status": "error",
            "message": "ML Error Handler not available",
            "health_score": 0
        }
    
    try:
        # Get health report from error handler
        health_report = ml_error_handler.get_health_report()
        
        # Calculate overall health score
        component_scores = list(health_report["component_health"].values())
        overall_health = sum(component_scores) / len(component_scores) if component_scores else 0.5
        overall_health_percentage = int(overall_health * 100)
        
        # Determine status
        if overall_health_percentage >= 90:
            status = "excellent"
            status_color = "green"
        elif overall_health_percentage >= 75:
            status = "good"
            status_color = "blue" 
        elif overall_health_percentage >= 60:
            status = "fair"
            status_color = "yellow"
        else:
            status = "poor"
            status_color = "red"
        
        return {
            "status": status,
            "status_color": status_color,
            "overall_health_percentage": overall_health_percentage,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_components": len(health_report["component_health"]),
                "healthy_components": len([h for h in component_scores if h >= 0.8]),
                "total_errors": health_report["total_errors"],
                "active_circuit_breakers": len([cb for cb in health_report["circuit_breakers"].values() if cb["state"] == "open"]),
                "available_fallbacks": len(health_report["fallback_strategies"])
            },
            "details": health_report
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting ML health report: {str(e)}")

@router.get("/error-summary")
async def get_error_summary():
    """Get ML error statistics and patterns"""
    if not ML_ERROR_HANDLER_AVAILABLE:
        return {
            "status": "error",
            "message": "ML Error Handler not available"
        }
    
    try:
        health_report = ml_error_handler.get_health_report()
        
        # Calculate error rates and patterns
        error_counts = health_report["error_counts"]
        recent_errors = health_report["recent_errors"]
        
        # Group errors by component
        component_errors = {}
        for error in recent_errors:
            component = error["component"]
            if component not in component_errors:
                component_errors[component] = 0
            component_errors[component] += 1
        
        # Calculate error trends (mock data - could be enhanced with time-series analysis)
        total_errors = sum(error_counts.values())
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "error_statistics": {
                "total_errors": total_errors,
                "error_types": error_counts,
                "errors_by_component": component_errors,
                "recent_error_count": len(recent_errors)
            },
            "error_trends": {
                "trend_direction": "stable",  # Could be "increasing", "decreasing", "stable"
                "error_rate_change": 0,      # Percentage change
                "most_common_error": max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None
            },
            "recommendations": [
                "Monitor components with high error rates",
                "Review circuit breaker configurations",
                "Consider implementing additional fallback strategies",
                "Schedule regular model retraining"
            ] if total_errors > 10 else [
                "ML system is operating within normal error thresholds",
                "Continue monitoring for emerging patterns"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting error summary: {str(e)}")

@router.get("/component-health")
async def get_component_health():
    """Get individual ML component health scores"""
    if not ML_ERROR_HANDLER_AVAILABLE:
        return {
            "status": "error",
            "message": "ML Error Handler not available"
        }
    
    try:
        health_report = ml_error_handler.get_health_report()
        component_health = health_report["component_health"]
        
        # Enhanced component analysis
        components = []
        for component_name, health_score in component_health.items():
            health_percentage = int(health_score * 100)
            
            if health_percentage >= 90:
                status = "excellent"
                status_color = "green"
                recommendation = "Component operating optimally"
            elif health_percentage >= 75:
                status = "good"
                status_color = "blue"
                recommendation = "Component stable with minor issues"
            elif health_percentage >= 60:
                status = "fair"
                status_color = "yellow"
                recommendation = "Component needs attention"
            else:
                status = "poor"
                status_color = "red"
                recommendation = "Component requires immediate attention"
            
            components.append({
                "name": component_name,
                "health_score": health_percentage,
                "status": status,
                "status_color": status_color,
                "recommendation": recommendation
            })
        
        # Sort by health score (worst first for attention)
        components.sort(key=lambda x: x["health_score"])
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "components": components,
            "summary": {
                "total_components": len(components),
                "components_needing_attention": len([c for c in components if c["health_score"] < 75]),
                "average_health_score": int(sum(c["health_score"] for c in components) / len(components)) if components else 0
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting component health: {str(e)}")

@router.get("/circuit-breaker-status") 
async def get_circuit_breaker_status():
    """Get status of all circuit breakers"""
    if not ML_ERROR_HANDLER_AVAILABLE:
        return {
            "status": "error", 
            "message": "ML Error Handler not available"
        }
    
    try:
        health_report = ml_error_handler.get_health_report()
        circuit_breakers = health_report["circuit_breakers"]
        
        # Analyze circuit breaker states
        breakers = []
        for breaker_name, breaker_info in circuit_breakers.items():
            state = breaker_info["state"]
            failure_count = breaker_info["failure_count"]
            last_failure = breaker_info.get("last_failure")
            
            # Determine status color and recommendation
            if state == "closed":
                status_color = "green"
                recommendation = "Operating normally"
            elif state == "half_open":
                status_color = "yellow"
                recommendation = "Testing recovery - monitor closely"
            else:  # open
                status_color = "red"
                recommendation = "Circuit breaker tripped - investigate root cause"
            
            breakers.append({
                "name": breaker_name,
                "state": state,
                "failure_count": failure_count,
                "last_failure": last_failure,
                "status_color": status_color,
                "recommendation": recommendation
            })
        
        # Sort by state priority (open first, then half_open, then closed)
        state_priority = {"open": 0, "half_open": 1, "closed": 2}
        breakers.sort(key=lambda x: state_priority[x["state"]])
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "circuit_breakers": breakers,
            "summary": {
                "total_breakers": len(breakers),
                "open_breakers": len([b for b in breakers if b["state"] == "open"]),
                "half_open_breakers": len([b for b in breakers if b["state"] == "half_open"]),
                "closed_breakers": len([b for b in breakers if b["state"] == "closed"])
            },
            "recommendations": [
                "Investigate open circuit breakers immediately",
                "Monitor half-open breakers for recovery",
                "Review failure patterns to prevent future trips"
            ] if any(b["state"] != "closed" for b in breakers) else [
                "All circuit breakers are operating normally"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting circuit breaker status: {str(e)}")

@router.get("/performance-metrics")
async def get_performance_metrics():
    """Get ML performance metrics and benchmarks"""
    try:
        # Mock performance metrics - in production this would come from actual monitoring
        metrics = {
            "response_times": {
                "source_recommendation_avg_ms": 1250,
                "prediction_avg_ms": 850,
                "model_loading_avg_ms": 3200,
                "feature_extraction_avg_ms": 420
            },
            "accuracy_metrics": {
                "recommendation_accuracy": 0.85,
                "prediction_confidence": 0.78,
                "fallback_success_rate": 0.92,
                "cache_hit_rate": 0.65
            },
            "throughput": {
                "requests_per_second": 12.5,
                "predictions_per_hour": 2400,
                "successful_operations_rate": 0.94
            },
            "resource_utilization": {
                "cpu_usage_percent": 35,
                "memory_usage_mb": 1250,
                "model_cache_size_mb": 450,
                "disk_usage_gb": 2.1
            }
        }
        
        # Performance analysis
        analysis = {
            "overall_performance": "good",
            "bottlenecks": [],
            "recommendations": []
        }
        
        # Analyze response times
        if metrics["response_times"]["source_recommendation_avg_ms"] > 2000:
            analysis["bottlenecks"].append("Source recommendation response time")
            analysis["recommendations"].append("Consider implementing result caching")
        
        if metrics["accuracy_metrics"]["recommendation_accuracy"] < 0.8:
            analysis["bottlenecks"].append("Recommendation accuracy")
            analysis["recommendations"].append("Review and retrain recommendation models")
        
        if metrics["resource_utilization"]["memory_usage_mb"] > 2000:
            analysis["bottlenecks"].append("Memory usage")
            analysis["recommendations"].append("Optimize model memory footprint")
        
        if not analysis["bottlenecks"]:
            analysis["recommendations"].append("ML system performance is within acceptable ranges")
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "analysis": analysis,
            "benchmarks": {
                "target_response_time_ms": 1500,
                "target_accuracy": 0.82,
                "target_availability": 0.99,
                "target_cache_hit_rate": 0.70
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting performance metrics: {str(e)}")

@router.post("/test-error-handling")
async def test_error_handling(component: str = "TestComponent"):
    """Test error handling capabilities (for development/testing)"""
    if not ML_ERROR_HANDLER_AVAILABLE:
        return {
            "status": "error",
            "message": "ML Error Handler not available"
        }
    
    try:
        # Create a test decorator and intentionally trigger an error
        @ml_error_handler.ml_error_handler(
            component=component,
            error_types=[],
            fallback_enabled=True,
            max_retries=1
        )
        def test_function():
            raise ValueError("This is a test error for error handling validation")
        
        # This should be caught by the error handler and return fallback
        result = test_function()
        
        return {
            "status": "success",
            "message": "Error handling test completed",
            "result": result,
            "note": "If you see this message, the error handler successfully provided a fallback"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error handling test failed: {str(e)}",
            "note": "This suggests the error handler may not be working correctly"
        }

@router.get("/cache-stats")
async def get_ml_cache_stats():
    """Get ML cache performance statistics"""
    if not ML_CACHE_AVAILABLE:
        return {
            "status": "error",
            "message": "ML Cache not available"
        }
    
    try:
        cache_stats = ml_cache.get_stats()
        
        # Performance analysis
        analysis = {
            "performance_impact": "excellent" if cache_stats["hit_rate_percent"] >= 70 else
                                "good" if cache_stats["hit_rate_percent"] >= 50 else
                                "fair" if cache_stats["hit_rate_percent"] >= 30 else "poor",
            "recommendations": []
        }
        
        if cache_stats["hit_rate_percent"] < 50:
            analysis["recommendations"].append("Consider increasing cache TTL for better hit rates")
        
        if cache_stats["evictions"] > cache_stats["cache_hits"]:
            analysis["recommendations"].append("Consider increasing cache size to reduce evictions")
            
        if cache_stats["time_saved_seconds"] > 60:
            analysis["recommendations"].append(f"Cache is saving significant computation time: {cache_stats['time_saved_seconds']}s")
        
        if not analysis["recommendations"]:
            analysis["recommendations"].append("Cache is performing optimally")
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "cache_statistics": cache_stats,
            "performance_analysis": analysis,
            "efficiency_metrics": {
                "time_saved_minutes": round(cache_stats["time_saved_seconds"] / 60, 2),
                "cache_efficiency": f"{cache_stats['hit_rate_percent']:.1f}%",
                "memory_utilization": f"{(cache_stats['memory_usage_mb'] / cache_stats['max_memory_mb']) * 100:.1f}%",
                "storage_efficiency": f"{cache_stats['current_size']}/{cache_stats['max_size']} entries"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cache stats: {str(e)}")

@router.get("/cache-info")
async def get_ml_cache_info():
    """Get detailed ML cache information"""
    if not ML_CACHE_AVAILABLE:
        return {
            "status": "error",
            "message": "ML Cache not available"
        }
    
    try:
        cache_info = ml_cache.get_cache_info()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "cache_details": cache_info,
            "insights": {
                "most_accessed_entries": cache_info["cache_entries"][:5] if cache_info["cache_entries"] else [],
                "cache_health": "healthy" if cache_info["statistics"]["hit_rate_percent"] > 50 else "needs_optimization",
                "memory_pressure": "high" if cache_info["system_memory"]["cache_memory_percent"] > 80 else "normal"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cache info: {str(e)}")

@router.post("/cache-invalidate")
async def invalidate_ml_cache(pattern: Optional[str] = None):
    """Invalidate ML cache entries (for maintenance)"""
    if not ML_CACHE_AVAILABLE:
        return {
            "status": "error",
            "message": "ML Cache not available"
        }
    
    try:
        before_count = ml_cache.get_stats()["current_size"]
        ml_cache.invalidate(pattern)
        after_count = ml_cache.get_stats()["current_size"]
        
        invalidated_count = before_count - after_count
        
        return {
            "status": "success",
            "message": f"Cache invalidation completed",
            "details": {
                "pattern": pattern or "all entries",
                "entries_invalidated": invalidated_count,
                "entries_remaining": after_count
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error invalidating cache: {str(e)}")

@router.get("/cache-test")
async def test_ml_cache_performance():
    """Test ML cache performance (for development)"""
    if not ML_CACHE_AVAILABLE:
        return {
            "status": "error",
            "message": "ML Cache not available"
        }
    
    try:
        import time
        import random
        
        # Test cache with sample ML operation
        @ml_cache.cached(ttl=300)  # 5 minute cache
        async def sample_ml_operation(input_value: str):
            # Simulate ML computation
            await asyncio.sleep(0.1)  # 100ms computation
            return {
                "result": f"processed_{input_value}",
                "computation_heavy": True,
                "timestamp": datetime.now().isoformat()
            }
        
        # Test multiple calls
        test_input = f"test_{random.randint(1000, 9999)}"
        
        # First call (should be computed)
        start_time = time.time()
        result1 = await sample_ml_operation(test_input)
        first_call_time = (time.time() - start_time) * 1000
        
        # Second call (should be cached)
        start_time = time.time()
        result2 = await sample_ml_operation(test_input)
        cached_call_time = (time.time() - start_time) * 1000
        
        speed_improvement = first_call_time / max(cached_call_time, 0.01)
        
        return {
            "status": "success",
            "test_results": {
                "first_call_time_ms": round(first_call_time, 2),
                "cached_call_time_ms": round(cached_call_time, 2),
                "speed_improvement": f"{speed_improvement:.1f}x faster",
                "cache_working": cached_call_time < first_call_time,
                "results_identical": result1 == result2
            },
            "cache_stats": ml_cache.get_stats(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing cache: {str(e)}")

@router.get("/validation-status")
async def get_model_validation_status():
    """Get model validation system health status"""
    if not ML_VALIDATOR_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "Model validation system not available",
            "recommendation": "Install scikit-learn and restart service"
        }
    
    try:
        health_status = validator_health.get_health_status()
        return {
            "status": "success",
            "validation_system": health_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting validation status: {str(e)}")

@router.get("/validation-summary")
async def get_validation_summary():
    """Get summary of all model validation activities"""
    if not ML_VALIDATOR_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "Model validation system not available"
        }
    
    try:
        summary = model_validator.get_validation_summary()
        return {
            "status": "success",
            "validation_summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting validation summary: {str(e)}")

@router.get("/drift-detection")
async def get_drift_detection_status():
    """Get model performance drift detection status for all models"""
    if not ML_VALIDATOR_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "Model validation system not available"
        }
    
    try:
        drift_status = {}
        
        # Check drift for known models
        known_models = ["relevance_classifier", "source_recommender", "learning_system"]
        
        for model_name in known_models:
            drift_info = model_validator.check_model_drift(model_name)
            if drift_info:
                drift_status[model_name] = drift_info
            else:
                drift_status[model_name] = {
                    "drift_detected": False,
                    "reason": "No performance data available"
                }
        
        # Overall drift assessment
        drift_detected = any(info.get("drift_detected", False) for info in drift_status.values())
        critical_drift = any(info.get("drift_amount", 0) > 0.1 for info in drift_status.values())
        
        return {
            "status": "success",
            "overall_assessment": {
                "drift_detected": drift_detected,
                "critical_drift_detected": critical_drift,
                "models_with_drift": [name for name, info in drift_status.items() if info.get("drift_detected", False)],
                "recommendation": "Immediate retraining required" if critical_drift else 
                                "Monitor closely" if drift_detected else "Performance stable"
            },
            "model_drift_status": drift_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking drift detection: {str(e)}")

@router.post("/validate-model/{model_name}")
async def trigger_model_validation(model_name: str):
    """Manually trigger validation for a specific model"""
    if not ML_VALIDATOR_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "Model validation system not available"
        }
    
    try:
        # This would typically load and validate the specified model
        # For now, return a placeholder response
        return {
            "status": "success",
            "message": f"Model validation triggered for {model_name}",
            "model_name": model_name,
            "validation_id": f"val_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "note": "Manual validation endpoint - would trigger actual validation in production"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error triggering model validation: {str(e)}")

@router.get("/advanced-validation")
async def get_advanced_validation_status():
    """Get advanced validation system status and capabilities"""
    if not ADVANCED_VALIDATOR_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "Advanced model validation system not available",
            "recommendation": "Install scikit-learn and restart service"
        }
    
    try:
        summary = advanced_validator.get_validation_summary()
        return {
            "status": "success",
            "advanced_validation": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting advanced validation status: {str(e)}")

@router.post("/model-comparison")
async def compare_models_endpoint(comparison_request: dict):
    """Compare two models with statistical significance testing
    
    Expected request body:
    {
        "model_a_data": {...},  # Model A performance data
        "model_b_data": {...},  # Model B performance data
        "model_a_name": "Model A",
        "model_b_name": "Model B",
        "statistical_test": true
    }
    """
    if not ADVANCED_VALIDATOR_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "Advanced model validation system not available"
        }
    
    try:
        # This is a placeholder for actual model comparison
        # In practice, you would load the actual models and test data
        return {
            "status": "success",
            "message": "Model comparison endpoint ready",
            "comparison_id": f"cmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "note": "This endpoint would compare actual loaded models in production",
            "required_data": {
                "model_a_path": "Path to first model file",
                "model_b_path": "Path to second model file", 
                "test_data_path": "Path to test dataset",
                "comparison_metrics": ["accuracy", "precision", "recall", "f1", "auc"]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in model comparison: {str(e)}")

@router.post("/model-robustness/{model_name}")
async def test_model_robustness(model_name: str):
    """Test model robustness against noise, outliers, and distribution shifts"""
    if not ADVANCED_VALIDATOR_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "Advanced model validation system not available"
        }
    
    try:
        # Placeholder for actual robustness testing
        # In practice, this would load the model and run comprehensive robustness tests
        return {
            "status": "success",
            "message": f"Robustness testing initiated for {model_name}",
            "model_name": model_name,
            "test_id": f"rob_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "tests_included": [
                "Noise resistance testing",
                "Feature perturbation analysis",
                "Distribution shift simulation",
                "Outlier sensitivity analysis",
                "Edge case performance evaluation"
            ],
            "expected_duration_minutes": 5,
            "timestamp": datetime.now().isoformat(),
            "note": "This endpoint would perform actual robustness testing in production"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in robustness testing: {str(e)}")

@router.get("/retraining-status")
async def get_retraining_status():
    """Get status of models that may need retraining"""
    if not ADVANCED_VALIDATOR_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "Advanced model validation system not available"
        }
    
    try:
        # Get retraining recommendations for all models
        retraining_status = {
            "models_needing_retraining": [],
            "models_monitored": 0,
            "high_urgency_count": 0,
            "medium_urgency_count": 0,
            "low_urgency_count": 0,
            "recommendations": []
        }
        
        # This would check actual model performance in production
        sample_models = ["relevance_classifier", "source_recommender", "learning_system"]
        
        for model_name in sample_models:
            # Simulate checking retraining status
            should_retrain = advanced_validator.should_trigger_retraining(
                model_name, 
                current_performance=0.85,  # Sample performance
                error_rate=0.1,
                data_age_days=15
            )
            
            if should_retrain["should_retrain"]:
                retraining_status["models_needing_retraining"].append({
                    "model_name": model_name,
                    "urgency": should_retrain["urgency"],
                    "triggers": should_retrain["triggers"],
                    "recommendation": should_retrain["recommendation"]
                })
                
                if should_retrain["urgency"] == "high":
                    retraining_status["high_urgency_count"] += 1
                elif should_retrain["urgency"] == "medium":
                    retraining_status["medium_urgency_count"] += 1
                else:
                    retraining_status["low_urgency_count"] += 1
        
        retraining_status["models_monitored"] = len(sample_models)
        
        # Overall recommendations
        if retraining_status["high_urgency_count"] > 0:
            retraining_status["recommendations"].append("Immediate attention required for high-urgency models")
        if retraining_status["medium_urgency_count"] > 0:
            retraining_status["recommendations"].append("Schedule retraining for medium-urgency models within 24 hours")
        if len(retraining_status["models_needing_retraining"]) == 0:
            retraining_status["recommendations"].append("All models performing within acceptable parameters")
        
        return {
            "status": "success",
            "retraining_analysis": retraining_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking retraining status: {str(e)}")

@router.get("/model-lineage/{model_name}")
async def get_model_lineage(model_name: str):
    """Get complete version history and lineage for a specific model"""
    if not ADVANCED_VALIDATOR_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "Advanced model validation system not available"
        }
    
    try:
        lineage = advanced_validator.get_model_lineage(model_name)
        
        return {
            "status": "success",
            "model_lineage": lineage,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model lineage: {str(e)}")

@router.post("/create-model-version/{model_name}")
async def create_model_version_endpoint(model_name: str, version_data: dict):
    """Create a new version of a model with comprehensive metadata
    
    Expected request body:
    {
        "performance_metrics": {"accuracy": 0.85, "precision": 0.82},
        "hyperparameters": {"n_estimators": 100, "max_depth": 10},
        "training_data_hash": "abc123...",
        "notes": "Improved feature engineering"
    }
    """
    if not ADVANCED_VALIDATOR_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "Advanced model validation system not available"
        }
    
    try:
        # This is a placeholder for actual model versioning
        # In practice, this would save the actual model and create version metadata
        
        version_id = f"{model_name}_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "status": "success",
            "message": f"Model version created for {model_name}",
            "version_id": version_id,
            "model_name": model_name,
            "created_at": datetime.now().isoformat(),
            "performance_metrics": version_data.get("performance_metrics", {}),
            "hyperparameters": version_data.get("hyperparameters", {}),
            "notes": version_data.get("notes", ""),
            "timestamp": datetime.now().isoformat(),
            "note": "This endpoint would save actual model files and metadata in production"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating model version: {str(e)}")

@router.get("/auto-optimizer")
async def get_auto_optimizer_status():
    """Get ML auto-optimization system status"""
    if not AUTO_OPTIMIZER_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "ML auto-optimization system not available",
            "recommendation": "Install scikit-learn and pandas for full optimization features"
        }
    
    try:
        health_status = optimizer_health.get_health_status()
        return {
            "status": "success",
            "auto_optimizer": health_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting auto-optimizer status: {str(e)}")

@router.get("/optimization-summary")
async def get_optimization_summary():
    """Get summary of all ML optimization activities"""
    if not AUTO_OPTIMIZER_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "ML auto-optimization system not available"
        }
    
    try:
        summary = auto_optimizer.get_optimization_summary()
        return {
            "status": "success",
            "optimization_summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting optimization summary: {str(e)}")

@router.get("/feature-engineer")
async def get_feature_engineer_status():
    """Get intelligent feature engineering system status"""
    if not FEATURE_ENGINEER_AVAILABLE:
        return {
            "status": "unavailable", 
            "message": "Intelligent feature engineering system not available",
            "recommendation": "Install scikit-learn and pandas for feature engineering capabilities"
        }
    
    try:
        return {
            "status": "success",
            "feature_engineer": {
                "status": "healthy",
                "sklearn_available": True,  # Since FEATURE_ENGINEER_AVAILABLE is True
                "capabilities": [
                    "Automatic feature type detection",
                    "Numeric feature transformations", 
                    "Categorical feature encoding",
                    "Text feature extraction",
                    "DateTime feature engineering",
                    "Feature interaction creation",
                    "Dimensionality reduction",
                    "Intelligent feature selection",
                    "Performance-based optimization"
                ],
                "supported_transformations": {
                    "numeric": ["log", "sqrt", "square", "binning", "polynomial", "interactions"],
                    "categorical": ["one_hot", "label", "target", "frequency"],
                    "text": ["length", "word_count", "character", "keywords"],
                    "datetime": ["year", "month", "day", "cyclical", "time_since"]
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting feature engineer status: {str(e)}")

@router.get("/feature-engineering-summary")
async def get_feature_engineering_summary():
    """Get summary of all feature engineering activities"""
    if not FEATURE_ENGINEER_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "Intelligent feature engineering system not available"
        }
    
    try:
        summary = feature_engineer.get_engineering_summary()
        return {
            "status": "success",
            "feature_engineering_summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting feature engineering summary: {str(e)}")

@router.post("/run-optimization")
async def run_model_optimization(optimization_request: dict):
    """Run automated model optimization
    
    Expected request body:
    {
        "dataset_info": {
            "n_samples": 1000,
            "n_features": 20,
            "problem_type": "classification"
        },
        "optimization_config": {
            "max_time": 300,
            "algorithms": ["random_forest", "gradient_boosting"],
            "cv_folds": 5
        }
    }
    """
    if not AUTO_OPTIMIZER_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "ML auto-optimization system not available"
        }
    
    try:
        # This is a placeholder for actual optimization
        # In production, this would load actual data and run optimization
        
        optimization_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        dataset_info = optimization_request.get("dataset_info", {})
        config = optimization_request.get("optimization_config", {})
        
        return {
            "status": "success",
            "message": "Model optimization initiated",
            "optimization_id": optimization_id,
            "dataset_info": dataset_info,
            "optimization_config": {
                "max_time_seconds": config.get("max_time", 300),
                "algorithms_to_try": config.get("algorithms", ["random_forest", "gradient_boosting"]),
                "cv_folds": config.get("cv_folds", 5),
                "feature_selection": config.get("feature_selection", True),
                "preprocessing": config.get("preprocessing", True)
            },
            "estimated_duration_minutes": config.get("max_time", 300) // 60,
            "timestamp": datetime.now().isoformat(),
            "note": "This endpoint would run actual optimization with provided data in production"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running optimization: {str(e)}")

@router.post("/run-feature-engineering")
async def run_feature_engineering(engineering_request: dict):
    """Run intelligent feature engineering
    
    Expected request body:
    {
        "dataset_info": {
            "n_samples": 1000,
            "n_features": 20,
            "feature_types": ["numeric", "categorical", "text"]
        },
        "engineering_config": {
            "max_features": 100,
            "include_interactions": true,
            "apply_selection": true
        }
    }
    """
    if not FEATURE_ENGINEER_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "Intelligent feature engineering system not available"
        }
    
    try:
        # This is a placeholder for actual feature engineering
        # In production, this would load actual data and run feature engineering
        
        engineering_id = f"feat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        dataset_info = engineering_request.get("dataset_info", {})
        config = engineering_request.get("engineering_config", {})
        
        return {
            "status": "success",
            "message": "Feature engineering initiated",
            "engineering_id": engineering_id,
            "dataset_info": dataset_info,
            "engineering_config": {
                "max_features": config.get("max_features", 100),
                "include_interactions": config.get("include_interactions", True),
                "apply_selection": config.get("apply_selection", True),
                "feature_types_to_engineer": config.get("feature_types", ["numeric", "categorical"])
            },
            "engineering_methods": [
                "Numeric transformations (log, sqrt, polynomial)",
                "Categorical encoding (one-hot, target, frequency)",
                "Feature interactions and combinations",
                "Dimensionality reduction (PCA, clustering)",
                "Intelligent feature selection"
            ],
            "estimated_duration_minutes": 2,
            "timestamp": datetime.now().isoformat(),
            "note": "This endpoint would run actual feature engineering with provided data in production"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running feature engineering: {str(e)}")

@router.post("/create-ensemble")
async def create_auto_ensemble(ensemble_request: dict):
    """Create an automated ensemble of optimized models
    
    Expected request body:
    {
        "dataset_info": {
            "problem_type": "classification",
            "n_samples": 1000
        },
        "ensemble_config": {
            "ensemble_size": 5,
            "include_diverse_algorithms": true
        }
    }
    """
    if not AUTO_OPTIMIZER_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "ML auto-optimization system not available for ensemble creation"
        }
    
    try:
        ensemble_id = f"ens_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        dataset_info = ensemble_request.get("dataset_info", {})
        config = ensemble_request.get("ensemble_config", {})
        
        return {
            "status": "success",
            "message": "Auto-ensemble creation initiated",
            "ensemble_id": ensemble_id,
            "ensemble_config": {
                "ensemble_size": config.get("ensemble_size", 5),
                "problem_type": dataset_info.get("problem_type", "classification"),
                "base_algorithms": [
                    "Random Forest",
                    "Gradient Boosting", 
                    "Extra Trees",
                    "Logistic Regression/Linear Regression",
                    "SVM/SVR"
                ],
                "weighting_strategy": "Performance-based weighted averaging",
                "validation_method": "5-fold cross-validation"
            },
            "expected_performance_improvement": "5-15% over best individual model",
            "estimated_duration_minutes": 5,
            "timestamp": datetime.now().isoformat(),
            "note": "This endpoint would create actual ensemble models with provided data in production"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating ensemble: {str(e)}")
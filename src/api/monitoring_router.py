"""
Monitoring API Router
Endpoints for frontend error reporting, user behavior tracking, and ML insights
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime

from loguru import logger
from research_synthesis.services.backend_error_monitor import backend_error_monitor
from research_synthesis.services.ml_error_learning_system import ml_error_learning_system
from research_synthesis.database.connection import get_mongodb, init_mongodb

router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


class FrontendError(BaseModel):
    sessionId: str
    timestamp: int
    errors: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    patterns: Dict[str, Any]
    userAgent: str
    url: str


class UserBehavior(BaseModel):
    userId: str
    sessionId: str
    features: Dict[str, Any]
    patterns: Dict[str, Any]
    timestamp: int


class SystemMetricsRequest(BaseModel):
    metrics: Dict[str, Any]
    timestamp: Optional[int] = None


@router.post("/frontend-errors")
async def receive_frontend_errors(
    error_data: FrontendError,
    background_tasks: BackgroundTasks
):
    """Receive and process frontend errors"""
    try:
        # Store frontend error data
        await store_frontend_errors(error_data)
        
        # Analyze critical errors immediately
        critical_errors = [e for e in error_data.errors 
                          if e.get('type') in ['javascript', 'network'] 
                          or e.get('severity') == 'critical']
        
        if critical_errors:
            # Schedule background analysis
            background_tasks.add_task(analyze_critical_frontend_errors, critical_errors, error_data.sessionId)
        
        # Get ML suggestions
        suggestions = await get_error_suggestions(error_data.errors)
        
        return {
            "success": True,
            "processed_errors": len(error_data.errors),
            "suggestions": suggestions,
            "session_id": error_data.sessionId
        }
        
    except Exception as e:
        logger.error(f"Frontend error processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/user-behavior")
async def receive_user_behavior(
    behavior_data: UserBehavior,
    background_tasks: BackgroundTasks
):
    """Receive and process user behavior data"""
    try:
        # Store behavior data
        await store_user_behavior(behavior_data)
        
        # Analyze behavior patterns
        background_tasks.add_task(analyze_user_patterns, behavior_data)
        
        # Get ML predictions
        predictions = await get_behavior_predictions(behavior_data)
        
        return {
            "success": True,
            "predictions": predictions,
            "user_id": behavior_data.userId,
            "session_id": behavior_data.sessionId
        }
        
    except Exception as e:
        logger.error(f"User behavior processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system-metrics")
async def receive_system_metrics(metrics_data: SystemMetricsRequest):
    """Receive system metrics for analysis"""
    try:
        # Process metrics
        timestamp = metrics_data.timestamp or int(datetime.now().timestamp() * 1000)
        
        # Store metrics
        await store_system_metrics({
            **metrics_data.metrics,
            "timestamp": timestamp
        })
        
        # Check for anomalies
        anomalies = await ml_error_learning_system.detect_anomalies(metrics_data.metrics)
        
        # Predict potential issues
        error_prediction = await ml_error_learning_system.predict_error_probability(
            metrics_data.metrics
        )
        
        return {
            "success": True,
            "anomalies": anomalies,
            "error_prediction": error_prediction,
            "timestamp": timestamp
        }
        
    except Exception as e:
        logger.error(f"System metrics processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/error-analysis/{error_id}")
async def get_error_analysis(error_id: str):
    """Get detailed analysis for a specific error"""
    try:
        await init_mongodb()
        db = get_mongodb()
        
        if not db:
            raise HTTPException(status_code=503, detail="Database not available")
        
        # Find error
        error = await db.error_analysis.find_one({"error_id": error_id})
        
        if not error:
            raise HTTPException(status_code=404, detail="Error not found")
        
        # Get ML insights
        ml_insights = await ml_error_learning_system.classify_root_cause(
            error.get("message", ""),
            error.get("stack_trace", ""),
            error.get("context", {})
        )
        
        return {
            "error": error,
            "ml_insights": ml_insights,
            "similar_errors": await find_similar_errors(error_id),
            "recommendations": error.get("recommendations", [])
        }
        
    except Exception as e:
        logger.error(f"Error analysis retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system-health")
async def get_system_health():
    """Get comprehensive system health status"""
    try:
        # Get backend error monitor status
        error_summary = backend_error_monitor.get_error_summary()
        
        # Get ML model status
        model_status = ml_error_learning_system.get_model_status()
        
        # Get recent metrics
        recent_metrics = await get_recent_system_metrics()
        
        # Calculate health score
        health_score = calculate_overall_health_score(error_summary, recent_metrics)
        
        return {
            "health_score": health_score,
            "error_summary": error_summary,
            "model_status": model_status,
            "recent_metrics": recent_metrics,
            "alerts": await get_active_alerts(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ml-insights")
async def get_ml_insights():
    """Get current ML insights and predictions"""
    try:
        insights = {
            "model_performance": {},
            "error_predictions": await get_current_error_predictions(),
            "behavior_insights": await get_behavior_insights(),
            "anomaly_alerts": await get_anomaly_alerts(),
            "recommendations": await get_ml_recommendations()
        }
        
        # Get model performance metrics
        for model_name in ml_error_learning_system.model_configs:
            if model_name in ml_error_learning_system.models:
                model = ml_error_learning_system.models[model_name]
                insights["model_performance"][model_name] = {
                    "last_trained": getattr(model, 'last_trained', None),
                    "training_samples": len(ml_error_learning_system.get_relevant_training_data(model_name)),
                    "ready": True
                }
            else:
                insights["model_performance"][model_name] = {"ready": False}
        
        return insights
        
    except Exception as e:
        logger.error(f"ML insights retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trigger-retraining")
async def trigger_model_retraining(background_tasks: BackgroundTasks):
    """Manually trigger model retraining"""
    try:
        def retrain_all_models():
            for model_name in ml_error_learning_system.model_configs:
                try:
                    ml_error_learning_system.retrain_model(model_name)
                    logger.info(f"Retrained model: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to retrain {model_name}: {e}")
        
        background_tasks.add_task(retrain_all_models)
        
        return {
            "success": True,
            "message": "Model retraining triggered",
            "models": list(ml_error_learning_system.model_configs.keys())
        }
        
    except Exception as e:
        logger.error(f"Model retraining trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit/recent")
async def get_recent_audit_logs():
    """Get recent audit logs for real-time display"""
    try:
        await init_mongodb()
        db = get_mongodb()
        
        if not db:
            return {"logs": []}
        
        # Get recent logs
        cursor = db.audit_logs.find().sort("timestamp", -1).limit(50)
        logs = []
        
        async for log in cursor:
            log['_id'] = str(log['_id'])
            logs.append(log)
        
        return {"logs": logs}
        
    except Exception as e:
        logger.error(f"Recent audit logs retrieval failed: {e}")
        return {"logs": [], "error": str(e)}


# Helper functions
async def store_frontend_errors(error_data: FrontendError):
    """Store frontend error data in database"""
    try:
        await init_mongodb()
        db = get_mongodb()
        
        if db:
            await db.frontend_errors.insert_one({
                "session_id": error_data.sessionId,
                "timestamp": datetime.fromtimestamp(error_data.timestamp / 1000).isoformat(),
                "errors": error_data.errors,
                "metrics": error_data.metrics,
                "patterns": error_data.patterns,
                "user_agent": error_data.userAgent,
                "url": error_data.url,
                "processed_at": datetime.now().isoformat()
            })
            
            # Also log individual errors for backend monitor
            for error in error_data.errors:
                backend_error_monitor.capture_error({
                    'error_type': 'FrontendError',
                    'event': error.get('message', 'Frontend error'),
                    'frontend_data': error,
                    'session_id': error_data.sessionId
                })
    
    except Exception as e:
        logger.error(f"Failed to store frontend errors: {e}")


async def store_user_behavior(behavior_data: UserBehavior):
    """Store user behavior data"""
    try:
        await init_mongodb()
        db = get_mongodb()
        
        if db:
            await db.user_behavior.insert_one({
                "user_id": behavior_data.userId,
                "session_id": behavior_data.sessionId,
                "timestamp": datetime.fromtimestamp(behavior_data.timestamp / 1000).isoformat(),
                "features": behavior_data.features,
                "patterns": behavior_data.patterns,
                "processed_at": datetime.now().isoformat()
            })
    
    except Exception as e:
        logger.error(f"Failed to store user behavior: {e}")


async def store_system_metrics(metrics: Dict[str, Any]):
    """Store system metrics"""
    try:
        await init_mongodb()
        db = get_mongodb()
        
        if db:
            await db.system_metrics_monitoring.insert_one({
                **metrics,
                "stored_at": datetime.now().isoformat()
            })
    
    except Exception as e:
        logger.error(f"Failed to store system metrics: {e}")


async def analyze_critical_frontend_errors(errors: List[Dict], session_id: str):
    """Analyze critical frontend errors"""
    for error in errors:
        try:
            # Perform root cause analysis
            root_cause = await ml_error_learning_system.classify_root_cause(
                error.get('message', ''),
                error.get('stack', ''),
                {'session_id': session_id}
            )
            
            # Store analysis
            await init_mongodb()
            db = get_mongodb()
            
            if db:
                await db.error_analysis.insert_one({
                    "error_id": error.get('id', 'unknown'),
                    "session_id": session_id,
                    "error_data": error,
                    "root_cause": root_cause,
                    "analyzed_at": datetime.now().isoformat(),
                    "type": "frontend_critical"
                })
        
        except Exception as e:
            logger.error(f"Critical error analysis failed: {e}")


async def analyze_user_patterns(behavior_data: UserBehavior):
    """Analyze user behavior patterns"""
    try:
        # Predict frustration
        frustration = await ml_error_learning_system.predict_user_frustration(
            {"patterns": behavior_data.patterns}
        )
        
        # Detect behavioral anomalies
        anomalies = await ml_error_learning_system.detect_anomalies({
            "behavior": behavior_data.features
        })
        
        # Store analysis
        await init_mongodb()
        db = get_mongodb()
        
        if db:
            await db.behavior_analysis.insert_one({
                "user_id": behavior_data.userId,
                "session_id": behavior_data.sessionId,
                "frustration_prediction": frustration,
                "anomalies": anomalies,
                "analyzed_at": datetime.now().isoformat()
            })
    
    except Exception as e:
        logger.error(f"User pattern analysis failed: {e}")


async def get_error_suggestions(errors: List[Dict]) -> List[Dict]:
    """Get ML-powered error suggestions"""
    suggestions = []
    
    for error in errors:
        try:
            # Get root cause classification
            root_cause = await ml_error_learning_system.classify_root_cause(
                error.get('message', ''),
                error.get('stack', '')
            )
            
            # Generate suggestion based on error type and root cause
            if error.get('type') == 'network':
                suggestions.append({
                    "type": "network_optimization",
                    "message": "Consider implementing retry logic and connection pooling",
                    "autoApply": False,
                    "confidence": root_cause.get('confidence', 0.5)
                })
            
            elif error.get('type') == 'performance':
                suggestions.append({
                    "type": "performance_optimization",
                    "action": "lazy-load-images",
                    "message": "Enable lazy loading for images to improve performance",
                    "autoApply": True,
                    "confidence": 0.9
                })
        
        except Exception as e:
            logger.error(f"Error suggestion generation failed: {e}")
    
    return suggestions


async def get_behavior_predictions(behavior_data: UserBehavior) -> Dict:
    """Get behavior predictions"""
    predictions = {}
    
    try:
        # Predict frustration
        frustration = await ml_error_learning_system.predict_user_frustration(
            {"patterns": behavior_data.patterns}
        )
        predictions["frustration"] = frustration
        
        # Predict next likely action (simplified)
        if behavior_data.patterns.get("clickPatterns"):
            predictions["nextLikelyAction"] = "click_interaction"
        elif behavior_data.patterns.get("scrollEvents"):
            predictions["nextLikelyAction"] = "scroll_behavior"
        else:
            predictions["nextLikelyAction"] = "navigation"
        
        # Predict churn risk (simplified)
        engagement = behavior_data.features.get("sessionFeatures", {}).get("engagementScore", 0)
        predictions["churnRisk"] = max(0, (100 - engagement) / 100.0)
        
        # Predict conversion probability
        conversion_events = len(behavior_data.patterns.get("conversionEvents", []))
        predictions["conversionProbability"] = min(1.0, conversion_events * 0.3)
    
    except Exception as e:
        logger.error(f"Behavior prediction failed: {e}")
    
    return predictions


async def find_similar_errors(error_id: str) -> List[Dict]:
    """Find similar errors"""
    try:
        await init_mongodb()
        db = get_mongodb()
        
        if not db:
            return []
        
        # Simple similarity search based on error messages
        cursor = db.error_analysis.find().limit(100)
        similar = []
        
        async for error in cursor:
            if error.get("error_id") != error_id:
                similar.append({
                    "error_id": error.get("error_id"),
                    "message": error.get("message", "")[:100],
                    "similarity": 0.5,  # Placeholder similarity score
                    "timestamp": error.get("analyzed_at")
                })
        
        return similar[:5]
    
    except Exception as e:
        logger.error(f"Similar error search failed: {e}")
        return []


async def get_recent_system_metrics() -> Dict:
    """Get recent system metrics"""
    try:
        await init_mongodb()
        db = get_mongodb()
        
        if not db:
            return {}
        
        # Get latest metrics
        cursor = db.system_metrics_monitoring.find().sort("timestamp", -1).limit(1)
        
        async for metric in cursor:
            metric['_id'] = str(metric['_id'])
            return metric
        
        return {}
    
    except Exception as e:
        logger.error(f"Recent metrics retrieval failed: {e}")
        return {}


def calculate_overall_health_score(error_summary: Dict, metrics: Dict) -> float:
    """Calculate overall system health score"""
    health_score = 100.0
    
    # Deduct for errors
    error_rate = error_summary.get('error_rate', 0)
    health_score -= min(error_rate * 2, 30)  # Max 30 point deduction
    
    # Deduct for resource usage
    if metrics.get('cpu', {}).get('percent', 0) > 80:
        health_score -= 10
    
    if metrics.get('memory', {}).get('percent', 0) > 85:
        health_score -= 10
    
    return max(0, health_score)


async def get_active_alerts() -> List[Dict]:
    """Get active system alerts"""
    try:
        await init_mongodb()
        db = get_mongodb()
        
        if not db:
            return []
        
        cursor = db.alerts.find({"acknowledged": False}).sort("timestamp", -1).limit(10)
        alerts = []
        
        async for alert in cursor:
            alert['_id'] = str(alert['_id'])
            alerts.append(alert)
        
        return alerts
    
    except Exception as e:
        logger.error(f"Active alerts retrieval failed: {e}")
        return []


async def get_current_error_predictions() -> List[Dict]:
    """Get current error predictions"""
    # Get current system state
    system_state = backend_error_monitor.get_current_system_state()
    
    # Predict errors
    prediction = await ml_error_learning_system.predict_error_probability(system_state)
    
    return [prediction] if prediction.get('probability', 0) > 0.3 else []


async def get_behavior_insights() -> Dict:
    """Get behavior insights"""
    try:
        await init_mongodb()
        db = get_mongodb()
        
        if not db:
            return {}
        
        # Get recent behavior analysis
        cursor = db.behavior_analysis.find().sort("analyzed_at", -1).limit(10)
        insights = {
            "high_frustration_users": 0,
            "anomalies_detected": 0,
            "average_engagement": 0
        }
        
        count = 0
        total_engagement = 0
        
        async for analysis in cursor:
            count += 1
            
            frustration = analysis.get("frustration_prediction", {})
            if frustration.get("frustration_level") == "high":
                insights["high_frustration_users"] += 1
            
            if analysis.get("anomalies", {}).get("is_anomaly", False):
                insights["anomalies_detected"] += 1
            
            # Calculate average engagement (would need more data)
            total_engagement += 50  # Placeholder
        
        if count > 0:
            insights["average_engagement"] = total_engagement / count
        
        return insights
    
    except Exception as e:
        logger.error(f"Behavior insights retrieval failed: {e}")
        return {}


async def get_anomaly_alerts() -> List[Dict]:
    """Get anomaly alerts"""
    return []  # Placeholder


async def get_ml_recommendations() -> List[Dict]:
    """Get ML-based recommendations"""
    return [
        {
            "type": "performance",
            "message": "Consider scaling up database connections",
            "priority": "medium",
            "confidence": 0.7
        }
    ]  # Placeholder
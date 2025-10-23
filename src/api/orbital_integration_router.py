"""
Orbital Integration Router - Bridges Research Synthesis Engine with OrbitScope ML
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import json
from loguru import logger

from research_synthesis.services.orbital_integration_service import OrbitalIntegrationService
from research_synthesis.database.connection import get_postgres_session
from pydantic import BaseModel


class OrbitalPosition(BaseModel):
    timestamp: datetime
    latitude: float
    longitude: float
    altitude_km: float
    velocity_kmh: Optional[float] = None


class OrbitalPrediction(BaseModel):
    satellite_id: str
    current_position: OrbitalPosition
    predicted_positions: List[OrbitalPosition]
    prediction_confidence: float
    model_version: str


class TLEData(BaseModel):
    satellite_id: str
    line1: str
    line2: str
    epoch: datetime


class OrbitalAnalysisRequest(BaseModel):
    satellite_ids: List[str]
    analysis_duration_hours: int = 24
    include_ml_predictions: bool = True
    include_sgp4_propagation: bool = True


router = APIRouter()


@router.get("/health")
async def orbital_health_check():
    """Health check for orbital integration services"""
    try:
        service = OrbitalIntegrationService()
        status = await service.check_orbital_systems_health()
        return {
            "status": "healthy",
            "orbital_ml_available": status.get("ml_available", False),
            "sgp4_available": status.get("sgp4_available", False),
            "mongodb_available": status.get("mongodb_available", False)
        }
    except Exception as e:
        logger.error(f"Orbital health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/satellites", response_model=List[Dict[str, Any]])
async def list_available_satellites():
    """Get list of satellites with available data"""
    try:
        service = OrbitalIntegrationService()
        satellites = await service.get_available_satellites()
        return satellites
    except Exception as e:
        logger.error(f"Failed to list satellites: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list satellites: {str(e)}")


@router.get("/satellites/{satellite_id}/current-position", response_model=OrbitalPosition)
async def get_current_satellite_position(satellite_id: str):
    """Get current position of specified satellite"""
    try:
        service = OrbitalIntegrationService()
        position = await service.get_current_position(satellite_id)
        
        if not position:
            raise HTTPException(status_code=404, detail=f"No current position found for satellite {satellite_id}")
        
        return OrbitalPosition(**position)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get current position for {satellite_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get position: {str(e)}")


@router.post("/satellites/{satellite_id}/predict", response_model=OrbitalPrediction)
async def predict_satellite_positions(
    satellite_id: str,
    hours_ahead: int = 6,
    prediction_interval_minutes: int = 5
):
    """Generate ML-based position predictions for satellite"""
    try:
        service = OrbitalIntegrationService()
        prediction = await service.generate_ml_predictions(
            satellite_id=satellite_id,
            hours_ahead=hours_ahead,
            interval_minutes=prediction_interval_minutes
        )
        
        if not prediction:
            raise HTTPException(
                status_code=404, 
                detail=f"Cannot generate predictions for satellite {satellite_id} - insufficient data"
            )
        
        return OrbitalPrediction(**prediction)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate predictions for {satellite_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/satellites/{satellite_id}/sgp4-propagate")
async def sgp4_propagate_orbit(
    satellite_id: str,
    hours_ahead: int = 24,
    step_minutes: int = 5
):
    """Generate SGP4-based orbital propagation"""
    try:
        service = OrbitalIntegrationService()
        propagation = await service.sgp4_propagate_orbit(
            satellite_id=satellite_id,
            hours_ahead=hours_ahead,
            step_minutes=step_minutes
        )
        
        if not propagation:
            raise HTTPException(
                status_code=404,
                detail=f"Cannot propagate orbit for {satellite_id} - TLE data not available"
            )
        
        return {
            "satellite_id": satellite_id,
            "propagation_method": "SGP4",
            "positions": propagation["positions"],
            "duration_hours": hours_ahead,
            "step_minutes": step_minutes,
            "generated_at": datetime.utcnow()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SGP4 propagation failed for {satellite_id}: {e}")
        raise HTTPException(status_code=500, detail=f"SGP4 propagation failed: {str(e)}")


@router.post("/analysis/comprehensive", response_model=Dict[str, Any])
async def comprehensive_orbital_analysis(request: OrbitalAnalysisRequest):
    """Perform comprehensive orbital analysis combining ML and SGP4 methods"""
    try:
        service = OrbitalIntegrationService()
        analysis = await service.comprehensive_orbital_analysis(
            satellite_ids=request.satellite_ids,
            duration_hours=request.analysis_duration_hours,
            include_ml=request.include_ml_predictions,
            include_sgp4=request.include_sgp4_propagation
        )
        
        return {
            "analysis_id": analysis["analysis_id"],
            "satellites_analyzed": len(request.satellite_ids),
            "analysis_duration_hours": request.analysis_duration_hours,
            "methods_used": analysis["methods_used"],
            "results": analysis["results"],
            "accuracy_metrics": analysis.get("accuracy_metrics", {}),
            "generated_at": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Comprehensive analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/satellites/{satellite_id}/validate-data")
async def validate_satellite_data(satellite_id: str):
    """Run validation framework on satellite data"""
    try:
        service = OrbitalIntegrationService()
        validation_result = await service.validate_satellite_data(satellite_id)
        
        return {
            "satellite_id": satellite_id,
            "validation_status": validation_result["status"],
            "data_points_validated": validation_result["data_points"],
            "issues_found": validation_result["issues"],
            "quality_score": validation_result["quality_score"],
            "recommendations": validation_result["recommendations"],
            "validated_at": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Data validation failed for {satellite_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.post("/satellites/{satellite_id}/train-model")
async def train_satellite_ml_model(satellite_id: str):
    """Train ML model for specific satellite"""
    try:
        service = OrbitalIntegrationService()
        training_result = await service.train_satellite_model(satellite_id)
        
        return {
            "satellite_id": satellite_id,
            "training_status": training_result["status"],
            "model_version": training_result["model_version"],
            "training_samples": training_result["training_samples"],
            "accuracy_metrics": training_result["accuracy_metrics"],
            "model_path": training_result["model_path"],
            "trained_at": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Model training failed for {satellite_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/data-sources/space-track/sync")
async def sync_space_track_data():
    """Synchronize TLE data from Space-Track.org"""
    try:
        service = OrbitalIntegrationService()
        sync_result = await service.sync_space_track_data()
        
        return {
            "sync_status": sync_result["status"],
            "satellites_updated": sync_result["satellites_updated"],
            "new_tle_records": sync_result["new_records"],
            "sync_timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Space-Track sync failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")


@router.get("/research-integration/orbital-research", response_model=Dict[str, Any])
async def get_orbital_research_data(
    topic: str = "orbital_mechanics",
    satellite_filter: Optional[List[str]] = None
):
    """Integrate orbital data with research synthesis for space industry intelligence"""
    try:
        service = OrbitalIntegrationService()
        research_data = await service.generate_orbital_research_intelligence(
            topic=topic,
            satellite_filter=satellite_filter
        )
        
        return {
            "research_topic": topic,
            "satellite_data_included": len(research_data.get("satellites", [])),
            "research_insights": research_data.get("insights", []),
            "orbital_trends": research_data.get("trends", {}),
            "related_sources": research_data.get("sources", []),
            "generated_at": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Orbital research integration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Research integration failed: {str(e)}")


@router.post("/integration/cross-system-sync")
async def cross_system_sync():
    """Synchronize data between Research Synthesis and OrbitScope ML systems"""
    try:
        service = OrbitalIntegrationService()
        sync_result = await service.cross_system_sync()
        
        return {
            "sync_status": "completed",
            "systems_synced": sync_result["systems"],
            "data_synchronized": sync_result["data_types"],
            "sync_metrics": sync_result["metrics"],
            "last_sync": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Cross-system sync failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")
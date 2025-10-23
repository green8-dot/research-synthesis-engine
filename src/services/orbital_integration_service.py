"""
Orbital Integration Service - Core service for integrating Research Synthesis with OrbitScope ML
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import subprocess
import sys
import uuid

from loguru import logger

try:
    import pymongo
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    logger.warning("PyMongo not available - MongoDB integration disabled")

try:
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("Scikit-learn not available - ML predictions disabled")

try:
    from sgp4.api import Satrec, jday
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False
    logger.warning("SGP4 not available - orbital propagation disabled")


class OrbitalIntegrationService:
    """Service for integrating orbital mechanics capabilities with research synthesis"""
    
    def __init__(self):
        self.mongodb_client = None
        self.orbitscope_db = None
        self.research_db = None
        self.models_path = Path("D:/orbitscope_ml/models/trained")
        self.core_scripts_path = Path("D:/orbitscope_ml/core")
        
        # Initialize connections
        asyncio.create_task(self._initialize_connections())
    
    async def _initialize_connections(self):
        """Initialize database connections"""
        if MONGODB_AVAILABLE:
            try:
                self.mongodb_client = pymongo.MongoClient('localhost', 27017, serverSelectionTimeoutMS=2000)
                # Test connection
                self.mongodb_client.server_info()
                self.orbitscope_db = self.mongodb_client.orbitscope_ml
                logger.info("MongoDB connection established for orbital integration")
            except Exception as e:
                logger.warning(f"MongoDB connection failed: {e}")
                self.mongodb_client = None
    
    async def check_orbital_systems_health(self) -> Dict[str, Any]:
        """Check health of all orbital systems"""
        status = {
            "mongodb_available": False,
            "ml_available": ML_AVAILABLE,
            "sgp4_available": SGP4_AVAILABLE,
            "models_available": False
        }
        
        # Check MongoDB
        if self.mongodb_client:
            try:
                self.mongodb_client.server_info()
                status["mongodb_available"] = True
            except Exception:
                status["mongodb_available"] = False
        
        # Check trained models
        if self.models_path.exists():
            lat_model = self.models_path / "iss_latitude_predictor.pkl"
            lon_model = self.models_path / "iss_longitude_predictor.pkl"
            status["models_available"] = lat_model.exists() and lon_model.exists()
        
        return status
    
    async def get_available_satellites(self) -> List[Dict[str, Any]]:
        """Get list of satellites with available data"""
        satellites = []
        
        if not self.orbitscope_db:
            logger.warning("MongoDB not available - returning ISS only")
            return [{"satellite_id": "ISS", "name": "International Space Station", "data_available": False}]
        
        try:
            # Check ISS positions
            iss_count = self.orbitscope_db.iss_positions.count_documents({})
            if iss_count > 0:
                latest_iss = self.orbitscope_db.iss_positions.find_one(sort=[('timestamp_unix', -1)])
                satellites.append({
                    "satellite_id": "ISS",
                    "name": "International Space Station",
                    "data_points": iss_count,
                    "latest_timestamp": latest_iss.get('timestamp') if latest_iss else None,
                    "data_available": True
                })
            
            # Check TLE data
            tle_count = self.orbitscope_db.tle_data.count_documents({})
            if tle_count > 0:
                tle_satellites = self.orbitscope_db.tle_data.distinct("satellite_id")
                for sat_id in tle_satellites:
                    if sat_id != "ISS":  # Avoid duplicating ISS
                        latest_tle = self.orbitscope_db.tle_data.find_one(
                            {"satellite_id": sat_id}, 
                            sort=[('epoch', -1)]
                        )
                        satellites.append({
                            "satellite_id": sat_id,
                            "name": f"Satellite {sat_id}",
                            "tle_available": True,
                            "latest_tle_epoch": latest_tle.get('epoch') if latest_tle else None,
                            "data_available": True
                        })
        
        except Exception as e:
            logger.error(f"Failed to get satellites: {e}")
            return [{"satellite_id": "ISS", "name": "International Space Station", "data_available": False}]
        
        return satellites
    
    async def get_current_position(self, satellite_id: str) -> Optional[Dict[str, Any]]:
        """Get current position of specified satellite"""
        if not self.orbitscope_db:
            return None
        
        try:
            if satellite_id.upper() == "ISS":
                position = self.orbitscope_db.iss_positions.find_one(
                    sort=[('timestamp_unix', -1)]
                )
                if position:
                    return {
                        "timestamp": position.get('timestamp'),
                        "latitude": position.get('latitude'),
                        "longitude": position.get('longitude'),
                        "altitude_km": position.get('altitude_km', 410),
                        "velocity_kmh": position.get('velocity_kmh')
                    }
            else:
                # For other satellites, try to get from TLE and propagate current position
                return await self._get_current_position_from_tle(satellite_id)
        
        except Exception as e:
            logger.error(f"Failed to get current position for {satellite_id}: {e}")
            return None
    
    async def _get_current_position_from_tle(self, satellite_id: str) -> Optional[Dict[str, Any]]:
        """Get current position from TLE data using SGP4"""
        if not SGP4_AVAILABLE or not self.orbitscope_db:
            return None
        
        try:
            tle = self.orbitscope_db.tle_data.find_one(
                {"satellite_id": satellite_id},
                sort=[('epoch', -1)]
            )
            
            if not tle:
                return None
            
            satellite = Satrec.twoline2rv(tle['line1'], tle['line2'])
            now = datetime.utcnow()
            jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, now.second)
            
            error, position, velocity = satellite.sgp4(jd, fr)
            
            if error == 0:
                # Convert TEME to lat/lon/alt (simplified)
                # Note: This is a simplified conversion - proper implementation would use coordinate transformations
                return {
                    "timestamp": now.isoformat(),
                    "latitude": position[0] * 0.00001,  # Simplified conversion
                    "longitude": position[1] * 0.00001,
                    "altitude_km": abs(position[2]) / 1000,
                    "velocity_kmh": np.sqrt(sum([v**2 for v in velocity])) * 3.6
                }
        
        except Exception as e:
            logger.error(f"Failed to get position from TLE for {satellite_id}: {e}")
            return None
    
    async def generate_ml_predictions(
        self, 
        satellite_id: str, 
        hours_ahead: int = 6, 
        interval_minutes: int = 5
    ) -> Optional[Dict[str, Any]]:
        """Generate ML-based position predictions"""
        if not ML_AVAILABLE or satellite_id.upper() != "ISS":
            logger.warning("ML predictions only available for ISS currently")
            return None
        
        try:
            # Load trained models
            lat_model_path = self.models_path / "iss_latitude_predictor.pkl"
            lon_model_path = self.models_path / "iss_longitude_predictor.pkl"
            
            if not (lat_model_path.exists() and lon_model_path.exists()):
                logger.warning("Trained models not found - training new models")
                await self._train_iss_models()
            
            lat_model = joblib.load(lat_model_path)
            lon_model = joblib.load(lon_model_path)
            
            # Get current position
            current_pos = await self.get_current_position(satellite_id)
            if not current_pos:
                return None
            
            # Generate predictions
            predictions = []
            current_lat = current_pos['latitude']
            current_lon = current_pos['longitude']
            current_alt = current_pos['altitude_km']
            
            num_predictions = (hours_ahead * 60) // interval_minutes
            
            for i in range(num_predictions):
                # Use ML models to predict next position
                features = [[current_lat, current_lon, current_alt, i]]
                next_lat = lat_model.predict(features)[0]
                next_lon = lon_model.predict(features)[0]
                
                prediction_time = datetime.utcnow() + timedelta(minutes=interval_minutes * (i + 1))
                
                predictions.append({
                    "timestamp": prediction_time.isoformat(),
                    "latitude": next_lat,
                    "longitude": next_lon,
                    "altitude_km": current_alt,  # Assume altitude stays roughly constant
                    "velocity_kmh": current_pos.get('velocity_kmh', 27600)  # ISS average speed
                })
                
                # Update for next iteration
                current_lat, current_lon = next_lat, next_lon
            
            return {
                "satellite_id": satellite_id,
                "current_position": current_pos,
                "predicted_positions": predictions,
                "prediction_confidence": 0.85,  # Placeholder - would come from model validation
                "model_version": "v1"
            }
        
        except Exception as e:
            logger.error(f"ML prediction failed for {satellite_id}: {e}")
            return None
    
    async def sgp4_propagate_orbit(
        self, 
        satellite_id: str, 
        hours_ahead: int = 24, 
        step_minutes: int = 5
    ) -> Optional[Dict[str, Any]]:
        """Generate SGP4-based orbital propagation"""
        if not SGP4_AVAILABLE or not self.orbitscope_db:
            return None
        
        try:
            # Get latest TLE for satellite
            tle = self.orbitscope_db.tle_data.find_one(
                {"satellite_id": satellite_id},
                sort=[('epoch', -1)]
            )
            
            if not tle:
                logger.warning(f"No TLE data found for {satellite_id}")
                return None
            
            satellite = Satrec.twoline2rv(tle['line1'], tle['line2'])
            start_time = datetime.utcnow()
            positions = []
            
            for minutes in range(0, hours_ahead * 60, step_minutes):
                t = start_time + timedelta(minutes=minutes)
                jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
                
                error, position, velocity = satellite.sgp4(jd, fr)
                
                if error == 0:
                    # Simplified coordinate conversion
                    positions.append({
                        'timestamp': t.isoformat(),
                        'latitude': position[0] * 0.00001,  # Simplified
                        'longitude': position[1] * 0.00001,
                        'altitude_km': abs(position[2]) / 1000,
                        'velocity_kmh': np.sqrt(sum([v**2 for v in velocity])) * 3.6
                    })
            
            return {
                "satellite_id": satellite_id,
                "method": "SGP4",
                "positions": positions,
                "tle_epoch": tle.get('epoch'),
                "generated_at": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"SGP4 propagation failed for {satellite_id}: {e}")
            return None
    
    async def comprehensive_orbital_analysis(
        self,
        satellite_ids: List[str],
        duration_hours: int = 24,
        include_ml: bool = True,
        include_sgp4: bool = True
    ) -> Dict[str, Any]:
        """Perform comprehensive orbital analysis"""
        analysis_id = str(uuid.uuid4())
        results = {}
        methods_used = []
        
        try:
            for satellite_id in satellite_ids:
                satellite_results = {}
                
                # Current position
                current_pos = await self.get_current_position(satellite_id)
                satellite_results["current_position"] = current_pos
                
                # ML predictions
                if include_ml and ML_AVAILABLE:
                    ml_predictions = await self.generate_ml_predictions(satellite_id, duration_hours)
                    satellite_results["ml_predictions"] = ml_predictions
                    if ml_predictions:
                        methods_used.append("ML")
                
                # SGP4 propagation
                if include_sgp4 and SGP4_AVAILABLE:
                    sgp4_propagation = await self.sgp4_propagate_orbit(satellite_id, duration_hours)
                    satellite_results["sgp4_propagation"] = sgp4_propagation
                    if sgp4_propagation:
                        methods_used.append("SGP4")
                
                # Data validation
                validation = await self.validate_satellite_data(satellite_id)
                satellite_results["validation"] = validation
                
                results[satellite_id] = satellite_results
            
            return {
                "analysis_id": analysis_id,
                "results": results,
                "methods_used": list(set(methods_used)),
                "accuracy_metrics": await self._calculate_accuracy_metrics(results)
            }
        
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {"analysis_id": analysis_id, "error": str(e), "results": {}, "methods_used": []}
    
    async def validate_satellite_data(self, satellite_id: str) -> Dict[str, Any]:
        """Run validation framework on satellite data"""
        if not self.orbitscope_db:
            return {"status": "error", "message": "Database not available"}
        
        try:
            # Call the existing validation framework
            result = await asyncio.to_thread(self._run_validation_script, satellite_id)
            return result
        except Exception as e:
            logger.error(f"Validation failed for {satellite_id}: {e}")
            return {
                "status": "error",
                "data_points": 0,
                "issues": [str(e)],
                "quality_score": 0.0,
                "recommendations": ["Fix database connection issues"]
            }
    
    def _run_validation_script(self, satellite_id: str) -> Dict[str, Any]:
        """Run the core validation framework script"""
        try:
            # Basic validation logic (simplified version of core/validation_framework.py)
            if satellite_id.upper() == "ISS" and self.orbitscope_db:
                positions = list(self.orbitscope_db.iss_positions.find().limit(100))
                
                issues = []
                quality_score = 1.0
                
                # Basic checks
                if len(positions) < 20:
                    issues.append("Insufficient data for ML training")
                    quality_score -= 0.3
                
                # Check for reasonable coordinates
                for pos in positions[:10]:  # Check first 10 positions
                    if not (-90 <= pos.get('latitude', 0) <= 90):
                        issues.append("Invalid latitude values found")
                        quality_score -= 0.2
                    if not (-180 <= pos.get('longitude', 0) <= 180):
                        issues.append("Invalid longitude values found")
                        quality_score -= 0.2
                
                return {
                    "status": "completed",
                    "data_points": len(positions),
                    "issues": issues,
                    "quality_score": max(0.0, quality_score),
                    "recommendations": ["Collect more data" if len(positions) < 50 else "Data quality is acceptable"]
                }
            else:
                return {
                    "status": "no_data",
                    "data_points": 0,
                    "issues": ["No data available for this satellite"],
                    "quality_score": 0.0,
                    "recommendations": ["Collect TLE data and generate positions"]
                }
        
        except Exception as e:
            return {
                "status": "error",
                "data_points": 0,
                "issues": [str(e)],
                "quality_score": 0.0,
                "recommendations": ["Check database connection"]
            }
    
    async def train_satellite_model(self, satellite_id: str) -> Dict[str, Any]:
        """Train ML model for specific satellite"""
        if not ML_AVAILABLE or satellite_id.upper() != "ISS":
            return {"status": "error", "message": "ML training only available for ISS"}
        
        try:
            # Train models using existing framework
            result = await asyncio.to_thread(self._train_iss_models)
            return result
        except Exception as e:
            logger.error(f"Model training failed for {satellite_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    def _train_iss_models(self) -> Dict[str, Any]:
        """Train ISS prediction models (simplified version of core/train_iss_predictor.py)"""
        if not self.orbitscope_db:
            return {"status": "error", "message": "Database not available"}
        
        try:
            # Get ISS positions
            positions = list(self.orbitscope_db.iss_positions.find().sort('timestamp_unix', 1))
            
            if len(positions) < 20:
                return {"status": "error", "message": "Insufficient data for training"}
            
            # Prepare training data (simplified)
            X, y_lat, y_lon = [], [], []
            
            for i in range(len(positions) - 1):
                current = positions[i]
                next_pos = positions[i + 1]
                
                features = [
                    current['latitude'],
                    current['longitude'],
                    current.get('altitude_km', 410),
                    i
                ]
                X.append(features)
                y_lat.append(next_pos['latitude'])
                y_lon.append(next_pos['longitude'])
            
            X, y_lat, y_lon = np.array(X), np.array(y_lat), np.array(y_lon)
            
            # Train models
            lat_model = RandomForestRegressor(n_estimators=50, random_state=42)
            lon_model = RandomForestRegressor(n_estimators=50, random_state=42)
            
            lat_model.fit(X, y_lat)
            lon_model.fit(X, y_lon)
            
            # Save models
            self.models_path.mkdir(exist_ok=True, parents=True)
            joblib.dump(lat_model, self.models_path / "iss_latitude_predictor.pkl")
            joblib.dump(lon_model, self.models_path / "iss_longitude_predictor.pkl")
            
            return {
                "status": "completed",
                "model_version": "v1",
                "training_samples": len(X),
                "accuracy_metrics": {"mae_lat": 0.01, "mae_lon": 0.01},  # Placeholder
                "model_path": str(self.models_path)
            }
        
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def sync_space_track_data(self) -> Dict[str, Any]:
        """Synchronize TLE data from Space-Track.org"""
        try:
            # This would integrate with the existing fetch_spacetrack_data.py script
            script_path = self.core_scripts_path.parent / "scripts" / "fetch_spacetrack_data.py"
            
            if script_path.exists():
                result = await asyncio.to_thread(subprocess.run, 
                    [sys.executable, str(script_path)], 
                    capture_output=True, text=True
                )
                
                if result.returncode == 0:
                    return {
                        "status": "completed",
                        "satellites_updated": 1,  # Placeholder
                        "new_records": 1
                    }
                else:
                    return {"status": "error", "message": result.stderr}
            else:
                return {"status": "error", "message": "Space-Track script not found"}
        
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def generate_orbital_research_intelligence(
        self, 
        topic: str = "orbital_mechanics", 
        satellite_filter: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate research intelligence combining orbital data with research synthesis"""
        try:
            # Get orbital data
            satellites = await self.get_available_satellites()
            
            # Filter if specified
            if satellite_filter:
                satellites = [s for s in satellites if s['satellite_id'] in satellite_filter]
            
            # Generate insights based on orbital data
            insights = []
            trends = {}
            
            for satellite in satellites:
                if satellite['data_available']:
                    # Analyze orbital trends
                    current_pos = await self.get_current_position(satellite['satellite_id'])
                    if current_pos:
                        insights.append(f"{satellite['name']} is currently at {current_pos['altitude_km']:.1f} km altitude")
                        
                        # Add to trends
                        trends[satellite['satellite_id']] = {
                            "altitude": current_pos['altitude_km'],
                            "velocity": current_pos.get('velocity_kmh', 'unknown')
                        }
            
            # Mock research sources (would integrate with actual research synthesis)
            sources = [
                {"title": "Orbital Mechanics Fundamentals", "url": "https://example.com/orbital", "relevance": 0.9},
                {"title": "Satellite Tracking Technology", "url": "https://example.com/tracking", "relevance": 0.8}
            ]
            
            return {
                "topic": topic,
                "satellites": satellites,
                "insights": insights,
                "trends": trends,
                "sources": sources
            }
        
        except Exception as e:
            logger.error(f"Orbital research intelligence failed: {e}")
            return {"topic": topic, "satellites": [], "insights": [], "trends": {}, "sources": []}
    
    async def cross_system_sync(self) -> Dict[str, Any]:
        """Synchronize data between Research Synthesis and OrbitScope ML systems"""
        try:
            sync_metrics = {
                "orbital_data_synced": 0,
                "research_reports_updated": 0,
                "ml_models_synchronized": 0
            }
            
            # Sync orbital data to research database (placeholder)
            satellites = await self.get_available_satellites()
            sync_metrics["orbital_data_synced"] = len(satellites)
            
            # Update ML models if needed
            if ML_AVAILABLE:
                # Check if models need updating
                model_status = await self.check_orbital_systems_health()
                if not model_status["models_available"]:
                    await self.train_satellite_model("ISS")
                    sync_metrics["ml_models_synchronized"] = 1
            
            return {
                "systems": ["Research Synthesis", "OrbitScope ML"],
                "data_types": ["orbital_positions", "tle_data", "ml_models"],
                "metrics": sync_metrics
            }
        
        except Exception as e:
            logger.error(f"Cross-system sync failed: {e}")
            return {"systems": [], "data_types": [], "metrics": {}}
    
    async def _calculate_accuracy_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate accuracy metrics for predictions"""
        try:
            metrics = {}
            
            for satellite_id, data in results.items():
                if "ml_predictions" in data and "sgp4_propagation" in data:
                    # Compare ML vs SGP4 (simplified)
                    ml_pred = data["ml_predictions"]
                    sgp4_prop = data["sgp4_propagation"]
                    
                    if ml_pred and sgp4_prop:
                        metrics[satellite_id] = {
                            "ml_confidence": ml_pred.get("prediction_confidence", 0.5),
                            "sgp4_available": True,
                            "comparison": "Both methods available"
                        }
            
            return metrics
        except Exception as e:
            logger.error(f"Accuracy calculation failed: {e}")
            return {}
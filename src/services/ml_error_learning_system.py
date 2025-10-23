"""
ML Error Learning System
Advanced ML system that learns from errors to predict and prevent future issues
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
import pickle
from pathlib import Path
import hashlib
import threading
import time

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import joblib

from loguru import logger
from research_synthesis.database.connection import get_mongodb, init_mongodb


class MLErrorLearningSystem:
    """ML system that learns from errors to predict and prevent future issues"""
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_extractors = {}
        
        # Data storage
        self.training_data = {
            'errors': [],
            'performance': [],
            'behavior': [],
            'system_metrics': []
        }
        
        # Model configurations
        self.model_configs = {
            'error_predictor': {
                'type': 'classification',
                'target': 'will_error_occur',
                'features': ['cpu_usage', 'memory_usage', 'error_history', 'request_rate']
            },
            'root_cause_classifier': {
                'type': 'classification',
                'target': 'root_cause_category',
                'features': ['error_message', 'stack_trace', 'system_state']
            },
            'anomaly_detector': {
                'type': 'anomaly_detection',
                'features': ['all_metrics']
            },
            'performance_predictor': {
                'type': 'regression',
                'target': 'response_time',
                'features': ['system_load', 'active_users', 'db_connections']
            },
            'user_frustration_detector': {
                'type': 'classification',
                'target': 'frustration_level',
                'features': ['click_patterns', 'scroll_behavior', 'error_frequency']
            }
        }
        
        # Learning parameters
        self.learning_rate = 0.1
        self.batch_size = 100
        self.min_training_samples = 50
        self.retrain_interval = 3600  # 1 hour
        
        # Pattern recognition
        self.error_patterns = defaultdict(list)
        self.solution_patterns = defaultdict(list)
        self.prediction_cache = {}
        
        self.init_system()
    
    def init_system(self):
        """Initialize the ML learning system"""
        self.load_existing_models()
        self.start_continuous_learning()
        self.setup_feature_extractors()
    
    def load_existing_models(self):
        """Load previously trained models"""
        model_dir = Path('models/ml_error_learning')
        
        if model_dir.exists():
            for model_name in self.model_configs.keys():
                model_path = model_dir / f'{model_name}.pkl'
                if model_path.exists():
                    try:
                        self.models[model_name] = joblib.load(model_path)
                        logger.info(f"Loaded model: {model_name}")
                    except Exception as e:
                        logger.error(f"Failed to load model {model_name}: {e}")
                
                # Load associated components
                vectorizer_path = model_dir / f'{model_name}_vectorizer.pkl'
                if vectorizer_path.exists():
                    try:
                        self.vectorizers[model_name] = joblib.load(vectorizer_path)
                    except Exception as e:
                        logger.error(f"Failed to load vectorizer for {model_name}: {e}")
                
                scaler_path = model_dir / f'{model_name}_scaler.pkl'
                if scaler_path.exists():
                    try:
                        self.scalers[model_name] = joblib.load(scaler_path)
                    except Exception as e:
                        logger.error(f"Failed to load scaler for {model_name}: {e}")
    
    def setup_feature_extractors(self):
        """Setup feature extraction functions"""
        self.feature_extractors = {
            'text_features': self.extract_text_features,
            'temporal_features': self.extract_temporal_features,
            'system_features': self.extract_system_features,
            'behavioral_features': self.extract_behavioral_features,
            'contextual_features': self.extract_contextual_features
        }
    
    def start_continuous_learning(self):
        """Start continuous learning in background"""
        def learning_loop():
            while True:
                try:
                    # Collect new training data
                    self.collect_training_data()
                    
                    # Update models if enough new data
                    for model_name in self.model_configs:
                        if self.should_retrain(model_name):
                            self.retrain_model(model_name)
                    
                    # Clean old predictions
                    self.clean_prediction_cache()
                    
                    # Save models periodically
                    self.save_all_models()
                    
                except Exception as e:
                    logger.error(f"Continuous learning error: {e}")
                
                time.sleep(self.retrain_interval)
        
        learning_thread = threading.Thread(target=learning_loop, daemon=True)
        learning_thread.start()
        logger.info("Started continuous learning system")
    
    async def collect_training_data(self):
        """Collect new training data from various sources"""
        try:
            await init_mongodb()
            db = get_mongodb()
            
            if not db:
                return
            
            # Get recent data
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            # Collect error data
            error_cursor = db.error_analysis.find({
                'timestamp': {'$gte': cutoff_time.isoformat()}
            }).limit(1000)
            
            async for error in error_cursor:
                self.training_data['errors'].append(error)
            
            # Collect performance data
            perf_cursor = db.system_metrics.find({
                'timestamp': {'$gte': cutoff_time.isoformat()}
            }).limit(1000)
            
            async for metric in perf_cursor:
                self.training_data['performance'].append(metric)
            
            # Collect user behavior data
            behavior_cursor = db.user_behavior.find({
                'timestamp': {'$gte': cutoff_time.isoformat()}
            }).limit(1000)
            
            async for behavior in behavior_cursor:
                self.training_data['behavior'].append(behavior)
            
            # Limit memory usage
            for data_type in self.training_data:
                if len(self.training_data[data_type]) > 5000:
                    self.training_data[data_type] = self.training_data[data_type][-5000:]
            
            logger.info(f"Collected training data: {sum(len(d) for d in self.training_data.values())} samples")
            
        except Exception as e:
            logger.error(f"Failed to collect training data: {e}")
    
    def should_retrain(self, model_name: str) -> bool:
        """Determine if a model should be retrained"""
        if model_name not in self.models:
            return len(self.training_data['errors']) >= self.min_training_samples
        
        # Check if enough new data since last training
        model = self.models[model_name]
        
        if hasattr(model, 'last_trained'):
            time_since_training = time.time() - model.last_trained
            if time_since_training < self.retrain_interval:
                return False
        
        # Check data quality and quantity
        relevant_data = self.get_relevant_training_data(model_name)
        return len(relevant_data) >= self.min_training_samples
    
    def get_relevant_training_data(self, model_name: str) -> List[Dict]:
        """Get training data relevant to a specific model"""
        config = self.model_configs[model_name]
        
        if model_name == 'error_predictor':
            return self.training_data['errors'] + self.training_data['performance']
        elif model_name == 'root_cause_classifier':
            return self.training_data['errors']
        elif model_name == 'anomaly_detector':
            return self.training_data['performance'] + self.training_data['behavior']
        elif model_name == 'performance_predictor':
            return self.training_data['performance']
        elif model_name == 'user_frustration_detector':
            return self.training_data['behavior']
        
        return []
    
    def retrain_model(self, model_name: str):
        """Retrain a specific model"""
        try:
            logger.info(f"Retraining model: {model_name}")
            
            # Get training data
            training_data = self.get_relevant_training_data(model_name)
            
            if len(training_data) < self.min_training_samples:
                logger.warning(f"Not enough training data for {model_name}: {len(training_data)}")
                return
            
            # Prepare features and targets
            X, y = self.prepare_training_data(model_name, training_data)
            
            if X is None or len(X) == 0:
                logger.warning(f"No valid features extracted for {model_name}")
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = self.create_model(model_name)
            model.fit(X_train, y_train)
            
            # Evaluate
            if len(X_test) > 0:
                y_pred = model.predict(X_test)
                
                if self.model_configs[model_name]['type'] == 'classification':
                    accuracy = accuracy_score(y_test, y_pred)
                    logger.info(f"{model_name} accuracy: {accuracy:.3f}")
                elif self.model_configs[model_name]['type'] == 'regression':
                    mse = np.mean((y_test - y_pred) ** 2)
                    logger.info(f"{model_name} MSE: {mse:.3f}")
            
            # Store model
            model.last_trained = time.time()
            self.models[model_name] = model
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(
                    self.get_feature_names(model_name),
                    model.feature_importances_
                ))
                logger.info(f"{model_name} top features: {sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]}")
            
        except Exception as e:
            logger.error(f"Failed to retrain model {model_name}: {e}")
    
    def prepare_training_data(self, model_name: str, training_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for a specific model"""
        config = self.model_configs[model_name]
        
        if model_name == 'error_predictor':
            return self.prepare_error_prediction_data(training_data)
        elif model_name == 'root_cause_classifier':
            return self.prepare_root_cause_data(training_data)
        elif model_name == 'anomaly_detector':
            return self.prepare_anomaly_data(training_data)
        elif model_name == 'performance_predictor':
            return self.prepare_performance_data(training_data)
        elif model_name == 'user_frustration_detector':
            return self.prepare_frustration_data(training_data)
        
        return None, None
    
    def prepare_error_prediction_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for error prediction model"""
        features = []
        targets = []
        
        for item in data:
            try:
                # Extract features
                feature_vector = []
                
                # System metrics
                if 'cpu' in item:
                    feature_vector.append(item['cpu'].get('percent', 0))
                    feature_vector.append(item['memory'].get('percent', 0))
                    feature_vector.append(item['disk'].get('percent', 0))
                else:
                    feature_vector.extend([0, 0, 0])  # Default values
                
                # Error history (count of recent errors)
                recent_errors = len([e for e in self.training_data['errors'] 
                                   if abs((datetime.fromisoformat(e.get('timestamp', '2000-01-01')) - 
                                          datetime.fromisoformat(item.get('timestamp', '2000-01-01'))).total_seconds()) < 3600])
                feature_vector.append(recent_errors)
                
                # Request rate (approximation)
                feature_vector.append(item.get('network', {}).get('bytes_recv', 0) / 1024)  # KB/s
                
                # Target: will error occur in next hour?
                target = 1 if 'error_id' in item else 0
                
                features.append(feature_vector)
                targets.append(target)
                
            except Exception as e:
                logger.debug(f"Skipping sample due to error: {e}")
                continue
        
        return np.array(features) if features else None, np.array(targets) if targets else None
    
    def prepare_root_cause_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for root cause classification"""
        texts = []
        targets = []
        
        for item in data:
            if 'message' in item and 'root_cause' in item:
                # Combine error message and context
                text = f"{item['message']} {item.get('context', {}).get('request', {}).get('path', '')}"
                texts.append(text)
                
                # Categorize root cause
                root_cause = item['root_cause']
                if isinstance(root_cause, dict):
                    primary_cause = root_cause.get('primary_cause', {})
                    if isinstance(primary_cause, dict):
                        category = primary_cause.get('function', 'unknown')
                    else:
                        category = str(primary_cause)[:50]
                else:
                    category = str(root_cause)[:50]
                
                targets.append(category)
        
        if not texts:
            return None, None
        
        # Vectorize texts
        if 'root_cause_classifier' not in self.vectorizers:
            self.vectorizers['root_cause_classifier'] = TfidfVectorizer(max_features=1000)
            X = self.vectorizers['root_cause_classifier'].fit_transform(texts)
        else:
            X = self.vectorizers['root_cause_classifier'].transform(texts)
        
        # Encode targets
        if 'root_cause_classifier' not in self.encoders:
            self.encoders['root_cause_classifier'] = LabelEncoder()
            y = self.encoders['root_cause_classifier'].fit_transform(targets)
        else:
            # Handle new labels
            try:
                y = self.encoders['root_cause_classifier'].transform(targets)
            except ValueError:
                # New labels found, refit encoder
                self.encoders['root_cause_classifier'].fit(targets)
                y = self.encoders['root_cause_classifier'].transform(targets)
        
        return X.toarray(), y
    
    def prepare_anomaly_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for anomaly detection"""
        features = []
        
        for item in data:
            feature_vector = []
            
            # System metrics
            if 'cpu' in item:
                feature_vector.extend([
                    item['cpu'].get('percent', 0),
                    item['memory'].get('percent', 0),
                    item['disk'].get('percent', 0),
                    len(item.get('process', {}).get('open_files', [])),
                    item.get('process', {}).get('num_threads', 0)
                ])
            
            # Behavioral metrics
            if 'features' in item:
                behavior = item['features']
                feature_vector.extend([
                    behavior.get('sessionFeatures', {}).get('interactionCount', 0),
                    behavior.get('sessionFeatures', {}).get('clickCount', 0),
                    behavior.get('sessionFeatures', {}).get('scrollEvents', 0),
                    behavior.get('sessionFeatures', {}).get('engagementScore', 0)
                ])
            
            if feature_vector:
                features.append(feature_vector)
        
        if not features:
            return None, None
        
        X = np.array(features)
        
        # Scale features
        if 'anomaly_detector' not in self.scalers:
            self.scalers['anomaly_detector'] = StandardScaler()
            X_scaled = self.scalers['anomaly_detector'].fit_transform(X)
        else:
            X_scaled = self.scalers['anomaly_detector'].transform(X)
        
        # For anomaly detection, we don't have explicit targets
        # The model will learn what's normal vs anomalous
        return X_scaled, None
    
    def prepare_performance_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for performance prediction"""
        features = []
        targets = []
        
        for item in data:
            if 'cpu' in item and 'response_time' in item:
                feature_vector = [
                    item['cpu'].get('percent', 0),
                    item['memory'].get('percent', 0),
                    item.get('active_connections', 0),
                    item.get('request_count', 0)
                ]
                
                features.append(feature_vector)
                targets.append(item['response_time'])
        
        return np.array(features) if features else None, np.array(targets) if targets else None
    
    def prepare_frustration_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for user frustration detection"""
        features = []
        targets = []
        
        for item in data:
            if 'patterns' in item:
                patterns = item['patterns']
                
                feature_vector = [
                    len(patterns.get('frustrationIndicators', [])),
                    patterns.get('engagementScore', 0),
                    len(patterns.get('clickPatterns', [])),
                    len(patterns.get('navigationPatterns', []))
                ]
                
                # Calculate frustration level
                frustration_count = len(patterns.get('frustrationIndicators', []))
                if frustration_count >= 3:
                    frustration_level = 'high'
                elif frustration_count >= 1:
                    frustration_level = 'medium'
                else:
                    frustration_level = 'low'
                
                features.append(feature_vector)
                targets.append(frustration_level)
        
        if not features:
            return None, None
        
        X = np.array(features)
        
        # Encode targets
        if 'user_frustration_detector' not in self.encoders:
            self.encoders['user_frustration_detector'] = LabelEncoder()
            y = self.encoders['user_frustration_detector'].fit_transform(targets)
        else:
            try:
                y = self.encoders['user_frustration_detector'].transform(targets)
            except ValueError:
                self.encoders['user_frustration_detector'].fit(targets)
                y = self.encoders['user_frustration_detector'].transform(targets)
        
        return X, y
    
    def create_model(self, model_name: str):
        """Create a new model instance"""
        config = self.model_configs[model_name]
        
        if config['type'] == 'classification':
            if model_name == 'root_cause_classifier':
                return MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
            else:
                return RandomForestClassifier(n_estimators=100, random_state=42)
        
        elif config['type'] == 'regression':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        
        elif config['type'] == 'anomaly_detection':
            return IsolationForest(contamination=0.1, random_state=42)
        
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def get_feature_names(self, model_name: str) -> List[str]:
        """Get feature names for a model"""
        config = self.model_configs[model_name]
        
        if model_name == 'error_predictor':
            return ['cpu_percent', 'memory_percent', 'disk_percent', 'recent_errors', 'request_rate']
        elif model_name == 'anomaly_detector':
            return ['cpu_percent', 'memory_percent', 'disk_percent', 'open_files', 'threads', 
                   'interactions', 'clicks', 'scroll_events', 'engagement']
        elif model_name == 'performance_predictor':
            return ['cpu_percent', 'memory_percent', 'connections', 'requests']
        elif model_name == 'user_frustration_detector':
            return ['frustration_indicators', 'engagement_score', 'click_patterns', 'nav_patterns']
        
        return config.get('features', [])
    
    # Feature extraction methods
    def extract_text_features(self, text: str, vectorizer_name: str = 'default') -> np.ndarray:
        """Extract features from text"""
        if vectorizer_name not in self.vectorizers:
            self.vectorizers[vectorizer_name] = TfidfVectorizer(max_features=100)
            return self.vectorizers[vectorizer_name].fit_transform([text]).toarray()[0]
        else:
            return self.vectorizers[vectorizer_name].transform([text]).toarray()[0]
    
    def extract_temporal_features(self, timestamp: str, context_window: int = 3600) -> List[float]:
        """Extract temporal features"""
        dt = datetime.fromisoformat(timestamp)
        
        return [
            dt.hour / 24.0,  # Hour of day normalized
            dt.weekday() / 7.0,  # Day of week normalized
            dt.day / 31.0,  # Day of month normalized
            dt.month / 12.0  # Month normalized
        ]
    
    def extract_system_features(self, system_metrics: Dict) -> List[float]:
        """Extract system-level features"""
        features = []
        
        if 'cpu' in system_metrics:
            features.extend([
                system_metrics['cpu'].get('percent', 0) / 100.0,
                system_metrics['memory'].get('percent', 0) / 100.0,
                system_metrics['disk'].get('percent', 0) / 100.0
            ])
        else:
            features.extend([0, 0, 0])
        
        return features
    
    def extract_behavioral_features(self, behavior_data: Dict) -> List[float]:
        """Extract user behavior features"""
        features = []
        
        if 'patterns' in behavior_data:
            patterns = behavior_data['patterns']
            features.extend([
                len(patterns.get('clickPatterns', [])),
                len(patterns.get('frustrationIndicators', [])),
                patterns.get('engagementScore', 0) / 100.0,
                len(patterns.get('conversionEvents', []))
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        return features
    
    def extract_contextual_features(self, context: Dict) -> List[float]:
        """Extract contextual features"""
        features = []
        
        # Request context
        if 'request' in context:
            request = context['request']
            features.extend([
                1 if request.get('method') == 'POST' else 0,
                1 if request.get('method') == 'GET' else 0,
                len(request.get('path', '')) / 100.0  # Normalized path length
            ])
        else:
            features.extend([0, 0, 0])
        
        return features
    
    # Prediction methods
    async def predict_error_probability(self, system_state: Dict, context: Dict = None) -> Dict:
        """Predict probability of error occurrence"""
        if 'error_predictor' not in self.models:
            return {'probability': 0.0, 'confidence': 0.0, 'features_used': []}
        
        try:
            # Extract features
            features = []
            features.extend(self.extract_system_features(system_state))
            features.extend(self.extract_temporal_features(datetime.now().isoformat()))
            
            if context:
                features.extend(self.extract_contextual_features(context))
            
            # Make prediction
            X = np.array([features])
            model = self.models['error_predictor']
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                probability = proba[1] if len(proba) > 1 else proba[0]
            else:
                probability = model.predict(X)[0]
            
            return {
                'probability': float(probability),
                'confidence': 0.8,  # Could be computed based on model performance
                'features_used': self.get_feature_names('error_predictor'),
                'recommendation': 'Monitor system closely' if probability > 0.7 else 'System stable'
            }
            
        except Exception as e:
            logger.error(f"Error prediction failed: {e}")
            return {'probability': 0.0, 'confidence': 0.0, 'error': str(e)}
    
    async def classify_root_cause(self, error_message: str, stack_trace: str = '', context: Dict = None) -> Dict:
        """Classify the root cause of an error"""
        if 'root_cause_classifier' not in self.models:
            return {'category': 'unknown', 'confidence': 0.0}
        
        try:
            # Prepare text
            text = f"{error_message} {stack_trace}"
            
            # Vectorize
            vectorizer = self.vectorizers.get('root_cause_classifier')
            if vectorizer:
                X = vectorizer.transform([text]).toarray()
                
                model = self.models['root_cause_classifier']
                encoder = self.encoders.get('root_cause_classifier')
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0]
                    pred_idx = np.argmax(proba)
                    confidence = proba[pred_idx]
                else:
                    pred_idx = model.predict(X)[0]
                    confidence = 0.5
                
                if encoder:
                    category = encoder.inverse_transform([pred_idx])[0]
                else:
                    category = str(pred_idx)
                
                return {
                    'category': category,
                    'confidence': float(confidence),
                    'possible_causes': self.get_similar_patterns(text)
                }
        
        except Exception as e:
            logger.error(f"Root cause classification failed: {e}")
        
        return {'category': 'unknown', 'confidence': 0.0}
    
    async def detect_anomalies(self, metrics: Dict) -> Dict:
        """Detect anomalies in system metrics"""
        if 'anomaly_detector' not in self.models:
            return {'is_anomaly': False, 'score': 0.0}
        
        try:
            # Extract features
            features = []
            features.extend(self.extract_system_features(metrics))
            
            # Add behavioral features if available
            if 'behavior' in metrics:
                features.extend(self.extract_behavioral_features(metrics['behavior']))
            
            # Scale features
            scaler = self.scalers.get('anomaly_detector')
            if scaler:
                X = scaler.transform([features])
                
                model = self.models['anomaly_detector']
                prediction = model.predict(X)[0]
                score = model.decision_function(X)[0]
                
                return {
                    'is_anomaly': prediction == -1,
                    'anomaly_score': float(score),
                    'threshold': 0.0,
                    'features_contributing': self.identify_anomaly_features(features, score)
                }
        
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
        
        return {'is_anomaly': False, 'score': 0.0}
    
    async def predict_user_frustration(self, behavior_data: Dict) -> Dict:
        """Predict user frustration level"""
        if 'user_frustration_detector' not in self.models:
            return {'frustration_level': 'unknown', 'confidence': 0.0}
        
        try:
            features = self.extract_behavioral_features(behavior_data)
            X = np.array([features])
            
            model = self.models['user_frustration_detector']
            encoder = self.encoders.get('user_frustration_detector')
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                pred_idx = np.argmax(proba)
                confidence = proba[pred_idx]
            else:
                pred_idx = model.predict(X)[0]
                confidence = 0.5
            
            if encoder:
                frustration_level = encoder.inverse_transform([pred_idx])[0]
            else:
                frustration_level = ['low', 'medium', 'high'][min(pred_idx, 2)]
            
            return {
                'frustration_level': frustration_level,
                'confidence': float(confidence),
                'interventions': self.suggest_interventions(frustration_level, behavior_data)
            }
        
        except Exception as e:
            logger.error(f"Frustration prediction failed: {e}")
        
        return {'frustration_level': 'unknown', 'confidence': 0.0}
    
    def get_similar_patterns(self, error_text: str) -> List[Dict]:
        """Find similar error patterns"""
        similar = []
        
        # Simple text similarity for now
        words = set(error_text.lower().split())
        
        for pattern_key, pattern_data in self.error_patterns.items():
            pattern_words = set(pattern_key.lower().split())
            similarity = len(words.intersection(pattern_words)) / len(words.union(pattern_words))
            
            if similarity > 0.3:
                similar.append({
                    'pattern': pattern_key,
                    'similarity': similarity,
                    'occurrences': len(pattern_data),
                    'solutions': self.solution_patterns.get(pattern_key, [])
                })
        
        return sorted(similar, key=lambda x: x['similarity'], reverse=True)[:5]
    
    def identify_anomaly_features(self, features: List[float], score: float) -> List[Dict]:
        """Identify which features contribute most to anomaly score"""
        feature_names = ['cpu', 'memory', 'disk', 'interactions', 'clicks', 'scrolls', 'engagement']
        
        # Simple approach: features that are far from expected ranges
        contributing = []
        
        for i, (feature, name) in enumerate(zip(features, feature_names)):
            if i < len(features):
                # Normalize and check if outside normal range
                if feature > 0.8 or feature < 0.1:  # Outside normal range
                    contributing.append({
                        'feature': name,
                        'value': feature,
                        'contribution': abs(feature - 0.5)  # Distance from normal
                    })
        
        return sorted(contributing, key=lambda x: x['contribution'], reverse=True)[:3]
    
    def suggest_interventions(self, frustration_level: str, behavior_data: Dict) -> List[str]:
        """Suggest interventions based on frustration level"""
        if frustration_level == 'high':
            return [
                'Offer immediate help or chat support',
                'Simplify the current process',
                'Provide alternative paths to complete task',
                'Show tutorial or tips'
            ]
        elif frustration_level == 'medium':
            return [
                'Provide subtle hints or tips',
                'Highlight important UI elements',
                'Offer optional assistance'
            ]
        else:
            return ['Monitor user progress']
    
    # Model management
    def save_all_models(self):
        """Save all trained models"""
        model_dir = Path('models/ml_error_learning')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            try:
                joblib.dump(model, model_dir / f'{model_name}.pkl')
                
                # Save associated components
                if model_name in self.vectorizers:
                    joblib.dump(self.vectorizers[model_name], model_dir / f'{model_name}_vectorizer.pkl')
                
                if model_name in self.scalers:
                    joblib.dump(self.scalers[model_name], model_dir / f'{model_name}_scaler.pkl')
                
                if model_name in self.encoders:
                    joblib.dump(self.encoders[model_name], model_dir / f'{model_name}_encoder.pkl')
                
            except Exception as e:
                logger.error(f"Failed to save model {model_name}: {e}")
    
    def clean_prediction_cache(self):
        """Clean old predictions from cache"""
        current_time = time.time()
        
        # Remove predictions older than 1 hour
        self.prediction_cache = {
            key: value for key, value in self.prediction_cache.items()
            if current_time - value.get('timestamp', 0) < 3600
        }
    
    def get_model_status(self) -> Dict:
        """Get status of all models"""
        status = {}
        
        for model_name in self.model_configs:
            model_status = {
                'trained': model_name in self.models,
                'last_training': getattr(self.models.get(model_name), 'last_trained', None),
                'training_samples': len(self.get_relevant_training_data(model_name)),
                'ready_for_prediction': model_name in self.models
            }
            
            if hasattr(self.models.get(model_name), 'feature_importances_'):
                model_status['top_features'] = dict(zip(
                    self.get_feature_names(model_name),
                    self.models[model_name].feature_importances_
                ))
            
            status[model_name] = model_status
        
        return status


# Global instance
ml_error_learning_system = MLErrorLearningSystem()
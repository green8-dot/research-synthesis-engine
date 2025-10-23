"""
Intelligent Feature Engineering Pipeline
Automatically discovers, creates, and optimizes features for ML models
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
from collections import defaultdict, Counter
import threading
import re

import numpy as np
import pandas as pd
from loguru import logger

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.preprocessing import (
        PolynomialFeatures, StandardScaler, MinMaxScaler,
        LabelEncoder, OneHotEncoder
    )
    from sklearn.feature_selection import (
        SelectKBest, f_classif, f_regression, mutual_info_classif,
        mutual_info_regression, RFE, SelectFromModel
    )
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available - feature engineering features limited")
    SKLEARN_AVAILABLE = False


@dataclass
class FeatureEngineering:
    """Result of feature engineering process"""
    original_features: int
    engineered_features: int
    feature_names: List[str]
    feature_types: Dict[str, str]
    feature_importance: List[Tuple[str, float]]
    engineering_methods: List[str]
    performance_improvement: float
    processing_time: float
    feature_statistics: Dict[str, Any]


class IntelligentFeatureEngineer:
    """Intelligent feature engineering pipeline with automated feature discovery"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("models/feature_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature engineering methods
        self.numeric_transformations = [
            'log_transform',
            'sqrt_transform',
            'square_transform',
            'reciprocal_transform',
            'polynomial_features',
            'interaction_features',
            'binning',
            'normalization',
            'standardization'
        ]
        
        self.categorical_transformations = [
            'one_hot_encoding',
            'label_encoding',
            'target_encoding',
            'frequency_encoding',
            'rare_category_grouping'
        ]
        
        self.text_transformations = [
            'length_features',
            'word_count_features',
            'character_features',
            'sentiment_features',
            'keyword_features',
            'n_gram_features'
        ]
        
        self.datetime_transformations = [
            'year_features',
            'month_features',
            'day_features',
            'hour_features',
            'weekday_features',
            'season_features',
            'time_since_features',
            'cyclical_features'
        ]
        
        # Feature quality metrics
        self.feature_quality_cache = {}
        self.engineering_history = []
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Intelligent Feature Engineer initialized with cache at {self.cache_dir}")
    
    def engineer_features(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray,
                         feature_names: Optional[List[str]] = None,
                         problem_type: str = 'auto',
                         max_features: int = 1000,
                         include_methods: Optional[List[str]] = None) -> FeatureEngineering:
        """
        Automatically engineer optimal features from dataset
        
        Args:
            X: Feature matrix or DataFrame
            y: Target vector
            feature_names: Names of original features
            problem_type: 'classification', 'regression', or 'auto'
            max_features: Maximum number of features to create
            include_methods: Specific methods to include
        """
        
        start_time = time.time()
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X.copy()
            feature_names = list(X_df.columns)
        
        original_features = len(feature_names)
        
        # Auto-detect problem type
        if problem_type == 'auto':
            unique_targets = len(np.unique(y))
            problem_type = 'classification' if unique_targets <= 20 else 'regression'
        
        logger.info(f"Starting feature engineering for {problem_type} problem")
        logger.info(f"Original features: {original_features}, Max new features: {max_features}")
        
        # Analyze feature types
        feature_types = self._analyze_feature_types(X_df)
        
        # Initialize feature engineering results
        engineered_df = X_df.copy()
        engineering_methods = []
        
        # Get baseline performance
        baseline_score = self._get_baseline_performance(X_df, y, problem_type)
        
        # Apply feature engineering methods
        if include_methods is None:
            include_methods = self._select_optimal_methods(X_df, y, feature_types, problem_type)
        
        # 1. Numeric feature engineering
        if any(ft == 'numeric' for ft in feature_types.values()):
            logger.info("Applying numeric feature engineering...")
            engineered_df, numeric_methods = self._engineer_numeric_features(
                engineered_df, feature_types, max_features // 4
            )
            engineering_methods.extend(numeric_methods)
        
        # 2. Categorical feature engineering
        if any(ft == 'categorical' for ft in feature_types.values()):
            logger.info("Applying categorical feature engineering...")
            engineered_df, categorical_methods = self._engineer_categorical_features(
                engineered_df, y, feature_types, problem_type, max_features // 4
            )
            engineering_methods.extend(categorical_methods)
        
        # 3. Text feature engineering
        if any(ft == 'text' for ft in feature_types.values()):
            logger.info("Applying text feature engineering...")
            engineered_df, text_methods = self._engineer_text_features(
                engineered_df, feature_types, max_features // 4
            )
            engineering_methods.extend(text_methods)
        
        # 4. DateTime feature engineering
        if any(ft == 'datetime' for ft in feature_types.values()):
            logger.info("Applying datetime feature engineering...")
            engineered_df, datetime_methods = self._engineer_datetime_features(
                engineered_df, feature_types
            )
            engineering_methods.extend(datetime_methods)
        
        # 5. Advanced feature interactions
        if original_features > 1:
            logger.info("Creating feature interactions...")
            engineered_df, interaction_methods = self._create_feature_interactions(
                engineered_df, y, problem_type, max_features // 4
            )
            engineering_methods.extend(interaction_methods)
        
        # 6. Dimensionality reduction features
        if engineered_df.shape[1] > 50:
            logger.info("Applying dimensionality reduction...")
            engineered_df, reduction_methods = self._apply_dimensionality_reduction(
                engineered_df, y, problem_type, min(20, max_features // 10)
            )
            engineering_methods.extend(reduction_methods)
        
        # Feature selection and optimization
        logger.info("Optimizing feature selection...")
        final_df, selected_features = self._optimize_feature_selection(
            engineered_df, y, problem_type, max_features
        )
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(final_df, y, problem_type)
        
        # Get final performance
        final_score = self._get_baseline_performance(final_df, y, problem_type)
        performance_improvement = final_score - baseline_score
        
        # Calculate feature statistics
        feature_statistics = self._calculate_feature_statistics(final_df, y)
        
        processing_time = time.time() - start_time
        
        # Create result
        result = FeatureEngineering(
            original_features=original_features,
            engineered_features=final_df.shape[1],
            feature_names=list(final_df.columns),
            feature_types=self._analyze_feature_types(final_df),
            feature_importance=feature_importance,
            engineering_methods=engineering_methods,
            performance_improvement=performance_improvement,
            processing_time=processing_time,
            feature_statistics=feature_statistics
        )
        
        # Cache result
        self._cache_engineering_result(result)
        
        with self.lock:
            self.engineering_history.append(result)
        
        logger.info(f"Feature engineering complete: {original_features} -> {final_df.shape[1]} features")
        logger.info(f"Performance improvement: {performance_improvement:+.4f} in {processing_time:.1f}s")
        
        return result
    
    def _analyze_feature_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analyze and classify feature types"""
        feature_types = {}
        
        for column in df.columns:
            col_data = df[column]
            
            # Skip if all null
            if col_data.isnull().all():
                feature_types[column] = 'null'
                continue
            
            # Check for datetime
            if pd.api.types.is_datetime64_any_dtype(col_data):
                feature_types[column] = 'datetime'
            # Check for numeric
            elif pd.api.types.is_numeric_dtype(col_data):
                feature_types[column] = 'numeric'
            # Check for text (strings with spaces or long strings)
            elif col_data.dtype == 'object':
                sample_values = col_data.dropna().head(100)
                if sample_values.empty:
                    feature_types[column] = 'null'
                else:
                    avg_length = sample_values.astype(str).str.len().mean()
                    has_spaces = sample_values.astype(str).str.contains(' ').any()
                    
                    if avg_length > 50 or has_spaces:
                        feature_types[column] = 'text'
                    else:
                        feature_types[column] = 'categorical'
            else:
                feature_types[column] = 'categorical'
        
        return feature_types
    
    def _select_optimal_methods(self, df: pd.DataFrame, y: np.ndarray,
                              feature_types: Dict[str, str], problem_type: str) -> List[str]:
        """Select optimal feature engineering methods based on data characteristics"""
        methods = []
        
        # Based on dataset size
        n_samples, n_features = df.shape
        
        if n_features < 10:
            methods.extend(['polynomial_features', 'interaction_features'])
        
        if n_samples > 1000:
            methods.extend(['pca_features', 'clustering_features'])
        
        # Based on feature types
        if any(ft == 'numeric' for ft in feature_types.values()):
            methods.extend(['log_transform', 'standardization'])
        
        if any(ft == 'categorical' for ft in feature_types.values()):
            methods.extend(['one_hot_encoding', 'target_encoding'])
        
        return list(set(methods))  # Remove duplicates
    
    def _engineer_numeric_features(self, df: pd.DataFrame, feature_types: Dict[str, str],
                                 max_features: int) -> Tuple[pd.DataFrame, List[str]]:
        """Engineer numeric features"""
        if not SKLEARN_AVAILABLE:
            return df, []
        
        result_df = df.copy()
        methods = []
        features_added = 0
        
        numeric_cols = [col for col, ftype in feature_types.items() if ftype == 'numeric']
        
        if not numeric_cols or features_added >= max_features:
            return result_df, methods
        
        for col in numeric_cols:
            if features_added >= max_features:
                break
                
            col_data = df[col].fillna(0)
            
            # Skip if all values are the same
            if col_data.nunique() <= 1:
                continue
            
            try:
                # Log transform (for positive values)
                if (col_data > 0).all():
                    result_df[f'{col}_log'] = np.log(col_data + 1)
                    methods.append('log_transform')
                    features_added += 1
                
                # Square root transform (for non-negative values)
                if (col_data >= 0).all() and features_added < max_features:
                    result_df[f'{col}_sqrt'] = np.sqrt(col_data)
                    methods.append('sqrt_transform')
                    features_added += 1
                
                # Square transform
                if features_added < max_features:
                    result_df[f'{col}_square'] = col_data ** 2
                    methods.append('square_transform')
                    features_added += 1
                
                # Binning
                if features_added < max_features and col_data.nunique() > 10:
                    result_df[f'{col}_binned'] = pd.cut(col_data, bins=5, labels=False)
                    methods.append('binning')
                    features_added += 1
                    
            except Exception as e:
                logger.warning(f"Error engineering numeric feature {col}: {e}")
                continue
        
        # Polynomial features (interactions)
        if len(numeric_cols) > 1 and features_added < max_features:
            try:
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                numeric_data = df[numeric_cols].fillna(0)
                
                if numeric_data.shape[1] <= 10:  # Limit to avoid explosion
                    poly_features = poly.fit_transform(numeric_data)
                    poly_names = poly.get_feature_names_out(numeric_cols)
                    
                    # Add only the interaction features (not the original ones)
                    start_idx = len(numeric_cols)
                    n_new_features = min(poly_features.shape[1] - start_idx, max_features - features_added)
                    
                    for i in range(start_idx, start_idx + n_new_features):
                        result_df[poly_names[i]] = poly_features[:, i]
                        features_added += 1
                    
                    methods.append('polynomial_features')
                    
            except Exception as e:
                logger.warning(f"Error creating polynomial features: {e}")
        
        return result_df, list(set(methods))
    
    def _engineer_categorical_features(self, df: pd.DataFrame, y: np.ndarray,
                                     feature_types: Dict[str, str], problem_type: str,
                                     max_features: int) -> Tuple[pd.DataFrame, List[str]]:
        """Engineer categorical features"""
        result_df = df.copy()
        methods = []
        features_added = 0
        
        categorical_cols = [col for col, ftype in feature_types.items() if ftype == 'categorical']
        
        if not categorical_cols or features_added >= max_features:
            return result_df, methods
        
        for col in categorical_cols:
            if features_added >= max_features:
                break
                
            col_data = df[col].fillna('missing')
            unique_values = col_data.nunique()
            
            try:
                # One-hot encoding for low cardinality
                if unique_values <= 10 and features_added + unique_values <= max_features:
                    dummies = pd.get_dummies(col_data, prefix=col, drop_first=True)
                    for dummy_col in dummies.columns:
                        result_df[dummy_col] = dummies[dummy_col]
                        features_added += 1
                    methods.append('one_hot_encoding')
                
                # Frequency encoding
                elif features_added < max_features:
                    freq_encoding = col_data.value_counts().to_dict()
                    result_df[f'{col}_frequency'] = col_data.map(freq_encoding)
                    methods.append('frequency_encoding')
                    features_added += 1
                
                # Target encoding for classification problems
                if problem_type == 'classification' and features_added < max_features:
                    target_mean = pd.DataFrame({'cat': col_data, 'target': y}).groupby('cat')['target'].mean()
                    result_df[f'{col}_target_encoded'] = col_data.map(target_mean)
                    methods.append('target_encoding')
                    features_added += 1
                    
            except Exception as e:
                logger.warning(f"Error engineering categorical feature {col}: {e}")
                continue
        
        return result_df, list(set(methods))
    
    def _engineer_text_features(self, df: pd.DataFrame, feature_types: Dict[str, str],
                              max_features: int) -> Tuple[pd.DataFrame, List[str]]:
        """Engineer text features"""
        result_df = df.copy()
        methods = []
        features_added = 0
        
        text_cols = [col for col, ftype in feature_types.items() if ftype == 'text']
        
        if not text_cols or features_added >= max_features:
            return result_df, methods
        
        for col in text_cols:
            if features_added >= max_features:
                break
                
            col_data = df[col].fillna('').astype(str)
            
            try:
                # Length features
                if features_added < max_features:
                    result_df[f'{col}_length'] = col_data.str.len()
                    methods.append('length_features')
                    features_added += 1
                
                # Word count features
                if features_added < max_features:
                    result_df[f'{col}_word_count'] = col_data.str.split().str.len()
                    methods.append('word_count_features')
                    features_added += 1
                
                # Character features
                if features_added < max_features:
                    result_df[f'{col}_digit_count'] = col_data.str.count(r'/d')
                    methods.append('character_features')
                    features_added += 1
                
                if features_added < max_features:
                    result_df[f'{col}_upper_count'] = col_data.str.count(r'[A-Z]')
                    features_added += 1
                
                # Keyword features (simple)
                if features_added < max_features:
                    common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of']
                    for word in common_words[:min(5, max_features - features_added)]:
                        result_df[f'{col}_has_{word}'] = col_data.str.contains(word, case=False).astype(int)
                        features_added += 1
                    methods.append('keyword_features')
                    
            except Exception as e:
                logger.warning(f"Error engineering text feature {col}: {e}")
                continue
        
        return result_df, list(set(methods))
    
    def _engineer_datetime_features(self, df: pd.DataFrame, feature_types: Dict[str, str]) -> Tuple[pd.DataFrame, List[str]]:
        """Engineer datetime features"""
        result_df = df.copy()
        methods = []
        
        datetime_cols = [col for col, ftype in feature_types.items() if ftype == 'datetime']
        
        if not datetime_cols:
            return result_df, methods
        
        for col in datetime_cols:
            try:
                dt_series = pd.to_datetime(df[col])
                
                # Basic datetime features
                result_df[f'{col}_year'] = dt_series.dt.year
                result_df[f'{col}_month'] = dt_series.dt.month
                result_df[f'{col}_day'] = dt_series.dt.day
                result_df[f'{col}_weekday'] = dt_series.dt.weekday
                result_df[f'{col}_hour'] = dt_series.dt.hour
                
                # Cyclical features
                result_df[f'{col}_month_sin'] = np.sin(2 * np.pi * dt_series.dt.month / 12)
                result_df[f'{col}_month_cos'] = np.cos(2 * np.pi * dt_series.dt.month / 12)
                
                methods.extend(['year_features', 'month_features', 'day_features', 
                               'weekday_features', 'hour_features', 'cyclical_features'])
                
            except Exception as e:
                logger.warning(f"Error engineering datetime feature {col}: {e}")
                continue
        
        return result_df, list(set(methods))
    
    def _create_feature_interactions(self, df: pd.DataFrame, y: np.ndarray,
                                   problem_type: str, max_features: int) -> Tuple[pd.DataFrame, List[str]]:
        """Create intelligent feature interactions"""
        result_df = df.copy()
        methods = []
        features_added = 0
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2 or features_added >= max_features:
            return result_df, methods
        
        try:
            # Calculate correlations with target
            correlations = {}
            for col in numeric_cols:
                if problem_type == 'classification':
                    # For classification, use point-biserial correlation approximation
                    corr = np.corrcoef(df[col].fillna(0), y)[0, 1]
                else:
                    corr = np.corrcoef(df[col].fillna(0), y)[0, 1]
                
                if not np.isnan(corr):
                    correlations[col] = abs(corr)
            
            # Sort columns by correlation with target
            sorted_cols = sorted(correlations.keys(), key=lambda x: correlations[x], reverse=True)[:10]
            
            # Create interactions between top correlated features
            for i, col1 in enumerate(sorted_cols):
                for j, col2 in enumerate(sorted_cols[i+1:], i+1):
                    if features_added >= max_features:
                        break
                    
                    col1_data = df[col1].fillna(0)
                    col2_data = df[col2].fillna(0)
                    
                    # Multiplication interaction
                    result_df[f'{col1}_x_{col2}'] = col1_data * col2_data
                    features_added += 1
                    
                    # Ratio interaction (avoid division by zero)
                    if features_added < max_features and (col2_data != 0).all():
                        result_df[f'{col1}_div_{col2}'] = col1_data / (col2_data + 1e-6)
                        features_added += 1
                    
                    # Difference interaction
                    if features_added < max_features:
                        result_df[f'{col1}_minus_{col2}'] = col1_data - col2_data
                        features_added += 1
            
            methods.append('interaction_features')
            
        except Exception as e:
            logger.warning(f"Error creating feature interactions: {e}")
        
        return result_df, list(set(methods))
    
    def _apply_dimensionality_reduction(self, df: pd.DataFrame, y: np.ndarray,
                                      problem_type: str, n_components: int) -> Tuple[pd.DataFrame, List[str]]:
        """Apply dimensionality reduction techniques"""
        if not SKLEARN_AVAILABLE:
            return df, []
        
        result_df = df.copy()
        methods = []
        
        # Get numeric data only
        numeric_data = df.select_dtypes(include=[np.number]).fillna(0)
        
        if numeric_data.shape[1] < 5:
            return result_df, methods
        
        try:
            # PCA
            pca = PCA(n_components=min(n_components, numeric_data.shape[1], numeric_data.shape[0]))
            pca_features = pca.fit_transform(numeric_data)
            
            for i in range(pca_features.shape[1]):
                result_df[f'pca_component_{i}'] = pca_features[:, i]
            
            methods.append('pca_features')
            
            # Clustering features (if enough samples)
            if numeric_data.shape[0] > 100:
                n_clusters = min(8, numeric_data.shape[0] // 20)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(numeric_data)
                result_df['cluster_id'] = cluster_labels
                
                # Distance to cluster centers
                distances = kmeans.transform(numeric_data)
                for i in range(n_clusters):
                    result_df[f'distance_to_cluster_{i}'] = distances[:, i]
                
                methods.append('clustering_features')
                
        except Exception as e:
            logger.warning(f"Error applying dimensionality reduction: {e}")
        
        return result_df, list(set(methods))
    
    def _optimize_feature_selection(self, df: pd.DataFrame, y: np.ndarray,
                                  problem_type: str, max_features: int) -> Tuple[pd.DataFrame, List[str]]:
        """Optimize feature selection using multiple methods"""
        if not SKLEARN_AVAILABLE:
            return df.iloc[:, :max_features], list(df.columns[:max_features])
        
        # Fill NaN values
        df_filled = df.fillna(0)
        
        if df_filled.shape[1] <= max_features:
            return df_filled, list(df_filled.columns)
        
        try:
            # Use mutual information for feature selection
            if problem_type == 'classification':
                selector = SelectKBest(mutual_info_classif, k=min(max_features, df_filled.shape[1]))
            else:
                selector = SelectKBest(mutual_info_regression, k=min(max_features, df_filled.shape[1]))
            
            selected_features = selector.fit_transform(df_filled, y)
            selected_indices = selector.get_support(indices=True)
            selected_columns = [df_filled.columns[i] for i in selected_indices]
            
            return pd.DataFrame(selected_features, columns=selected_columns, index=df.index), selected_columns
            
        except Exception as e:
            logger.warning(f"Error in feature selection: {e}")
            return df_filled.iloc[:, :max_features], list(df_filled.columns[:max_features])
    
    def _calculate_feature_importance(self, df: pd.DataFrame, y: np.ndarray,
                                    problem_type: str) -> List[Tuple[str, float]]:
        """Calculate feature importance using tree-based models"""
        if not SKLEARN_AVAILABLE:
            return []
        
        try:
            df_filled = df.fillna(0)
            
            if problem_type == 'classification':
                model = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            
            model.fit(df_filled, y)
            
            # Get feature importance
            importances = model.feature_importances_
            feature_names = df_filled.columns
            
            # Create importance pairs and sort
            importance_pairs = list(zip(feature_names, importances))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return importance_pairs[:20]  # Top 20 features
            
        except Exception as e:
            logger.warning(f"Error calculating feature importance: {e}")
            return []
    
    def _calculate_feature_statistics(self, df: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive feature statistics"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            stats = {
                'total_features': len(df.columns),
                'numeric_features': len(numeric_df.columns),
                'categorical_features': len(df.select_dtypes(exclude=[np.number]).columns),
                'missing_values_count': df.isnull().sum().sum(),
                'missing_values_percent': (df.isnull().sum().sum() / df.size) * 100,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'correlation_with_target': {}
            }
            
            # Calculate correlations with target for numeric features
            for col in numeric_df.columns:
                try:
                    corr = np.corrcoef(numeric_df[col].fillna(0), y)[0, 1]
                    if not np.isnan(corr):
                        stats['correlation_with_target'][col] = float(corr)
                except:
                    continue
            
            return stats
            
        except Exception as e:
            logger.warning(f"Error calculating feature statistics: {e}")
            return {}
    
    def _get_baseline_performance(self, X: pd.DataFrame, y: np.ndarray, problem_type: str) -> float:
        """Get baseline model performance"""
        if not SKLEARN_AVAILABLE:
            return 0.0
        
        try:
            X_filled = X.fillna(0)
            
            if problem_type == 'classification':
                model = RandomForestClassifier(n_estimators=10, random_state=42)
                scores = cross_val_score(model, X_filled, y, cv=3, scoring='accuracy')
            else:
                model = RandomForestRegressor(n_estimators=10, random_state=42)
                scores = cross_val_score(model, X_filled, y, cv=3, scoring='r2')
            
            return float(scores.mean())
            
        except Exception as e:
            logger.warning(f"Error calculating baseline performance: {e}")
            return 0.0
    
    def _cache_engineering_result(self, result: FeatureEngineering):
        """Cache feature engineering result"""
        try:
            cache_file = self.cache_dir / f"feature_engineering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert to JSON-serializable format
            result_dict = asdict(result)
            
            with open(cache_file, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
                
            logger.debug(f"Cached feature engineering result to {cache_file}")
            
        except Exception as e:
            logger.warning(f"Could not cache feature engineering result: {e}")
    
    def get_engineering_summary(self) -> Dict[str, Any]:
        """Get summary of all feature engineering activities"""
        with self.lock:
            if not self.engineering_history:
                return {
                    "status": "no_engineering",
                    "total_engineering_runs": 0
                }
            
            total_runs = len(self.engineering_history)
            avg_improvement = np.mean([r.performance_improvement for r in self.engineering_history])
            best_improvement = max(r.performance_improvement for r in self.engineering_history)
            
            # Method usage statistics
            method_usage = defaultdict(int)
            for result in self.engineering_history:
                for method in result.engineering_methods:
                    method_usage[method] += 1
            
            return {
                "total_engineering_runs": total_runs,
                "average_performance_improvement": float(avg_improvement),
                "best_performance_improvement": float(best_improvement),
                "most_used_methods": dict(sorted(method_usage.items(), key=lambda x: x[1], reverse=True)[:5]),
                "features_available": [
                    "Automatic feature type detection",
                    "Numeric feature transformations",
                    "Categorical feature encoding",
                    "Text feature extraction",
                    "DateTime feature engineering",
                    "Feature interaction creation",
                    "Dimensionality reduction",
                    "Intelligent feature selection",
                    "Performance-based optimization"
                ]
            }


# Global feature engineer instance
feature_engineer = IntelligentFeatureEngineer()
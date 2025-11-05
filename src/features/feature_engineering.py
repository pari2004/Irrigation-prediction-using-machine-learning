"""
Feature engineering pipeline for irrigation prediction model.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings


class IrrigationFeatureEngineer:
    """Feature engineering for irrigation prediction."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.is_fitted = False
    
    def create_lag_features(self, df: pd.DataFrame, columns: List[str], 
                           lags: List[int], group_col: str = 'zone_id') -> pd.DataFrame:
        """Create lag features for specified columns."""
        df_with_lags = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    lag_col = f"{col}_lag_{lag}"
                    df_with_lags[lag_col] = df_with_lags.groupby(group_col)[col].shift(lag)
        
        return df_with_lags
    
    def create_rolling_features(self, df: pd.DataFrame, columns: List[str],
                               windows: List[int], group_col: str = 'zone_id') -> pd.DataFrame:
        """Create rolling statistics features."""
        df_with_rolling = df.copy()
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    # Rolling mean
                    mean_col = f"{col}_rolling_mean_{window}d"
                    df_with_rolling[mean_col] = (
                        df_with_rolling.groupby(group_col)[col]
                        .rolling(window=window, min_periods=1)
                        .mean()
                        .reset_index(level=0, drop=True)
                    )
                    
                    # Rolling sum for cumulative variables
                    if col in ['rain_mm', 'irrigation_mm_next_day', 'ET0_mm']:
                        sum_col = f"{col}_rolling_sum_{window}d"
                        df_with_rolling[sum_col] = (
                            df_with_rolling.groupby(group_col)[col]
                            .rolling(window=window, min_periods=1)
                            .sum()
                            .reset_index(level=0, drop=True)
                        )
                    
                    # Rolling std for variability
                    if window > 2:
                        std_col = f"{col}_rolling_std_{window}d"
                        df_with_rolling[std_col] = (
                            df_with_rolling.groupby(group_col)[col]
                            .rolling(window=window, min_periods=2)
                            .std()
                            .reset_index(level=0, drop=True)
                        )
        
        return df_with_rolling
    
    def create_soil_moisture_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create soil moisture derived features."""
        df_enhanced = df.copy()
        
        # Soil moisture trend (current vs 3-day average)
        theta_cols = [f'theta_{i}' if i > 0 else 'theta0' for i in range(7)]
        available_theta_cols = [col for col in theta_cols if col in df.columns]
        
        if len(available_theta_cols) >= 3:
            # 3-day average
            df_enhanced['theta_3d_mean'] = df_enhanced[available_theta_cols[:3]].mean(axis=1)
            
            # Trend (current - 3-day average)
            df_enhanced['theta_trend_3d'] = df_enhanced['theta0'] - df_enhanced['theta_3d_mean']
            
            # Volatility (std of recent measurements)
            df_enhanced['theta_volatility_3d'] = df_enhanced[available_theta_cols[:3]].std(axis=1)
        
        # Soil water deficit features
        if all(col in df.columns for col in ['theta0', 'field_capacity_theta', 'wilting_point_theta']):
            # Available water content
            df_enhanced['awc_current'] = (
                df_enhanced['theta0'] - df_enhanced['wilting_point_theta']
            ) / (df_enhanced['field_capacity_theta'] - df_enhanced['wilting_point_theta'])
            
            # Deficit from field capacity
            df_enhanced['deficit_from_fc'] = (
                df_enhanced['field_capacity_theta'] - df_enhanced['theta0']
            )
            
            # Stress indicator (below 50% available water)
            df_enhanced['stress_indicator'] = (df_enhanced['awc_current'] < 0.5).astype(int)
            
            # Days to wilting point (simplified)
            if 'ET0_mm' in df.columns and 'Kc_stage' in df.columns:
                daily_et = df_enhanced['ET0_mm'] * df_enhanced['Kc_stage']
                available_mm = (
                    (df_enhanced['theta0'] - df_enhanced['wilting_point_theta']) * 
                    df_enhanced['root_depth_m'] * 1000
                )
                df_enhanced['days_to_wilting'] = np.where(
                    daily_et > 0, available_mm / daily_et, 999
                ).clip(0, 30)  # Cap at 30 days
        
        return df_enhanced
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weather-derived features."""
        df_enhanced = df.copy()
        
        # Temperature features
        if all(col in df.columns for col in ['tmax', 'tmin']):
            df_enhanced['temp_range'] = df_enhanced['tmax'] - df_enhanced['tmin']
            df_enhanced['temp_mean'] = (df_enhanced['tmax'] + df_enhanced['tmin']) / 2
        
        # Vapor pressure deficit (simplified)
        if all(col in df.columns for col in ['tmax', 'tmin', 'RH_mean']):
            tmean = (df_enhanced['tmax'] + df_enhanced['tmin']) / 2
            es = 0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))  # Saturation VP
            ea = es * df_enhanced['RH_mean'] / 100  # Actual VP
            df_enhanced['vpd'] = es - ea
        
        # Atmospheric demand
        if all(col in df.columns for col in ['ET0_mm', 'Kc_stage']):
            df_enhanced['atmospheric_demand'] = df_enhanced['ET0_mm'] * df_enhanced['Kc_stage']
        
        # Water balance features
        if all(col in df.columns for col in ['rain_mm', 'ET0_mm', 'Kc_stage']):
            et_crop = df_enhanced['ET0_mm'] * df_enhanced['Kc_stage']
            df_enhanced['water_deficit'] = np.maximum(0, et_crop - df_enhanced['rain_mm'])
            df_enhanced['water_surplus'] = np.maximum(0, df_enhanced['rain_mm'] - et_crop)
        
        # Forecast features
        if all(col in df.columns for col in ['forecast_ET0_mm', 'forecast_rain_mm', 'Kc_stage']):
            forecast_et_crop = df_enhanced['forecast_ET0_mm'] * df_enhanced['Kc_stage']
            df_enhanced['forecast_water_deficit'] = np.maximum(
                0, forecast_et_crop - df_enhanced['forecast_rain_mm']
            )
        
        return df_enhanced
    
    def create_irrigation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create irrigation management features."""
        df_enhanced = df.copy()
        
        # Irrigation intensity
        if 'irrigation_mm_last_7d' in df.columns:
            df_enhanced['irrigation_intensity_7d'] = df_enhanced['irrigation_mm_last_7d'] / 7
        
        # Time since irrigation categories
        if 'days_since_irrigation' in df.columns:
            df_enhanced['irrigation_urgency'] = pd.cut(
                df_enhanced['days_since_irrigation'],
                bins=[-1, 0, 2, 5, 10, 999],
                labels=['today', 'recent', 'moderate', 'overdue', 'critical']
            ).astype(str)
        
        # Irrigation efficiency (if we have area)
        if all(col in df.columns for col in ['last_irrigation_mm', 'area_m2']):
            df_enhanced['last_irrigation_liters'] = (
                df_enhanced['last_irrigation_mm'] * df_enhanced['area_m2']
            )
        
        return df_enhanced
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from date."""
        df_enhanced = df.copy()
        
        if 'date' in df.columns:
            df_enhanced['date'] = pd.to_datetime(df_enhanced['date'])
            
            # Basic temporal features
            df_enhanced['day_of_year'] = df_enhanced['date'].dt.dayofyear
            df_enhanced['month'] = df_enhanced['date'].dt.month
            df_enhanced['week_of_year'] = df_enhanced['date'].dt.isocalendar().week
            
            # Seasonal features (sinusoidal encoding)
            df_enhanced['day_sin'] = np.sin(2 * np.pi * df_enhanced['day_of_year'] / 365.25)
            df_enhanced['day_cos'] = np.cos(2 * np.pi * df_enhanced['day_of_year'] / 365.25)
            
            # Growing season indicator (simplified)
            df_enhanced['growing_season'] = (
                (df_enhanced['month'] >= 4) & (df_enhanced['month'] <= 10)
            ).astype(int)
        
        return df_enhanced
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        df_enhanced = df.copy()
        
        # Crop stage interactions
        if all(col in df.columns for col in ['Kc_stage', 'ET0_mm']):
            df_enhanced['kc_et0_interaction'] = df_enhanced['Kc_stage'] * df_enhanced['ET0_mm']
        
        # Soil moisture and weather interactions
        if all(col in df.columns for col in ['theta0', 'ET0_mm']):
            df_enhanced['theta_et_interaction'] = df_enhanced['theta0'] * df_enhanced['ET0_mm']
        
        # Stress and demand interaction
        if all(col in df.columns for col in ['awc_current', 'atmospheric_demand']):
            df_enhanced['stress_demand_interaction'] = (
                (1 - df_enhanced['awc_current']) * df_enhanced['atmospheric_demand']
            )
        
        return df_enhanced
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                  categorical_cols: List[str] = None) -> pd.DataFrame:
        """Encode categorical features."""
        if categorical_cols is None:
            categorical_cols = ['crop_type', 'growth_stage', 'zone_id', 'irrigation_urgency']
        
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = self.encoders[col].fit_transform(
                        df_encoded[col].astype(str)
                    )
                else:
                    # Handle unseen categories
                    unique_vals = df_encoded[col].astype(str).unique()
                    known_vals = self.encoders[col].classes_
                    
                    # Map unknown values to a default
                    df_encoded[col] = df_encoded[col].astype(str).apply(
                        lambda x: x if x in known_vals else known_vals[0]
                    )
                    df_encoded[f'{col}_encoded'] = self.encoders[col].transform(df_encoded[col])
        
        return df_encoded
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit feature engineering pipeline and transform data."""
        df_processed = df.copy()
        
        # Create all features
        df_processed = self.create_temporal_features(df_processed)
        df_processed = self.create_soil_moisture_features(df_processed)
        df_processed = self.create_weather_features(df_processed)
        df_processed = self.create_irrigation_features(df_processed)
        
        # Create lag and rolling features
        weather_cols = ['ET0_mm', 'rain_mm', 'tmax', 'tmin']
        df_processed = self.create_lag_features(df_processed, weather_cols, [1, 2, 3])
        df_processed = self.create_rolling_features(df_processed, weather_cols, [3, 7])
        
        # Soil moisture rolling features
        theta_cols = ['theta0']
        df_processed = self.create_rolling_features(df_processed, theta_cols, [3, 7])
        
        # Create interaction features
        df_processed = self.create_interaction_features(df_processed)
        
        # Encode categorical features
        df_processed = self.encode_categorical_features(df_processed)
        
        # Store feature names (excluding target and metadata)
        exclude_cols = [
            'date', 'zone_id', 'irrigation_mm_next_day', 'crop_type', 
            'growth_stage', 'irrigation_urgency'
        ]
        self.feature_names = [
            col for col in df_processed.columns 
            if col not in exclude_cols and not col.endswith('_encoded')
        ]
        
        # Add encoded categorical features
        encoded_cols = [col for col in df_processed.columns if col.endswith('_encoded')]
        self.feature_names.extend(encoded_cols)
        
        self.is_fitted = True
        return df_processed
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before transform")
        
        # Apply same transformations as fit_transform
        return self.fit_transform(df)  # For simplicity, reuse fit_transform logic
    
    def get_feature_names(self) -> List[str]:
        """Get list of engineered feature names."""
        return self.feature_names.copy()
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Group features by type for importance analysis."""
        groups = {
            'soil_moisture': [],
            'weather': [],
            'irrigation_history': [],
            'temporal': [],
            'interactions': [],
            'categorical': []
        }
        
        for feature in self.feature_names:
            if 'theta' in feature or 'awc' in feature or 'deficit' in feature:
                groups['soil_moisture'].append(feature)
            elif any(x in feature for x in ['ET0', 'rain', 'temp', 'RH', 'wind', 'solar', 'vpd']):
                groups['weather'].append(feature)
            elif 'irrigation' in feature or 'days_since' in feature:
                groups['irrigation_history'].append(feature)
            elif any(x in feature for x in ['day_', 'month', 'week', 'season']):
                groups['temporal'].append(feature)
            elif 'interaction' in feature:
                groups['interactions'].append(feature)
            elif 'encoded' in feature:
                groups['categorical'].append(feature)
            else:
                # Default to weather if unclear
                groups['weather'].append(feature)
        
        return groups

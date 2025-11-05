"""
Data schema definitions and validation for the irrigation prediction system.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
import pandas as pd
import numpy as np


class IrrigationDataSchema(BaseModel):
    """Schema for irrigation prediction dataset."""
    
    # Temporal
    date: datetime
    zone_id: str
    area_m2: float = Field(gt=0, description="Zone area in square meters")
    
    # Crop information
    crop_type: str
    growth_stage: str = Field(description="early/mid/late or specific stage")
    Kc_stage: float = Field(ge=0, le=2.0, description="Crop coefficient for current stage")
    
    # Weather (current/yesterday)
    ET0_mm: float = Field(ge=0, description="Reference evapotranspiration")
    rain_mm: float = Field(ge=0, description="Rainfall")
    tmax: float = Field(description="Maximum temperature (°C)")
    tmin: float = Field(description="Minimum temperature (°C)")
    RH_mean: float = Field(ge=0, le=100, description="Mean relative humidity (%)")
    wind_2m: float = Field(ge=0, description="Wind speed at 2m (m/s)")
    solar_rad: float = Field(ge=0, description="Solar radiation (MJ/m²/day)")
    
    # Weather forecast (D+1)
    forecast_ET0_mm: float = Field(ge=0, description="Forecast ET0 for next day")
    forecast_rain_mm: float = Field(ge=0, description="Forecast rainfall for next day")
    
    # Soil properties
    root_depth_m: float = Field(gt=0, le=3.0, description="Effective root depth (m)")
    field_capacity_theta: float = Field(ge=0, le=1.0, description="Field capacity (VWC)")
    wilting_point_theta: float = Field(ge=0, le=1.0, description="Wilting point (VWC)")
    bulk_density: Optional[float] = Field(ge=0.8, le=2.0, description="Soil bulk density (g/cm³)")
    
    # Soil moisture history (volumetric water content)
    theta0: float = Field(ge=0, le=1.0, description="Current day soil moisture")
    theta_1: float = Field(ge=0, le=1.0, description="1 day ago soil moisture")
    theta_2: float = Field(ge=0, le=1.0, description="2 days ago soil moisture")
    theta_3: float = Field(ge=0, le=1.0, description="3 days ago soil moisture")
    theta_4: float = Field(ge=0, le=1.0, description="4 days ago soil moisture")
    theta_5: float = Field(ge=0, le=1.0, description="5 days ago soil moisture")
    theta_6: float = Field(ge=0, le=1.0, description="6 days ago soil moisture")
    
    # Irrigation history
    irrigation_mm_last_7d: float = Field(ge=0, description="Total irrigation in last 7 days")
    last_irrigation_mm: float = Field(ge=0, description="Last irrigation amount")
    days_since_irrigation: int = Field(ge=0, description="Days since last irrigation")
    
    # Target variable
    irrigation_mm_next_day: float = Field(ge=0, description="Target irrigation for next day")
    
    @validator('field_capacity_theta', 'wilting_point_theta')
    def validate_soil_water_bounds(cls, v, values):
        """Ensure field capacity > wilting point."""
        if 'wilting_point_theta' in values and 'field_capacity_theta' in values:
            if values['field_capacity_theta'] <= values['wilting_point_theta']:
                raise ValueError("Field capacity must be greater than wilting point")
        return v
    
    @validator('theta0', 'theta_1', 'theta_2', 'theta_3', 'theta_4', 'theta_5', 'theta_6')
    def validate_soil_moisture_realistic(cls, v, values):
        """Ensure soil moisture is within reasonable bounds."""
        if 'field_capacity_theta' in values and v > values['field_capacity_theta'] * 1.1:
            raise ValueError(f"Soil moisture {v} exceeds field capacity by too much")
        return v


class PredictionInput(BaseModel):
    """Schema for prediction input (without target)."""
    
    # All fields from IrrigationDataSchema except irrigation_mm_next_day
    date: datetime
    zone_id: str
    area_m2: float = Field(gt=0)
    crop_type: str
    growth_stage: str
    Kc_stage: float = Field(ge=0, le=2.0)
    ET0_mm: float = Field(ge=0)
    rain_mm: float = Field(ge=0)
    tmax: float
    tmin: float
    RH_mean: float = Field(ge=0, le=100)
    wind_2m: float = Field(ge=0)
    solar_rad: float = Field(ge=0)
    forecast_ET0_mm: float = Field(ge=0)
    forecast_rain_mm: float = Field(ge=0)
    root_depth_m: float = Field(gt=0, le=3.0)
    field_capacity_theta: float = Field(ge=0, le=1.0)
    wilting_point_theta: float = Field(ge=0, le=1.0)
    bulk_density: Optional[float] = Field(ge=0.8, le=2.0)
    theta0: float = Field(ge=0, le=1.0)
    theta_1: float = Field(ge=0, le=1.0)
    theta_2: float = Field(ge=0, le=1.0)
    theta_3: float = Field(ge=0, le=1.0)
    theta_4: float = Field(ge=0, le=1.0)
    theta_5: float = Field(ge=0, le=1.0)
    theta_6: float = Field(ge=0, le=1.0)
    irrigation_mm_last_7d: float = Field(ge=0)
    last_irrigation_mm: float = Field(ge=0)
    days_since_irrigation: int = Field(ge=0)


class PredictionOutput(BaseModel):
    """Schema for model predictions."""
    
    zone_id: str
    date: datetime
    irrigation_mm_raw: float = Field(description="Raw model prediction")
    irrigation_mm_safe: float = Field(description="Post-processed safe prediction")
    irrigation_liters: float = Field(description="Irrigation volume in liters")
    confidence_score: Optional[float] = Field(ge=0, le=1.0, description="Prediction confidence")
    
    # Diagnostic information
    physics_baseline_mm: float = Field(description="Physics-based baseline")
    ml_residual_mm: float = Field(description="ML-predicted residual")
    max_allowed_mm: float = Field(description="Maximum irrigation to avoid FC violation")
    
    # Soil moisture projections
    projected_theta_tomorrow: float = Field(description="Projected soil moisture after irrigation")
    stress_risk: str = Field(description="low/medium/high stress risk assessment")


def validate_dataframe(df: pd.DataFrame, schema_class=IrrigationDataSchema) -> List[Dict[str, Any]]:
    """
    Validate a pandas DataFrame against the schema.
    
    Args:
        df: DataFrame to validate
        schema_class: Pydantic schema class to validate against
        
    Returns:
        List of validation errors (empty if all valid)
    """
    errors = []
    
    for idx, row in df.iterrows():
        try:
            schema_class(**row.to_dict())
        except Exception as e:
            errors.append({
                'row_index': idx,
                'error': str(e),
                'row_data': row.to_dict()
            })
    
    return errors


def get_feature_columns() -> List[str]:
    """Get list of feature columns for ML models."""
    return [
        "ET0_mm", "forecast_ET0_mm", "rain_mm", "forecast_rain_mm", "Kc_stage",
        "theta0", "theta_1", "theta_2", "theta_3", "theta_4", "theta_5", "theta_6",
        "root_depth_m", "field_capacity_theta", "wilting_point_theta",
        "irrigation_mm_last_7d", "last_irrigation_mm", "days_since_irrigation",
        "tmax", "tmin", "RH_mean", "wind_2m", "solar_rad", "area_m2"
    ]


def get_target_column() -> str:
    """Get target column name."""
    return "irrigation_mm_next_day"


# Crop coefficient lookup table (simplified)
CROP_COEFFICIENTS = {
    "tomato": {"early": 0.6, "mid": 1.15, "late": 0.8},
    "corn": {"early": 0.3, "mid": 1.2, "late": 0.6},
    "wheat": {"early": 0.4, "mid": 1.15, "late": 0.4},
    "lettuce": {"early": 0.7, "mid": 1.0, "late": 0.95},
    "potato": {"early": 0.5, "mid": 1.15, "late": 0.75},
    "default": {"early": 0.5, "mid": 1.0, "late": 0.7}
}


def get_crop_coefficient(crop_type: str, growth_stage: str) -> float:
    """Get crop coefficient for given crop and stage."""
    crop_data = CROP_COEFFICIENTS.get(crop_type.lower(), CROP_COEFFICIENTS["default"])
    return crop_data.get(growth_stage.lower(), crop_data["mid"])

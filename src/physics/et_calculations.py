"""
FAO-56 Penman-Monteith ET0 calculations and related physics-based functions.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional


def penman_monteith_et0(tmax: float, tmin: float, rh_mean: float, 
                       wind_2m: float, solar_rad: float, 
                       elevation: float = 100.0, latitude: float = 40.0,
                       day_of_year: int = 180) -> float:
    """
    Calculate reference evapotranspiration (ET0) using FAO-56 Penman-Monteith equation.
    
    Args:
        tmax: Maximum temperature (°C)
        tmin: Minimum temperature (°C)
        rh_mean: Mean relative humidity (%)
        wind_2m: Wind speed at 2m height (m/s)
        solar_rad: Solar radiation (MJ/m²/day)
        elevation: Elevation above sea level (m)
        latitude: Latitude (degrees)
        day_of_year: Day of year (1-365)
    
    Returns:
        ET0 in mm/day
    """
    # Mean temperature
    tmean = (tmax + tmin) / 2.0
    
    # Atmospheric pressure (kPa)
    P = 101.3 * ((293 - 0.0065 * elevation) / 293) ** 5.26
    
    # Psychrometric constant (kPa/°C)
    gamma = 0.665 * P
    
    # Saturation vapor pressure (kPa)
    es_tmax = 0.6108 * np.exp(17.27 * tmax / (tmax + 237.3))
    es_tmin = 0.6108 * np.exp(17.27 * tmin / (tmin + 237.3))
    es = (es_tmax + es_tmin) / 2.0
    
    # Actual vapor pressure (kPa)
    ea = es * rh_mean / 100.0
    
    # Slope of saturation vapor pressure curve (kPa/°C)
    delta = 4098 * (0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))) / (tmean + 237.3) ** 2
    
    # Net radiation calculation (simplified)
    # Solar declination
    dr = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
    delta_rad = 0.409 * np.sin(2 * np.pi * day_of_year / 365 - 1.39)
    
    # Sunset hour angle
    lat_rad = np.pi * latitude / 180
    ws = np.arccos(-np.tan(lat_rad) * np.tan(delta_rad))
    
    # Extraterrestrial radiation
    Ra = (24 * 60 / np.pi) * 0.082 * dr * (
        ws * np.sin(lat_rad) * np.sin(delta_rad) + 
        np.cos(lat_rad) * np.cos(delta_rad) * np.sin(ws)
    )
    
    # Clear sky solar radiation
    Rs0 = (0.75 + 2e-5 * elevation) * Ra
    
    # Net shortwave radiation
    Rns = (1 - 0.23) * solar_rad  # Albedo = 0.23 for grass reference
    
    # Net longwave radiation (simplified)
    Rnl = (4.903e-9 * ((tmax + 273.16)**4 + (tmin + 273.16)**4) / 2 * 
           (0.34 - 0.14 * np.sqrt(ea)) * 
           (1.35 * min(solar_rad / Rs0, 1.0) - 0.35))
    
    # Net radiation
    Rn = Rns - Rnl
    
    # Soil heat flux (assumed negligible for daily calculations)
    G = 0.0
    
    # ET0 calculation
    numerator = 0.408 * delta * (Rn - G) + gamma * 900 / (tmean + 273) * wind_2m * (es - ea)
    denominator = delta + gamma * (1 + 0.34 * wind_2m)
    
    et0 = numerator / denominator
    
    return max(0.0, et0)


def calculate_crop_et(et0: float, kc: float) -> float:
    """
    Calculate crop evapotranspiration.
    
    Args:
        et0: Reference evapotranspiration (mm/day)
        kc: Crop coefficient
    
    Returns:
        Crop ET in mm/day
    """
    return et0 * kc


def effective_rainfall(rainfall: float, et_crop: float, 
                      runoff_factor: float = 0.1) -> float:
    """
    Calculate effective rainfall using simple method.
    
    Args:
        rainfall: Daily rainfall (mm)
        et_crop: Crop evapotranspiration (mm)
        runoff_factor: Fraction lost to runoff
    
    Returns:
        Effective rainfall (mm)
    """
    # Simple method: effective rain is limited by crop demand and runoff
    max_effective = et_crop * 1.5  # Can exceed daily ET slightly
    effective = rainfall * (1 - runoff_factor)
    return min(effective, max_effective)


def soil_water_balance(theta_prev: float, irrigation: float, rain_eff: float,
                      et_crop: float, field_capacity: float, wilting_point: float,
                      root_depth: float, drainage_rate: float = 0.1) -> dict:
    """
    Simple daily soil water balance.
    
    Args:
        theta_prev: Previous day soil moisture (VWC)
        irrigation: Irrigation applied (mm)
        rain_eff: Effective rainfall (mm)
        et_crop: Crop evapotranspiration (mm)
        field_capacity: Field capacity (VWC)
        wilting_point: Wilting point (VWC)
        root_depth: Root depth (m)
        drainage_rate: Drainage rate when above FC (fraction/day)
    
    Returns:
        Dictionary with water balance components
    """
    # Convert VWC to mm of water
    theta_prev_mm = theta_prev * root_depth * 1000
    fc_mm = field_capacity * root_depth * 1000
    wp_mm = wilting_point * root_depth * 1000
    
    # Water balance
    water_in = irrigation + rain_eff
    water_out = et_crop
    
    # New soil water content
    theta_new_mm = theta_prev_mm + water_in - water_out
    
    # Drainage if above field capacity
    drainage = 0.0
    if theta_new_mm > fc_mm:
        excess = theta_new_mm - fc_mm
        drainage = excess * drainage_rate
        theta_new_mm -= drainage
    
    # Cannot go below wilting point (plant death)
    theta_new_mm = max(wp_mm, theta_new_mm)
    
    # Convert back to VWC
    theta_new = theta_new_mm / (root_depth * 1000)
    
    return {
        'theta_new': theta_new,
        'theta_new_mm': theta_new_mm,
        'water_in': water_in,
        'water_out': water_out,
        'drainage': drainage,
        'stress': theta_new < wilting_point * 1.1  # Stress threshold
    }


def calculate_irrigation_need(theta_current: float, field_capacity: float,
                             wilting_point: float, root_depth: float,
                             et_forecast: float, rain_forecast: float,
                             mad_factor: float = 0.5) -> dict:
    """
    Calculate irrigation need based on soil moisture and forecast.
    
    Args:
        theta_current: Current soil moisture (VWC)
        field_capacity: Field capacity (VWC)
        wilting_point: Wilting point (VWC)
        root_depth: Root depth (m)
        et_forecast: Forecast ET for next day (mm)
        rain_forecast: Forecast rainfall for next day (mm)
        mad_factor: Management allowed depletion (0-1)
    
    Returns:
        Dictionary with irrigation recommendations
    """
    # Convert to mm
    theta_mm = theta_current * root_depth * 1000
    fc_mm = field_capacity * root_depth * 1000
    wp_mm = wilting_point * root_depth * 1000
    
    # Total available water and readily available water
    taw = fc_mm - wp_mm
    raw = taw * mad_factor
    
    # Management threshold
    threshold_mm = fc_mm - raw
    
    # Effective rainfall forecast
    rain_eff = effective_rainfall(rain_forecast, et_forecast)
    
    # Projected soil moisture after ET and rain
    theta_projected_mm = theta_mm - et_forecast + rain_eff
    
    # Irrigation need
    if theta_projected_mm < threshold_mm:
        # Irrigate to field capacity
        irrigation_need = fc_mm - theta_projected_mm
    else:
        irrigation_need = 0.0
    
    # Ensure non-negative
    irrigation_need = max(0.0, irrigation_need)
    
    return {
        'irrigation_mm': irrigation_need,
        'theta_projected': theta_projected_mm / (root_depth * 1000),
        'taw_mm': taw,
        'raw_mm': raw,
        'threshold_mm': threshold_mm,
        'depletion_fraction': (fc_mm - theta_mm) / taw,
        'stress_risk': theta_projected_mm < threshold_mm
    }


class PhysicsBaseline:
    """Physics-based irrigation baseline model."""
    
    def __init__(self, mad_factor: float = 0.5, elevation: float = 100.0,
                 latitude: float = 40.0):
        """
        Initialize physics baseline model.
        
        Args:
            mad_factor: Management allowed depletion fraction
            elevation: Site elevation (m)
            latitude: Site latitude (degrees)
        """
        self.mad_factor = mad_factor
        self.elevation = elevation
        self.latitude = latitude
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict irrigation needs using physics-based approach.
        
        Args:
            data: DataFrame with required columns
        
        Returns:
            Array of irrigation predictions (mm)
        """
        predictions = []
        
        for _, row in data.iterrows():
            # Calculate ET0 if not provided
            if 'ET0_mm' in row and not pd.isna(row['ET0_mm']):
                et0 = row['ET0_mm']
            else:
                et0 = penman_monteith_et0(
                    row['tmax'], row['tmin'], row['RH_mean'],
                    row['wind_2m'], row['solar_rad'],
                    self.elevation, self.latitude,
                    row.get('day_of_year', 180)
                )
            
            # Crop ET
            et_crop = calculate_crop_et(et0, row['Kc_stage'])
            
            # Forecast ET
            et_forecast = calculate_crop_et(row['forecast_ET0_mm'], row['Kc_stage'])
            
            # Calculate irrigation need
            result = calculate_irrigation_need(
                row['theta0'], row['field_capacity_theta'],
                row['wilting_point_theta'], row['root_depth_m'],
                et_forecast, row['forecast_rain_mm'],
                self.mad_factor
            )
            
            predictions.append(result['irrigation_mm'])
        
        return np.array(predictions)
    
    def get_diagnostics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get detailed diagnostics for physics-based predictions."""
        diagnostics = []
        
        for _, row in data.iterrows():
            # Calculate all components
            et0 = row.get('ET0_mm', penman_monteith_et0(
                row['tmax'], row['tmin'], row['RH_mean'],
                row['wind_2m'], row['solar_rad'],
                self.elevation, self.latitude
            ))
            
            et_crop = calculate_crop_et(et0, row['Kc_stage'])
            et_forecast = calculate_crop_et(row['forecast_ET0_mm'], row['Kc_stage'])
            rain_eff = effective_rainfall(row['forecast_rain_mm'], et_forecast)
            
            irrigation_result = calculate_irrigation_need(
                row['theta0'], row['field_capacity_theta'],
                row['wilting_point_theta'], row['root_depth_m'],
                et_forecast, row['forecast_rain_mm'],
                self.mad_factor
            )
            
            diagnostics.append({
                'et0_calculated': et0,
                'et_crop': et_crop,
                'et_forecast': et_forecast,
                'rain_effective': rain_eff,
                'irrigation_physics': irrigation_result['irrigation_mm'],
                'depletion_fraction': irrigation_result['depletion_fraction'],
                'stress_risk': irrigation_result['stress_risk'],
                'taw_mm': irrigation_result['taw_mm']
            })
        
        return pd.DataFrame(diagnostics)

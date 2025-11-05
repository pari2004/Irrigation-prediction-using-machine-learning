"""
Generate synthetic irrigation data for testing and development.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random
from .data_schema import get_crop_coefficient, CROP_COEFFICIENTS


class IrrigationDataGenerator:
    """Generate realistic synthetic irrigation data."""
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator."""
        np.random.seed(seed)
        random.seed(seed)
        
        # Define realistic parameter ranges
        self.crop_types = list(CROP_COEFFICIENTS.keys())[:-1]  # Exclude 'default'
        self.growth_stages = ["early", "mid", "late"]
        
        # Soil texture classes with typical properties
        self.soil_types = {
            "sandy": {"fc": 0.15, "wp": 0.06, "bd": 1.6},
            "loam": {"fc": 0.25, "wp": 0.12, "bd": 1.4},
            "clay": {"fc": 0.35, "wp": 0.18, "bd": 1.3}
        }
        
        # Climate zones with typical weather patterns
        self.climate_zones = {
            "arid": {"et0_mean": 6.0, "rain_freq": 0.1, "rain_mean": 8.0},
            "semi_arid": {"et0_mean": 4.5, "rain_freq": 0.2, "rain_mean": 12.0},
            "temperate": {"et0_mean": 3.5, "rain_freq": 0.3, "rain_mean": 15.0}
        }
    
    def generate_weather_sequence(self, days: int, climate: str = "semi_arid") -> Dict[str, List[float]]:
        """Generate realistic weather sequence."""
        climate_params = self.climate_zones[climate]
        
        # Generate correlated weather variables
        et0_base = climate_params["et0_mean"]
        
        weather = {
            "ET0_mm": [],
            "rain_mm": [],
            "tmax": [],
            "tmin": [],
            "RH_mean": [],
            "wind_2m": [],
            "solar_rad": []
        }
        
        for day in range(days):
            # Seasonal variation (simplified sinusoidal)
            seasonal_factor = 0.8 + 0.4 * np.sin(2 * np.pi * day / 365)
            
            # ET0 with daily variation
            et0 = max(0.5, et0_base * seasonal_factor + np.random.normal(0, 0.8))
            weather["ET0_mm"].append(et0)
            
            # Temperature (correlated with ET0)
            tmax = 15 + 15 * seasonal_factor + np.random.normal(0, 3)
            tmin = tmax - 8 - np.random.exponential(2)
            weather["tmax"].append(tmax)
            weather["tmin"].append(tmin)
            
            # Rainfall (stochastic with dry spells)
            if np.random.random() < climate_params["rain_freq"]:
                rain = np.random.exponential(climate_params["rain_mean"])
            else:
                rain = 0.0
            weather["rain_mm"].append(rain)
            
            # Other variables
            weather["RH_mean"].append(max(20, min(95, 60 + np.random.normal(0, 15))))
            weather["wind_2m"].append(max(0.5, np.random.exponential(2.0)))
            weather["solar_rad"].append(max(5, 25 * seasonal_factor + np.random.normal(0, 3)))
        
        return weather
    
    def simulate_soil_moisture(self, weather: Dict, soil_props: Dict, 
                             irrigation_schedule: List[float], crop_kc: List[float]) -> List[float]:
        """Simulate soil moisture using simple water balance."""
        fc = soil_props["field_capacity_theta"]
        wp = soil_props["wilting_point_theta"]
        root_depth = soil_props["root_depth_m"]
        
        # Convert to mm of water in root zone
        fc_mm = fc * root_depth * 1000
        wp_mm = wp * root_depth * 1000
        
        theta_mm = [fc_mm * 0.8]  # Start at 80% of field capacity
        
        for i in range(len(weather["ET0_mm"])):
            # Water balance components
            et_crop = weather["ET0_mm"][i] * crop_kc[i]
            effective_rain = min(weather["rain_mm"][i], et_crop * 1.5)  # Simple effective rain
            irrigation = irrigation_schedule[i]
            
            # Simple drainage when above FC
            theta_new = theta_mm[-1] + irrigation + effective_rain - et_crop
            
            # Drainage above field capacity
            if theta_new > fc_mm:
                drainage = (theta_new - fc_mm) * 0.3  # 30% drains quickly
                theta_new -= drainage
            
            # Cannot go below wilting point (plant stress/death)
            theta_new = max(wp_mm, theta_new)
            
            theta_mm.append(theta_new)
        
        # Convert back to volumetric water content
        theta_vwc = [t / (root_depth * 1000) for t in theta_mm[1:]]  # Skip initial value
        return theta_vwc
    
    def generate_irrigation_schedule(self, days: int, et_sequence: List[float], 
                                   rain_sequence: List[float], kc_sequence: List[float],
                                   management_style: str = "moderate") -> List[float]:
        """Generate realistic irrigation schedule based on management style."""
        irrigation = []
        
        # Management parameters
        if management_style == "conservative":
            threshold_factor = 0.7  # Irrigate more frequently
            irrigation_factor = 1.2  # Apply more water
        elif management_style == "aggressive":
            threshold_factor = 1.3  # Wait longer to irrigate
            irrigation_factor = 0.8  # Apply less water
        else:  # moderate
            threshold_factor = 1.0
            irrigation_factor = 1.0
        
        days_since_irrigation = 0
        
        for i in range(days):
            et_crop = et_sequence[i] * kc_sequence[i]
            effective_rain = min(rain_sequence[i], et_crop)
            water_deficit = max(0, et_crop - effective_rain)
            
            days_since_irrigation += 1
            
            # Decision logic (simplified)
            should_irrigate = (
                (water_deficit > 3.0 * threshold_factor) or  # High water demand
                (days_since_irrigation > 5 and water_deficit > 1.0) or  # Long dry spell
                (days_since_irrigation > 10)  # Maximum interval
            )
            
            if should_irrigate and rain_sequence[i] < 2.0:  # Don't irrigate if raining
                # Apply irrigation to meet crop demand plus buffer
                irrig_amount = water_deficit * irrigation_factor + np.random.normal(0, 1.0)
                irrig_amount = max(0, min(25, irrig_amount))  # Cap at 25mm
                irrigation.append(irrig_amount)
                days_since_irrigation = 0
            else:
                irrigation.append(0.0)
        
        return irrigation
    
    def generate_dataset(self, 
                        start_date: str = "2023-01-01",
                        days: int = 365,
                        zones: List[str] = None,
                        climate: str = "semi_arid") -> pd.DataFrame:
        """Generate complete synthetic dataset."""
        
        if zones is None:
            zones = [f"zone_{i:02d}" for i in range(1, 6)]  # 5 zones by default
        
        all_data = []
        
        for zone_id in zones:
            # Random zone properties
            crop_type = random.choice(self.crop_types)
            soil_type = random.choice(list(self.soil_types.keys()))
            soil_props = self.soil_types[soil_type].copy()
            
            # Add some variation to soil properties
            soil_props["field_capacity_theta"] = soil_props["fc"] + np.random.normal(0, 0.02)
            soil_props["wilting_point_theta"] = soil_props["wp"] + np.random.normal(0, 0.01)
            soil_props["root_depth_m"] = 0.3 + np.random.exponential(0.4)  # 0.3-1.5m typical
            soil_props["bulk_density"] = soil_props["bd"] + np.random.normal(0, 0.1)
            
            # Ensure FC > WP
            soil_props["wilting_point_theta"] = min(
                soil_props["wilting_point_theta"], 
                soil_props["field_capacity_theta"] - 0.05
            )
            
            # Zone area
            area_m2 = 1000 + np.random.exponential(2000)  # 0.1-0.5 hectare typical
            
            # Generate weather
            weather = self.generate_weather_sequence(days + 7, climate)  # Extra days for forecast
            
            # Generate crop coefficient sequence (simplified seasonal pattern)
            stage_duration = days // 3
            kc_sequence = []
            for day in range(days):
                if day < stage_duration:
                    stage = "early"
                elif day < 2 * stage_duration:
                    stage = "mid"
                else:
                    stage = "late"
                
                kc_base = get_crop_coefficient(crop_type, stage)
                kc_daily = kc_base + np.random.normal(0, 0.05)  # Small daily variation
                kc_sequence.append(max(0.2, min(1.5, kc_daily)))
            
            # Generate irrigation schedule
            management_style = random.choice(["conservative", "moderate", "aggressive"])
            irrigation_schedule = self.generate_irrigation_schedule(
                days, weather["ET0_mm"][:days], weather["rain_mm"][:days], 
                kc_sequence, management_style
            )
            
            # Simulate soil moisture
            theta_sequence = self.simulate_soil_moisture(
                {k: v[:days] for k, v in weather.items()}, 
                soil_props, irrigation_schedule, kc_sequence
            )
            
            # Create DataFrame for this zone
            zone_data = []
            for day in range(7, days):  # Start from day 7 to have history
                date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=day)
                
                # Determine growth stage
                if day < stage_duration:
                    growth_stage = "early"
                elif day < 2 * stage_duration:
                    growth_stage = "mid"
                else:
                    growth_stage = "late"
                
                # Calculate irrigation history
                irrigation_last_7d = sum(irrigation_schedule[day-7:day])
                last_irrigation_mm = 0
                days_since_irrigation = 0
                for d in range(day-1, max(0, day-15), -1):  # Look back up to 15 days
                    if irrigation_schedule[d] > 0:
                        last_irrigation_mm = irrigation_schedule[d]
                        days_since_irrigation = day - d - 1
                        break
                else:
                    days_since_irrigation = 15  # Cap at 15 days
                
                row = {
                    "date": date,
                    "zone_id": zone_id,
                    "area_m2": area_m2,
                    "crop_type": crop_type,
                    "growth_stage": growth_stage,
                    "Kc_stage": kc_sequence[day],
                    "ET0_mm": weather["ET0_mm"][day],
                    "forecast_ET0_mm": weather["ET0_mm"][day + 1],
                    "rain_mm": weather["rain_mm"][day],
                    "forecast_rain_mm": weather["rain_mm"][day + 1],
                    "tmax": weather["tmax"][day],
                    "tmin": weather["tmin"][day],
                    "RH_mean": weather["RH_mean"][day],
                    "wind_2m": weather["wind_2m"][day],
                    "solar_rad": weather["solar_rad"][day],
                    "root_depth_m": soil_props["root_depth_m"],
                    "field_capacity_theta": soil_props["field_capacity_theta"],
                    "wilting_point_theta": soil_props["wilting_point_theta"],
                    "bulk_density": soil_props["bulk_density"],
                    "theta0": theta_sequence[day],
                    "theta_1": theta_sequence[day-1],
                    "theta_2": theta_sequence[day-2],
                    "theta_3": theta_sequence[day-3],
                    "theta_4": theta_sequence[day-4],
                    "theta_5": theta_sequence[day-5],
                    "theta_6": theta_sequence[day-6],
                    "irrigation_mm_last_7d": irrigation_last_7d,
                    "last_irrigation_mm": last_irrigation_mm,
                    "days_since_irrigation": days_since_irrigation,
                    "irrigation_mm_next_day": irrigation_schedule[day + 1] if day + 1 < len(irrigation_schedule) else 0
                }
                
                zone_data.append(row)
            
            all_data.extend(zone_data)
        
        return pd.DataFrame(all_data)


def main():
    """Generate sample dataset and save to file."""
    generator = IrrigationDataGenerator(seed=42)
    
    # Generate training data (1 year)
    print("Generating training dataset...")
    train_df = generator.generate_dataset(
        start_date="2023-01-01",
        days=365,
        zones=[f"zone_{i:02d}" for i in range(1, 11)],  # 10 zones
        climate="semi_arid"
    )
    
    # Generate test data (3 months)
    print("Generating test dataset...")
    test_df = generator.generate_dataset(
        start_date="2024-01-01",
        days=90,
        zones=[f"zone_{i:02d}" for i in range(1, 6)],  # 5 zones
        climate="semi_arid"
    )
    
    # Save datasets
    train_df.to_csv("data/sample/irrigation_train.csv", index=False)
    test_df.to_csv("data/sample/irrigation_test.csv", index=False)
    
    print(f"Generated training dataset: {len(train_df)} rows")
    print(f"Generated test dataset: {len(test_df)} rows")
    print("Datasets saved to data/sample/")
    
    # Display sample statistics
    print("\nTraining data summary:")
    print(train_df.describe())


if __name__ == "__main__":
    main()

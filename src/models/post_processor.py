"""
Post-processing safety layer for irrigation predictions.
Ensures no field capacity violations and respects physical constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import warnings


class IrrigationPostProcessor:
    """Post-processing safety layer for irrigation predictions."""
    
    def __init__(self, 
                 min_irrigation_mm: float = 0.5,
                 max_irrigation_mm: float = 50.0,
                 fc_safety_margin: float = 0.02,
                 drainage_rate: float = 0.1):
        """
        Initialize post-processor.
        
        Args:
            min_irrigation_mm: Minimum irrigation amount (system constraint)
            max_irrigation_mm: Maximum irrigation amount per day
            fc_safety_margin: Safety margin below field capacity (VWC)
            drainage_rate: Drainage rate when above FC (fraction/day)
        """
        self.min_irrigation_mm = min_irrigation_mm
        self.max_irrigation_mm = max_irrigation_mm
        self.fc_safety_margin = fc_safety_margin
        self.drainage_rate = drainage_rate
    
    def calculate_max_safe_irrigation(self, 
                                    current_theta: np.ndarray,
                                    field_capacity: np.ndarray,
                                    root_depth: np.ndarray,
                                    et_forecast: np.ndarray,
                                    rain_forecast: np.ndarray) -> np.ndarray:
        """
        Calculate maximum safe irrigation to avoid FC violations.
        
        Args:
            current_theta: Current soil moisture (VWC)
            field_capacity: Field capacity (VWC)
            root_depth: Root depth (m)
            et_forecast: Forecast ET for next day (mm)
            rain_forecast: Forecast rainfall (mm)
            
        Returns:
            Maximum safe irrigation amounts (mm)
        """
        # Convert to mm of water in root zone
        current_mm = current_theta * root_depth * 1000
        fc_mm = field_capacity * root_depth * 1000
        
        # Apply safety margin
        safe_fc_mm = (field_capacity - self.fc_safety_margin) * root_depth * 1000
        
        # Calculate effective rainfall (simplified)
        effective_rain = np.minimum(rain_forecast, et_forecast * 1.5)
        
        # Water balance: current + irrigation + rain - ET <= safe_FC
        # Therefore: irrigation <= safe_FC - current - rain + ET
        max_irrigation = safe_fc_mm - current_mm - effective_rain + et_forecast
        
        # Ensure non-negative
        max_irrigation = np.maximum(0, max_irrigation)
        
        # Apply system constraints
        max_irrigation = np.minimum(max_irrigation, self.max_irrigation_mm)
        
        return max_irrigation
    
    def apply_minimum_runtime_constraint(self, 
                                       irrigation_mm: np.ndarray,
                                       area_m2: np.ndarray,
                                       flow_rate_lpm: Optional[np.ndarray] = None,
                                       min_runtime_minutes: float = 2.0) -> np.ndarray:
        """
        Apply minimum runtime constraint for irrigation systems.
        
        Args:
            irrigation_mm: Irrigation amounts (mm)
            area_m2: Zone areas (m²)
            flow_rate_lpm: Flow rates (L/min) per zone
            min_runtime_minutes: Minimum runtime for valves
            
        Returns:
            Adjusted irrigation amounts
        """
        if flow_rate_lpm is None:
            # Use default flow rate based on area (rough estimate)
            flow_rate_lpm = np.sqrt(area_m2) * 2.0  # 2 L/min per sqrt(m²)
        
        # Convert irrigation to volume (liters)
        volume_liters = irrigation_mm * area_m2
        
        # Calculate runtime in minutes
        runtime_minutes = volume_liters / flow_rate_lpm
        
        # Apply minimum runtime constraint
        adjusted_runtime = np.where(
            (runtime_minutes > 0) & (runtime_minutes < min_runtime_minutes),
            min_runtime_minutes,
            runtime_minutes
        )
        
        # Convert back to irrigation depth
        adjusted_volume = adjusted_runtime * flow_rate_lpm
        adjusted_irrigation = adjusted_volume / area_m2
        
        return adjusted_irrigation
    
    def round_to_deliverable_increments(self, 
                                      irrigation_mm: np.ndarray,
                                      increment_mm: float = 0.5) -> np.ndarray:
        """
        Round irrigation amounts to deliverable increments.
        
        Args:
            irrigation_mm: Irrigation amounts (mm)
            increment_mm: Minimum deliverable increment
            
        Returns:
            Rounded irrigation amounts
        """
        # Round to nearest increment
        rounded = np.round(irrigation_mm / increment_mm) * increment_mm
        
        # Ensure minimum threshold
        rounded = np.where(
            (irrigation_mm > 0) & (rounded < self.min_irrigation_mm),
            self.min_irrigation_mm,
            rounded
        )
        
        return rounded
    
    def check_stress_prevention(self, 
                              irrigation_mm: np.ndarray,
                              current_theta: np.ndarray,
                              wilting_point: np.ndarray,
                              root_depth: np.ndarray,
                              et_forecast: np.ndarray,
                              rain_forecast: np.ndarray,
                              stress_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check if irrigation prevents plant stress.
        
        Args:
            irrigation_mm: Proposed irrigation amounts
            current_theta: Current soil moisture (VWC)
            wilting_point: Wilting point (VWC)
            root_depth: Root depth (m)
            et_forecast: Forecast ET (mm)
            rain_forecast: Forecast rainfall (mm)
            stress_threshold: Stress threshold (fraction of available water)
            
        Returns:
            Tuple of (stress_risk_flags, recommended_adjustments)
        """
        # Convert to mm
        current_mm = current_theta * root_depth * 1000
        wp_mm = wilting_point * root_depth * 1000
        
        # Effective rainfall
        effective_rain = np.minimum(rain_forecast, et_forecast * 1.5)
        
        # Projected soil moisture after irrigation
        projected_mm = current_mm + irrigation_mm + effective_rain - et_forecast
        
        # Available water above wilting point
        available_mm = projected_mm - wp_mm
        
        # Stress risk if available water is below threshold
        stress_risk = available_mm < (current_mm - wp_mm) * stress_threshold
        
        # Recommended adjustment to prevent stress
        target_mm = wp_mm + (current_mm - wp_mm) * stress_threshold
        adjustment_needed = np.maximum(0, target_mm - projected_mm)
        
        return stress_risk, adjustment_needed
    
    def process_predictions(self, 
                          predictions: np.ndarray,
                          metadata: pd.DataFrame,
                          apply_all_constraints: bool = True) -> Dict[str, np.ndarray]:
        """
        Apply all post-processing constraints to predictions.
        
        Args:
            predictions: Raw model predictions (mm)
            metadata: DataFrame with soil, weather, and system metadata
            apply_all_constraints: Whether to apply all constraints
            
        Returns:
            Dictionary with processed predictions and diagnostics
        """
        # Extract required columns
        required_cols = [
            'theta0', 'field_capacity_theta', 'wilting_point_theta',
            'root_depth_m', 'forecast_ET0_mm', 'forecast_rain_mm',
            'Kc_stage', 'area_m2'
        ]
        
        missing_cols = [col for col in required_cols if col not in metadata.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Calculate forecast ET
        et_forecast = metadata['forecast_ET0_mm'] * metadata['Kc_stage']
        
        # Step 1: Calculate maximum safe irrigation
        max_safe = self.calculate_max_safe_irrigation(
            metadata['theta0'].values,
            metadata['field_capacity_theta'].values,
            metadata['root_depth_m'].values,
            et_forecast.values,
            metadata['forecast_rain_mm'].values
        )
        
        # Step 2: Apply field capacity constraint
        fc_constrained = np.minimum(predictions, max_safe)
        
        # Step 3: Check stress prevention
        stress_risk, stress_adjustment = self.check_stress_prevention(
            fc_constrained,
            metadata['theta0'].values,
            metadata['wilting_point_theta'].values,
            metadata['root_depth_m'].values,
            et_forecast.values,
            metadata['forecast_rain_mm'].values
        )
        
        # Step 4: Apply stress prevention if needed
        stress_adjusted = fc_constrained + stress_adjustment
        
        # Step 5: Re-apply FC constraint after stress adjustment
        final_constrained = np.minimum(stress_adjusted, max_safe)
        
        # Step 6: Apply system constraints
        if apply_all_constraints:
            # Minimum runtime constraint (if flow rate data available)
            if 'flow_rate_lpm' in metadata.columns:
                runtime_adjusted = self.apply_minimum_runtime_constraint(
                    final_constrained,
                    metadata['area_m2'].values,
                    metadata['flow_rate_lpm'].values
                )
            else:
                runtime_adjusted = self.apply_minimum_runtime_constraint(
                    final_constrained,
                    metadata['area_m2'].values
                )
            
            # Round to deliverable increments
            final_predictions = self.round_to_deliverable_increments(runtime_adjusted)
        else:
            final_predictions = final_constrained
        
        # Ensure absolute bounds
        final_predictions = np.clip(final_predictions, 0, self.max_irrigation_mm)
        
        # Calculate volumes in liters
        volumes_liters = final_predictions * metadata['area_m2'].values
        
        # Diagnostics
        results = {
            'irrigation_mm_raw': predictions,
            'irrigation_mm_fc_safe': fc_constrained,
            'irrigation_mm_final': final_predictions,
            'irrigation_liters': volumes_liters,
            'max_safe_mm': max_safe,
            'stress_risk': stress_risk,
            'stress_adjustment_mm': stress_adjustment,
            'fc_violation_prevented': predictions > max_safe,
            'adjustment_applied': np.abs(final_predictions - predictions) > 0.1
        }
        
        return results
    
    def validate_predictions(self, 
                           predictions: np.ndarray,
                           metadata: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate predictions against physical constraints.
        
        Args:
            predictions: Irrigation predictions (mm)
            metadata: Metadata DataFrame
            
        Returns:
            Validation report
        """
        # Process predictions to get diagnostics
        results = self.process_predictions(predictions, metadata, apply_all_constraints=False)
        
        # Validation metrics
        n_total = len(predictions)
        n_fc_violations = np.sum(results['fc_violation_prevented'])
        n_stress_risk = np.sum(results['stress_risk'])
        n_negative = np.sum(predictions < 0)
        n_excessive = np.sum(predictions > self.max_irrigation_mm)
        
        validation_report = {
            'total_predictions': n_total,
            'fc_violations': n_fc_violations,
            'fc_violation_rate': n_fc_violations / n_total,
            'stress_risk_count': n_stress_risk,
            'stress_risk_rate': n_stress_risk / n_total,
            'negative_predictions': n_negative,
            'excessive_predictions': n_excessive,
            'mean_adjustment': np.mean(np.abs(results['irrigation_mm_final'] - predictions)),
            'max_adjustment': np.max(np.abs(results['irrigation_mm_final'] - predictions)),
            'total_water_raw_liters': np.sum(predictions * metadata['area_m2']),
            'total_water_safe_liters': np.sum(results['irrigation_liters']),
            'water_savings_liters': np.sum(predictions * metadata['area_m2']) - np.sum(results['irrigation_liters'])
        }
        
        return validation_report
    
    def create_irrigation_schedule(self, 
                                 predictions: np.ndarray,
                                 metadata: pd.DataFrame,
                                 schedule_horizon_days: int = 7) -> pd.DataFrame:
        """
        Create detailed irrigation schedule with timing and volumes.
        
        Args:
            predictions: Irrigation predictions (mm)
            metadata: Metadata DataFrame
            schedule_horizon_days: Days to schedule ahead
            
        Returns:
            Detailed irrigation schedule DataFrame
        """
        # Process predictions
        results = self.process_predictions(predictions, metadata)
        
        # Create schedule DataFrame
        schedule = pd.DataFrame({
            'date': metadata.get('date', pd.date_range('2024-01-01', periods=len(predictions))),
            'zone_id': metadata.get('zone_id', [f'zone_{i}' for i in range(len(predictions))]),
            'irrigation_mm': results['irrigation_mm_final'],
            'irrigation_liters': results['irrigation_liters'],
            'runtime_minutes': results['irrigation_liters'] / metadata.get('flow_rate_lpm', 10.0),
            'priority': np.where(results['stress_risk'], 'high', 
                               np.where(results['irrigation_mm_final'] > 5, 'medium', 'low')),
            'notes': np.where(results['fc_violation_prevented'], 'FC-limited', 
                            np.where(results['stress_adjustment_mm'] > 0, 'stress-adjusted', 'normal'))
        })
        
        # Sort by priority and irrigation amount
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        schedule['priority_num'] = schedule['priority'].map(priority_order)
        schedule = schedule.sort_values(['priority_num', 'irrigation_mm'], ascending=[False, False])
        schedule = schedule.drop('priority_num', axis=1)
        
        return schedule

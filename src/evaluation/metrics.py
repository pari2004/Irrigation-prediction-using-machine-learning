"""
Comprehensive evaluation metrics for irrigation prediction models.
Includes agronomic KPIs and asymmetric error analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


class IrrigationEvaluator:
    """Comprehensive evaluation framework for irrigation models."""
    
    def __init__(self, 
                 under_irrigation_threshold: float = 1.0,
                 over_irrigation_threshold: float = 1.0,
                 stress_threshold_vwc: float = 0.1):
        """
        Initialize evaluator.
        
        Args:
            under_irrigation_threshold: Threshold for significant under-irrigation (mm)
            over_irrigation_threshold: Threshold for significant over-irrigation (mm)
            stress_threshold_vwc: VWC threshold below which stress occurs
        """
        self.under_threshold = under_irrigation_threshold
        self.over_threshold = over_irrigation_threshold
        self.stress_threshold = stress_threshold_vwc
    
    def basic_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic regression metrics."""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mean_error': np.mean(y_pred - y_true),
            'median_error': np.median(y_pred - y_true),
            'std_error': np.std(y_pred - y_true)
        }
    
    def asymmetric_error_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate asymmetric error metrics."""
        error = y_pred - y_true
        
        # Under-irrigation metrics (y_pred < y_true)
        under_mask = error < -self.under_threshold
        under_errors = error[under_mask]
        
        # Over-irrigation metrics (y_pred > y_true)
        over_mask = error > self.over_threshold
        over_errors = error[over_mask]
        
        return {
            'under_irrigation_rate': np.mean(under_mask),
            'over_irrigation_rate': np.mean(over_mask),
            'under_irrigation_mae': np.mean(np.abs(under_errors)) if len(under_errors) > 0 else 0,
            'over_irrigation_mae': np.mean(np.abs(over_errors)) if len(over_errors) > 0 else 0,
            'under_irrigation_severity': np.mean(under_errors) if len(under_errors) > 0 else 0,
            'over_irrigation_severity': np.mean(over_errors) if len(over_errors) > 0 else 0,
            'asymmetric_score': np.mean(under_mask) * 2 + np.mean(over_mask)  # Weighted score
        }
    
    def water_use_efficiency_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   area_m2: np.ndarray) -> Dict[str, float]:
        """Calculate water use efficiency metrics."""
        # Total water volumes
        total_actual_liters = np.sum(y_true * area_m2)
        total_predicted_liters = np.sum(y_pred * area_m2)
        
        # Water use error
        water_use_error = (total_predicted_liters - total_actual_liters) / total_actual_liters * 100
        
        # Daily water use statistics
        daily_actual = y_true * area_m2
        daily_predicted = y_pred * area_m2
        daily_error = daily_predicted - daily_actual
        
        return {
            'total_water_actual_liters': total_actual_liters,
            'total_water_predicted_liters': total_predicted_liters,
            'water_use_error_pct': water_use_error,
            'water_savings_liters': total_actual_liters - total_predicted_liters,
            'daily_water_mae_liters': mean_absolute_error(daily_actual, daily_predicted),
            'daily_water_rmse_liters': np.sqrt(mean_squared_error(daily_actual, daily_predicted)),
            'water_efficiency_score': 100 - abs(water_use_error)  # Higher is better
        }
    
    def agronomic_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    soil_moisture_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate agronomic performance metrics."""
        if soil_moisture_data.empty:
            return {}
        
        required_cols = ['theta0', 'field_capacity_theta', 'wilting_point_theta', 'root_depth_m']
        if not all(col in soil_moisture_data.columns for col in required_cols):
            return {}
        
        # Simulate soil moisture trajectories
        actual_trajectory = self._simulate_soil_moisture(y_true, soil_moisture_data)
        predicted_trajectory = self._simulate_soil_moisture(y_pred, soil_moisture_data)
        
        # Field capacity violations
        fc_violations_actual = np.sum(actual_trajectory > soil_moisture_data['field_capacity_theta'])
        fc_violations_predicted = np.sum(predicted_trajectory > soil_moisture_data['field_capacity_theta'])
        
        # Stress events (below threshold)
        stress_events_actual = np.sum(actual_trajectory < self.stress_threshold)
        stress_events_predicted = np.sum(predicted_trajectory < self.stress_threshold)
        
        # Available water content
        awc_actual = self._calculate_awc(actual_trajectory, soil_moisture_data)
        awc_predicted = self._calculate_awc(predicted_trajectory, soil_moisture_data)
        
        return {
            'fc_violations_actual': fc_violations_actual,
            'fc_violations_predicted': fc_violations_predicted,
            'fc_violation_difference': fc_violations_predicted - fc_violations_actual,
            'stress_events_actual': stress_events_actual,
            'stress_events_predicted': stress_events_predicted,
            'stress_event_difference': stress_events_predicted - stress_events_actual,
            'mean_awc_actual': np.mean(awc_actual),
            'mean_awc_predicted': np.mean(awc_predicted),
            'awc_mae': mean_absolute_error(awc_actual, awc_predicted),
            'soil_moisture_mae': mean_absolute_error(actual_trajectory, predicted_trajectory)
        }
    
    def temporal_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   dates: pd.Series) -> Dict[str, Any]:
        """Calculate temporal performance metrics."""
        if len(dates) != len(y_true):
            return {}
        
        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'y_true': y_true,
            'y_pred': y_pred,
            'error': y_pred - y_true
        })
        
        # Monthly performance
        df['month'] = df['date'].dt.month
        monthly_mae = df.groupby('month')['error'].apply(lambda x: np.mean(np.abs(x)))
        
        # Seasonal performance
        df['season'] = df['date'].dt.month.map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })
        seasonal_mae = df.groupby('season')['error'].apply(lambda x: np.mean(np.abs(x)))
        
        # Growing season vs non-growing season
        growing_months = [4, 5, 6, 7, 8, 9, 10]
        df['growing_season'] = df['month'].isin(growing_months)
        growing_mae = df[df['growing_season']]['error'].apply(lambda x: np.mean(np.abs(x)))
        non_growing_mae = df[~df['growing_season']]['error'].apply(lambda x: np.mean(np.abs(x)))
        
        return {
            'monthly_mae': monthly_mae.to_dict(),
            'seasonal_mae': seasonal_mae.to_dict(),
            'growing_season_mae': growing_mae.iloc[0] if not growing_mae.empty else 0,
            'non_growing_season_mae': non_growing_mae.iloc[0] if not non_growing_mae.empty else 0,
            'temporal_consistency': 1 / (1 + np.std(monthly_mae))  # Higher is better
        }
    
    def zone_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                               zone_ids: pd.Series) -> Dict[str, Any]:
        """Calculate per-zone performance metrics."""
        if len(zone_ids) != len(y_true):
            return {}
        
        df = pd.DataFrame({
            'zone_id': zone_ids,
            'y_true': y_true,
            'y_pred': y_pred,
            'error': y_pred - y_true
        })
        
        # Per-zone MAE
        zone_mae = df.groupby('zone_id')['error'].apply(lambda x: np.mean(np.abs(x)))
        
        # Zone consistency
        zone_consistency = 1 / (1 + np.std(zone_mae))
        
        return {
            'zone_mae': zone_mae.to_dict(),
            'zone_consistency': zone_consistency,
            'worst_zone': zone_mae.idxmax(),
            'best_zone': zone_mae.idxmin(),
            'zone_performance_range': zone_mae.max() - zone_mae.min()
        }
    
    def comprehensive_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray,
                               metadata: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Perform comprehensive evaluation with all metrics."""
        results = {}
        
        # Basic regression metrics
        results['basic_metrics'] = self.basic_regression_metrics(y_true, y_pred)
        
        # Asymmetric error metrics
        results['asymmetric_metrics'] = self.asymmetric_error_metrics(y_true, y_pred)
        
        if metadata is not None:
            # Water use efficiency
            if 'area_m2' in metadata.columns:
                results['water_efficiency'] = self.water_use_efficiency_metrics(
                    y_true, y_pred, metadata['area_m2'].values
                )
            
            # Agronomic performance
            results['agronomic_metrics'] = self.agronomic_performance_metrics(
                y_true, y_pred, metadata
            )
            
            # Temporal performance
            if 'date' in metadata.columns:
                results['temporal_metrics'] = self.temporal_performance_metrics(
                    y_true, y_pred, metadata['date']
                )
            
            # Zone performance
            if 'zone_id' in metadata.columns:
                results['zone_metrics'] = self.zone_performance_metrics(
                    y_true, y_pred, metadata['zone_id']
                )
        
        # Overall score (weighted combination)
        overall_score = self._calculate_overall_score(results)
        results['overall_score'] = overall_score
        
        return results
    
    def _simulate_soil_moisture(self, irrigation: np.ndarray, 
                              soil_data: pd.DataFrame) -> np.ndarray:
        """Simulate soil moisture trajectory given irrigation schedule."""
        # Simplified soil water balance
        theta = soil_data['theta0'].values.copy()
        
        for i in range(len(irrigation)):
            if i == 0:
                continue
            
            # Water balance components (simplified)
            et_crop = soil_data.iloc[i].get('ET0_mm', 3.0) * soil_data.iloc[i].get('Kc_stage', 1.0)
            rain = soil_data.iloc[i].get('rain_mm', 0.0)
            
            # Convert to mm
            theta_mm = theta[i-1] * soil_data.iloc[i]['root_depth_m'] * 1000
            fc_mm = soil_data.iloc[i]['field_capacity_theta'] * soil_data.iloc[i]['root_depth_m'] * 1000
            wp_mm = soil_data.iloc[i]['wilting_point_theta'] * soil_data.iloc[i]['root_depth_m'] * 1000
            
            # Water balance
            theta_new_mm = theta_mm + irrigation[i] + rain - et_crop
            
            # Drainage above FC
            if theta_new_mm > fc_mm:
                theta_new_mm = fc_mm + (theta_new_mm - fc_mm) * 0.7  # 30% drains
            
            # Cannot go below WP
            theta_new_mm = max(wp_mm, theta_new_mm)
            
            # Convert back to VWC
            theta[i] = theta_new_mm / (soil_data.iloc[i]['root_depth_m'] * 1000)
        
        return theta
    
    def _calculate_awc(self, theta: np.ndarray, soil_data: pd.DataFrame) -> np.ndarray:
        """Calculate available water content."""
        fc = soil_data['field_capacity_theta'].values
        wp = soil_data['wilting_point_theta'].values
        
        awc = (theta - wp) / (fc - wp)
        return np.clip(awc, 0, 1)
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Calculate weighted overall performance score."""
        score = 0.0
        weights = {
            'mae': 0.3,
            'asymmetric_score': 0.25,
            'water_efficiency_score': 0.2,
            'temporal_consistency': 0.15,
            'zone_consistency': 0.1
        }
        
        # Basic MAE (inverted and normalized)
        if 'basic_metrics' in results:
            mae = results['basic_metrics'].get('mae', 10)
            score += weights['mae'] * max(0, 100 - mae * 10)  # Scale MAE
        
        # Asymmetric score (inverted)
        if 'asymmetric_metrics' in results:
            asym_score = results['asymmetric_metrics'].get('asymmetric_score', 1)
            score += weights['asymmetric_score'] * max(0, 100 - asym_score * 50)
        
        # Water efficiency
        if 'water_efficiency' in results:
            water_eff = results['water_efficiency'].get('water_efficiency_score', 50)
            score += weights['water_efficiency_score'] * water_eff
        
        # Temporal consistency
        if 'temporal_metrics' in results:
            temp_cons = results['temporal_metrics'].get('temporal_consistency', 0.5)
            score += weights['temporal_consistency'] * temp_cons * 100
        
        # Zone consistency
        if 'zone_metrics' in results:
            zone_cons = results['zone_metrics'].get('zone_consistency', 0.5)
            score += weights['zone_consistency'] * zone_cons * 100
        
        return score
    
    def create_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Create a formatted evaluation report."""
        report = "IRRIGATION MODEL EVALUATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Basic metrics
        if 'basic_metrics' in results:
            report += "BASIC REGRESSION METRICS:\n"
            for metric, value in results['basic_metrics'].items():
                report += f"  {metric.upper()}: {value:.3f}\n"
            report += "\n"
        
        # Asymmetric metrics
        if 'asymmetric_metrics' in results:
            report += "ASYMMETRIC ERROR ANALYSIS:\n"
            for metric, value in results['asymmetric_metrics'].items():
                report += f"  {metric.replace('_', ' ').title()}: {value:.3f}\n"
            report += "\n"
        
        # Water efficiency
        if 'water_efficiency' in results:
            report += "WATER USE EFFICIENCY:\n"
            for metric, value in results['water_efficiency'].items():
                if isinstance(value, (int, float)):
                    report += f"  {metric.replace('_', ' ').title()}: {value:.1f}\n"
            report += "\n"
        
        # Overall score
        if 'overall_score' in results:
            report += f"OVERALL PERFORMANCE SCORE: {results['overall_score']:.1f}/100\n"
        
        return report

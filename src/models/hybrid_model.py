"""
Hybrid physics + ML model for irrigation prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import joblib
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
# from catboost import CatBoostRegressor

# Import our custom modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.et_calculations import PhysicsBaseline
from features.feature_engineering import IrrigationFeatureEngineer
from models.loss_functions import (
    AsymmetricLoss, create_sample_weights, 
    evaluate_asymmetric_performance
)


class HybridIrrigationModel:
    """Hybrid physics + ML model for irrigation prediction."""
    
    def __init__(self, 
                 model_type: str = 'xgboost',
                 physics_weight: float = 0.3,
                 asymmetric_alpha: float = 2.0,
                 asymmetric_beta: float = 1.0,
                 mad_factor: float = 0.5):
        """
        Initialize hybrid model.
        
        Args:
            model_type: 'xgboost', 'lightgbm', or 'catboost'
            physics_weight: Weight for physics baseline in ensemble (0-1)
            asymmetric_alpha: Under-irrigation penalty weight
            asymmetric_beta: Over-irrigation penalty weight
            mad_factor: Management allowed depletion factor for physics model
        """
        self.model_type = model_type
        self.physics_weight = physics_weight
        self.asymmetric_alpha = asymmetric_alpha
        self.asymmetric_beta = asymmetric_beta
        
        # Initialize components
        self.physics_model = PhysicsBaseline(mad_factor=mad_factor)
        self.feature_engineer = IrrigationFeatureEngineer()
        self.ml_model = None
        self.asymmetric_loss = AsymmetricLoss(asymmetric_alpha, asymmetric_beta)
        
        # Model parameters
        self.model_params = self._get_default_params()
        
        # Training history
        self.training_history = {}
        self.feature_importance = {}
        self.is_fitted = False
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for each model type."""
        if self.model_type == 'xgboost':
            return {
                'n_estimators': 800,
                'max_depth': 6,
                'learning_rate': 0.03,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_lambda': 1.0,
                'reg_alpha': 0.1,
                'random_state': 42,
                'n_jobs': -1
            }
        elif self.model_type == 'lightgbm':
            return {
                'n_estimators': 800,
                'max_depth': 6,
                'learning_rate': 0.03,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_lambda': 1.0,
                'reg_alpha': 0.1,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
        elif self.model_type == 'catboost':
            return {
                'iterations': 800,
                'depth': 6,
                'learning_rate': 0.03,
                'l2_leaf_reg': 1.0,
                'random_state': 42,
                'verbose': False
            }
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _create_ml_model(self) -> Any:
        """Create ML model instance."""
        if self.model_type == 'xgboost':
            return xgb.XGBRegressor(**self.model_params)
        elif self.model_type == 'lightgbm':
            return lgb.LGBMRegressor(**self.model_params)
        elif self.model_type == 'catboost':
            raise ImportError("CatBoost is not installed. Please install it to use this model type.")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _prepare_features(self, df: pd.DataFrame, fit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and physics baseline."""
        # Feature engineering
        if fit:
            df_features = self.feature_engineer.fit_transform(df)
        else:
            df_features = self.feature_engineer.transform(df)
        
        # Get physics baseline predictions
        physics_pred = self.physics_model.predict(df)
        
        # Get ML features
        feature_names = self.feature_engineer.get_feature_names()
        X_ml = df_features[feature_names].values
        
        # Add physics prediction as a feature
        X_combined = np.column_stack([X_ml, physics_pred.reshape(-1, 1)])
        
        return X_combined, physics_pred
    
    def fit(self, df: pd.DataFrame, 
            validation_split: float = 0.2,
            use_sample_weights: bool = True) -> Dict[str, Any]:
        """
        Fit the hybrid model.
        
        Args:
            df: Training dataframe
            validation_split: Fraction for validation
            use_sample_weights: Whether to use asymmetric sample weights
            
        Returns:
            Training history dictionary
        """
        print("Preparing features and physics baseline...")
        
        # Prepare features
        X_combined, physics_pred = self._prepare_features(df, fit=True)
        y = df['irrigation_mm_next_day'].values
        
        # Calculate residuals (ML target)
        y_residual = y - physics_pred
        
        # Train-validation split
        X_train, X_val, y_train, y_val, y_res_train, y_res_val, phys_train, phys_val = train_test_split(
            X_combined, y, y_residual, physics_pred, 
            test_size=validation_split, random_state=42, shuffle=True
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        
        # Create sample weights for asymmetric loss
        sample_weights = None
        if use_sample_weights:
            # Calculate ET demand for weighting
            et_demand = df['ET0_mm'].values * df['Kc_stage'].values
            stress_risk = (df['theta0'] < df['wilting_point_theta'] * 1.2).astype(int)
            
            all_weights = create_sample_weights(
                y, et_demand, stress_risk, 
                self.asymmetric_alpha, self.asymmetric_beta
            )
            
            # Split weights
            _, _, weights_train, weights_val = train_test_split(
                X_combined, all_weights, test_size=validation_split, 
                random_state=42, shuffle=True
            )
            sample_weights = weights_train
        
        # Create and train ML model
        print(f"Training {self.model_type} model...")
        self.ml_model = self._create_ml_model()
        
        # Fit model with sample weights if supported
        if self.model_type == 'xgboost' and sample_weights is not None:
            self.ml_model.fit(
                X_train, y_res_train,
                sample_weight=sample_weights,
                eval_set=[(X_val, y_res_val)],
                verbose=False
            )
        elif self.model_type == 'lightgbm' and sample_weights is not None:
            self.ml_model.fit(
                X_train, y_res_train,
                sample_weight=sample_weights,
                eval_set=[(X_val, y_res_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
        elif self.model_type == 'catboost' and sample_weights is not None:
            raise ImportError("CatBoost is not installed. Please install it to use this model type.")
        else:
            self.ml_model.fit(X_train, y_res_train)
        
        # Make predictions
        y_res_pred_train = self.ml_model.predict(X_train)
        y_res_pred_val = self.ml_model.predict(X_val)
        
        # Combine with physics predictions
        y_pred_train = phys_train + y_res_pred_train
        y_pred_val = phys_val + y_res_pred_val
        
        # Evaluate performance
        train_metrics = evaluate_asymmetric_performance(y_train, y_pred_train)
        val_metrics = evaluate_asymmetric_performance(y_val, y_pred_val)
        
        # Store feature importance
        if hasattr(self.ml_model, 'feature_importances_'):
            feature_names = self.feature_engineer.get_feature_names() + ['physics_baseline']
            self.feature_importance = dict(zip(feature_names, self.ml_model.feature_importances_))
        
        # Training history
        self.training_history = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'physics_baseline_mae': mean_absolute_error(y_train, phys_train),
            'ml_residual_mae': mean_absolute_error(y_res_train, y_res_pred_train),
            'hybrid_improvement': train_metrics['mae'] - mean_absolute_error(y_train, phys_train)
        }
        
        self.is_fitted = True
        
        print("Training completed!")
        print(f"Physics baseline MAE: {self.training_history['physics_baseline_mae']:.3f}")
        print(f"Hybrid model MAE: {train_metrics['mae']:.3f}")
        print(f"Validation MAE: {val_metrics['mae']:.3f}")
        
        return self.training_history

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the hybrid model.

        Args:
            df: Input dataframe

        Returns:
            Array of irrigation predictions (mm)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Prepare features
        X_combined, physics_pred = self._prepare_features(df, fit=False)

        # ML residual prediction
        ml_residual = self.ml_model.predict(X_combined)

        # Combine physics and ML predictions
        hybrid_pred = physics_pred + ml_residual

        # Ensure non-negative predictions
        hybrid_pred = np.maximum(0, hybrid_pred)

        return hybrid_pred

    def predict_with_diagnostics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions with detailed diagnostics.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with predictions and diagnostics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Prepare features
        X_combined, physics_pred = self._prepare_features(df, fit=False)

        # ML residual prediction
        ml_residual = self.ml_model.predict(X_combined)

        # Combine predictions
        hybrid_pred = physics_pred + ml_residual
        hybrid_pred_safe = np.maximum(0, hybrid_pred)

        # Create diagnostics dataframe
        diagnostics = pd.DataFrame({
            'zone_id': df['zone_id'],
            'date': df['date'],
            'physics_baseline_mm': physics_pred,
            'ml_residual_mm': ml_residual,
            'hybrid_raw_mm': hybrid_pred,
            'hybrid_safe_mm': hybrid_pred_safe,
            'irrigation_liters': hybrid_pred_safe * df['area_m2'],
            'current_theta': df['theta0'],
            'field_capacity': df['field_capacity_theta'],
            'et_demand': df['ET0_mm'] * df['Kc_stage'],
            'forecast_rain': df['forecast_rain_mm']
        })

        # Add confidence scores (simplified)
        if hasattr(self.ml_model, 'predict_proba'):
            # For models that support uncertainty estimation
            diagnostics['confidence_score'] = 0.8  # Placeholder
        else:
            # Use feature importance and prediction magnitude as proxy
            diagnostics['confidence_score'] = np.clip(
                1.0 - np.abs(ml_residual) / (physics_pred + 1e-6), 0.1, 0.9
            )

        return diagnostics

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance ranking."""
        if not self.feature_importance:
            raise ValueError("Model must be fitted to get feature importance")

        importance_df = pd.DataFrame([
            {'feature': name, 'importance': importance}
            for name, importance in self.feature_importance.items()
        ]).sort_values('importance', ascending=False)

        return importance_df.head(top_n)

    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        model_data = {
            'ml_model': self.ml_model,
            'feature_engineer': self.feature_engineer,
            'physics_model': self.physics_model,
            'model_type': self.model_type,
            'physics_weight': self.physics_weight,
            'asymmetric_alpha': self.asymmetric_alpha,
            'asymmetric_beta': self.asymmetric_beta,
            'training_history': self.training_history,
            'feature_importance': self.feature_importance,
            'is_fitted': self.is_fitted
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'HybridIrrigationModel':
        """Load a trained model."""
        model_data = joblib.load(filepath)

        # Create new instance
        model = cls(
            model_type=model_data['model_type'],
            physics_weight=model_data['physics_weight'],
            asymmetric_alpha=model_data['asymmetric_alpha'],
            asymmetric_beta=model_data['asymmetric_beta']
        )

        # Restore state
        model.ml_model = model_data['ml_model']
        model.feature_engineer = model_data['feature_engineer']
        model.physics_model = model_data['physics_model']
        model.training_history = model_data['training_history']
        model.feature_importance = model_data['feature_importance']
        model.is_fitted = model_data['is_fitted']

        return model

    def cross_validate(self, df: pd.DataFrame, cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation."""
        from sklearn.model_selection import KFold

        # Prepare features
        X_combined, physics_pred = self._prepare_features(df, fit=True)
        y = df['irrigation_mm_next_day'].values
        y_residual = y - physics_pred

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        cv_scores = {
            'mae': [],
            'rmse': [],
            'asymmetric_score': []
        }

        for train_idx, val_idx in kf.split(X_combined):
            # Split data
            X_train, X_val = X_combined[train_idx], X_combined[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            y_res_train, y_res_val = y_residual[train_idx], y_residual[val_idx]
            phys_train, phys_val = physics_pred[train_idx], physics_pred[val_idx]

            # Train model
            fold_model = self._create_ml_model()
            fold_model.fit(X_train, y_res_train)

            # Predict
            y_res_pred = fold_model.predict(X_val)
            y_pred = phys_val + y_res_pred

            # Evaluate
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            asym_metrics = evaluate_asymmetric_performance(y_val, y_pred)

            cv_scores['mae'].append(mae)
            cv_scores['rmse'].append(rmse)
            cv_scores['asymmetric_score'].append(asym_metrics['asymmetric_score'])

        # Calculate statistics
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)

        return cv_results

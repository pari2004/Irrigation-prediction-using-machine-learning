#!/usr/bin/env python3
"""
Quick demo of the ML-driven irrigation prediction system.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from utils.data_generator import IrrigationDataGenerator
from physics.et_calculations import PhysicsBaseline
from features.feature_engineering import IrrigationFeatureEngineer
from models.post_processor import IrrigationPostProcessor

# Simple XGBoost model
import xgboost as xgb
from sklearn.metrics import mean_absolute_error


def quick_demo():
    """Run a quick demonstration of the irrigation system."""
    print("🌱 ML-Driven Precision Irrigation System - Quick Demo")
    print("=" * 60)
    
    # Step 1: Generate sample data
    print("\n1. Generating sample irrigation data...")
    generator = IrrigationDataGenerator(seed=42)
    
    # Generate training data (3 months, 3 zones)
    train_df = generator.generate_dataset(
        start_date="2023-01-01",
        days=90,
        zones=["zone_A", "zone_B", "zone_C"],
        climate="semi_arid"
    )
    
    # Generate test data (2 weeks, 3 zones)
    test_df = generator.generate_dataset(
        start_date="2023-04-01",
        days=14,
        zones=["zone_A", "zone_B", "zone_C"],
        climate="semi_arid"
    )
    
    print(f"   Training data: {len(train_df)} samples")
    print(f"   Test data: {len(test_df)} samples")
    
    # Display sample data
    print(f"\n   Sample training data:")
    sample_cols = ['date', 'zone_id', 'crop_type', 'theta0', 'ET0_mm', 'irrigation_mm_next_day']
    print(train_df[sample_cols].head(3).to_string(index=False))
    
    # Step 2: Physics baseline
    print("\n2. Computing physics-based baseline...")
    physics_model = PhysicsBaseline(mad_factor=0.5)
    
    train_physics = physics_model.predict(train_df)
    test_physics = physics_model.predict(test_df)
    
    physics_mae = mean_absolute_error(test_df['irrigation_mm_next_day'], test_physics)
    print(f"   Physics baseline MAE: {physics_mae:.3f} mm")
    
    # Step 3: Feature engineering
    print("\n3. Engineering features...")
    feature_engineer = IrrigationFeatureEngineer()
    
    # Fit on training data
    train_features = feature_engineer.fit_transform(train_df)
    test_features = feature_engineer.transform(test_df)
    
    # Get feature columns and handle missing values
    feature_cols = feature_engineer.get_feature_names()
    
    # Select only numeric columns that exist in both datasets
    available_features = []
    for col in feature_cols:
        if col in train_features.columns and col in test_features.columns:
            if train_features[col].dtype in ['int64', 'float64'] and test_features[col].dtype in ['int64', 'float64']:
                available_features.append(col)
    
    print(f"   Using {len(available_features)} features")
    
    # Prepare ML data
    X_train = train_features[available_features].fillna(0)
    y_train = train_df['irrigation_mm_next_day']
    X_test = test_features[available_features].fillna(0)
    y_test = test_df['irrigation_mm_next_day']
    
    # Step 4: Train ML model
    print("\n4. Training XGBoost model...")
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        n_jobs=1  # Use single thread to avoid issues
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    ml_predictions = model.predict(X_test)
    ml_mae = mean_absolute_error(y_test, ml_predictions)
    print(f"   ML model MAE: {ml_mae:.3f} mm")
    
    # Step 5: Hybrid approach
    print("\n5. Creating hybrid predictions...")
    
    # Simple hybrid: 70% ML + 30% physics
    hybrid_predictions = 0.7 * ml_predictions + 0.3 * test_physics
    hybrid_mae = mean_absolute_error(y_test, hybrid_predictions)
    print(f"   Hybrid model MAE: {hybrid_mae:.3f} mm")
    
    # Step 6: Post-processing
    print("\n6. Applying safety constraints...")
    post_processor = IrrigationPostProcessor(
        min_irrigation_mm=0.5,
        max_irrigation_mm=25.0,
        fc_safety_margin=0.02
    )
    
    processed_results = post_processor.process_predictions(hybrid_predictions, test_df)
    
    final_mae = mean_absolute_error(y_test, processed_results['irrigation_mm_final'])
    print(f"   Final safe predictions MAE: {final_mae:.3f} mm")
    
    # Step 7: Basic evaluation
    print("\n7. Basic evaluation...")
    
    # Calculate basic metrics
    errors = processed_results['irrigation_mm_final'] - y_test.values
    under_irrigation = np.sum(errors < -1.0) / len(errors)
    over_irrigation = np.sum(errors > 1.0) / len(errors)
    
    print(f"   Under-irrigation rate (>1mm): {under_irrigation:.1%}")
    print(f"   Over-irrigation rate (>1mm): {over_irrigation:.1%}")
    
    total_water = np.sum(processed_results['irrigation_liters'])
    fc_violations_prevented = np.sum(processed_results['fc_violation_prevented'])
    
    print(f"   Total water recommended: {total_water:,.0f} L")
    print(f"   FC violations prevented: {fc_violations_prevented}")
    
    # Step 8: Show sample predictions
    print("\n8. Sample predictions:")
    print("   Zone    Date        Actual  Physics   ML    Hybrid  Final   Volume(L)")
    print("   " + "-" * 70)
    
    for i in range(min(8, len(test_df))):
        row = test_df.iloc[i]
        actual = y_test.iloc[i]
        physics = test_physics[i]
        ml_pred = ml_predictions[i]
        hybrid = hybrid_predictions[i]
        final = processed_results['irrigation_mm_final'][i]
        volume = processed_results['irrigation_liters'][i]
        
        print(f"   {row['zone_id']:<6} {row['date'].strftime('%Y-%m-%d')} "
              f"{actual:6.1f}  {physics:6.1f}  {ml_pred:6.1f}  {hybrid:6.1f}  {final:6.1f}  {volume:8.0f}")
    
    # Step 9: Feature importance
    print(f"\n9. Top 10 most important features:")
    
    if len(available_features) > 0:
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for _, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']:<25}: {row['importance']:.4f}")
    
    # Step 10: Summary
    print(f"\n10. Summary:")
    print(f"    Physics baseline:     {physics_mae:.3f} mm MAE")
    print(f"    ML model:            {ml_mae:.3f} mm MAE")
    print(f"    Hybrid model:        {hybrid_mae:.3f} mm MAE")
    print(f"    Final safe model:    {final_mae:.3f} mm MAE")
    
    if physics_mae > 0:
        improvement = (physics_mae - final_mae) / physics_mae * 100
        print(f"    Improvement over physics: {improvement:.1f}%")
    
    print(f"\n✅ Demo completed successfully!")
    
    # Save results for dashboard
    print(f"\n💾 Saving results...")
    os.makedirs("data/sample", exist_ok=True)
    
    # Create dashboard data
    dashboard_data = test_df.copy()
    dashboard_data['predicted_irrigation_mm'] = processed_results['irrigation_mm_final']
    dashboard_data['predicted_liters'] = processed_results['irrigation_liters']
    dashboard_data['physics_baseline_mm'] = test_physics
    dashboard_data['ml_prediction_mm'] = ml_predictions
    dashboard_data['hybrid_prediction_mm'] = hybrid_predictions
    
    dashboard_data.to_csv("data/sample/demo_results.csv", index=False)
    print(f"   Saved demo results to: data/sample/demo_results.csv")
    
    # Save model summary
    summary = {
        'physics_mae': physics_mae,
        'ml_mae': ml_mae,
        'hybrid_mae': hybrid_mae,
        'final_mae': final_mae,
        'total_water_liters': float(total_water),
        'fc_violations_prevented': int(fc_violations_prevented),
        'under_irrigation_rate': float(under_irrigation),
        'over_irrigation_rate': float(over_irrigation),
        'num_features': len(available_features),
        'num_test_samples': len(test_df)
    }
    
    import json
    with open("data/sample/demo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"   Saved summary to: data/sample/demo_summary.json")
    
    return {
        'train_df': train_df,
        'test_df': test_df,
        'model': model,
        'predictions': processed_results,
        'summary': summary
    }


if __name__ == "__main__":
    try:
        # Run the demo
        results = quick_demo()
        
        print(f"\n🎯 Next steps:")
        print(f"   1. Check the results in data/sample/")
        print(f"   2. Try running: python example_usage.py")
        print(f"   3. Explore the generated data and model predictions")
        print(f"   4. Modify parameters and re-run to see different results")
        
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nPlease check the error and try again.")

#!/usr/bin/env python3
"""
Simple demo of the ML-driven irrigation prediction system.
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
from evaluation.metrics import IrrigationEvaluator

# Simple XGBoost model without the complex hybrid wrapper
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def simple_demo():
    """Run a simple demonstration of the irrigation system."""
    print("🌱 ML-Driven Precision Irrigation System - Simple Demo")
    print("=" * 60)
    
    # Step 1: Generate sample data
    print("\n1. Generating sample irrigation data...")
    generator = IrrigationDataGenerator(seed=42)
    
    # Generate training data (6 months, 5 zones)
    train_df = generator.generate_dataset(
        start_date="2023-01-01",
        days=180,
        zones=[f"zone_{i:02d}" for i in range(1, 6)],
        climate="semi_arid"
    )
    
    # Generate test data (1 month, 3 zones)
    test_df = generator.generate_dataset(
        start_date="2023-07-01",
        days=30,
        zones=[f"zone_{i:02d}" for i in range(1, 4)],
        climate="semi_arid"
    )
    
    print(f"   Training data: {len(train_df)} samples")
    print(f"   Test data: {len(test_df)} samples")
    
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
    
    # Get feature columns
    feature_cols = feature_engineer.get_feature_names()
    print(f"   Created {len(feature_cols)} features")
    
    # Prepare ML data
    X_train = train_features[feature_cols].fillna(0)
    y_train = train_df['irrigation_mm_next_day']
    X_test = test_features[feature_cols].fillna(0)
    y_test = test_df['irrigation_mm_next_day']
    
    # Step 4: Train ML model
    print("\n4. Training XGBoost model...")
    
    # Simple XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    ml_predictions = model.predict(X_test)
    ml_mae = mean_absolute_error(y_test, ml_predictions)
    print(f"   ML model MAE: {ml_mae:.3f} mm")
    
    # Step 5: Hybrid approach (simple combination)
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
    
    # Step 7: Evaluation
    print("\n7. Comprehensive evaluation...")
    evaluator = IrrigationEvaluator()
    
    results = evaluator.comprehensive_evaluation(
        y_test.values,
        processed_results['irrigation_mm_final'],
        test_df
    )
    
    print(f"   Under-irrigation rate: {results['asymmetric_metrics']['under_irrigation_rate']:.1%}")
    print(f"   Over-irrigation rate: {results['asymmetric_metrics']['over_irrigation_rate']:.1%}")
    
    if 'water_efficiency' in results:
        print(f"   Water use efficiency: {results['water_efficiency']['water_efficiency_score']:.1f}%")
    
    # Step 8: Show sample predictions
    print("\n8. Sample predictions:")
    print("   Zone       Date        Actual  Physics   ML    Hybrid  Final   Volume(L)")
    print("   " + "-" * 75)
    
    for i in range(min(10, len(test_df))):
        row = test_df.iloc[i]
        actual = y_test.iloc[i]
        physics = test_physics[i]
        ml_pred = ml_predictions[i]
        hybrid = hybrid_predictions[i]
        final = processed_results['irrigation_mm_final'][i]
        volume = processed_results['irrigation_liters'][i]
        
        print(f"   {row['zone_id']:<8} {row['date'].strftime('%Y-%m-%d')} "
              f"{actual:6.1f}  {physics:6.1f}  {ml_pred:6.1f}  {hybrid:6.1f}  {final:6.1f}  {volume:8.0f}")
    
    # Step 9: Feature importance
    print(f"\n9. Top 10 most important features:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
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
    
    improvement = (physics_mae - final_mae) / physics_mae * 100
    print(f"    Improvement over physics: {improvement:.1f}%")
    
    total_water = np.sum(processed_results['irrigation_liters'])
    fc_violations_prevented = np.sum(processed_results['fc_violation_prevented'])
    
    print(f"    Total water recommended: {total_water:,.0f} L")
    print(f"    FC violations prevented: {fc_violations_prevented}")
    
    print(f"\n✅ Demo completed successfully!")
    
    return {
        'train_df': train_df,
        'test_df': test_df,
        'model': model,
        'feature_engineer': feature_engineer,
        'post_processor': post_processor,
        'results': results,
        'predictions': processed_results
    }


def create_simple_dashboard_data(demo_results):
    """Create simple data for dashboard visualization."""
    print(f"\n📊 Creating dashboard data...")
    
    test_df = demo_results['test_df']
    predictions = demo_results['predictions']
    
    # Create summary statistics
    summary = {
        'total_zones': test_df['zone_id'].nunique(),
        'total_days': test_df['date'].nunique(),
        'total_water_liters': np.sum(predictions['irrigation_liters']),
        'avg_irrigation_mm': np.mean(predictions['irrigation_mm_final']),
        'fc_violations_prevented': np.sum(predictions['fc_violation_prevented']),
        'stress_risk_zones': np.sum(predictions['stress_risk'])
    }
    
    print(f"   Dashboard summary:")
    for key, value in summary.items():
        print(f"     {key}: {value}")
    
    # Save sample data for dashboard
    os.makedirs("data/sample", exist_ok=True)
    
    # Combine test data with predictions
    dashboard_data = test_df.copy()
    dashboard_data['predicted_irrigation_mm'] = predictions['irrigation_mm_final']
    dashboard_data['predicted_liters'] = predictions['irrigation_liters']
    dashboard_data['fc_violation_prevented'] = predictions['fc_violation_prevented']
    dashboard_data['stress_risk'] = predictions['stress_risk']
    
    dashboard_data.to_csv("data/sample/dashboard_data.csv", index=False)
    print(f"   Saved dashboard data to: data/sample/dashboard_data.csv")
    
    return summary


if __name__ == "__main__":
    try:
        # Run the demo
        demo_results = simple_demo()
        
        # Create dashboard data
        summary = create_simple_dashboard_data(demo_results)
        
        print(f"\n🎯 Next steps:")
        print(f"   1. Run 'streamlit run dashboard.py' to see the interactive dashboard")
        print(f"   2. Check the generated data in data/sample/")
        print(f"   3. Explore the model predictions and feature importance")
        print(f"   4. Try modifying parameters and re-running the demo")
        
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nPlease check the error and try again.")

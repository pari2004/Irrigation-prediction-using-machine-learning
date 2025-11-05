#!/usr/bin/env python3
"""
Example usage of the ML-driven irrigation prediction system.
Demonstrates the complete workflow from data generation to predictions.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append('src')

from utils.data_generator import IrrigationDataGenerator
from models.hybrid_model import HybridIrrigationModel
from models.post_processor import IrrigationPostProcessor
from evaluation.metrics import IrrigationEvaluator


def example_1_basic_usage():
    """Example 1: Basic model training and prediction."""
    print("="*60)
    print("EXAMPLE 1: Basic Model Training and Prediction")
    print("="*60)
    
    # Step 1: Generate sample data
    print("1. Generating sample data...")
    generator = IrrigationDataGenerator(seed=42)
    
    train_df = generator.generate_dataset(
        start_date="2023-01-01",
        days=180,  # 6 months
        zones=["zone_01", "zone_02", "zone_03"],
        climate="semi_arid"
    )
    
    test_df = generator.generate_dataset(
        start_date="2023-07-01",
        days=30,  # 1 month
        zones=["zone_01", "zone_02", "zone_03"],
        climate="semi_arid"
    )
    
    print(f"   Training data: {len(train_df)} samples")
    print(f"   Test data: {len(test_df)} samples")
    
    # Step 2: Train model
    print("\n2. Training hybrid model...")
    model = HybridIrrigationModel(
        model_type='xgboost',
        asymmetric_alpha=2.0,
        asymmetric_beta=1.0
    )
    
    training_history = model.fit(train_df, validation_split=0.2)
    
    print(f"   Training MAE: {training_history['train_metrics']['mae']:.3f} mm")
    print(f"   Validation MAE: {training_history['val_metrics']['mae']:.3f} mm")
    
    # Step 3: Make predictions
    print("\n3. Making predictions...")
    predictions = model.predict(test_df)
    
    # Step 4: Evaluate performance
    print("\n4. Evaluating performance...")
    evaluator = IrrigationEvaluator()
    results = evaluator.comprehensive_evaluation(
        test_df['irrigation_mm_next_day'].values,
        predictions,
        test_df
    )
    
    print(f"   Test MAE: {results['basic_metrics']['mae']:.3f} mm")
    print(f"   Under-irrigation rate: {results['asymmetric_metrics']['under_irrigation_rate']:.1%}")
    print(f"   Over-irrigation rate: {results['asymmetric_metrics']['over_irrigation_rate']:.1%}")
    
    return model, test_df, predictions


def example_2_post_processing():
    """Example 2: Post-processing with safety constraints."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Post-Processing with Safety Constraints")
    print("="*60)
    
    # Use model from example 1
    model, test_df, raw_predictions = example_1_basic_usage()
    
    # Apply post-processing
    print("\n1. Applying post-processing safety constraints...")
    post_processor = IrrigationPostProcessor(
        min_irrigation_mm=0.5,
        max_irrigation_mm=25.0,
        fc_safety_margin=0.02
    )
    
    processed_results = post_processor.process_predictions(raw_predictions, test_df)
    
    # Compare raw vs processed
    print("\n2. Comparing raw vs post-processed predictions...")
    
    raw_total = np.sum(raw_predictions * test_df['area_m2'])
    processed_total = np.sum(processed_results['irrigation_liters'])
    fc_violations_prevented = np.sum(processed_results['fc_violation_prevented'])
    
    print(f"   Raw total water: {raw_total:,.0f} L")
    print(f"   Processed total water: {processed_total:,.0f} L")
    print(f"   Water saved: {raw_total - processed_total:,.0f} L")
    print(f"   FC violations prevented: {fc_violations_prevented}")
    
    # Create irrigation schedule
    print("\n3. Creating irrigation schedule...")
    schedule = post_processor.create_irrigation_schedule(
        processed_results['irrigation_mm_final'], 
        test_df
    )
    
    print("   Sample schedule (first 5 entries):")
    print(schedule.head()[['date', 'zone_id', 'irrigation_mm', 'irrigation_liters', 'priority']].to_string(index=False))
    
    return processed_results, schedule


def example_3_detailed_analysis():
    """Example 3: Detailed model analysis and diagnostics."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Detailed Model Analysis")
    print("="*60)
    
    # Generate fresh data for analysis
    generator = IrrigationDataGenerator(seed=123)
    analysis_df = generator.generate_dataset(
        start_date="2024-01-01",
        days=60,
        zones=["zone_A", "zone_B"],
        climate="semi_arid"
    )
    
    # Train a model for analysis
    model = HybridIrrigationModel(model_type='xgboost')
    model.fit(analysis_df[:40])  # Use first 40 days for training
    
    # Get detailed diagnostics
    print("\n1. Getting detailed prediction diagnostics...")
    test_data = analysis_df[40:]  # Last 20 days for testing
    diagnostics = model.predict_with_diagnostics(test_data)
    
    # Feature importance analysis
    print("\n2. Feature importance analysis...")
    importance_df = model.get_feature_importance(top_n=10)
    print("   Top 10 most important features:")
    for _, row in importance_df.iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Zone-specific analysis
    print("\n3. Zone-specific performance...")
    for zone in test_data['zone_id'].unique():
        zone_data = test_data[test_data['zone_id'] == zone]
        zone_diag = diagnostics[diagnostics['zone_id'] == zone]
        
        mae = np.mean(np.abs(zone_diag['hybrid_safe_mm'] - zone_data['irrigation_mm_next_day']))
        avg_physics = np.mean(zone_diag['physics_baseline_mm'])
        avg_ml = np.mean(zone_diag['ml_residual_mm'])
        
        print(f"   {zone}:")
        print(f"     MAE: {mae:.2f} mm")
        print(f"     Avg physics baseline: {avg_physics:.2f} mm")
        print(f"     Avg ML adjustment: {avg_ml:.2f} mm")
    
    # Prediction examples
    print("\n4. Example predictions with explanations...")
    for i in range(3):
        row = test_data.iloc[i]
        diag = diagnostics.iloc[i]
        
        print(f"\n   Example {i+1}: {row['zone_id']} on {row['date'].strftime('%Y-%m-%d')}")
        print(f"     Soil moisture: {row['theta0']:.3f} (FC: {row['field_capacity_theta']:.3f})")
        print(f"     ET demand: {row['ET0_mm'] * row['Kc_stage']:.1f} mm")
        print(f"     Physics says: {diag['physics_baseline_mm']:.1f} mm")
        print(f"     ML adjusts by: {diag['ml_residual_mm']:.1f} mm")
        print(f"     Final recommendation: {diag['hybrid_safe_mm']:.1f} mm")
        print(f"     Actual irrigation: {row['irrigation_mm_next_day']:.1f} mm")
        print(f"     Error: {diag['hybrid_safe_mm'] - row['irrigation_mm_next_day']:.1f} mm")


def example_4_real_time_simulation():
    """Example 4: Simulate real-time daily predictions."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Real-Time Daily Prediction Simulation")
    print("="*60)
    
    # Setup
    generator = IrrigationDataGenerator(seed=456)
    
    # Generate historical data for training
    historical_df = generator.generate_dataset(
        start_date="2023-01-01",
        days=300,
        zones=["field_north", "field_south", "field_east"],
        climate="semi_arid"
    )
    
    # Train model
    print("1. Training model on historical data...")
    model = HybridIrrigationModel(model_type='lightgbm')
    model.fit(historical_df)
    
    # Simulate real-time predictions for a week
    print("\n2. Simulating daily predictions for one week...")
    
    current_date = pd.to_datetime("2024-01-01")
    post_processor = IrrigationPostProcessor()
    
    weekly_schedule = []
    
    for day in range(7):
        prediction_date = current_date + pd.Timedelta(days=day)
        
        # Generate "current" conditions for this day
        daily_data = generator.generate_dataset(
            start_date=prediction_date.strftime("%Y-%m-%d"),
            days=1,
            zones=["field_north", "field_south", "field_east"],
            climate="semi_arid"
        )
        
        # Make predictions
        predictions = model.predict(daily_data)
        processed = post_processor.process_predictions(predictions, daily_data)
        
        # Create daily schedule
        daily_schedule = pd.DataFrame({
            'date': daily_data['date'],
            'zone_id': daily_data['zone_id'],
            'irrigation_mm': processed['irrigation_mm_final'],
            'irrigation_liters': processed['irrigation_liters'],
            'soil_moisture': daily_data['theta0'],
            'et_demand': daily_data['ET0_mm'] * daily_data['Kc_stage'],
            'stress_risk': processed['stress_risk']
        })
        
        weekly_schedule.append(daily_schedule)
        
        # Print daily summary
        total_water = daily_schedule['irrigation_liters'].sum()
        stress_zones = daily_schedule['stress_risk'].sum()
        
        print(f"   {prediction_date.strftime('%Y-%m-%d')}: {total_water:,.0f} L total, {stress_zones} zones at risk")
    
    # Combine weekly schedule
    full_schedule = pd.concat(weekly_schedule, ignore_index=True)
    
    print(f"\n3. Weekly summary:")
    print(f"   Total water used: {full_schedule['irrigation_liters'].sum():,.0f} L")
    print(f"   Average daily irrigation: {full_schedule['irrigation_mm'].mean():.1f} mm")
    print(f"   Days with stress risk: {full_schedule.groupby('date')['stress_risk'].any().sum()}")
    
    return full_schedule


def main():
    """Run all examples."""
    print("ML-DRIVEN PRECISION IRRIGATION SYSTEM")
    print("Example Usage Demonstrations")
    print("=" * 60)
    
    try:
        # Run examples
        example_1_basic_usage()
        example_2_post_processing()
        example_3_detailed_analysis()
        weekly_schedule = example_4_real_time_simulation()
        
        # Save example outputs
        print(f"\n" + "="*60)
        print("SAVING EXAMPLE OUTPUTS")
        print("="*60)
        
        os.makedirs("examples/outputs", exist_ok=True)
        weekly_schedule.to_csv("examples/outputs/weekly_schedule_example.csv", index=False)
        print("   Saved: examples/outputs/weekly_schedule_example.csv")
        
        print("\n✅ All examples completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python train_model.py --generate-data' to train a full model")
        print("2. Run 'streamlit run dashboard.py' to see the interactive dashboard")
        print("3. Check docs/USAGE_GUIDE.md for detailed usage instructions")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Please check your installation and try again.")


if __name__ == "__main__":
    main()

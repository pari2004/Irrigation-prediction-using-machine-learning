#!/usr/bin/env python3
"""
Training script for the hybrid irrigation prediction model.
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from utils.data_generator import IrrigationDataGenerator
from utils.data_schema import validate_dataframe, IrrigationDataSchema
from models.hybrid_model import HybridIrrigationModel
from models.post_processor import IrrigationPostProcessor
from evaluation.metrics import IrrigationEvaluator


def generate_sample_data():
    """Generate sample training and test data."""
    print("Generating sample data...")
    
    generator = IrrigationDataGenerator(seed=42)
    
    # Generate training data (1 year, 10 zones)
    train_df = generator.generate_dataset(
        start_date="2023-01-01",
        days=365,
        zones=[f"zone_{i:02d}" for i in range(1, 11)],
        climate="semi_arid"
    )
    
    # Generate test data (3 months, 5 zones)
    test_df = generator.generate_dataset(
        start_date="2024-01-01",
        days=90,
        zones=[f"zone_{i:02d}" for i in range(1, 6)],
        climate="semi_arid"
    )
    
    # Save datasets
    os.makedirs("data/sample", exist_ok=True)
    train_df.to_csv("data/sample/irrigation_train.csv", index=False)
    test_df.to_csv("data/sample/irrigation_test.csv", index=False)
    
    print(f"Generated training dataset: {len(train_df)} rows")
    print(f"Generated test dataset: {len(test_df)} rows")
    
    return train_df, test_df


def validate_data(df, dataset_name):
    """Validate dataset against schema."""
    print(f"Validating {dataset_name} data...")
    
    errors = validate_dataframe(df, IrrigationDataSchema)
    if errors:
        print(f"Found {len(errors)} validation errors in {dataset_name}:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  Row {error['row_index']}: {error['error']}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
        return False
    else:
        print(f"{dataset_name} data validation passed!")
        return True


def train_and_evaluate_model(train_df, test_df, model_type='xgboost'):
    """Train and evaluate the hybrid model."""
    print(f"\nTraining {model_type} hybrid model...")
    
    # Initialize model
    model = HybridIrrigationModel(
        model_type=model_type,
        physics_weight=0.3,
        asymmetric_alpha=2.0,
        asymmetric_beta=1.0,
        mad_factor=0.5
    )
    
    # Train model
    training_history = model.fit(
        train_df, 
        validation_split=0.2,
        use_sample_weights=True
    )
    
    # Make predictions on test set
    print("Making predictions on test set...")
    test_predictions = model.predict(test_df)
    
    # Get detailed diagnostics
    test_diagnostics = model.predict_with_diagnostics(test_df)
    
    # Post-process predictions
    print("Applying post-processing safety constraints...")
    post_processor = IrrigationPostProcessor(
        min_irrigation_mm=0.5,
        max_irrigation_mm=50.0,
        fc_safety_margin=0.02
    )
    
    post_processed = post_processor.process_predictions(
        test_predictions, 
        test_df
    )
    
    # Evaluate performance
    print("Evaluating model performance...")
    evaluator = IrrigationEvaluator(
        under_irrigation_threshold=1.0,
        over_irrigation_threshold=1.0
    )
    
    # Evaluate raw predictions
    raw_results = evaluator.comprehensive_evaluation(
        test_df['irrigation_mm_next_day'].values,
        test_predictions,
        test_df
    )
    
    # Evaluate post-processed predictions
    processed_results = evaluator.comprehensive_evaluation(
        test_df['irrigation_mm_next_day'].values,
        post_processed['irrigation_mm_final'],
        test_df
    )
    
    # Print evaluation reports
    print("\n" + "="*60)
    print("RAW MODEL PREDICTIONS:")
    print(evaluator.create_evaluation_report(raw_results))
    
    print("\n" + "="*60)
    print("POST-PROCESSED PREDICTIONS:")
    print(evaluator.create_evaluation_report(processed_results))
    
    # Feature importance
    print("\n" + "="*60)
    print("TOP 10 MOST IMPORTANT FEATURES:")
    feature_importance = model.get_feature_importance(top_n=10)
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    model_path = f"models/hybrid_{model_type}_model.pkl"
    os.makedirs("models", exist_ok=True)
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")
    
    return {
        'model': model,
        'training_history': training_history,
        'raw_results': raw_results,
        'processed_results': processed_results,
        'test_diagnostics': test_diagnostics,
        'post_processed': post_processed
    }


def compare_models(train_df, test_df):
    """Compare different model types."""
    print("\n" + "="*60)
    print("COMPARING DIFFERENT MODEL TYPES")
    print("="*60)
    
    model_types = ['xgboost', 'lightgbm', 'catboost']
    results = {}
    
    for model_type in model_types:
        print(f"\n--- Training {model_type.upper()} ---")
        try:
            result = train_and_evaluate_model(train_df, test_df, model_type)
            results[model_type] = result
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            continue
    
    # Compare results
    if len(results) > 1:
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY:")
        print("="*60)
        
        comparison_df = pd.DataFrame({
            model_type: {
                'MAE': result['processed_results']['basic_metrics']['mae'],
                'RMSE': result['processed_results']['basic_metrics']['rmse'],
                'Under-irrigation Rate': result['processed_results']['asymmetric_metrics']['under_irrigation_rate'],
                'Over-irrigation Rate': result['processed_results']['asymmetric_metrics']['over_irrigation_rate'],
                'Overall Score': result['processed_results']['overall_score']
            }
            for model_type, result in results.items()
        }).T
        
        print(comparison_df.round(4))
        
        # Find best model
        best_model = comparison_df['Overall Score'].idxmax()
        print(f"\nBest performing model: {best_model.upper()}")
    
    return results


def create_prediction_examples(model, test_df, n_examples=5):
    """Create example predictions with explanations."""
    print(f"\n" + "="*60)
    print(f"EXAMPLE PREDICTIONS (showing {n_examples} cases):")
    print("="*60)
    
    # Get diagnostics for all test data
    diagnostics = model.predict_with_diagnostics(test_df)
    
    # Select diverse examples
    examples_idx = np.linspace(0, len(test_df)-1, n_examples, dtype=int)
    
    for i, idx in enumerate(examples_idx):
        row = test_df.iloc[idx]
        diag = diagnostics.iloc[idx]
        
        print(f"\nExample {i+1}: Zone {row['zone_id']} on {row['date'].strftime('%Y-%m-%d')}")
        print(f"  Crop: {row['crop_type']} ({row['growth_stage']} stage)")
        print(f"  Current soil moisture: {row['theta0']:.3f} VWC")
        print(f"  Field capacity: {row['field_capacity_theta']:.3f} VWC")
        print(f"  ET demand: {row['ET0_mm'] * row['Kc_stage']:.1f} mm")
        print(f"  Forecast rain: {row['forecast_rain_mm']:.1f} mm")
        print(f"  Physics baseline: {diag['physics_baseline_mm']:.1f} mm")
        print(f"  ML adjustment: {diag['ml_residual_mm']:.1f} mm")
        print(f"  Final recommendation: {diag['hybrid_safe_mm']:.1f} mm ({diag['irrigation_liters']:.0f} L)")
        print(f"  Actual irrigation: {row['irrigation_mm_next_day']:.1f} mm")
        print(f"  Confidence: {diag['confidence_score']:.2f}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train irrigation prediction model')
    parser.add_argument('--data-path', type=str, help='Path to training data CSV')
    parser.add_argument('--test-path', type=str, help='Path to test data CSV')
    parser.add_argument('--model-type', type=str, default='xgboost', 
                       choices=['xgboost', 'lightgbm', 'catboost'],
                       help='Type of ML model to use')
    parser.add_argument('--compare-models', action='store_true',
                       help='Compare all model types')
    parser.add_argument('--generate-data', action='store_true',
                       help='Generate synthetic data instead of loading from file')
    
    args = parser.parse_args()
    
    print("ML-Driven Precision Irrigation Model Training")
    print("=" * 50)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load or generate data
    if args.generate_data or (not args.data_path):
        train_df, test_df = generate_sample_data()
    else:
        print(f"Loading training data from {args.data_path}")
        train_df = pd.read_csv(args.data_path, parse_dates=['date'])
        
        if args.test_path:
            print(f"Loading test data from {args.test_path}")
            test_df = pd.read_csv(args.test_path, parse_dates=['date'])
        else:
            # Split training data
            print("Splitting training data for testing...")
            split_idx = int(len(train_df) * 0.8)
            test_df = train_df[split_idx:].copy()
            train_df = train_df[:split_idx].copy()
    
    # Validate data
    train_valid = validate_data(train_df, "training")
    test_valid = validate_data(test_df, "test")
    
    if not (train_valid and test_valid):
        print("Data validation failed. Please check your data.")
        return
    
    # Train and evaluate
    if args.compare_models:
        results = compare_models(train_df, test_df)
        # Use best model for examples
        if results:
            best_model_name = max(results.keys(), 
                                key=lambda k: results[k]['processed_results']['overall_score'])
            best_model = results[best_model_name]['model']
            create_prediction_examples(best_model, test_df)
    else:
        result = train_and_evaluate_model(train_df, test_df, args.model_type)
        create_prediction_examples(result['model'], test_df)
    
    print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

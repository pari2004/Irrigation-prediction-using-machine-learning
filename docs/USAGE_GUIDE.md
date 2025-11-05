# Usage Guide: ML-Driven Precision Irrigation System

## Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd finalproject

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Data and Train Model
```bash
# Generate synthetic data and train model
python train_model.py --generate-data --compare-models

# This will:
# - Generate training and test datasets
# - Train XGBoost, LightGBM, and CatBoost models
# - Compare performance and save the best model
```

### 3. Run Dashboard
```bash
# Start the interactive dashboard
streamlit run dashboard.py

# Open browser to http://localhost:8501
```

## Detailed Usage

### Training a Model

#### Using Synthetic Data
```bash
# Basic training with XGBoost
python train_model.py --generate-data --model-type xgboost

# Compare all model types
python train_model.py --generate-data --compare-models
```

#### Using Your Own Data
```bash
# Train with custom data
python train_model.py --data-path data/my_irrigation_data.csv --test-path data/my_test_data.csv

# Your CSV must include these columns:
# date, zone_id, area_m2, crop_type, growth_stage, Kc_stage,
# ET0_mm, forecast_ET0_mm, rain_mm, forecast_rain_mm,
# tmax, tmin, RH_mean, wind_2m, solar_rad,
# theta0, theta_1, theta_2, theta_3, theta_4, theta_5, theta_6,
# root_depth_m, field_capacity_theta, wilting_point_theta,
# irrigation_mm_last_7d, last_irrigation_mm, days_since_irrigation,
# irrigation_mm_next_day
```

### Making Predictions

#### Python API
```python
import pandas as pd
from src.models.hybrid_model import HybridIrrigationModel
from src.models.post_processor import IrrigationPostProcessor

# Load trained model
model = HybridIrrigationModel.load_model("models/hybrid_xgboost_model.pkl")

# Prepare input data
input_data = pd.DataFrame({
    'date': ['2024-08-24'],
    'zone_id': ['zone_01'],
    'area_m2': [5000],
    'crop_type': ['tomato'],
    'growth_stage': ['mid'],
    'Kc_stage': [1.15],
    'ET0_mm': [4.5],
    'forecast_ET0_mm': [4.2],
    'rain_mm': [0.0],
    'forecast_rain_mm': [2.0],
    'tmax': [28.5],
    'tmin': [18.2],
    'RH_mean': [65.0],
    'wind_2m': [2.1],
    'solar_rad': [22.5],
    'theta0': [0.25],
    'theta_1': [0.26],
    'theta_2': [0.24],
    'theta_3': [0.23],
    'theta_4': [0.25],
    'theta_5': [0.27],
    'theta_6': [0.26],
    'root_depth_m': [0.6],
    'field_capacity_theta': [0.35],
    'wilting_point_theta': [0.15],
    'irrigation_mm_last_7d': [12.5],
    'last_irrigation_mm': [3.2],
    'days_since_irrigation': [2]
})

# Make predictions
predictions = model.predict(input_data)
print(f"Recommended irrigation: {predictions[0]:.1f} mm")

# Get detailed diagnostics
diagnostics = model.predict_with_diagnostics(input_data)
print(diagnostics[['physics_baseline_mm', 'ml_residual_mm', 'hybrid_safe_mm']])

# Apply post-processing safety constraints
post_processor = IrrigationPostProcessor()
safe_predictions = post_processor.process_predictions(predictions, input_data)
print(f"Safe irrigation: {safe_predictions['irrigation_mm_final'][0]:.1f} mm")
print(f"Volume: {safe_predictions['irrigation_liters'][0]:.0f} L")
```

#### Batch Processing
```python
# Process multiple zones/days
batch_data = pd.read_csv("data/prediction_input.csv")
batch_predictions = model.predict(batch_data)

# Create irrigation schedule
schedule = post_processor.create_irrigation_schedule(batch_predictions, batch_data)
schedule.to_csv("irrigation_schedule.csv", index=False)
```

### Data Preparation

#### Required Data Format
Your data must be in CSV format with the following columns:

**Temporal & Spatial:**
- `date`: Date (YYYY-MM-DD format)
- `zone_id`: Unique zone identifier
- `area_m2`: Zone area in square meters

**Crop Information:**
- `crop_type`: Crop name (e.g., 'tomato', 'corn', 'wheat')
- `growth_stage`: Growth stage ('early', 'mid', 'late')
- `Kc_stage`: Crop coefficient for current stage (0.2-1.5)

**Weather (Current Day):**
- `ET0_mm`: Reference evapotranspiration (mm/day)
- `rain_mm`: Rainfall (mm)
- `tmax`: Maximum temperature (°C)
- `tmin`: Minimum temperature (°C)
- `RH_mean`: Mean relative humidity (%)
- `wind_2m`: Wind speed at 2m height (m/s)
- `solar_rad`: Solar radiation (MJ/m²/day)

**Weather Forecast:**
- `forecast_ET0_mm`: Forecast ET0 for next day
- `forecast_rain_mm`: Forecast rainfall for next day

**Soil Properties:**
- `root_depth_m`: Effective root depth (meters)
- `field_capacity_theta`: Field capacity (volumetric water content)
- `wilting_point_theta`: Wilting point (volumetric water content)

**Soil Moisture History:**
- `theta0`: Current day soil moisture (VWC)
- `theta_1` to `theta_6`: Soil moisture 1-6 days ago (VWC)

**Irrigation History:**
- `irrigation_mm_last_7d`: Total irrigation in last 7 days (mm)
- `last_irrigation_mm`: Last irrigation amount (mm)
- `days_since_irrigation`: Days since last irrigation

**Target (for training only):**
- `irrigation_mm_next_day`: Actual irrigation applied next day (mm)

#### Data Quality Requirements
- **No missing values** in required columns
- **Realistic ranges:** 
  - Soil moisture: 0.05-0.50 VWC
  - ET0: 0.5-12.0 mm/day
  - Temperature: -10 to 50°C
  - Irrigation: 0-50 mm/day
- **Temporal consistency:** Data should be in chronological order
- **Soil physics:** Field capacity > wilting point

### Model Evaluation

#### Performance Metrics
```python
from src.evaluation.metrics import IrrigationEvaluator

evaluator = IrrigationEvaluator()
results = evaluator.comprehensive_evaluation(y_true, y_pred, metadata)

# Print evaluation report
print(evaluator.create_evaluation_report(results))
```

#### Key Metrics to Monitor
- **MAE (Mean Absolute Error):** Target <2.0 mm
- **Under-irrigation Rate:** Target <5%
- **Over-irrigation Rate:** Target <10%
- **Water Use Efficiency:** Target >90%
- **Field Capacity Violations:** Target <2%

### Dashboard Features

#### Overview Tab
- Daily irrigation schedule visualization
- Zone performance comparison
- Water use summary statistics

#### Zone Details Tab
- Soil moisture trajectory plots
- Zone-specific predictions and diagnostics
- Historical irrigation patterns

#### Schedule Tab
- Detailed irrigation schedule with priorities
- Filterable by zone, date, priority
- Downloadable CSV export

#### Performance Tab
- Feature importance analysis
- Prediction error distributions
- Model performance metrics

### Troubleshooting

#### Common Issues

**1. Model Not Found Error**
```
Error: Model not found at models/hybrid_xgboost_model.pkl
Solution: Run training script first: python train_model.py --generate-data
```

**2. Data Validation Errors**
```
Error: Field capacity must be greater than wilting point
Solution: Check soil property values in your data
```

**3. Missing Weather Data**
```
Error: Missing required columns: ['ET0_mm', 'forecast_ET0_mm']
Solution: Calculate ET0 using weather data or use weather service API
```

**4. Unrealistic Predictions**
```
Issue: Model predicts very high irrigation amounts
Solution: Check soil moisture sensors for calibration drift
```

#### Performance Optimization

**For Large Datasets:**
- Use batch prediction instead of single predictions
- Consider data sampling for very large historical datasets
- Use parallel processing for multiple zones

**For Real-time Applications:**
- Cache model in memory
- Pre-compute static features
- Use asynchronous data fetching

### Integration with Irrigation Systems

#### Hardware Requirements
- **Soil Moisture Sensors:** Calibrated VWC sensors at multiple depths
- **Weather Station:** Local weather measurements or API access
- **Flow Control:** Precise irrigation valves with flow measurement
- **Data Logger:** Automated data collection system

#### Software Integration
```python
# Example integration with irrigation controller
def daily_irrigation_update():
    # Fetch current data
    current_data = fetch_sensor_data()
    weather_forecast = fetch_weather_forecast()
    
    # Prepare input
    input_df = prepare_model_input(current_data, weather_forecast)
    
    # Make predictions
    predictions = model.predict(input_df)
    safe_predictions = post_processor.process_predictions(predictions, input_df)
    
    # Send to irrigation controller
    for zone_id, irrigation_mm in zip(input_df['zone_id'], safe_predictions['irrigation_mm_final']):
        send_irrigation_command(zone_id, irrigation_mm)
    
    # Log results
    log_predictions(input_df, safe_predictions)
```

### Best Practices

#### Data Collection
- **Sensor Maintenance:** Regular calibration and cleaning
- **Data Quality Checks:** Automated validation and anomaly detection
- **Backup Systems:** Redundant sensors for critical measurements

#### Model Management
- **Regular Retraining:** Monthly or seasonal model updates
- **Performance Monitoring:** Daily accuracy tracking
- **Version Control:** Track model versions and performance

#### Operational Use
- **Gradual Deployment:** Start with test zones before full implementation
- **Human Oversight:** Always have manual override capabilities
- **Documentation:** Keep detailed logs of decisions and outcomes

### Support and Resources

- **Documentation:** See docs/ folder for detailed technical documentation
- **Examples:** Check examples/ folder for usage examples
- **Issues:** Report problems via GitHub issues
- **Community:** Join discussions in project forums

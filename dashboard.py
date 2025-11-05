#!/usr/bin/env python3
"""
Interactive dashboard for irrigation predictions and monitoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append('src')

from models.hybrid_model import HybridIrrigationModel
from models.post_processor import IrrigationPostProcessor
from utils.data_generator import IrrigationDataGenerator


def load_model():
    """Load trained model."""
    model_path = "models/hybrid_xgboost_model.pkl"
    if os.path.exists(model_path):
        return HybridIrrigationModel.load_model(model_path)
    else:
        st.error(f"Model not found at {model_path}. Please train a model first.")
        return None


def generate_demo_data():
    """Generate demo data for dashboard."""
    generator = IrrigationDataGenerator(seed=42)
    
    # Generate recent data (last 30 days)
    df = generator.generate_dataset(
        start_date=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        days=30,
        zones=[f"zone_{i:02d}" for i in range(1, 6)],
        climate="semi_arid"
    )
    
    return df


def create_soil_moisture_plot(df, zone_id):
    """Create soil moisture trajectory plot."""
    zone_data = df[df['zone_id'] == zone_id].copy()
    zone_data = zone_data.sort_values('date')
    
    fig = go.Figure()
    
    # Soil moisture
    fig.add_trace(go.Scatter(
        x=zone_data['date'],
        y=zone_data['theta0'],
        mode='lines+markers',
        name='Soil Moisture',
        line=dict(color='blue', width=2)
    ))
    
    # Field capacity line
    fig.add_hline(
        y=zone_data['field_capacity_theta'].iloc[0],
        line_dash="dash",
        line_color="green",
        annotation_text="Field Capacity"
    )
    
    # Wilting point line
    fig.add_hline(
        y=zone_data['wilting_point_theta'].iloc[0],
        line_dash="dash",
        line_color="red",
        annotation_text="Wilting Point"
    )
    
    fig.update_layout(
        title=f"Soil Moisture Trajectory - {zone_id}",
        xaxis_title="Date",
        yaxis_title="Volumetric Water Content",
        hovermode='x unified'
    )
    
    return fig


def create_irrigation_schedule_plot(df, predictions):
    """Create irrigation schedule visualization."""
    df_plot = df.copy()
    df_plot['predicted_irrigation'] = predictions
    
    # Group by date and sum across zones
    daily_schedule = df_plot.groupby('date').agg({
        'irrigation_mm_next_day': 'sum',
        'predicted_irrigation': 'sum',
        'area_m2': 'sum'
    }).reset_index()
    
    # Convert to liters
    daily_schedule['actual_liters'] = daily_schedule['irrigation_mm_next_day'] * daily_schedule['area_m2']
    daily_schedule['predicted_liters'] = daily_schedule['predicted_irrigation'] * daily_schedule['area_m2']
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Daily Irrigation (mm)', 'Daily Water Volume (L)'),
        vertical_spacing=0.1
    )
    
    # Irrigation depth
    fig.add_trace(
        go.Bar(x=daily_schedule['date'], y=daily_schedule['irrigation_mm_next_day'], 
               name='Actual', marker_color='lightblue'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=daily_schedule['date'], y=daily_schedule['predicted_irrigation'], 
               name='Predicted', marker_color='orange', opacity=0.7),
        row=1, col=1
    )
    
    # Water volume
    fig.add_trace(
        go.Bar(x=daily_schedule['date'], y=daily_schedule['actual_liters'], 
               name='Actual (L)', marker_color='lightblue', showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=daily_schedule['date'], y=daily_schedule['predicted_liters'], 
               name='Predicted (L)', marker_color='orange', opacity=0.7, showlegend=False),
        row=2, col=1
    )
    
    fig.update_layout(
        title="Irrigation Schedule Comparison",
        height=600,
        barmode='group'
    )
    
    return fig


def create_zone_performance_plot(df, predictions):
    """Create zone performance comparison."""
    df_plot = df.copy()
    df_plot['predicted_irrigation'] = predictions
    df_plot['error'] = df_plot['predicted_irrigation'] - df_plot['irrigation_mm_next_day']
    
    # Calculate zone statistics
    zone_stats = df_plot.groupby('zone_id').agg({
        'error': ['mean', 'std'],
        'irrigation_mm_next_day': 'mean',
        'predicted_irrigation': 'mean'
    }).round(3)
    
    zone_stats.columns = ['Mean Error', 'Error Std', 'Actual Mean', 'Predicted Mean']
    zone_stats = zone_stats.reset_index()
    
    fig = px.scatter(
        zone_stats, 
        x='Actual Mean', 
        y='Predicted Mean',
        size='Error Std',
        hover_data=['Mean Error'],
        text='zone_id',
        title="Zone Performance: Predicted vs Actual Irrigation"
    )
    
    # Add perfect prediction line
    max_val = max(zone_stats['Actual Mean'].max(), zone_stats['Predicted Mean'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode='lines',
        line=dict(dash='dash', color='red'),
        name='Perfect Prediction'
    ))
    
    fig.update_traces(textposition="top center")
    fig.update_layout(
        xaxis_title="Actual Mean Irrigation (mm)",
        yaxis_title="Predicted Mean Irrigation (mm)"
    )
    
    return fig


def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="Irrigation Prediction Dashboard",
        page_icon="💧",
        layout="wide"
    )
    
    st.title("🌱 ML-Driven Precision Irrigation Dashboard")
    st.markdown("Real-time irrigation recommendations and monitoring")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Demo Data", "Upload CSV"]
    )
    
    if data_source == "Demo Data":
        df = generate_demo_data()
        st.sidebar.success("Demo data loaded")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, parse_dates=['date'])
            st.sidebar.success("Data uploaded successfully")
        else:
            st.sidebar.warning("Please upload a CSV file")
            st.stop()
    
    # Make predictions
    with st.spinner("Making predictions..."):
        predictions = model.predict(df)
        diagnostics = model.predict_with_diagnostics(df)
    
    # Post-processing
    post_processor = IrrigationPostProcessor()
    post_processed = post_processor.process_predictions(predictions, df)
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_water = np.sum(post_processed['irrigation_liters'])
        st.metric("Total Water (L)", f"{total_water:,.0f}")
    
    with col2:
        avg_irrigation = np.mean(post_processed['irrigation_mm_final'])
        st.metric("Avg Irrigation (mm)", f"{avg_irrigation:.1f}")
    
    with col3:
        fc_violations = np.sum(post_processed['fc_violation_prevented'])
        st.metric("FC Violations Prevented", f"{fc_violations}")
    
    with col4:
        stress_risk = np.sum(post_processed['stress_risk'])
        st.metric("Zones at Stress Risk", f"{stress_risk}")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🌱 Zone Details", "📅 Schedule", "📈 Performance"])
    
    with tab1:
        st.subheader("Irrigation Schedule Overview")
        
        # Irrigation schedule plot
        schedule_fig = create_irrigation_schedule_plot(df, post_processed['irrigation_mm_final'])
        st.plotly_chart(schedule_fig, use_container_width=True)
        
        # Zone performance plot
        performance_fig = create_zone_performance_plot(df, post_processed['irrigation_mm_final'])
        st.plotly_chart(performance_fig, use_container_width=True)
    
    with tab2:
        st.subheader("Zone-Specific Analysis")
        
        # Zone selector
        selected_zone = st.selectbox("Select Zone", df['zone_id'].unique())
        
        # Soil moisture plot
        soil_fig = create_soil_moisture_plot(df, selected_zone)
        st.plotly_chart(soil_fig, use_container_width=True)
        
        # Zone details table
        zone_data = df[df['zone_id'] == selected_zone].copy()
        zone_diagnostics = diagnostics[diagnostics['zone_id'] == selected_zone].copy()
        
        # Merge data
        zone_summary = pd.merge(
            zone_data[['date', 'zone_id', 'theta0', 'irrigation_mm_next_day']],
            zone_diagnostics[['date', 'zone_id', 'physics_baseline_mm', 'ml_residual_mm', 'hybrid_safe_mm']],
            on=['date', 'zone_id']
        )
        
        st.subheader(f"Detailed Predictions for {selected_zone}")
        st.dataframe(zone_summary.round(3), use_container_width=True)
    
    with tab3:
        st.subheader("Irrigation Schedule")
        
        # Create schedule table
        schedule_df = post_processor.create_irrigation_schedule(
            post_processed['irrigation_mm_final'], df
        )
        
        # Filter by priority
        priority_filter = st.multiselect(
            "Filter by Priority",
            options=['high', 'medium', 'low'],
            default=['high', 'medium', 'low']
        )
        
        filtered_schedule = schedule_df[schedule_df['priority'].isin(priority_filter)]
        
        st.dataframe(filtered_schedule, use_container_width=True)
        
        # Download button
        csv = filtered_schedule.to_csv(index=False)
        st.download_button(
            label="Download Schedule as CSV",
            data=csv,
            file_name=f"irrigation_schedule_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with tab4:
        st.subheader("Model Performance")
        
        # Feature importance
        if hasattr(model, 'get_feature_importance'):
            importance_df = model.get_feature_importance(top_n=15)
            
            fig_importance = px.bar(
                importance_df, 
                x='importance', 
                y='feature',
                orientation='h',
                title="Feature Importance"
            )
            fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Error distribution
        errors = post_processed['irrigation_mm_final'] - df['irrigation_mm_next_day']
        
        fig_error = px.histogram(
            x=errors,
            nbins=30,
            title="Prediction Error Distribution",
            labels={'x': 'Error (mm)', 'y': 'Frequency'}
        )
        st.plotly_chart(fig_error, use_container_width=True)
        
        # Performance metrics
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Absolute Error", f"{mae:.2f} mm")
        with col2:
            st.metric("Root Mean Square Error", f"{rmse:.2f} mm")


if __name__ == "__main__":
    main()

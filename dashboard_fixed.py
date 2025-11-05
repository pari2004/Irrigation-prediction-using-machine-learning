#!/usr/bin/env python3
"""
Fixed dashboard for irrigation predictions and monitoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime


def load_project_data():
    """Load project results data."""
    try:
        # Load main project data
        if os.path.exists("data/processed/test_predictions.csv"):
            # Load predictions and test data
            predictions_df = pd.read_csv("data/processed/test_predictions.csv", parse_dates=['date'])
            test_df = pd.read_csv("data/processed/irrigation_test.csv", parse_dates=['date'])
            
            # Merge data
            demo_data = pd.merge(test_df, predictions_df, on=['zone_id', 'date'], how='left')
            
            # Rename columns for dashboard
            demo_data['predicted_irrigation_mm'] = demo_data['hybrid_safe_mm']
            demo_data['predicted_liters'] = demo_data['irrigation_liters']
            demo_data['ml_prediction_mm'] = demo_data['ml_residual_mm'] + demo_data['physics_baseline_mm']
            demo_data['hybrid_prediction_mm'] = demo_data['hybrid_raw_mm']
            
            # Calculate summary
            errors = demo_data['predicted_irrigation_mm'] - demo_data['irrigation_mm_next_day']
            summary = {
                'total_water_liters': demo_data['predicted_liters'].sum(),
                'final_mae': abs(errors).mean(),
                'physics_mae': abs(demo_data['physics_baseline_mm'] - demo_data['irrigation_mm_next_day']).mean(),
                'under_irrigation_rate': (errors < -1.0).mean(),
                'over_irrigation_rate': (errors > 1.0).mean(),
                'fc_violations_prevented': 0,
                'num_features': 90,
                'num_test_samples': len(demo_data)
            }
            
            return demo_data, summary
            
        else:
            st.error("Project data not found. Please run 'python run_project.py' first.")
            return None, None
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


def create_irrigation_comparison_plot(df):
    """Create comparison plot of different prediction methods."""
    fig = go.Figure()
    
    # Actual irrigation
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['irrigation_mm_next_day'],
        mode='markers+lines',
        name='Actual',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Physics baseline
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['physics_baseline_mm'],
        mode='lines',
        name='Physics Baseline',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    # Final prediction
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['predicted_irrigation_mm'],
        mode='markers+lines',
        name='Final Hybrid',
        line=dict(color='purple', width=3),
        marker=dict(size=8, symbol='diamond')
    ))
    
    fig.update_layout(
        title="Irrigation Predictions Comparison",
        xaxis_title="Date",
        yaxis_title="Irrigation (mm)",
        hovermode='x unified',
        height=500
    )
    
    return fig


def create_soil_moisture_plot(df, zone_id):
    """Create soil moisture plot for a specific zone."""
    zone_data = df[df['zone_id'] == zone_id].copy()
    zone_data = zone_data.sort_values('date')
    
    fig = go.Figure()
    
    # Current soil moisture
    fig.add_trace(go.Scatter(
        x=zone_data['date'],
        y=zone_data['theta0'],
        mode='lines+markers',
        name='Soil Moisture',
        line=dict(color='blue', width=2)
    ))
    
    # Field capacity
    if 'field_capacity_theta' in zone_data.columns:
        fig.add_hline(
            y=zone_data['field_capacity_theta'].iloc[0],
            line_dash="dash",
            line_color="green",
            annotation_text="Field Capacity"
        )
    
    # Wilting point
    if 'wilting_point_theta' in zone_data.columns:
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
        hovermode='x unified',
        height=400
    )
    
    return fig


def create_performance_metrics_plot(summary):
    """Create performance metrics comparison."""
    metrics = ['Physics Baseline', 'Final Hybrid']
    mae_values = [summary['physics_mae'], summary['final_mae']]
    
    colors = ['red', 'purple']
    
    fig = go.Figure(data=[
        go.Bar(x=metrics, y=mae_values, marker_color=colors)
    ])
    
    fig.update_layout(
        title="Model Performance Comparison (MAE)",
        xaxis_title="Model Type",
        yaxis_title="Mean Absolute Error (mm)",
        height=400
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
    
    # Load data
    demo_data, summary = load_project_data()
    
    if demo_data is None or summary is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("Dashboard Controls")
    st.sidebar.success("Project data loaded successfully!")
    
    # Display summary metrics
    st.header("📊 Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Water (L)", 
            f"{summary['total_water_liters']:,.0f}",
            help="Total water recommended across all zones"
        )
    
    with col2:
        st.metric(
            "Model Accuracy", 
            f"{summary['final_mae']:.2f} mm MAE",
            delta=f"{summary['physics_mae'] - summary['final_mae']:.2f} mm improvement",
            help="Mean Absolute Error of final model vs physics baseline"
        )
    
    with col3:
        st.metric(
            "Under-irrigation Rate", 
            f"{summary['under_irrigation_rate']:.1%}",
            help="Percentage of predictions that under-irrigate by >1mm"
        )
    
    with col4:
        improvement = (summary['physics_mae'] - summary['final_mae']) / summary['physics_mae'] * 100
        st.metric(
            "Improvement", 
            f"{improvement:.1f}%",
            help="Improvement over physics-only baseline"
        )
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📈 Predictions", "🌱 Zone Analysis", "📋 Data"])
    
    with tab1:
        st.subheader("Irrigation Predictions Comparison")
        
        # Prediction comparison plot
        comparison_fig = create_irrigation_comparison_plot(demo_data)
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Performance metrics
        st.subheader("Model Performance")
        performance_fig = create_performance_metrics_plot(summary)
        st.plotly_chart(performance_fig, use_container_width=True)
        
        # Key insights
        st.subheader("Key Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Model Performance:**
            - Physics baseline: {summary['physics_mae']:.2f} mm MAE
            - Final model: {summary['final_mae']:.2f} mm MAE
            - Improvement: {improvement:.1f}%
            """)
        
        with col2:
            st.warning(f"""
            **Irrigation Patterns:**
            - Under-irrigation rate: {summary['under_irrigation_rate']:.1%}
            - Over-irrigation rate: {summary['over_irrigation_rate']:.1%}
            - Features used: {summary['num_features']}
            """)
    
    with tab2:
        st.subheader("Zone-Specific Analysis")
        
        # Zone selector
        zones = demo_data['zone_id'].unique()
        selected_zone = st.selectbox("Select Zone for Analysis", zones)
        
        # Zone data
        zone_data = demo_data[demo_data['zone_id'] == selected_zone]
        
        # Soil moisture plot
        soil_fig = create_soil_moisture_plot(demo_data, selected_zone)
        st.plotly_chart(soil_fig, use_container_width=True)
        
        # Zone details
        st.subheader(f"Details for {selected_zone}")
        
        # Calculate zone summary statistics safely
        avg_irrigation = zone_data['predicted_irrigation_mm'].mean()
        total_water = zone_data['predicted_liters'].sum()
        avg_soil_moisture = zone_data['theta0'].mean()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Irrigation", f"{avg_irrigation:.1f} mm")
        
        with col2:
            st.metric("Total Water", f"{total_water:,.0f} L")
        
        with col3:
            st.metric("Avg Soil Moisture", f"{avg_soil_moisture:.3f} VWC")
    
    with tab3:
        st.subheader("Raw Data")
        
        # Data filters
        col1, col2 = st.columns(2)
        
        with col1:
            zone_filter = st.multiselect(
                "Filter by Zone",
                options=demo_data['zone_id'].unique(),
                default=demo_data['zone_id'].unique()
            )
        
        with col2:
            date_range = st.date_input(
                "Date Range",
                value=(demo_data['date'].min(), demo_data['date'].max()),
                min_value=demo_data['date'].min(),
                max_value=demo_data['date'].max()
            )
        
        # Filter data
        filtered_data = demo_data[
            (demo_data['zone_id'].isin(zone_filter)) &
            (demo_data['date'] >= pd.to_datetime(date_range[0])) &
            (demo_data['date'] <= pd.to_datetime(date_range[1]))
        ]
        
        # Display key columns
        display_cols = [
            'date', 'zone_id', 'crop_type', 'theta0', 'ET0_mm', 
            'irrigation_mm_next_day', 'predicted_irrigation_mm', 
            'predicted_liters', 'physics_baseline_mm'
        ]
        
        available_cols = [col for col in display_cols if col in filtered_data.columns]
        st.dataframe(filtered_data[available_cols], use_container_width=True)
        
        # Download button
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name=f"irrigation_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("🌱 **ML-Driven Precision Irrigation System** | Built with Streamlit and Plotly")


if __name__ == "__main__":
    main()

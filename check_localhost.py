#!/usr/bin/env python3
"""
Check localhost services for ML-driven irrigation project.
"""

import requests
import json
from datetime import datetime


def check_service(url, name):
    """Check if a service is running."""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ {name}: Running at {url}")
            return True
        else:
            print(f"⚠️ {name}: Responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {name}: Not accessible at {url}")
        return False


def main():
    """Check all localhost services."""
    print("🔍 ML-Driven Precision Irrigation - Localhost Status Check")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    services = [
        ("http://localhost:8501", "Streamlit Dashboard"),
        ("http://localhost:8000", "Project Web Server"),
        ("http://localhost:8000/api/status", "API Status Endpoint"),
        ("http://localhost:8000/api/metrics", "API Metrics Endpoint")
    ]
    
    running_services = 0
    total_services = len(services)
    
    for url, name in services:
        if check_service(url, name):
            running_services += 1
    
    print()
    print("📊 SERVICE SUMMARY:")
    print(f"   Running: {running_services}/{total_services}")
    print(f"   Status: {'🟢 All Systems Operational' if running_services == total_services else '🟡 Some Services Down'}")
    
    if running_services > 0:
        print()
        print("🌐 AVAILABLE SERVICES:")
        if check_service("http://localhost:8501", ""):
            print("   📊 Dashboard: http://localhost:8501")
        if check_service("http://localhost:8000", ""):
            print("   🏠 Project Home: http://localhost:8000")
            print("   📋 API Status: http://localhost:8000/api/status")
            print("   📈 API Metrics: http://localhost:8000/api/metrics")
    
    # Try to get project metrics
    try:
        response = requests.get("http://localhost:8000/api/metrics", timeout=5)
        if response.status_code == 200:
            metrics = response.json()
            print()
            print("📈 PROJECT METRICS:")
            print(f"   Model Accuracy: {metrics.get('final_mae', 'N/A')} mm MAE")
            print(f"   Improvement: {metrics.get('improvement_pct', 'N/A')}%")
            print(f"   Water Managed: {metrics.get('total_water_liters', 'N/A'):,} L")
            print(f"   Under-irrigation: {metrics.get('under_irrigation_rate', 'N/A'):.1%}")
            print(f"   Over-irrigation: {metrics.get('over_irrigation_rate', 'N/A'):.1%}")
    except:
        pass
    
    print()
    print("🎯 QUICK ACCESS:")
    print("   Dashboard: http://localhost:8501")
    print("   Project Home: http://localhost:8000")
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Start all localhost services for ML-driven irrigation project.
"""

import subprocess
import time
import webbrowser
import os
import sys
from datetime import datetime


def start_service(command, name, wait_time=3):
    """Start a service and wait."""
    print(f"🚀 Starting {name}...")
    try:
        process = subprocess.Popen(command, shell=True, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        time.sleep(wait_time)
        
        if process.poll() is None:  # Process is still running
            print(f"✅ {name} started successfully")
            return process
        else:
            print(f"❌ {name} failed to start")
            return None
    except Exception as e:
        print(f"❌ Error starting {name}: {e}")
        return None


def check_url(url, max_attempts=5):
    """Check if URL is accessible."""
    import requests
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False


def main():
    """Start all localhost services."""
    print("🌱 ML-DRIVEN PRECISION IRRIGATION - LOCALHOST STARTUP")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Directory: {os.getcwd()}")
    print()
    
    # Check if required files exist
    required_files = [
        "dashboard_fixed.py",
        "web_server.py",
        "data/processed/test_predictions.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n🔧 Please run 'python run_project.py' first to generate data.")
        return
    
    print("✅ All required files found")
    print()
    
    # Start services
    services = []
    
    # 1. Start Streamlit Dashboard
    print("📊 Starting Streamlit Dashboard...")
    streamlit_cmd = f"{sys.executable} -m streamlit run dashboard_fixed.py --server.port 8501 --server.headless true"
    streamlit_process = start_service(streamlit_cmd, "Streamlit Dashboard", 5)
    if streamlit_process:
        services.append(("Streamlit Dashboard", "http://localhost:8501", streamlit_process))
    
    # 2. Start Web Server
    print("🌐 Starting Project Web Server...")
    webserver_cmd = f"{sys.executable} web_server.py"
    webserver_process = start_service(webserver_cmd, "Web Server", 3)
    if webserver_process:
        services.append(("Project Web Server", "http://localhost:8000", webserver_process))
    
    print()
    print("⏳ Waiting for services to initialize...")
    time.sleep(5)
    
    # Check service status
    print("🔍 Checking service status...")
    working_services = []
    
    for name, url, process in services:
        if check_url(url):
            print(f"✅ {name}: {url}")
            working_services.append((name, url))
        else:
            print(f"⚠️ {name}: Starting up... {url}")
            working_services.append((name, url))  # Add anyway, might be slow to start
    
    print()
    print("🎉 LOCALHOST SERVICES READY!")
    print("=" * 60)
    
    if working_services:
        print("🌐 Available Services:")
        for name, url in working_services:
            print(f"   📍 {name}: {url}")
        
        print()
        print("🚀 Opening services in browser...")
        
        # Open main project page
        webbrowser.open("http://localhost:8000")
        time.sleep(2)
        
        # Open dashboard
        webbrowser.open("http://localhost:8501")
        
        print()
        print("📋 Quick Access URLs:")
        print("   🏠 Project Home: http://localhost:8000")
        print("   📊 Dashboard: http://localhost:8501")
        print("   📈 API Status: http://localhost:8000/api/status")
        print("   📊 API Metrics: http://localhost:8000/api/metrics")
        
        print()
        print("🎯 FEATURES AVAILABLE:")
        print("   ✅ Interactive irrigation dashboard")
        print("   ✅ Real-time prediction visualization")
        print("   ✅ Zone-specific soil moisture analysis")
        print("   ✅ Performance metrics and comparisons")
        print("   ✅ Project documentation and files")
        print("   ✅ API endpoints for integration")
        print("   ✅ Downloadable reports and data")
        
        print()
        print("🛑 To stop services: Press Ctrl+C in this terminal")
        print("💡 To restart: Run 'python start_localhost.py' again")
        
        try:
            print("\n⏳ Services running... (Press Ctrl+C to stop)")
            while True:
                time.sleep(10)
                # Check if processes are still running
                active_processes = 0
                for _, _, process in services:
                    if process and process.poll() is None:
                        active_processes += 1
                
                if active_processes == 0:
                    print("⚠️ All services stopped")
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping services...")
            for name, _, process in services:
                if process and process.poll() is None:
                    print(f"   Stopping {name}...")
                    process.terminate()
            print("✅ All services stopped")
    
    else:
        print("❌ No services started successfully")
        print("🔧 Try running the services manually:")
        print("   streamlit run dashboard_fixed.py --server.port 8501")
        print("   python web_server.py")


if __name__ == "__main__":
    main()

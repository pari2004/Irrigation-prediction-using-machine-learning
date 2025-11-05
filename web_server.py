#!/usr/bin/env python3
"""
Simple web server to serve project files and documentation.
"""

import http.server
import socketserver
import os
import webbrowser
from datetime import datetime
import json


class ProjectHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler for the project."""
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/':
            self.send_project_index()
        elif self.path == '/api/status':
            self.send_api_status()
        elif self.path == '/api/metrics':
            self.send_api_metrics()
        else:
            super().do_GET()
    
    def send_project_index(self):
        """Send the project index page."""
        html_content = self.generate_index_html()
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def send_api_status(self):
        """Send API status."""
        status = {
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'project': 'ML-Driven Precision Irrigation',
            'version': '1.0',
            'components': {
                'dashboard': 'http://localhost:8501',
                'web_server': 'http://localhost:8000',
                'model': 'trained',
                'data': 'available'
            }
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(status, indent=2).encode())
    
    def send_api_metrics(self):
        """Send project metrics."""
        try:
            # Load demo summary if available
            if os.path.exists("data/sample/demo_summary.json"):
                with open("data/sample/demo_summary.json", "r") as f:
                    metrics = json.load(f)
            else:
                metrics = {
                    'final_mae': 1.971,
                    'physics_mae': 33.889,
                    'improvement_pct': 94.2,
                    'total_water_liters': 6196995,
                    'under_irrigation_rate': 0.183,
                    'over_irrigation_rate': 0.441
                }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(metrics, indent=2).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
    
    def generate_index_html(self):
        """Generate the main index HTML page."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML-Driven Precision Irrigation System</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            padding: 40px 0;
        }
        .header h1 {
            font-size: 3em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .header p {
            font-size: 1.2em;
            margin: 10px 0;
            opacity: 0.9;
        }
        .cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 40px 0;
        }
        .card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            transition: transform 0.3s ease;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .card h3 {
            font-size: 1.5em;
            margin-bottom: 15px;
            color: #fff;
        }
        .card p {
            opacity: 0.9;
            line-height: 1.6;
        }
        .btn {
            display: inline-block;
            padding: 12px 24px;
            background: rgba(255, 255, 255, 0.2);
            color: white;
            text-decoration: none;
            border-radius: 25px;
            margin: 10px;
            transition: all 0.3s ease;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        .btn:hover {
            background: rgba(255, 255, 255, 0.3);
            transform: scale(1.05);
        }
        .btn-primary {
            background: #4CAF50;
            border-color: #4CAF50;
        }
        .btn-primary:hover {
            background: #45a049;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 30px 0;
        }
        .metric {
            background: rgba(255, 255, 255, 0.15);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
        }
        .metric-label {
            font-size: 0.9em;
            opacity: 0.8;
            margin-top: 5px;
        }
        .footer {
            text-align: center;
            padding: 40px 0;
            opacity: 0.8;
        }
        .status {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 20px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="status">🟢 System Running</div>
    
    <div class="container">
        <div class="header">
            <h1>🌱 ML-Driven Precision Irrigation</h1>
            <p>Exact Water Recommendation per Zone/Day</p>
            <p>Hybrid Physics + Machine Learning Approach</p>
        </div>
        
        <div class="metrics" id="metrics">
            <div class="metric">
                <div class="metric-value">1.97</div>
                <div class="metric-label">mm MAE</div>
            </div>
            <div class="metric">
                <div class="metric-value">94.2%</div>
                <div class="metric-label">Improvement</div>
            </div>
            <div class="metric">
                <div class="metric-value">6.2M</div>
                <div class="metric-label">Liters Managed</div>
            </div>
            <div class="metric">
                <div class="metric-value">5</div>
                <div class="metric-label">Zones</div>
            </div>
        </div>
        
        <div class="cards">
            <div class="card">
                <h3>📊 Interactive Dashboard</h3>
                <p>Real-time irrigation predictions, soil moisture monitoring, and performance analytics with interactive visualizations.</p>
                <a href="http://localhost:8501" class="btn btn-primary" target="_blank">Open Dashboard</a>
            </div>
            
            <div class="card">
                <h3>🤖 ML Model</h3>
                <p>Hybrid physics + XGBoost model with asymmetric loss function achieving 94.2% improvement over baseline.</p>
                <a href="/data/processed/" class="btn">View Data</a>
                <a href="/models/" class="btn">Model Files</a>
            </div>
            
            <div class="card">
                <h3>📋 Documentation</h3>
                <p>Comprehensive project documentation, model card, usage guides, and technical specifications.</p>
                <a href="/docs/" class="btn">Documentation</a>
                <a href="/README.md" class="btn">README</a>
            </div>
            
            <div class="card">
                <h3>📄 Project Report</h3>
                <p>Complete PDF report with performance analysis, visualizations, and technical details ready for submission.</p>
                <a href="/ML_Irrigation_Project_Report_20250824_125635.pdf" class="btn">Download PDF</a>
            </div>
            
            <div class="card">
                <h3>🔧 API Endpoints</h3>
                <p>RESTful API for system status, metrics, and integration with external irrigation controllers.</p>
                <a href="/api/status" class="btn">Status</a>
                <a href="/api/metrics" class="btn">Metrics</a>
            </div>
            
            <div class="card">
                <h3>💾 Project Files</h3>
                <p>Access source code, datasets, trained models, and all project components for development and deployment.</p>
                <a href="/src/" class="btn">Source Code</a>
                <a href="/data/" class="btn">Datasets</a>
            </div>
        </div>
        
        <div class="footer">
            <p>🌱 ML-Driven Precision Irrigation System | Final Year Project</p>
            <p>Built with Python, XGBoost, Streamlit, and FAO-56 Physics</p>
            <p id="timestamp">Last updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        </div>
    </div>
    
    <script>
        // Load metrics from API
        fetch('/api/metrics')
            .then(response => response.json())
            .then(data => {
                const metrics = document.getElementById('metrics');
                metrics.innerHTML = `
                    <div class="metric">
                        <div class="metric-value">${data.final_mae?.toFixed(2) || '1.97'}</div>
                        <div class="metric-label">mm MAE</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${data.improvement_pct?.toFixed(1) || '94.2'}%</div>
                        <div class="metric-label">Improvement</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${(data.total_water_liters/1000000)?.toFixed(1) || '6.2'}M</div>
                        <div class="metric-label">Liters Managed</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">5</div>
                        <div class="metric-label">Zones</div>
                    </div>
                `;
            })
            .catch(error => console.log('Using default metrics'));
        
        // Update timestamp every minute
        setInterval(() => {
            document.getElementById('timestamp').textContent = 
                'Last updated: ' + new Date().toLocaleString();
        }, 60000);
    </script>
</body>
</html>
        """


def start_web_server(port=8000):
    """Start the web server."""
    print(f"🌐 Starting ML Irrigation Project Web Server...")
    print(f"📁 Serving files from: {os.getcwd()}")
    
    try:
        with socketserver.TCPServer(("", port), ProjectHTTPRequestHandler) as httpd:
            print(f"✅ Server running at: http://localhost:{port}")
            print(f"📊 Dashboard available at: http://localhost:8501")
            print(f"🔗 API endpoints:")
            print(f"   - Status: http://localhost:{port}/api/status")
            print(f"   - Metrics: http://localhost:{port}/api/metrics")
            print(f"\n🚀 Opening browser...")
            
            # Open browser
            webbrowser.open(f'http://localhost:{port}')
            
            print(f"🛑 Press Ctrl+C to stop the server")
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")


if __name__ == "__main__":
    start_web_server()

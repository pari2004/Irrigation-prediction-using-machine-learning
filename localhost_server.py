#!/usr/bin/env python3
"""
Enhanced localhost server for ML-driven irrigation project with model card support.
"""

import http.server
import socketserver
import os
import webbrowser
import json
import markdown
from datetime import datetime
from urllib.parse import unquote


class EnhancedProjectHandler(http.server.SimpleHTTPRequestHandler):
    """Enhanced HTTP handler with markdown support and project navigation."""
    
    def do_GET(self):
        """Handle GET requests with enhanced features."""
        path = unquote(self.path)
        
        if path == '/':
            self.serve_project_home()
        elif path == '/model-card':
            self.serve_model_card()
        elif path == '/docs':
            self.serve_documentation_index()
        elif path.endswith('.md'):
            self.serve_markdown_file(path)
        elif path == '/api/status':
            self.serve_api_status()
        elif path == '/api/metrics':
            self.serve_api_metrics()
        elif path == '/api/files':
            self.serve_file_list()
        else:
            super().do_GET()
    
    def serve_project_home(self):
        """Serve enhanced project home page."""
        html = self.generate_project_home()
        self.send_html_response(html)
    
    def serve_model_card(self):
        """Serve the model card as HTML."""
        try:
            # Try to read from docs directory first
            model_card_path = "docs/MODEL_CARD.md"
            if not os.path.exists(model_card_path):
                # Fallback to current directory
                model_card_path = "MODEL_CARD.md"
            
            if os.path.exists(model_card_path):
                with open(model_card_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Convert markdown to HTML
                html_content = markdown.markdown(content, extensions=['tables', 'fenced_code'])
                
                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>ML Irrigation Model Card</title>
                    <style>
                        body {{ font-family: 'Segoe UI', sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
                        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                        h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
                        h3 {{ color: #7f8c8d; }}
                        code {{ background: #f8f9fa; padding: 2px 4px; border-radius: 3px; }}
                        pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                        th {{ background-color: #f2f2f2; font-weight: bold; }}
                        .nav {{ background: #3498db; color: white; padding: 10px; margin: -20px -20px 20px -20px; }}
                        .nav a {{ color: white; text-decoration: none; margin-right: 20px; }}
                        .nav a:hover {{ text-decoration: underline; }}
                    </style>
                </head>
                <body>
                    <div class="nav">
                        <a href="/">🏠 Home</a>
                        <a href="/model-card">📋 Model Card</a>
                        <a href="/docs">📚 Documentation</a>
                        <a href="http://localhost:8501" target="_blank">📊 Dashboard</a>
                    </div>
                    {html_content}
                </body>
                </html>
                """
                self.send_html_response(html)
            else:
                self.send_error(404, "Model card not found")
        except Exception as e:
            self.send_error(500, f"Error loading model card: {e}")
    
    def serve_documentation_index(self):
        """Serve documentation index."""
        docs_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Project Documentation</title>
            <style>
                body { font-family: 'Segoe UI', sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }
                .nav { background: #3498db; color: white; padding: 10px; margin: -20px -20px 20px -20px; }
                .nav a { color: white; text-decoration: none; margin-right: 20px; }
                .doc-card { background: #f8f9fa; padding: 20px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #3498db; }
                .doc-card h3 { margin-top: 0; color: #2c3e50; }
                .doc-card a { color: #3498db; text-decoration: none; font-weight: bold; }
                .doc-card a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <div class="nav">
                <a href="/">🏠 Home</a>
                <a href="/model-card">📋 Model Card</a>
                <a href="/docs">📚 Documentation</a>
                <a href="http://localhost:8501" target="_blank">📊 Dashboard</a>
            </div>
            
            <h1>📚 Project Documentation</h1>
            
            <div class="doc-card">
                <h3>📋 Model Card</h3>
                <p>Comprehensive model documentation including architecture, performance, limitations, and usage guidelines.</p>
                <a href="/model-card">View Model Card →</a>
            </div>
            
            <div class="doc-card">
                <h3>📖 Usage Guide</h3>
                <p>Detailed instructions for using the irrigation prediction system, API documentation, and integration examples.</p>
                <a href="/docs/USAGE_GUIDE.md">View Usage Guide →</a>
            </div>
            
            <div class="doc-card">
                <h3>📊 Project Report</h3>
                <p>Complete PDF report with performance analysis, visualizations, and technical details.</p>
                <a href="/ML_Irrigation_Project_Report_20250824_125635.pdf">Download PDF Report →</a>
            </div>
            
            <div class="doc-card">
                <h3>💾 Source Code</h3>
                <p>Access to all source code, models, and datasets used in the project.</p>
                <a href="/src/">Browse Source Code →</a>
            </div>
            
            <div class="doc-card">
                <h3>📈 Live Dashboard</h3>
                <p>Interactive dashboard with real-time predictions, visualizations, and performance metrics.</p>
                <a href="http://localhost:8501" target="_blank">Open Dashboard →</a>
            </div>
        </body>
        </html>
        """
        self.send_html_response(docs_html)
    
    def serve_markdown_file(self, path):
        """Serve any markdown file as HTML."""
        try:
            file_path = path.lstrip('/')
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                html_content = markdown.markdown(content, extensions=['tables', 'fenced_code'])
                
                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>{os.path.basename(file_path)}</title>
                    <style>
                        body {{ font-family: 'Segoe UI', sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
                        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                        h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
                        code {{ background: #f8f9fa; padding: 2px 4px; border-radius: 3px; }}
                        pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                        .nav {{ background: #3498db; color: white; padding: 10px; margin: -20px -20px 20px -20px; }}
                        .nav a {{ color: white; text-decoration: none; margin-right: 20px; }}
                    </style>
                </head>
                <body>
                    <div class="nav">
                        <a href="/">🏠 Home</a>
                        <a href="/model-card">📋 Model Card</a>
                        <a href="/docs">📚 Documentation</a>
                        <a href="http://localhost:8501" target="_blank">📊 Dashboard</a>
                    </div>
                    {html_content}
                </body>
                </html>
                """
                self.send_html_response(html)
            else:
                self.send_error(404, f"File not found: {file_path}")
        except Exception as e:
            self.send_error(500, f"Error loading file: {e}")
    
    def serve_api_status(self):
        """Serve API status."""
        status = {
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'project': 'ML-Driven Precision Irrigation',
            'version': '1.0',
            'services': {
                'web_server': 'http://localhost:8000',
                'dashboard': 'http://localhost:8501',
                'model_card': 'http://localhost:8000/model-card',
                'documentation': 'http://localhost:8000/docs'
            }
        }
        self.send_json_response(status)
    
    def serve_api_metrics(self):
        """Serve project metrics."""
        try:
            # Default metrics
            metrics = {
                'final_mae': 1.971,
                'physics_mae': 33.889,
                'improvement_pct': 94.2,
                'total_water_liters': 6196995,
                'under_irrigation_rate': 0.183,
                'over_irrigation_rate': 0.441,
                'zones': 5,
                'prediction_days': 83
            }
            
            # Try to load actual metrics if available
            if os.path.exists("data/sample/demo_summary.json"):
                with open("data/sample/demo_summary.json", "r") as f:
                    actual_metrics = json.load(f)
                    metrics.update(actual_metrics)
            
            self.send_json_response(metrics)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status=500)
    
    def serve_file_list(self):
        """Serve list of project files."""
        try:
            files = {
                'documentation': [],
                'data': [],
                'models': [],
                'source': []
            }
            
            # Scan for files
            for root, dirs, filenames in os.walk('.'):
                for filename in filenames:
                    filepath = os.path.join(root, filename).replace('\\', '/')
                    if filepath.startswith('./docs/'):
                        files['documentation'].append(filepath)
                    elif filepath.startswith('./data/'):
                        files['data'].append(filepath)
                    elif filepath.startswith('./models/'):
                        files['models'].append(filepath)
                    elif filepath.startswith('./src/'):
                        files['source'].append(filepath)
            
            self.send_json_response(files)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status=500)
    
    def generate_project_home(self):
        """Generate enhanced project home page."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML-Driven Precision Irrigation System</title>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; margin: 0; padding: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; min-height: 100vh; }}
                .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
                .header {{ text-align: center; padding: 40px 0; }}
                .header h1 {{ font-size: 3em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
                .nav {{ background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 20px 0; text-align: center; }}
                .nav a {{ color: white; text-decoration: none; margin: 0 20px; padding: 10px 20px; background: rgba(255,255,255,0.2); border-radius: 20px; display: inline-block; }}
                .nav a:hover {{ background: rgba(255,255,255,0.3); }}
                .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 40px 0; }}
                .card {{ background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); border-radius: 15px; padding: 30px; text-align: center; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 30px 0; }}
                .metric {{ background: rgba(255,255,255,0.15); padding: 20px; border-radius: 10px; text-align: center; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #4CAF50; }}
                .btn {{ display: inline-block; padding: 12px 24px; background: rgba(255,255,255,0.2); color: white; text-decoration: none; border-radius: 25px; margin: 10px; }}
                .btn:hover {{ background: rgba(255,255,255,0.3); }}
                .btn-primary {{ background: #4CAF50; }}
                .status {{ position: fixed; top: 20px; right: 20px; background: #4CAF50; padding: 10px 20px; border-radius: 20px; }}
            </style>
        </head>
        <body>
            <div class="status">🟢 System Running</div>
            
            <div class="container">
                <div class="header">
                    <h1>🌱 ML-Driven Precision Irrigation</h1>
                    <p>Exact Water Recommendation per Zone/Day</p>
                </div>
                
                <div class="nav">
                    <a href="/">🏠 Home</a>
                    <a href="/model-card">📋 Model Card</a>
                    <a href="/docs">📚 Documentation</a>
                    <a href="http://localhost:8501" target="_blank">📊 Dashboard</a>
                    <a href="/api/status">🔧 API</a>
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
                        <p>Real-time irrigation predictions and analytics</p>
                        <a href="http://localhost:8501" class="btn btn-primary" target="_blank">Open Dashboard</a>
                    </div>
                    
                    <div class="card">
                        <h3>📋 Model Card</h3>
                        <p>Complete model documentation and specifications</p>
                        <a href="/model-card" class="btn">View Model Card</a>
                    </div>
                    
                    <div class="card">
                        <h3>📚 Documentation</h3>
                        <p>Usage guides and technical documentation</p>
                        <a href="/docs" class="btn">Browse Docs</a>
                    </div>
                    
                    <div class="card">
                        <h3>📄 Project Report</h3>
                        <p>Complete PDF report with analysis</p>
                        <a href="/ML_Irrigation_Project_Report_20250824_125635.pdf" class="btn">Download PDF</a>
                    </div>
                </div>
            </div>
            
            <script>
                // Load live metrics
                fetch('/api/metrics')
                    .then(response => response.json())
                    .then(data => {{
                        document.getElementById('metrics').innerHTML = `
                            <div class="metric">
                                <div class="metric-value">${{data.final_mae?.toFixed(2) || '1.97'}}</div>
                                <div class="metric-label">mm MAE</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">${{data.improvement_pct?.toFixed(1) || '94.2'}}%</div>
                                <div class="metric-label">Improvement</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">${{(data.total_water_liters/1000000)?.toFixed(1) || '6.2'}}M</div>
                                <div class="metric-label">Liters Managed</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">${{data.zones || '5'}}</div>
                                <div class="metric-label">Zones</div>
                            </div>
                        `;
                    }})
                    .catch(() => console.log('Using default metrics'));
            </script>
        </body>
        </html>
        """
    
    def send_html_response(self, html):
        """Send HTML response."""
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def send_json_response(self, data, status=200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())


def start_enhanced_server(port=8000):
    """Start the enhanced localhost server."""
    print(f"🌱 Starting Enhanced ML Irrigation Project Server...")
    print(f"📁 Serving from: {os.getcwd()}")
    
    try:
        with socketserver.TCPServer(("", port), EnhancedProjectHandler) as httpd:
            print(f"✅ Server running at: http://localhost:{port}")
            print(f"📋 Model Card: http://localhost:{port}/model-card")
            print(f"📚 Documentation: http://localhost:{port}/docs")
            print(f"📊 Dashboard: http://localhost:8501")
            print(f"🔧 API Status: http://localhost:{port}/api/status")
            
            # Open browser
            webbrowser.open(f'http://localhost:{port}')
            
            print(f"🛑 Press Ctrl+C to stop")
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print(f"🛑 Server stopped")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    start_enhanced_server()

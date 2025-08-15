"""
HTML Templates Module
=====================

HTML templates for web interface.
"""

from datetime import date


def get_home_template(face_system) -> str:
    """Generate home page HTML template"""
    return f"""
    <html>
        <head>
            <title>Simple Face Recognition System</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .endpoint {{ background: #e8f4f8; padding: 20px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #2196F3; }}
                h1 {{ color: #333; text-align: center; }}
                h2 {{ color: #666; }}
                .success {{ color: #4CAF50; }}
                .error {{ color: #f44336; }}
                .info {{ color: #2196F3; }}
                .warning {{ color: #ff9800; background: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                ul {{ line-height: 1.6; }}
                .status {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎯 Simple Face Recognition System</h1>
                <p style="text-align: center; color: #666;">A Face ID-like authentication system using OpenCV</p>
                
                <div class="warning">
                    <strong>Note:</strong> This version uses OpenCV face detection with histogram comparison. 
                    While not as accurate as deep learning models, it works without complex dependencies.
                </div>
                
                <div class="endpoint">
                    <h2>📡 Available Endpoints:</h2>
                    <ul>
                        <li><strong>POST /register/{{name}}</strong> - Register a new face (captures 3 samples)</li>
                        <li><strong>GET /verify</strong> - Start face verification</li>
                        <li><strong>GET /users</strong> - List registered users</li>
                        <li><strong>DELETE /users/{{name}}</strong> - Delete a registered user</li>
                        <li><strong>GET /verifications/today</strong> - Get today's verification logs</li>
                        <li><strong>GET /verifications/user/{{name}}</strong> - Get user verification history</li>
                        <li><strong>GET /verifications/status/{{name}}</strong> - Check if user verified today</li>
                    </ul>
                </div>
                
                <div class="endpoint">
                    <h2>📋 Instructions:</h2>
                    <ol>
                        <li>Use <code>/register/{{name}}</code> to register new faces</li>
                        <li>Use <code>/verify</code> to start real-time verification</li>
                        <li>During registration: Press SPACE to capture (3 times), ESC to cancel</li>
                        <li>During verification: Press ESC to exit</li>
                        <li>Ensure good lighting and face the camera directly</li>
                    </ol>
                </div>
                
                <div class="status">
                    <h2>📊 Current Status:</h2>
                    <p><strong>Registered Users:</strong> <span class="info">{len(face_system.known_names)}</span></p>
                    <p><strong>Users:</strong> {", ".join(face_system.known_names) if face_system.known_names else "None"}</p>
                    <p><strong>Recognition Method:</strong> OpenCV + Histogram Comparison</p>
                    <p><strong>Verification Logging:</strong> <span class="success">✅ Enabled</span></p>
                    <p><strong>Today's Verifications:</strong> <span class="info">{len(face_system.verification_db.get_today_verifications())}</span></p>
                </div>
                
                <div class="endpoint">
                    <h2>🔧 Usage Tips:</h2>
                    <ul>
                        <li>Register in good lighting conditions</li>
                        <li>Keep your face centered and at normal distance</li>
                        <li>Multiple face samples improve accuracy</li>
                        <li>Similarity threshold is set to 0.7 (adjustable)</li>
                        <li><strong>NEW:</strong> Successful verifications are automatically logged to database</li>
                        <li><strong>NEW:</strong> Each user can only be logged once per day</li>
                        <li><strong>NEW:</strong> View verification logs via API endpoints</li>
                    </ul>
                </div>
            </div>
        </body>
    </html>
    """

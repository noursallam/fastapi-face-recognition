"""
Face Recognition System - Main Application
==========================================

Main FastAPI application entry point using modular structure.
"""

from fastapi import FastAPI
import uvicorn

from app import (
    SimpleFaceRecognitionSystem,
    setup_routes,
    APP_TITLE,
    APP_DESCRIPTION,
    HOST,
    PORT
)


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    # Create FastAPI app
    app = FastAPI(
        title=APP_TITLE,
        description=APP_DESCRIPTION,
        version="2.0.0"
    )
    
    # Initialize face recognition system
    face_system = SimpleFaceRecognitionSystem()
    
    # Setup API routes
    setup_routes(app, face_system)
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    print("Starting Simple Face Recognition System...")
    print("This version uses OpenCV face detection with histogram comparison")
    print("✅ Verification logging enabled - successful verifications will be stored in database")
    print("📊 Database file: verification_logs.db")
    print("Make sure you have a working webcam connected.")
    print(f"Access the web interface at: http://localhost:{PORT}")
    
    uvicorn.run(app, host=HOST, port=PORT)

"""
API Routes Module
=================

FastAPI routes for face recognition system.
"""

from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from datetime import date
from typing import Dict, Any

from ..core.face_recognition import SimpleFaceRecognitionSystem
from ..templates import get_home_template


def setup_routes(app, face_system: SimpleFaceRecognitionSystem):
    """Setup all API routes"""
    
    @app.get("/", response_class=HTMLResponse)
    async def home():
        """Home page with instructions"""
        return get_home_template(face_system)

    @app.post("/register/{name}")
    async def register_user(name: str) -> Dict[str, Any]:
        """Register a new user's face"""
        if not name.strip():
            raise HTTPException(status_code=400, detail="Name cannot be empty")
        
        success, message = face_system.register_face(name.strip())
        
        if success:
            return {"status": "success", "message": message}
        else:
            raise HTTPException(status_code=400, detail=message)

    @app.get("/verify")
    async def verify_faces() -> Dict[str, Any]:
        """Start face verification mode (legacy endpoint)"""
        success, message = face_system.verify_face()
        
        if success:
            return {"status": "success", "message": message}
        else:
            raise HTTPException(status_code=500, detail=message)

    @app.get("/verify/detailed")
    async def verify_faces_detailed() -> Dict[str, Any]:
        """Face verification with detailed results (recommended for web clients)"""
        success, message, data = face_system.verify_face_headless()
        
        if success:
            return {
                "status": "success", 
                "message": message,
                "verification_data": data
            }
        else:
            return {
                "status": "failure",
                "message": message,
                "verification_data": data
            }

    @app.get("/users")
    async def get_users() -> Dict[str, Any]:
        """Get list of registered users"""
        return {
            "registered_users": face_system.known_names,
            "total_count": len(face_system.known_names),
            "recognition_method": "OpenCV + Histogram Comparison"
        }

    @app.delete("/users/{name}")
    async def delete_user(name: str) -> Dict[str, Any]:
        """Delete a registered user"""
        if name not in face_system.known_names:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Find and remove user
        index = face_system.known_names.index(name)
        face_system.known_names.pop(index)
        face_system.known_faces.pop(index)
        face_system.save_faces()
        
        return {"status": "success", "message": f"User {name} deleted successfully"}

    @app.get("/verifications/today")
    async def get_today_verifications() -> Dict[str, Any]:
        """Get all verifications for today"""
        verifications = face_system.verification_db.get_today_verifications()
        return {
            "date": str(date.today()),
            "total_verifications": len(verifications),
            "verifications": verifications
        }

    @app.get("/verifications/user/{user_name}")
    async def get_user_verifications(user_name: str, days: int = 30) -> Dict[str, Any]:
        """Get verification history for a specific user"""
        if user_name not in face_system.known_names:
            raise HTTPException(status_code=404, detail="User not found")
        
        history = face_system.verification_db.get_user_verification_history(user_name, days)
        has_verified_today = face_system.verification_db.has_verified_today(user_name)
        
        return {
            "user_name": user_name,
            "has_verified_today": has_verified_today,
            "days_requested": days,
            "total_records": len(history),
            "verification_history": history
        }

    @app.get("/verifications/status/{user_name}")
    async def check_verification_status(user_name: str) -> Dict[str, Any]:
        """Check if a user has verified today"""
        if user_name not in face_system.known_names:
            raise HTTPException(status_code=404, detail="User not found")
        
        has_verified = face_system.verification_db.has_verified_today(user_name)
        return {
            "user_name": user_name,
            "date": str(date.today()),
            "has_verified_today": has_verified,
            "status": "verified" if has_verified else "not_verified"
        }

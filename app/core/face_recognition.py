"""
Face Recognition Core Module
============================

Core face recognition functionality using OpenCV.
"""

import cv2
import pickle
import os
import numpy as np
from typing import Dict, List, Tuple
from ..config import (
    ENCODINGS_FILE, 
    FACE_RECOGNITION_THRESHOLD,
    FACE_RESIZE_DIMENSIONS,
    REQUIRED_FACE_SAMPLES,
    CAMERA_INDEX,
    CAMERA_FLIP_HORIZONTAL,
    FACE_CASCADE_SCALE_FACTOR,
    FACE_CASCADE_MIN_NEIGHBORS,
    HEADLESS_MODE
)
from ..database.verification_db import VerificationDatabase


class SimpleFaceRecognitionSystem:
    """Simple face recognition system using OpenCV"""
    
    def __init__(self):
        self.known_faces = []
        self.known_names = []
        self.load_faces()
        
        # Initialize OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize verification database
        self.verification_db = VerificationDatabase()
    
    def load_faces(self):
        """Load face data from file if it exists"""
        if os.path.exists(ENCODINGS_FILE):
            try:
                with open(ENCODINGS_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data.get('faces', [])
                    self.known_names = data.get('names', [])
                print(f"Loaded {len(self.known_names)} registered faces")
            except Exception as e:
                print(f"Error loading face data: {e}")
                self.known_faces = []
                self.known_names = []
    
    def save_faces(self):
        """Save face data to file"""
        try:
            data = {
                'faces': self.known_faces,
                'names': self.known_names
            }
            with open(ENCODINGS_FILE, 'wb') as f:
                pickle.dump(data, f)
            print("Face data saved successfully")
        except Exception as e:
            print(f"Error saving face data: {e}")
    
    def extract_face_features(self, face_img):
        """Extract simple features from face image"""
        # Resize to standard size
        face_resized = cv2.resize(face_img, FACE_RESIZE_DIMENSIONS)
        # Convert to grayscale
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        # Calculate histogram as feature
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        # Normalize histogram
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    
    def compare_faces(self, face_features, threshold=FACE_RECOGNITION_THRESHOLD):
        """Compare face features with known faces"""
        if not self.known_faces:
            return False, "Unknown", 1.0
        
        best_match_idx = -1
        best_similarity = 0
        
        for i, known_features in enumerate(self.known_faces):
            # Calculate correlation coefficient
            similarity = cv2.compareHist(face_features, known_features, cv2.HISTCMP_CORREL)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = i
        
        if best_similarity > threshold:
            return True, self.known_names[best_match_idx], best_similarity
        else:
            return False, "Unknown", best_similarity
    
    def register_face(self, name: str) -> Tuple[bool, str]:
        """Register a new face using webcam"""
        if name in self.known_names:
            return False, "Name already registered"
        
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            return False, "Could not open camera"
        
        print(f"Registration mode for {name}. Press SPACE to capture, ESC to cancel.")
        
        captured_faces = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            if CAMERA_FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                FACE_CASCADE_SCALE_FACTOR, 
                FACE_CASCADE_MIN_NEIGHBORS
            )
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]
                
                # Show face count
                cv2.putText(
                    frame, 
                    f"Faces captured: {len(captured_faces)}/{REQUIRED_FACE_SAMPLES}", 
                    (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
            
            # Display instructions
            cv2.putText(frame, f"Registering: {name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to capture, ESC to cancel", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Face Registration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Spacebar to capture
                if len(faces) > 0:
                    # Get the largest face
                    largest_face = max(faces, key=lambda face: face[2] * face[3])
                    x, y, w, h = largest_face
                    face_roi = frame[y:y+h, x:x+w]
                    
                    if face_roi.size > 0:
                        captured_faces.append(face_roi.copy())
                        print(f"Captured face {len(captured_faces)}/{REQUIRED_FACE_SAMPLES}")
                        
                        if len(captured_faces) >= REQUIRED_FACE_SAMPLES:
                            # Process captured faces
                            face_features_list = []
                            for face in captured_faces:
                                features = self.extract_face_features(face)
                                face_features_list.append(features)
                            
                            # Average the features
                            avg_features = np.mean(face_features_list, axis=0)
                            
                            # Save the registration
                            self.known_faces.append(avg_features)
                            self.known_names.append(name)
                            self.save_faces()
                            
                            cap.release()
                            cv2.destroyAllWindows()
                            return True, f"Successfully registered {name} with {len(captured_faces)} face samples"
                    else:
                        print("Invalid face region, try again")
                else:
                    print("No face detected, try again")
            elif key == 27:  # ESC to cancel
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return False, "Registration cancelled"
    
    def verify_face_headless(self, max_attempts: int = 10) -> Tuple[bool, str, Dict]:
        """Headless face verification - captures frames and returns results without GUI"""
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            return False, "Could not open camera", {}
        
        verification_results = []
        
        for attempt in range(max_attempts):
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Flip frame horizontally for mirror effect
            if CAMERA_FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                FACE_CASCADE_SCALE_FACTOR, 
                FACE_CASCADE_MIN_NEIGHBORS
            )
            
            # Process each face found
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]
                
                if face_roi.size > 0:
                    # Extract features
                    face_features = self.extract_face_features(face_roi)
                    
                    # Compare with known faces
                    is_match, name, similarity = self.compare_faces(face_features)
                    
                    result = {
                        "attempt": attempt + 1,
                        "face_detected": True,
                        "is_match": is_match,
                        "name": name,
                        "similarity": float(similarity),
                        "access_granted": is_match and name != "Unknown"
                    }
                    
                    # Log successful verification to database
                    if is_match and name != "Unknown":
                        self.verification_db.log_verification(name, similarity)
                        result["logged"] = True
                    else:
                        result["logged"] = False
                    
                    verification_results.append(result)
                    
                    # If we found a match, we can stop early
                    if is_match and name != "Unknown":
                        cap.release()
                        return True, f"Verification successful for {name}", {
                            "user": name,
                            "similarity": float(similarity),
                            "access_granted": True,
                            "attempts": attempt + 1,
                            "all_results": verification_results
                        }
        
        cap.release()
        
        # If we got here, no successful match was found
        if verification_results:
            best_result = max(verification_results, key=lambda x: x["similarity"])
            return False, f"Verification failed. Best match: {best_result['name']} (similarity: {best_result['similarity']:.2f})", {
                "access_granted": False,
                "attempts": max_attempts,
                "best_match": best_result,
                "all_results": verification_results
            }
        else:
            return False, "No faces detected during verification", {
                "access_granted": False,
                "attempts": max_attempts,
                "faces_detected": 0,
                "all_results": verification_results
            }

    def verify_face(self) -> Tuple[bool, str]:
        """Face verification - uses GUI mode or headless mode based on configuration"""
        if HEADLESS_MODE:
            success, message, data = self.verify_face_headless()
            return success, message
        else:
            return self.verify_face_gui()
    
    def verify_face_gui(self) -> Tuple[bool, str]:
        """Interactive face verification using webcam with GUI display"""
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            return False, "Could not open camera"
        
        print("Verification mode. Press ESC to exit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            if CAMERA_FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                FACE_CASCADE_SCALE_FACTOR, 
                FACE_CASCADE_MIN_NEIGHBORS
            )
            
            # Process each face found
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]
                
                if face_roi.size > 0:
                    # Extract features
                    face_features = self.extract_face_features(face_roi)
                    
                    # Compare with known faces
                    is_match, name, similarity = self.compare_faces(face_features)
                    
                    # Log successful verification to database
                    if is_match and name != "Unknown":
                        self.verification_db.log_verification(name, similarity)
                        verification_status = "✅ VERIFIED & LOGGED"
                    else:
                        verification_status = "❌ ACCESS DENIED"
                    
                    color = (0, 255, 0) if is_match else (0, 0, 255)  # Green for match, Red for no match
                    access_text = "Access Granted" if is_match else "Access Denied"
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw name and access status
                    cv2.putText(frame, name, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.putText(frame, access_text, (x, y+h+25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f"Similarity: {similarity:.2f}", (x, y+h+50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.putText(frame, verification_status, (x, y+h+75), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Display registered users count
            cv2.putText(frame, f"Registered Users: {len(self.known_names)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press ESC to exit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            try:
                cv2.imshow('Face Verification', frame)
            except cv2.error as e:
                cap.release()
                return False, f"GUI display error: {str(e)}. Consider using headless mode."
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return True, "Verification completed"

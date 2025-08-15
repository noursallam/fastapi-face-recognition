"""
Configuration and Constants
===========================

Application configuration and constant values.
"""

import os
from pathlib import Path

# File paths
ENCODINGS_FILE = "face_data.pkl"
DATABASE_FILE = "verification_logs.db"

# Face recognition settings
FACE_RECOGNITION_THRESHOLD = 0.7
FACE_RESIZE_DIMENSIONS = (100, 100)
REQUIRED_FACE_SAMPLES = 3

# Camera settings
CAMERA_INDEX = 0
CAMERA_FLIP_HORIZONTAL = True

# OpenCV settings
FACE_CASCADE_SCALE_FACTOR = 1.3
FACE_CASCADE_MIN_NEIGHBORS = 5

# Server settings
HOST = "0.0.0.0"
PORT = 8000

# App metadata
APP_TITLE = "Face Recognition System"
APP_DESCRIPTION = "Face ID-like authentication system with OpenCV"

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Database settings
DEFAULT_HISTORY_DAYS = 30

# GUI settings
HEADLESS_MODE = True  # Set to False for desktop GUI mode

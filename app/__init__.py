"""
Face Recognition App Package
============================

A modular face recognition system with FastAPI and OpenCV.

Structure:
- core/: Core face recognition functionality
- api/: FastAPI endpoints and routes
- database/: Database operations and models
- templates/: HTML templates and static content
"""

from .core import SimpleFaceRecognitionSystem
from .api import setup_routes
from .config import APP_TITLE, APP_DESCRIPTION, HOST, PORT

__version__ = "2.0.0"
__author__ = "Face Recognition System"

__all__ = [
    'SimpleFaceRecognitionSystem',
    'setup_routes',
    'APP_TITLE',
    'APP_DESCRIPTION',
    'HOST',
    'PORT'
]

from .core import SimpleFaceRecognitionSystem
from .api import setup_routes
from .config import APP_TITLE, APP_DESCRIPTION, HOST, PORT

__all__ = [
    'SimpleFaceRecognitionSystem',
    'setup_routes',
    'APP_TITLE',
    'APP_DESCRIPTION',
    'HOST',
    'PORT'
]

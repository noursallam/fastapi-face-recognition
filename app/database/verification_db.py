"""
Verification Database Module
============================

Handles all database operations for verification logging.
"""

import sqlite3
from datetime import datetime, date
from typing import Dict, List
from ..config import DATABASE_FILE


class VerificationDatabase:
    """Database handler for verification logs"""
    
    def __init__(self, db_file=DATABASE_FILE):
        self.db_file = db_file
        self.init_database()
    
    def init_database(self):
        """Initialize the verification logs database"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Create verification_logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS verification_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT NOT NULL,
                    verification_date DATE NOT NULL,
                    verification_time DATETIME NOT NULL,
                    similarity_score REAL NOT NULL,
                    UNIQUE(user_name, verification_date)
                )
            ''')
            
            # Create index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_user_date 
                ON verification_logs(user_name, verification_date)
            ''')
            
            conn.commit()
            conn.close()
            print("✅ Verification database initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing database: {e}")
    
    def log_verification(self, user_name: str, similarity_score: float) -> bool:
        """Log a successful verification"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            today = date.today()
            now = datetime.now()
            
            # Insert or update verification log for today
            cursor.execute('''
                INSERT OR REPLACE INTO verification_logs 
                (user_name, verification_date, verification_time, similarity_score)
                VALUES (?, ?, ?, ?)
            ''', (user_name, today, now, similarity_score))
            
            conn.commit()
            conn.close()
            print(f"✅ Logged verification for {user_name} on {today}")
            return True
        except Exception as e:
            print(f"❌ Error logging verification: {e}")
            return False
    
    def get_today_verifications(self) -> List[Dict]:
        """Get all verifications for today"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            today = date.today()
            cursor.execute('''
                SELECT user_name, verification_time, similarity_score
                FROM verification_logs
                WHERE verification_date = ?
                ORDER BY verification_time DESC
            ''', (today,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "user_name": row[0],
                    "verification_time": row[1],
                    "similarity_score": row[2]
                })
            
            conn.close()
            return results
        except Exception as e:
            print(f"❌ Error getting today's verifications: {e}")
            return []
    
    def get_user_verification_history(self, user_name: str, days: int = 30) -> List[Dict]:
        """Get verification history for a specific user"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT verification_date, verification_time, similarity_score
                FROM verification_logs
                WHERE user_name = ?
                ORDER BY verification_date DESC, verification_time DESC
                LIMIT ?
            ''', (user_name, days))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "verification_date": row[0],
                    "verification_time": row[1],
                    "similarity_score": row[2]
                })
            
            conn.close()
            return results
        except Exception as e:
            print(f"❌ Error getting user verification history: {e}")
            return []
    
    def has_verified_today(self, user_name: str) -> bool:
        """Check if user has verified today"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            today = date.today()
            cursor.execute('''
                SELECT COUNT(*) FROM verification_logs
                WHERE user_name = ? AND verification_date = ?
            ''', (user_name, today))
            
            count = cursor.fetchone()[0]
            conn.close()
            return count > 0
        except Exception as e:
            print(f"❌ Error checking today's verification: {e}")
            return False

import sqlite3
from datetime import datetime

DB_PATH = "mandarin_progress.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create User Profiles Table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  created_at TEXT)''')
                  
    # Create Assessment Logs Table (Linked to User)
    c.execute('''CREATE TABLE IF NOT EXISTS attempts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT,
                  target_word TEXT,
                  overall_score REAL,
                  whisper_transcript TEXT,
                  dsp_latency_ms REAL,
                  timestamp TEXT,
                  FOREIGN KEY(username) REFERENCES users(username))''')
                  
    conn.commit()
    conn.close()

def log_attempt(username, target_word, score, transcript, latency):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Ensure user exists, if not, create them silently
    c.execute("INSERT OR IGNORE INTO users (username, created_at) VALUES (?, ?)", 
              (username, datetime.now().isoformat()))
              
    # Log the exact attempt
    c.execute('''INSERT INTO attempts 
                 (username, target_word, overall_score, whisper_transcript, dsp_latency_ms, timestamp) 
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (username, target_word, score, transcript, latency, datetime.now().isoformat()))
              
    conn.commit()
    conn.close()

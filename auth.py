import sqlite3
import hashlib

DB_PATH = "users.db"

def get_db():
    return sqlite3.connect(DB_PATH)

def create_user_table():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username, password):
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)", 
            (username, hash_password(password))
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(username, password):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT * FROM users WHERE username=? AND password=?", 
        (username, hash_password(password))
    )
    user = c.fetchone()
    conn.close()
    return user

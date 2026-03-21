import sqlite3
import json
import time
from typing import List, Dict, Any

DB_NAME = "rlm_chat_history.db"

def init_db():
    """Initialize the SQLite database with required tables."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Table for conversations (sessions)
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            created_at REAL
        )
    ''')
    
    # Table for messages (linked to conversations)
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            role TEXT,
            content TEXT,
            tokens INTEGER,
            timestamp REAL,
            model TEXT DEFAULT 'rlm',
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        )
    ''')

    # Migrate existing DBs that lack the model column
    try:
        c.execute("ALTER TABLE messages ADD COLUMN model TEXT DEFAULT 'rlm'")
    except Exception:
        pass  # Column already exists
    
    conn.commit()
    conn.close()

def create_conversation(title: str = "New Chat") -> int:
    """Creates a new conversation and returns its ID."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO conversations (title, created_at) VALUES (?, ?)", (title, time.time()))
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return new_id

def get_all_conversations() -> List[Dict]:
    """Returns a list of all conversations, sorted by newest first."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM conversations ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def save_message(conversation_id: int, role: str, content: str, tokens: int = 0, model: str = 'rlm'):
    """Saves a single message to the database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO messages (conversation_id, role, content, tokens, timestamp, model)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (conversation_id, role, content, tokens, time.time(), model))
    
    # Auto-update title if it's the first user message
    if role == "user":
        # Check if title is generic "New Chat"
        c.execute("SELECT title FROM conversations WHERE id = ?", (conversation_id,))
        current_title = c.fetchone()[0]
        if current_title == "New Chat":
            # Generate simple title from first 30 chars
            new_title = content[:30] + "..." if len(content) > 30 else content
            c.execute("UPDATE conversations SET title = ? WHERE id = ?", (new_title, conversation_id))
            
    conn.commit()
    conn.close()

def load_messages(conversation_id: int, models: list = None) -> List[Dict[str, Any]]:
    """Loads messages for a conversation, optionally filtered by model list."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    if models:
        placeholders = ",".join("?" * len(models))
        c.execute(
            f"SELECT role, content, tokens, timestamp, model FROM messages "
            f"WHERE conversation_id = ? AND model IN ({placeholders}) ORDER BY id ASC",
            [conversation_id] + models,
        )
    else:
        c.execute(
            "SELECT role, content, tokens, timestamp, model FROM messages "
            "WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        )
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def delete_conversation(conversation_id: int):
    """Deletes a conversation and its messages."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    c.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()

# Initialize on module import
init_db()
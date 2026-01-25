import sqlite3
import json
import os
import datetime
from pathlib import Path

class ChatHistoryDB:
    """Quản lý lịch sử chat trong SQLite."""
    
    def __init__(self, db_path=None):
        """Khởi tạo kết nối đến cơ sở dữ liệu SQLite."""
        if db_path is None:
            # Tạo thư mục data nếu chưa tồn tại
            data_dir = Path(__file__).parent.parent / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = data_dir / "chat_history.db"
        
        self.db_path = str(db_path)
        self._create_tables_if_not_exist()
    
    def _create_tables_if_not_exist(self):
        """Tạo các bảng cần thiết nếu chưa tồn tại."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Bảng lưu trữ các phiên chat
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            title TEXT,
            metadata TEXT
        )
        ''')
        
        # Bảng lưu trữ các tin nhắn trong phiên chat
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_session(self, session_id=None, title=None, metadata=None):
        """Tạo một phiên chat mới."""
        if session_id is None:
            # Tạo session_id từ timestamp nếu không được cung cấp
            session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute(
            "INSERT OR IGNORE INTO chat_sessions (session_id, title, metadata) VALUES (?, ?, ?)",
            (session_id, title, metadata_json)
        )
        
        conn.commit()
        conn.close()
        
        return session_id
    
    def add_message(self, session_id, role, content):
        """Thêm một tin nhắn vào phiên chat."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Kiểm tra xem phiên chat có tồn tại không
        cursor.execute("SELECT 1 FROM chat_sessions WHERE session_id = ?", (session_id,))
        if not cursor.fetchone():
            # Tạo phiên chat nếu chưa tồn tại
            self.create_session(session_id)
        
        cursor.execute(
            "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        
        conn.commit()
        conn.close()
    
    def get_session_messages(self, session_id):
        """Lấy tất cả tin nhắn của một phiên chat."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT role, content, timestamp FROM chat_messages WHERE session_id = ? ORDER BY timestamp",
            (session_id,)
        )
        
        messages = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return messages
    
    def get_all_sessions(self):
        """Lấy tất cả các phiên chat."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, session_id, created_at, title FROM chat_sessions ORDER BY created_at DESC"
        )
        
        sessions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return sessions
    
    def delete_session(self, session_id):
        """Xóa một phiên chat và tất cả tin nhắn của nó."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Xóa tin nhắn trước
        cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        # Sau đó xóa phiên chat
        cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        
        conn.commit()
        conn.close()

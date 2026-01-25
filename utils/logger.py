import os
import logging
import sys
from datetime import datetime

class Logger:
    """
    Logger class to handle logging to file instead of terminal.
    Each time logging is started, the old log file is cleared.
    Ensures proper handling of Unicode characters.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        self.log_file = os.path.join(self.log_dir, 'agent_log.txt')
        
        # Create logs directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # Clear the log file
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        
        # Đảm bảo stdout và stderr sử dụng UTF-8
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
        if sys.stderr.encoding != 'utf-8':
            sys.stderr.reconfigure(encoding='utf-8')
            
        # Tạo file handler với encoding UTF-8
        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        
        # Cấu hình logger
        self.logger = logging.getLogger('agent_logger')
        self.logger.setLevel(logging.INFO)
        
        # Xóa các handler cũ nếu có
        if self.logger.handlers:
            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)
                
        # Thêm file handler mới
        self.logger.addHandler(file_handler)
        self._initialized = True
    
    def info(self, message):
        """Log an info message"""
        self.logger.info(message)
        
    def debug(self, message):
        """Log a debug message"""
        self.logger.debug(message)
        
    def warning(self, message):
        """Log a warning message"""
        self.logger.warning(message)
        
    def error(self, message):
        """Log an error message"""
        self.logger.error(message)
        
    def critical(self, message):
        """Log a critical message"""
        self.logger.critical(message)

# Singleton instance
logger = Logger()

def log(message, level='info'):
    """
    Log a message with the specified level.
    Ensures proper handling of Unicode characters.
    
    Args:
        message: The message to log
        level: The log level (info, debug, warning, error, critical)
    """
    # Đảm bảo message là Unicode string
    if not isinstance(message, str):
        message = str(message)
    
    # Xử lý các trường hợp escape sequence có thể gây lỗi
    try:
        # Thử encode và decode để đảm bảo UTF-8 hợp lệ
        message = message.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    except Exception as e:
        message = f"[Error encoding log message: {str(e)}]"
    
    # Gọi hàm log tương ứng
    if level == 'info':
        logger.info(message)
    elif level == 'debug':
        logger.debug(message)
    elif level == 'warning':
        logger.warning(message)
    elif level == 'error':
        logger.error(message)
    elif level == 'critical':
        logger.critical(message)
    else:
        logger.info(message)

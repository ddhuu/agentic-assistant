import os
import traceback
import datetime
from typing import Optional

def log_error(error: Exception, message: Optional[str] = None, log_file: str = "error_log.txt", keep_latest_only: bool = True):
    """
    Ghi log lỗi vào file .txt
    
    Args:
        error: Exception cần ghi log
        message: Thông điệp bổ sung (nếu có)
        log_file: Tên file log (mặc định là error_log.txt)
        keep_latest_only: Chỉ giữ log mới nhất (mặc định là True)
    """
    try:
        # Tạo thư mục logs nếu chưa tồn tại
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Đường dẫn đầy đủ đến file log
        log_path = os.path.join(logs_dir, log_file)
        
        # Lấy thời gian hiện tại
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Tạo nội dung log
        log_content = f"=== ERROR LOG: {current_time} ===\n"
        
        if message:
            log_content += f"Message: {message}\n"
        
        log_content += f"Error Type: {type(error).__name__}\n"
        log_content += f"Error Message: {str(error)}\n"
        log_content += "Traceback:\n"
        log_content += traceback.format_exc()
        log_content += "\n\n"
        
        # Chế độ ghi: 'w' để ghi đè (chỉ giữ log mới nhất) hoặc 'a' để ghi thêm
        mode = "w" if keep_latest_only else "a"
        
        # Ghi log vào file
        with open(log_path, mode, encoding="utf-8") as f:
            f.write(log_content)
            
        print(f"Error logged to {log_path}")
        return log_path
        
    except Exception as e:
        print(f"Failed to log error: {str(e)}")
        return None

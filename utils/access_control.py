import os
import json
import logging

logger = logging.getLogger(__name__)

class AccessControlManager:
    """Quản lý quyền truy cập file dựa trên vai trò người dùng"""
    
    def __init__(self, access_control_file=None):
        """Khởi tạo quản lý quyền truy cập"""
        if access_control_file is None:
            # Đường dẫn mặc định đến file cấu hình trong thư mục config
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            access_control_file = os.path.join(current_dir, "config", "access_control.json")
        self.access_control_file = access_control_file
        self.access_rules = self.load_access_rules()
    
    def load_access_rules(self):
        """Tải quy tắc quyền truy cập từ file"""
        try:
            if os.path.exists(self.access_control_file):
                with open(self.access_control_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"File quyền truy cập {self.access_control_file} không tồn tại. Sử dụng quy tắc mặc định.")
                return {
                    "file_types": {"default": ["admin"]},
                    "folders": {},
                    "special_files": {}
                }
        except Exception as e:
            logger.error(f"Lỗi khi tải quy tắc quyền truy cập: {e}")
            return {"file_types": {"default": ["admin"]}, "folders": {}, "special_files": {}}
    
    def check_file_access(self, file_path, user_role):
        """Kiểm tra quyền truy cập file dựa trên vai trò người dùng
        
        Args:
            file_path (str): Đường dẫn đầy đủ đến file
            user_role (str): Vai trò của người dùng
            
        Returns:
            bool: True nếu người dùng có quyền truy cập, False nếu không
            str: Thông báo lý do từ chối truy cập (nếu từ chối)
        """
        if not file_path or not os.path.exists(file_path):
            return False, f"File không tồn tại: {file_path}"
        
        # Admin luôn có quyền truy cập tất cả file
        if user_role == "admin":
            return True, ""
        
        # Kiểm tra file đặc biệt
        file_name = os.path.basename(file_path)
        if file_name in self.access_rules.get("special_files", {}):
            allowed_roles = self.access_rules["special_files"][file_name]
            if user_role in allowed_roles:
                return True, ""
            else:
                return False, f"Bạn không có quyền truy cập file đặc biệt: {file_name}. Cần vai trò: {', '.join(allowed_roles)}"
        
        # Kiểm tra thư mục
        parent_folder = os.path.basename(os.path.dirname(file_path))
        if parent_folder in self.access_rules.get("folders", {}):
            allowed_roles = self.access_rules["folders"][parent_folder]
            if user_role in allowed_roles:
                return True, ""
            else:
                return False, f"Bạn không có quyền truy cập thư mục: {parent_folder}. Cần vai trò: {', '.join(allowed_roles)}"
        
        # Kiểm tra loại file
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in self.access_rules.get("file_types", {}):
            allowed_roles = self.access_rules["file_types"][file_ext]
            if user_role in allowed_roles:
                return True, ""
            else:
                return False, f"Bạn không có quyền truy cập file loại: {file_ext}. Cần vai trò: {', '.join(allowed_roles)}"
        
        # Sử dụng quy tắc mặc định
        default_roles = self.access_rules.get("file_types", {}).get("default", ["admin"])
        if user_role in default_roles:
            return True, ""
        else:
            return False, f"Bạn không có quyền truy cập file này. Cần vai trò: {', '.join(default_roles)}"

    def log_access_attempt(self, file_path, user_role, success, reason=""):
        """Ghi log nỗ lực truy cập"""
        status = "SUCCESS" if success else "DENIED"
        logger.info(f"ACCESS {status}: User role '{user_role}' accessing '{file_path}'. {reason}")

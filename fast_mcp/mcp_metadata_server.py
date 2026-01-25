import json
import os
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mcp_metadata_server")

# Khởi tạo MCP server
mcp = FastMCP("metadata_server")
logger.info("Khởi động MCP Metadata Server...")

# Cấu hình
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")

# Đảm bảo thư mục data tồn tại
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)
    logger.info(f"Đã tạo thư mục dữ liệu: {DATA_DIR}")
else:
    logger.info(f"Thư mục dữ liệu đã tồn tại: {DATA_DIR}")

# Khởi tạo file metadata nếu chưa tồn tại
if not os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump({}, f, ensure_ascii=False, indent=2)
    logger.info(f"Đã tạo file metadata mới: {METADATA_FILE}")
else:
    logger.info(f"File metadata đã tồn tại: {METADATA_FILE}")

def load_metadata() -> Dict[str, Any]:
    """Đọc dữ liệu từ file JSON."""
    try:
        if not os.path.exists(METADATA_FILE):
            logger.warning("File metadata không tồn tại, tạo mới")
            return {}
            
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                logger.warning("File metadata rỗng, khởi tạo dữ liệu mới")
                return {}
            
            data = json.loads(content)
            logger.debug(f"Đã tải {len(data)} bản ghi metadata")
            return data
    except json.JSONDecodeError as e:
        logger.warning(f"Lỗi JSON khi đọc file metadata: {e}, khởi tạo dữ liệu mới")
        return {}
    except FileNotFoundError as e:
        logger.warning(f"File metadata không tìm thấy: {e}")
        return {}
    except Exception as e:
        logger.error(f"Lỗi không xác định khi đọc metadata: {e}")
        return {}

def save_metadata(data: Dict[str, Any]) -> bool:
    """Lưu dữ liệu vào file JSON."""
    try:
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Tạo file tạm thời để đảm bảo tính toàn vẹn dữ liệu
        temp_file = METADATA_FILE + '.tmp'
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        
        # Di chuyển file tạm thời thành file chính
        if os.path.exists(METADATA_FILE):
            os.remove(METADATA_FILE)
        os.rename(temp_file, METADATA_FILE)
        
        logger.info(f"Đã lưu {len(data)} bản ghi vào metadata")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi lưu metadata: {e}")
        # Xóa file tạm thời nếu có lỗi
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        return False

# Tải dữ liệu ban đầu
metadata_store = load_metadata()

class FileMetadata(BaseModel):
    id: str
    filename: str
    label: str
    content: Optional[str] = None
    created_at: str
    updated_at: str
    additional_metadata: Optional[Dict] = None

@mcp.tool()
async def save_metadata_to_json(
    filename: str, 
    label: str, 
    content: Optional[str] = None,
    additional_metadata: Optional[Dict] = None
) -> str:
    """
    Lưu metadata vào file JSON.
    
    Args:
        filename: Tên file
        label: Nhãn phân loại
        content: Nội dung file (tùy chọn)
        additional_metadata: Metadata bổ sung (tùy chọn)
    
    Returns:
        JSON string chứa kết quả
    """
    try:
        logger.info(f"Đang lưu metadata cho file: {filename}, label: {label}")
        metadata_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        metadata = {
            "id": metadata_id,
            "filename": filename,
            "label": label,
            "content": content,
            "created_at": now,
            "updated_at": now,
            "additional_metadata": additional_metadata or {}
        }
        
        # Tải dữ liệu hiện tại
        data = load_metadata()
        data[metadata_id] = metadata
        
        # Lưu vào file JSON
        if save_metadata(data):
            # Cập nhật bộ nhớ tạm
            global metadata_store
            metadata_store = data
            logger.info(f"Đã lưu metadata với ID: {metadata_id}")
            
            result = {
                "status": "success",
                "message": "Metadata đã được lưu vào file JSON",
                "metadata": metadata
            }
        else:
            result = {
                "status": "error",
                "message": "Không thể lưu metadata vào file"
            }
            
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Lỗi trong save_metadata_to_json: {e}")
        result = {
            "status": "error",
            "message": f"Lỗi khi lưu metadata: {str(e)}"
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

@mcp.tool()
async def get_metadata(metadata_id: str) -> str:
    """
    Lấy metadata từ file JSON theo ID.
    
    Args:
        metadata_id: ID của metadata cần lấy
    
    Returns:
        JSON string chứa metadata hoặc thông báo lỗi
    """
    try:
        logger.info(f"Đang tìm metadata với ID: {metadata_id}")
        data = load_metadata()
        metadata = data.get(metadata_id)
        
        if not metadata:
            logger.warning(f"Không tìm thấy metadata với ID: {metadata_id}")
            result = {
                "status": "error",
                "message": f"Không tìm thấy metadata với ID {metadata_id}"
            }
        else:
            logger.info(f"Đã tìm thấy metadata cho file: {metadata.get('filename')}")
            result = {
                "status": "success",
                "metadata": metadata
            }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Lỗi khi đọc metadata: {e}")
        result = {
            "status": "error",
            "message": f"Lỗi khi đọc metadata: {str(e)}"
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

@mcp.tool()
async def search_metadata(
    filename: Optional[str] = None, 
    label: Optional[str] = None
) -> str:
    """
    Tìm kiếm metadata theo tên file hoặc nhãn.
    
    Args:
        filename: Tên file để tìm kiếm (tùy chọn)
        label: Nhãn để tìm kiếm (tùy chọn)
    
    Returns:
        JSON string chứa kết quả tìm kiếm
    """
    try:
        search_criteria = []
        if filename:
            search_criteria.append(f"filename='{filename}'")
        if label:
            search_criteria.append(f"label='{label}'")
        
        criteria_str = " và ".join(search_criteria) if search_criteria else "tất cả"
        logger.info(f"Đang tìm kiếm metadata với tiêu chí: {criteria_str}")
        
        data = load_metadata()
        results = []
        
        for metadata_id, metadata in data.items():
            should_include = False
            
            if filename and filename.lower() in metadata.get("filename", "").lower():
                should_include = True
            elif label and label.lower() in metadata.get("label", "").lower():
                should_include = True
            elif not filename and not label:
                should_include = True
            
            if should_include:
                results.append(metadata)
        
        logger.info(f"Tìm thấy {len(results)} kết quả")
        result = {
            "status": "success",
            "count": len(results),
            "results": results
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Lỗi khi tìm kiếm metadata: {e}")
        result = {
            "status": "error",
            "message": f"Lỗi khi tìm kiếm metadata: {str(e)}"
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

@mcp.tool()
async def list_all_metadata() -> str:
    """
    Liệt kê tất cả metadata có trong hệ thống.
    
    Returns:
        JSON string chứa tất cả metadata
    """
    try:
        logger.info("Đang liệt kê tất cả metadata")
        data = load_metadata()
        
        result = {
            "status": "success",
            "count": len(data),
            "metadata": list(data.values())
        }
        
        logger.info(f"Đã liệt kê {len(data)} bản ghi metadata")
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"Lỗi khi liệt kê metadata: {e}")
        result = {
            "status": "error",
            "message": f"Lỗi khi liệt kê metadata: {str(e)}"
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

@mcp.resource("metadata://files/{metadata_id}")
async def get_metadata_resource(metadata_id: str) -> str:
    """
    Lấy metadata dưới dạng resource.
    
    Args:
        metadata_id: ID của metadata
    
    Returns:
        String representation của metadata
    """
    logger.info(f"Đang truy cập metadata resource với ID: {metadata_id}")
    data = load_metadata()
    metadata = data.get(metadata_id)
    
    if not metadata:
        logger.warning(f"Không tìm thấy metadata resource với ID: {metadata_id}")
        return f"Không tìm thấy metadata với ID {metadata_id}"
    
    logger.info(f"Đã tìm thấy metadata resource cho file: {metadata.get('filename')}")
    return json.dumps(metadata, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Chạy MCP server
    logger.info("=== MCP Metadata Server ===")
    logger.info(f"Dữ liệu được lưu tại: {DATA_DIR}")
    logger.info(f"File metadata: {METADATA_FILE}")
    logger.info("Đang khởi động server...")
    
    try:
        # Chạy MCP server với stdio transport
        mcp.run(transport='stdio')
    except KeyboardInterrupt:
        logger.info("Server đã dừng bởi người dùng")
    except Exception as e:
        logger.error(f"Server gặp lỗi: {e}")
        raise
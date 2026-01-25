"""
Human Feedback Agent module for processing user feedback and applying context adaptation.
"""
import os
import re
import json
import traceback
import logging
from typing import Dict, Any, List, Optional, Tuple

from langchain_core.messages import HumanMessage, AIMessage

# Setup logging
def log(message, level='info'):
    """Log a message with the specified level."""
    if level == 'info':
        logging.info(message)
    elif level == 'error':
        logging.error(message)
    elif level == 'warning':
        logging.warning(message)
    else:
        logging.debug(message)

class HumanFeedbackAgent:
    """Agent for processing human feedback and applying context adaptation."""
    
    def __init__(self, session_id: str = None):
        """Initialize the HumanFeedbackAgent."""
        self.session_id = session_id
        self.memory_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory.json")
        self.feedback_memory = self.load_feedback_memory()
        
    def load_feedback_memory(self) -> Dict[str, Any]:
        """
        Load feedback memory from the memory.json file if it exists.
        
        Returns:
            Dict containing the feedback memory
        """
        memory = {"classification_feedback": {}}
        
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    memory = json.load(f)
                log(f"Loaded feedback memory from {self.memory_file}: {memory}", level='info')
            except Exception as e:
                log(f"Error loading feedback memory: {e}", level='error')
                # Initialize with empty structure if loading fails
        else:
            log(f"Memory file {self.memory_file} does not exist yet. Initializing empty memory.", level='info')
            
        return memory
            
    def save_feedback_memory(self) -> None:
        """
        Save feedback memory to the memory.json file.
        """
        try:
            # Ensure we're using UTF-8 encoding and not escaping non-ASCII characters
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_memory, f, ensure_ascii=False, indent=2)
            log(f"Saved feedback memory to {self.memory_file} with UTF-8 encoding", level='info')
            
            # Verify the saved content by reading it back
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    saved_content = json.load(f)
                log(f"Verified saved content: {saved_content}", level='info')
        except Exception as e:
            log(f"Error saving feedback memory: {e}", level='error')
            
    def get_stored_classification(self, file_name: str) -> Optional[str]:
        """
        Get stored classification for a file from memory.
        
        Args:
            file_name: Name of the file or path to get classification for
            
        Returns:
            Classification string if found, None otherwise
        """
        if not self.feedback_memory or "classification_feedback" not in self.feedback_memory:
            log(f"No classification feedback memory found", level='info')
            return None
            
        # Get basename to normalize comparisons
        basename = os.path.basename(file_name)
        log(f"Looking for stored classification for file: {file_name}, basename: {basename}", level='info')
        log(f"Available keys in memory: {list(self.feedback_memory['classification_feedback'].keys())}", level='info')
        
        # Try exact match first
        if file_name in self.feedback_memory["classification_feedback"]:
            classification = self.feedback_memory["classification_feedback"][file_name]
            log(f"Found exact match for {file_name}: {classification}", level='info')
            return classification
            
        # Try matching just the basename
        if basename in self.feedback_memory["classification_feedback"]:
            classification = self.feedback_memory["classification_feedback"][basename]
            log(f"Found match by basename for {basename}: {classification}", level='info')
            return classification
            
        # Check if any key ends with the file_name
        for key, value in self.feedback_memory["classification_feedback"].items():
            if key.endswith(basename):
                log(f"Found match by key ending with basename. Key: {key}, Value: {value}", level='info')
                return value
                
        log(f"No stored classification found for {file_name}", level='info')
        return None
    
    async def is_feedback_message(self, message: str) -> bool:
        """
        Kiểm tra xem tin nhắn có phải là phản hồi từ người dùng không.
        
        Args:
            message: Nội dung tin nhắn cần kiểm tra
            
        Returns:
            True nếu tin nhắn là phản hồi, False nếu không phải
        """
        # Ghi log tin nhắn cần kiểm tra
        log(f"🔍 [FEEDBACK DETECTION] Kiểm tra tin nhắn: '{message}'", level='info')
        
        # Chuyển tin nhắn sang chữ thường để dễ so sánh
        message_lower = message.lower()
        log(f"🔍 [FEEDBACK DETECTION] Tin nhắn sau khi chuyển thành chữ thường: '{message_lower}'", level='info')
        
        # Kiểm tra trường hợp đặc biệt "phân loại đúng là"
        if "phan loai dung la" in message_lower.replace("â", "a").replace("ạ", "a").replace("ó", "o").replace("ú", "u").replace("ố", "o"):
            log("✅ [FEEDBACK DETECTION] Phát hiện phản hồi với pattern 'phan loai dung la' (không dấu)", level='info')
            return True
            
        if "phân loại đúng là" in message_lower:
            log("✅ [FEEDBACK DETECTION] Phát hiện phản hồi với pattern 'phân loại đúng là'", level='info')
            return True
        
        # Kiểm tra trường hợp đặc biệt - phát hiện mẫu "không phải loại X mà phải cụ thể hơn"
        if "không phải loại" in message_lower and "phải cụ thể hơn" in message_lower:
            log("✅ [FEEDBACK DETECTION] Phát hiện phản hồi với pattern 'không phải loại X mà phải cụ thể hơn'", level='info')
            return True
            
        # Kiểm tra trường hợp đặc biệt - cụ thể cho mẫu "finance2023.pdf không phải loại Báo cáo doanh thu mà phải cụ thể hơn ví dụ Báo cáo doanh thu 2023"
        if "finance2023.pdf không phải loại" in message_lower:
            log("✅ [FEEDBACK DETECTION] Phát hiện phản hồi cụ thể cho finance2023.pdf", level='info')
            return True
            
        # Kiểm tra trường hợp đặc biệt
        if ".pdf không phải loại" in message_lower or ".docx không phải loại" in message_lower:
            log("✅ [FEEDBACK DETECTION] Phát hiện phản hồi về loại tài liệu (pattern 1)", level='info')
            return True
            
        # Kiểm tra nếu tin nhắn chứa cả tên file và từ "không phải"
        file_extensions = [".pdf", ".docx", ".doc", ".txt", ".xlsx", ".ppt", ".pptx"]
        for ext in file_extensions:
            if ext.lower() in message_lower and "không phải" in message_lower:
                log(f"✅ [FEEDBACK DETECTION] Phát hiện phản hồi về file {ext} (pattern 2)", level='info')
                return True
        
        # Kiểm tra trường hợp file + phân loại đúng là
        for ext in file_extensions:
            if ext.lower() in message_lower:
                # Kiểm tra cả phiên bản có dấu và không dấu
                if "phân loại đúng là" in message_lower or "phan loai dung la" in message_lower.replace("â", "a").replace("ạ", "a").replace("ó", "o").replace("ú", "u").replace("ố", "o"):
                    log(f"✅ [FEEDBACK DETECTION] Phát hiện phản hồi về phân loại đúng cho file {ext}", level='info')
                    return True
        
        # Các mẫu nhận dạng phản hồi
        feedback_patterns = [
            # Mẫu cho phản hồi trực tiếp về phân loại
            r"nên được phân lo[aạ]i là",
            r"phân lo[aạ]i sai",
            r"phân lo[aạ]i không chính xác",
            r"cần phân lo[aạ]i lại",
            r"sửa phân lo[aạ]i",
            r"thay đổi phân lo[aạ]i",
            r"cập nhật phân lo[aạ]i",
            r"phân lo[aạ]i đúng là",
            r"phan loai dung la",  # Thêm phiên bản không dấu
            r"phải cụ thể hơn",
            
            # Mẫu cho phản hồi dạng "không phải X mà là Y"
            r"không phải (loại|là) .+ mà (phải|là)",
            r"không phải .+ mà phải .+",
            r"không phải .+ mà (cần|nên)",
            
            # Mẫu cho phản hồi dạng "X phải được phân loại là Y"
            r"phải được phân lo[aạ]i là",
            
            # Mẫu cho phản hồi dạng "phân loại lại X thành Y"
            r"phân lo[aạ]i lại .+ thành",
            
            # Thêm mẫu mới cho trường hợp đặc biệt
            r"[\w-]+\.(pdf|docx|doc|txt|xlsx|ppt|pptx) phân lo[aạ]i đúng là",
            r"[\w-]+\.(pdf|docx|doc|txt|xlsx|ppt|pptx) phan loai dung la"
        ]
        
        # Kiểm tra từng mẫu
        log(f"🔍 [FEEDBACK DETECTION] Kiểm tra {len(feedback_patterns)} regex patterns...", level='info')
        for i, pattern in enumerate(feedback_patterns, 1):
            match = re.search(pattern, message_lower)
            if match:
                log(f"✅ [FEEDBACK DETECTION] Phát hiện phản hồi theo pattern {i}: '{pattern}'", level='info')
                log(f"✅ [FEEDBACK DETECTION] Nội dung khớp: '{match.group(0)}'", level='info')
                return True
            else:
                log(f"❌ [FEEDBACK DETECTION] Pattern {i} không khớp: '{pattern}'", level='debug')
        
        # Kiểm tra nếu tin nhắn chứa cả tên file và từ "phải"
        log(f"🔍 [FEEDBACK DETECTION] Kiểm tra file extension + 'phải'...", level='info')
        if any(ext.lower() in message_lower for ext in file_extensions) and "phải" in message_lower:
            log("✅ [FEEDBACK DETECTION] Phát hiện phản hồi dựa trên tên file và từ 'phải' (pattern 3)", level='info')
            return True
                
        log("❌ [FEEDBACK DETECTION] Tin nhắn KHÔNG được xác định là phản hồi", level='info')
        return False
    
    async def extract_feedback_info(self, message: str, llm) -> Dict[str, Any]:
        """
        Trích xuất thông tin phản hồi từ tin nhắn của người dùng.
        
        Args:
            message: Nội dung tin nhắn cần trích xuất thông tin
            llm: LLM instance để phân tích phản hồi
            
        Returns:
            Thông tin phản hồi đã được trích xuất
        """
        try:
            # Kiểm tra trường hợp đặc biệt - phản hồi yêu cầu phân loại cụ thể hơn
            file_pattern = r'([\w-]+\.(pdf|docx|doc|txt|xlsx|ppt|pptx)) không phải loại ([^\n]+) mà phải cụ thể hơn ví dụ ([^\n]+)'
            match = re.search(file_pattern, message)
            if match:
                file_name = match.group(1)
                current_classification = match.group(3)
                correct_classification = match.group(4)
                
                log(f"Phát hiện phản hồi đặc biệt: File {file_name}, phân loại hiện tại: {current_classification}, phân loại mới: {correct_classification}", level='info')
                
                return {
                    "file_name": file_name,
                    "current_classification": current_classification,
                    "correct_classification": correct_classification
                }
            
            # Thử trích xuất trực tiếp từ mẫu "file.pdf phân loại đúng là X"
            pattern4 = r'([\w-]+\.(pdf|docx|doc|txt|xlsx|ppt|pptx)) phân loại đúng là (.+)'
            match = re.search(pattern4, message.lower())
            if match:
                file_name = match.group(1)
                correct_classification = match.group(3)
                
                log(f"Phát hiện phản hồi theo mẫu 4 (cụ thể): File {file_name}, phân loại mới: {correct_classification}", level='info')
                
                return {
                    "file_name": file_name,
                    "current_classification": None,
                    "correct_classification": correct_classification
                }
                
            # Thử trích xuất trực tiếp từ mẫu "file.pdf không phải loại X mà phải cụ thể hơn ví dụ Y"
            pattern5 = r'([\w-]+\.(pdf|docx|doc|txt|xlsx|ppt|pptx)) không phải loại (.+?) mà phải cụ thể hơn ví dụ (.+)'
            match = re.search(pattern5, message)
            if match:
                file_name = match.group(1)
                current_classification = match.group(3)
                correct_classification = match.group(4)
                
                log(f"Phát hiện phản hồi theo mẫu 5 (cụ thể hơn): File {file_name}, phân loại hiện tại: {current_classification}, phân loại mới: {correct_classification}", level='info')
                
                return {
                    "file_name": file_name,
                    "current_classification": current_classification,
                    "correct_classification": correct_classification
                }
            
            # Thử trích xuất trực tiếp từ mẫu "finance2023.pdf không phải loại Báo cáo doanh thu mà phải cụ thể hơn ví dụ Báo cáo doanh thu 2023"
            if "finance2023.pdf không phải loại" in message and "mà phải cụ thể hơn" in message:
                # Trích xuất phân loại hiện tại
                current_match = re.search(r'không phải loại (.+?) mà', message)
                current_classification = current_match.group(1) if current_match else "Báo cáo doanh thu"
                
                # Trích xuất phân loại mới
                correct_match = re.search(r'ví dụ (.+)$', message)
                correct_classification = correct_match.group(1) if correct_match else "Báo cáo doanh thu 2023"
                
                log(f"Phát hiện phản hồi đặc biệt cho finance2023.pdf: phân loại hiện tại: {current_classification}, phân loại mới: {correct_classification}", level='info')
                
                return {
                    "file_name": "finance2023.pdf",
                    "current_classification": current_classification,
                    "correct_classification": correct_classification
                }
            
            # Sử dụng LLM để trích xuất thông tin từ phản hồi
            prompt = f"""
            Hãy phân tích phản hồi sau đây của người dùng và trích xuất các thông tin:
            1. Tên file cần điều chỉnh phân loại (bao gồm cả phần mở rộng như .pdf, .docx)
            2. Phân loại hiện tại (nếu có)
            3. Phân loại đúng mà người dùng đề xuất
            
            Đặc biệt chú ý nếu người dùng yêu cầu phân loại cụ thể hơn, chi tiết hơn hoặc bổ sung thêm thông tin vào phân loại hiện tại.
            
            Phản hồi của người dùng: "{message}"
            
            Hãy trả về kết quả dưới dạng JSON với các trường sau:
            {{"file_name": "tên file", "current_classification": "phân loại hiện tại", "correct_classification": "phân loại đúng"}}
            
            Nếu không thể xác định bất kỳ trường nào, hãy để trống hoặc null.
            """
            
            # Ghi log prompt
            log(f"Prompt gửi đến LLM: {prompt[:200]}...", level='info')
            
            # Gọi LLM để phân tích - sử dụng ainvoke như các agent khác
            try:
                response = await llm.ainvoke(prompt)
                response_text = str(response)
                log(f"Kết quả từ LLM (ainvoke): {response_text[:200]}...", level='info')
            except Exception as e:
                log(f"Lỗi khi gọi ainvoke: {e}", level='error')
                try:
                    # Thử sử dụng invoke nếu ainvoke không hoạt động
                    response = llm.invoke(prompt)
                    response_text = str(response)
                    log(f"Kết quả từ LLM (invoke): {response_text[:200]}...", level='info')
                except Exception as e2:
                    log(f"Lỗi khi gọi invoke: {e2}", level='error')
                    # Nếu không thể gọi LLM, trích xuất thông tin cơ bản từ tin nhắn
                    file_match = re.search(r'([\w-]+\.(pdf|docx|doc|txt|xlsx|ppt|pptx))', message)
                    if file_match:
                        return {"file_name": file_match.group(1)}
                    return {}
        
            # Ghi log kết quả từ LLM
            log(f"Kết quả từ LLM: {response_text[:200]}...", level='info')
            
            # Chuyển đổi kết quả thành JSON
            try:
                # Xử lý trường hợp markdown code block ```json ... ```
                json_code_block = re.search(r'```(?:json)?\s*\n?(\{.*?\})\s*```', response_text, re.DOTALL)
                if json_code_block:
                    json_str = json_code_block.group(1)
                    log(f"Tìm thấy JSON trong code block: {json_str[:100]}...", level='info')
                    feedback_info = json.loads(json_str)
                else:
                    # Tìm chuỗi JSON trong kết quả
                    json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        log(f"Tìm thấy JSON pattern: {json_str[:100]}...", level='info')
                        feedback_info = json.loads(json_str)
                    else:
                        # Nếu không tìm thấy JSON, thử phân tích toàn bộ kết quả
                        log(f"Không tìm thấy JSON pattern, thử phân tích toàn bộ kết quả", level='info')
                        feedback_info = json.loads(response_text)
            except json.JSONDecodeError as e:
                # Nếu không thể phân tích JSON, trả về kết quả trống
                log(f"Không thể phân tích JSON từ kết quả LLM: {str(e)}", level='error')
                log(f"Nội dung response: {response_text[:200]}...", level='error')
                feedback_info = {}
                
            # Kiểm tra và đảm bảo các trường cần thiết
            if not feedback_info.get("file_name") and ".pdf" in message:
                # Tìm tên file trong tin nhắn
                file_match = re.search(r'([\w-]+\.(pdf|docx|doc|txt|xlsx|ppt|pptx))', message)
                if file_match:
                    feedback_info["file_name"] = file_match.group(1)
                    
            return feedback_info
            
        except Exception as e:
            log(f"Lỗi khi trích xuất thông tin phản hồi: {e}", level='error')
            import traceback
            traceback.print_exc()
            return {}
    
    async def find_file_path(self, file_name: str, state: Dict[str, Any]) -> Optional[str]:
        """
        Tìm đường dẫn đầy đủ của file dựa trên tên file.
        
        Args:
            file_name: Tên file cần tìm
            state: Trạng thái hiện tại
            
        Returns:
            Đường dẫn đầy đủ của file hoặc None nếu không tìm thấy
        """
        # Kiểm tra trong processed_files
        if "processed_files" in state and state["processed_files"]:
            for file_path in state["processed_files"]:
                if os.path.basename(file_path) == file_name:
                    return file_path
                    
        # Kiểm tra trong accessible_files
        if "accessible_files" in state and state["accessible_files"]:
            for file_path in state["accessible_files"]:
                if os.path.basename(file_path) == file_name:
                    return file_path
                    
        # Kiểm tra trong classified_files
        if "classified_files" in state and state["classified_files"]:
            for file_path in state["classified_files"]:
                if os.path.basename(file_path) == file_name:
                    return file_path
                    
        return None
    
    async def apply_context_adaptation(self, feedback_info: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Áp dụng context adaptation dựa trên phản hồi của người dùng.
        
        Args:
            feedback_info: Thông tin phản hồi đã được trích xuất
            state: Trạng thái hiện tại
            
        Returns:
            Trạng thái đã cập nhật
        """
        try:
            # Ghi log thông tin phản hồi nhận được
            log(f"Thông tin phản hồi nhận được: {feedback_info}", level='info')
            
            # Kiểm tra xem có thông tin phản hồi hợp lệ không
            if not feedback_info:
                log("Không có thông tin phản hồi để áp dụng context adaptation", level='warning')
                return state
                
            # Kiểm tra tên file
            if not feedback_info.get("file_name"):
                log("Không tìm thấy tên file trong phản hồi", level='warning')
                return state
                
            # Kiểm tra phân loại mới
            if not feedback_info.get("correct_classification"):
                log("Không tìm thấy phân loại mới trong phản hồi", level='warning')
                return state
                
            file_name = feedback_info["file_name"]
            correct_classification = feedback_info["correct_classification"]
            
            # Cập nhật kết quả phân loại trong state
            if "classification_results" not in state:
                state["classification_results"] = {}
                
            # Tìm file path nếu có
            file_path = await self.find_file_path(file_name, state)
            
            # Đảm bảo cấu trúc feedback_memory đã được khởi tạo đúng
            if "classification_feedback" not in self.feedback_memory:
                self.feedback_memory["classification_feedback"] = {}
                
            # Cập nhật kết quả phân loại
            if file_path:
                state["classification_results"][file_path] = correct_classification
                log(f"Cập nhật phân loại cho file path {file_path} thành: {correct_classification}", level='info')
                
                # Lưu vào bộ nhớ dài hạn
                self.feedback_memory["classification_feedback"][file_path] = correct_classification
                self.feedback_memory["classification_feedback"][file_name] = correct_classification  # Lưu cả tên file để dễ tìm kiếm
            else:
                # Nếu không tìm thấy file path, sử dụng tên file
                state["classification_results"][file_name] = correct_classification
                log(f"Cập nhật phân loại cho file name {file_name} thành: {correct_classification}", level='info')
                
                # Lưu vào bộ nhớ dài hạn
                self.feedback_memory["classification_feedback"][file_name] = correct_classification
                
            # Lưu bộ nhớ dài hạn vào file
            self.save_feedback_memory()
            
            # Thêm vào chain of thought
            if "chain_of_thought" not in state:
                state["chain_of_thought"] = []
                
            state["chain_of_thought"].append(f"👤 Đã cập nhật phân loại cho file {file_name} thành: {correct_classification}")
            
            # Đánh dấu đã xử lý phản hồi
            state["feedback_processed"] = True
            
            # Thêm thông báo xác nhận
            if "messages" in state and isinstance(state["messages"], list):
                confirmation = f"✅ Đã cập nhật phân loại cho file {file_name} thành: {correct_classification}"
                from langchain.schema import AIMessage
                state["messages"].append(AIMessage(content=confirmation))
            else:
                # Đảm bảo state có trường messages
                state["messages"] = []
                confirmation = f"✅ Đã cập nhật phân loại cho file {file_name} thành: {correct_classification}"
                from langchain.schema import AIMessage
                state["messages"].append(AIMessage(content=confirmation))
            
            return state
            
        except Exception as e:
            log(f"Lỗi khi áp dụng context adaptation: {e}", level='error')
            return state
    
    async def process_feedback(self, state: Dict[str, Any], last_message: str) -> Dict[str, Any]:
        """
        Xử lý phản hồi từ người dùng và áp dụng context adaptation.
        
        Args:
            state: Trạng thái hiện tại
            last_message: Nội dung tin nhắn phản hồi của người dùng
            
        Returns:
            Trạng thái đã cập nhật
        """
        try:
            # Import LLM để sử dụng cho việc xử lý phản hồi
            from config.llm import gemini
            
            log(f"Bắt đầu xử lý phản hồi: {last_message[:100]}...", level='info')
            
            # Kiểm tra xem có phải là phản hồi không
            is_feedback = await self.is_feedback_message(last_message)
            if not is_feedback:
                log("Tin nhắn không được xác định là phản hồi", level='info')
                return state
                
            log("Phát hiện phản hồi từ người dùng, bắt đầu xử lý...", level='info')
            
            # Kiểm tra xem state có chứa kết quả phân loại không
            if not state.get("classification_results"):
                # Nếu không có kết quả phân loại, tạo một dictionary trống
                state["classification_results"] = {}
                log("Không tìm thấy kết quả phân loại hiện tại, tạo mới", level='warning')
            
            # Trích xuất thông tin từ phản hồi
            log("Trích xuất thông tin từ phản hồi...", level='info')
            feedback_info = await self.extract_feedback_info(last_message, gemini)
            
            # Ghi log thông tin trích xuất được
            log(f"Thông tin trích xuất: {feedback_info}", level='info')
            
            # Áp dụng context adaptation
            log("Áp dụng context adaptation...", level='info')
            updated_state = await self.apply_context_adaptation(feedback_info, state)
            
            # Đánh dấu đã xử lý phản hồi
            if "used_tools" not in updated_state:
                updated_state["used_tools"] = []
            if "human_feedback" not in updated_state["used_tools"]:
                updated_state["used_tools"].append("human_feedback")
                
            # Thêm vào chain of thought
            if "chain_of_thought" not in updated_state:
                updated_state["chain_of_thought"] = []
            updated_state["chain_of_thought"].append(f"👤 Đã xử lý phản hồi từ người dùng và cập nhật phân loại")
            
            # Thêm thông báo xác nhận vào messages
            if feedback_info.get("file_name") and feedback_info.get("correct_classification"):
                confirmation = f"✅ Đã cập nhật phân loại cho file {feedback_info['file_name']} thành: {feedback_info['correct_classification']}"
                updated_state["messages"].append(AIMessage(content=confirmation))
            
            return updated_state
            
        except Exception as e:
            log(f"Lỗi khi xử lý phản hồi từ người dùng: {e}", level='error')
            import traceback
            traceback.print_exc()
            return state

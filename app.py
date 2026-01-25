import gradio as gr
import asyncio
import time
import nest_asyncio
import platform
import os
import sys
import traceback
import threading
from typing import Optional, List, Tuple, Dict
import uuid
import logging
from contextlib import asynccontextmanager
import atexit
import gc
import warnings
import hashlib
import json
from dotenv import load_dotenv
load_dotenv()

# Import security configuration
from config.security import security_config

# CRITICAL: Fix for Python 3.13 compatibility
import inspect

def safe_cleandoc(doc):
    """Safe implementation of inspect.cleandoc that handles all edge cases"""
    if doc is None:
        return ''
    
    if isinstance(doc, list):
        if not doc:
            return ''
        try:
            doc = '\n'.join(str(item) for item in doc)
        except Exception:
            return ''
    
    if isinstance(doc, dict):
        try:
            doc = str(doc)
        except Exception:
            return ''
    
    if not isinstance(doc, str):
        try:
            doc = str(doc)
        except Exception:
            return ''
    
    if not doc or not doc.strip():
        return ''
        
    try:
        if not hasattr(doc, 'expandtabs'):
            doc = str(doc)
        
        lines = doc.expandtabs().splitlines()
        if not lines:
            return ''
        
        indent = float('inf')
        for line in lines[1:]:
            stripped = line.lstrip()
            if stripped:
                indent = min(indent, len(line) - len(stripped))
        
        trimmed = [lines[0].strip()]
        if indent != float('inf') and indent > 0:
            for line in lines[1:]:
                if len(line) >= indent:
                    trimmed.append(line[indent:].rstrip())
                else:
                    trimmed.append(line.rstrip())
        else:
            for line in lines[1:]:
                trimmed.append(line.rstrip())
        
        while trimmed and not trimmed[-1]:
            trimmed.pop()
        
        return '\n'.join(trimmed)
        
    except Exception:
        try:
            result = str(doc).strip()
            return result
        except Exception:
            return ''

# Force override the inspect.cleandoc function
inspect.cleandoc = safe_cleandoc

# Configure logging FIRST
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning, module="grpc")

# Authentication Configuration
USERS_FILE = "users.json"

# Get default admin credentials from secure config
DEFAULT_ADMIN = {
    "username": security_config.default_admin_username,
    "password": security_config.default_admin_password,
    "role": "admin"
}

class AuthManager:
    """Simple authentication manager"""
    
    def __init__(self):
        self.users = self.load_users()
        self.ensure_default_admin()
    
    def load_users(self):
        """Load users from file"""
        try:
            if os.path.exists(USERS_FILE):
                with open(USERS_FILE, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            return {}
    
    def save_users(self):
        """Save users to file"""
        try:
            with open(USERS_FILE, 'w') as f:
                json.dump(self.users, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Error saving users: {e}")
            return False
    
    def hash_password(self, password: str):
        """Hash password with salt"""
        salt = os.urandom(32)
        key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return salt.hex() + ':' + key.hex()
    
    def verify_password(self, password: str, hashed: str):
        """Verify password against hash"""
        try:
            salt_hex, key_hex = hashed.split(':')
            salt = bytes.fromhex(salt_hex)
            key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
            return key.hex() == key_hex
        except Exception:
            return False
    
    def ensure_default_admin(self):
        """Ensure default admin user exists"""
        if not self.users or "admin" not in self.users:
            self.users["admin"] = {
                "username": DEFAULT_ADMIN["username"],
                "password": self.hash_password(DEFAULT_ADMIN["password"]),
                "role": DEFAULT_ADMIN["role"],
                "created_at": time.time()
            }
            self.save_users()
            logger.info("Created default admin user")
    
    def authenticate(self, username: str, password: str):
        """Authenticate user"""
        if username in self.users:
            user = self.users[username]
            if self.verify_password(password, user["password"]):
                user["last_login"] = time.time()
                
                if "role" not in user:
                    user["role"] = "user"
                    logger.info(f"Assigned default role 'user' to {username}")
                    self.save_users()
                
                user_info = user.copy()
                return True, user_info
        return False, None

# Initialize authentication manager
auth_manager = AuthManager()

# Add root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# IMPORTANT: Lazy import to prevent crash during startup
# MultiAgentSystem will be imported only when needed
# from agents.multi_agents import MultiAgentSystem  # MOVED TO LAZY IMPORT

from utils.file_upload import FileUploadManager


# Event loop configuration
if platform.system() == 'Windows':
    try:
        if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        else:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception as e:
        print(f"Cannot set event loop policy: {e}")

nest_asyncio.apply()

def safe_str_conversion(data):
    """Safely convert any data to string"""
    if data is None:
        return ""
    
    if isinstance(data, str):
        return data
    
    if isinstance(data, list):
        try:
            if all(isinstance(item, str) for item in data):
                return '\n'.join(data)
            return '\n'.join(str(item) for item in data)
        except Exception:
            return str(data)
    
    if isinstance(data, dict):
        try:
            if 'content' in data:
                return safe_str_conversion(data['content'])
            return str(data)
        except Exception:
            return str(data)
    
    try:
        return str(data)
    except Exception:
        return "<Unable to convert to string>"

def safe_progress_update(progress, value, desc=""):
    """Safely update progress without causing errors"""
    try:
        if progress is not None and hasattr(progress, '__call__'):
            progress(value, desc=desc)
        return True
    except Exception as e:
        logger.warning(f"Progress update error: {e}")
        return False

class SingletonAsyncSystem:
    """Singleton pattern for managing async system"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.system = None
        self.loop = None
        self.loop_thread = None
        self.initialized = False
        self.initialization_error = None
        self._initialized = True
        
    def start_event_loop(self):
        """Start persistent event loop in background thread"""
        if self.loop and not self.loop.is_closed():
            return
            
        def run_loop():
            try:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                self.loop.run_forever()
            except Exception as e:
                logger.error(f"Event loop error: {e}")
            finally:
                if self.loop and not self.loop.is_closed():
                    self.loop.close()
        
        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()
        
        timeout = 10
        start_time = time.time()
        while (not self.loop or not self.loop.is_running()) and (time.time() - start_time) < timeout:
            time.sleep(0.1)
            
        if not self.loop or not self.loop.is_running():
            raise RuntimeError("Failed to start event loop")
    
    async def _initialize_system(self):
        """Internal method to initialize the system"""
        try:
            # Lazy import to prevent crash during app startup
            from agents.multi_agents import MultiAgentSystem
            
            if self.system is None:
                self.system = MultiAgentSystem()
            
            logger.info("Initializing multi-agent system with access control...")
            result = await self.system.initialize()
            
            if result:
                self.initialized = True
                self.initialization_error = None
                logger.info("System initialized successfully with role-based access control!")
                return True, None
            else:
                error_msg = "System initialization returned False"
                self.initialization_error = error_msg
                return False, error_msg
                
        except Exception as e:
            error_msg = f"System initialization error: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.initialization_error = error_msg
            return False, error_msg
    
    def initialize_system(self):
        """Initialize the system using persistent event loop"""
        if not self.loop or not self.loop.is_running():
            self.start_event_loop()
        
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._initialize_system(), 
                self.loop
            )
            success, error = future.result(timeout=150)
            return success, error
        except Exception as e:
            error_msg = f"Failed to initialize system: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return False, error_msg
    
    async def _process_message(self, message, request_id, user_role="user"):
        """Internal method to process message"""
        try:
            if not self.initialized:
                await self._initialize_system()
                
            if not self.initialized:
                return {"error": "System initialization failed"}
            
            if not self.system:
                return {"error": "System not available"}
            
            logger.info(f"Processing message with request_id: {request_id} and user_role: {user_role}")
            
            result = await self.system.run(message, session_id=request_id, user_role=user_role)
            
            chain_of_thought = []
            if hasattr(self.system, "chain_of_thought") and self.system.chain_of_thought:
                chain_of_thought = self.system.chain_of_thought
            elif "chain_of_thought" in result:
                chain_of_thought = result["chain_of_thought"]
            
            result["chain_of_thought"] = chain_of_thought
            
            if isinstance(result, dict):
                if 'content' in result:
                    try:
                        if 'chain_of_thought' not in result:
                            result['chain_of_thought'] = chain_of_thought
                        elif isinstance(result['chain_of_thought'], list):
                            pass
                        else:
                            result['chain_of_thought'] = [safe_str_conversion(result['chain_of_thought'])]
                    except Exception as cot_error:
                        logger.warning(f"Chain of thought processing error: {cot_error}")
                        result['chain_of_thought'] = ["Chain of thought processing error"]
                return result
            else:
                content = safe_str_conversion(result)
                return {"content": content, "chain_of_thought": ["Response processed successfully"]}
            
        except Exception as e:
            error_msg = f"Message processing error: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"error": error_msg}
    
    def process_message(self, message, request_id, user_role="user"):
        """Process message using persistent event loop"""
        if not self.loop or not self.loop.is_running():
            return {"error": "Event loop not running"}
        
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._process_message(message, request_id, user_role), 
                self.loop
            )
            result = future.result(timeout=200)
            return result
        except Exception as e:
            error_msg = f"Failed to process message: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"error": error_msg}
    
    def reset_system(self):
        """Reset the entire system"""
        try:
            self.system = None
            self.initialized = False
            self.initialization_error = None
            gc.collect()
            return "System reset successfully. Please reinitialize."
        except Exception as e:
            logger.error(f"Reset error: {e}")
            return f"Reset error: {str(e)}"
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.loop and self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
            if self.loop_thread and self.loop_thread.is_alive():
                self.loop_thread.join(timeout=5)
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

# Create singleton instance
async_system = SingletonAsyncSystem()

class MultiAgentChatbot:
    """Simplified chatbot class using singleton async system"""
    
    def __init__(self):
        self.chat_history = []
        self.chain_of_thought = []
        
    def initialize_system(self):
        """Initialize the system"""
        return async_system.initialize_system()
    
    def get_system_status(self):
        """Get current system status"""
        if not async_system.initialized:
            if async_system.initialization_error:
                return f"❌ System Error: {async_system.initialization_error}", "error"
            else:
                return "⚪ System not initialized", "warning"
        else:
            return "✅ System Ready", "success"
    
    def reset_system(self):
        """Reset the system"""
        message = async_system.reset_system()
        self.chat_history = []
        self.chain_of_thought = []
        return message, [], [], "⚪ System reset. Please reinitialize."
    
    def process_message(self, message: str, history: List, progress=None, current_user=None):
        """Process a user message and return the response"""
        try:
            if not async_system.initialized:
                return history, self.chain_of_thought, "❌ System not initialized. Please initialize first."
            
            if not message or not message.strip():
                return history, self.chain_of_thought, "Please enter a message."
            
            safe_progress_update(progress, 0.1, "Starting request processing...")
            
            user_message = {"role": "user", "content": message}
            new_history = history + [user_message] if history is not None else [user_message]
            
            self.chain_of_thought = []
            request_id = f"request_{int(time.time())}_{str(uuid.uuid4())[:8]}"
            
            user_role = "user"
            if current_user and isinstance(current_user, dict) and "role" in current_user:
                user_role = current_user["role"]
                logger.info(f"Using authenticated user role: {user_role}")
            
            safe_progress_update(progress, 0.3, "Processing with multi-agent system...")
            
            logger.info(f"Processing message with request_id: {request_id} and user_role: {user_role}")
            result = async_system.process_message(message, request_id, user_role=user_role)
            
            safe_progress_update(progress, 0.8, "Processing response...")
            
            logger.info(f"Received result type: {type(result)}")
            
            if result is None:
                error_msg = "No response received from system"
                logger.error(error_msg)
                assistant_message = {"role": "assistant", "content": f"❌ {error_msg}"}
                new_history.append(assistant_message)
                self.chain_of_thought.append(f"❌ Error: {error_msg}")
                return new_history, self.chain_of_thought, ""
            
            if isinstance(result, dict) and "error" in result:
                error_msg = result.get("error", "Unknown error occurred")
                logger.error(f"Error in result: {error_msg}")
                assistant_message = {"role": "assistant", "content": f"❌ {error_msg}"}
                new_history.append(assistant_message)
                self.chain_of_thought.append(f"❌ Error: {error_msg}")
                return new_history, self.chain_of_thought, ""
            
            response_content = safe_str_conversion(result.get("content", "Sorry, I couldn't process your request."))
            
            if isinstance(result, dict) and result.get("used_tools") and "rag" in result.get("used_tools"):
                response_content = f"🗂️ {response_content}"
            
            try:
                if "chain_of_thought" in result and result["chain_of_thought"]:
                    if isinstance(result["chain_of_thought"], list):
                        self.chain_of_thought = [safe_str_conversion(step) for step in result["chain_of_thought"]]
                    else:
                        self.chain_of_thought = [safe_str_conversion(result["chain_of_thought"])]
                else:
                    self.chain_of_thought = ["Response processed successfully"]
            except Exception as cot_error:
                logger.warning(f"Chain of thought processing error: {cot_error}")
                self.chain_of_thought = ["Chain of thought processing completed with minor issues"]
            
            assistant_message = {"role": "assistant", "content": response_content}
            new_history.append(assistant_message)
            
            safe_progress_update(progress, 1.0, "Complete!")
            
            logger.info("Message processed successfully")
            return new_history, self.chain_of_thought, ""
            
        except Exception as e:
            error_message = f"❌ Processing error: {str(e)}"
            logger.error(f"Exception in process_message: {e}")
            logger.error(traceback.format_exc())
            
            try:
                assistant_message = {"role": "assistant", "content": error_message}
                if history is not None:
                    new_history = history + [{"role": "user", "content": message}, assistant_message]
                else:
                    new_history = [{"role": "user", "content": message}, assistant_message]
            except Exception as history_error:
                logger.error(f"History handling error: {history_error}")
                new_history = [{"role": "assistant", "content": error_message}]
            
            self.chain_of_thought = [f"❌ Error: {str(e)}"]
            
            return new_history, self.chain_of_thought, ""

# Create chatbot instance (lazy initialization - system will be initialized on first use)
chatbot = MultiAgentChatbot()

# Create file upload manager instance
file_upload_manager = FileUploadManager(upload_dir=security_config.data_dir)

# Flag to track if we should initialize system in background
_system_initializing = False

def format_chain_of_thought(chain_of_thought):
    """Format chain of thought for display"""
    if not chain_of_thought:
        return "Waiting for processing..."
    
    formatted_steps = []
    for step in chain_of_thought:
        step_str = safe_str_conversion(step)
        formatted_steps.append(step_str)
    
    return "\n\n".join(formatted_steps)

# Register cleanup on exit
atexit.register(async_system.cleanup)

# Enhanced CSS with modern loading screen
css = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

/* Force remove all backgrounds first */
*, *::before, *::after {
    background-color: transparent !important;
    background-image: none !important;
}

/* Apply gradient to main containers */
body, html, .gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif !important;
    min-height: 100vh !important;
    background-attachment: fixed !important;
    color: #ffffff !important;
}

/* Remove all Gradio default backgrounds */
.gradio-container div,
.gradio-container .gr-column,
.gradio-container .gr-row,
.gradio-container .gr-group,
.gradio-container .gr-block,
.gradio-container .gr-box,
.gradio-container .gr-form,
.gradio-container .gr-panel,
.gradio-container .contain,
.gradio-container .flex,
.gradio-container .gap {
    background: transparent !important;
}

/* FORCE ALL TEXT TO BE WHITE */
.gradio-container *,
.gradio-container p,
.gradio-container span,
.gradio-container div,
.gradio-container label,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container h5,
.gradio-container h6 {
    color: #ffffff !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* App logo styling - simplified and centered */
.app-logo {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin: 2rem auto;
    padding: 1rem;
}

.logo-icon-simple {
    width: 120px;
    height: 120px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3rem;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    border: none;
    position: relative;
}

.logo-icon-simple::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
    animation: logoShine 4s infinite;
}

@keyframes logoShine {
    0% { left: -100%; }
    100% { left: 100%; }
}

/* Modern Loading Screen */
.loading-screen {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 80vh;
    text-align: center;
    animation: fadeIn 0.8s ease-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

.loading-logo {
    width: 140px;
    height: 140px;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 4rem;
    margin-bottom: 2rem;
    position: relative;
    animation: pulse 2s infinite;
    box-shadow: 0 15px 35px rgba(59, 130, 246, 0.4);
}

@keyframes pulse {
    0%, 100% { transform: scale(1); box-shadow: 0 15px 35px rgba(59, 130, 246, 0.4); }
    50% { transform: scale(1.05); box-shadow: 0 20px 45px rgba(139, 92, 246, 0.6); }
}

.loading-title {
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, #ffffff, #e2e8f0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: none;
}

.loading-subtitle {
    font-size: 1.2rem;
    opacity: 0.8;
    margin-bottom: 3rem;
    font-weight: 500;
}

.loading-progress {
    width: 300px;
    height: 6px;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 2rem;
}

.loading-bar {
    height: 100%;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #3b82f6);
    background-size: 200% 100%;
    border-radius: 10px;
    animation: loading 2s ease-in-out infinite;
}

@keyframes loading {
    0% { width: 0%; background-position: 200% 0; }
    50% { width: 70%; background-position: 50% 0; }
    100% { width: 100%; background-position: 0% 0; }
}

.loading-steps {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    align-items: center;
}

.loading-step {
    display: flex;
    align-items: center;
    gap: 1rem;
    font-size: 1rem;
    opacity: 0.7;
    transition: all 0.3s ease;
}

.loading-step.active {
    opacity: 1;
    transform: translateX(10px);
}

.loading-step-icon {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
}

.loading-step.active .loading-step-icon {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    animation: bounce 0.6s ease-out;
}

@keyframes bounce {
    0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
    40% { transform: translateY(-10px); }
    60% { transform: translateY(-5px); }
}

/* Login form styling */
.login-form {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px) !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3) !important;
    max-width: 450px !important;
    margin: 2rem auto !important;
    animation: fadeInUp 0.6s ease-out !important;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Input styling with white text */
input[type="text"], input[type="password"], textarea {
    background: rgba(255, 255, 255, 0.15) !important;
    border: 2px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    padding: 16px 20px !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    backdrop-filter: blur(10px) !important;
}

input[type="text"]:focus, input[type="password"]:focus, textarea:focus {
    border-color: #3b82f6 !important;
    background: rgba(255, 255, 255, 0.2) !important;
    outline: none !important;
    box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.2) !important;
    transform: translateY(-2px) !important;
}

input::placeholder, textarea::placeholder {
    color: rgba(255, 255, 255, 0.7) !important;
    font-weight: 400 !important;
}

/* Button styling */
.gr-button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    padding: 14px 28px !important;
    font-size: 15px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4) !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
}

.gr-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 35px rgba(59, 130, 246, 0.5) !important;
}

.gr-button-secondary {
    background: rgba(255, 255, 255, 0.15) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    color: #ffffff !important;
    backdrop-filter: blur(10px) !important;
}

.gr-button-secondary:hover {
    background: rgba(255, 255, 255, 0.25) !important;
    transform: translateY(-1px) !important;
}

/* Glass panels with enhanced borders */
.chatbot, .gr-chatbot {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 2px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2) !important;
    color: #ffffff !important;
}

.control-panel {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px) !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2) !important;
}

.chain-display {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 2px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    color: #ffffff !important;
    backdrop-filter: blur(10px) !important;
}

/* Typography with consistent white colors */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3) !important;
}

p, span, div {
    color: #ffffff !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

label {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
}

/* Message styling with enhanced contrast */
.user-message {
    background: rgba(59, 130, 246, 0.9) !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    margin-bottom: 0.5rem !important;
    color: #ffffff !important;
    font-weight: 900 !important;
    border: 2px solid rgba(59, 130, 246, 0.6) !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2) !important;
}

.assistant-message {
    background: rgba(139, 92, 246, 0.9) !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    margin-bottom: 0.5rem !important;
    color: #ffffff !important;
    font-weight: 900 !important;
    border: 2px solid rgba(139, 92, 246, 0.6) !important;
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.2) !important;
}

/* Ensure text is always visible in messages */
.user-message p,
.assistant-message p,
.user-message div,
.assistant-message div {
    color: #ffffff !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
    margin: 0.25rem 0 !important;
    line-height: 1.5 !important;
}

/* Specific overrides for chat interface */
.gr-chatbot .message-content,
.gr-chatbot .user-message,
.gr-chatbot .assistant-message {
    color: #ffffff !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
}

/* Make sure all text in chat is white and readable */
.gr-chatbot,
.gr-chatbot *,
.gr-chatbot .message,
.gr-chatbot .message * {
    color: #ffffff !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
}

/* User info with enhanced styling */
.user-info {
    background: rgba(255, 255, 255, 0.05) !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 10px 20px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    backdrop-filter: blur(15px) !important;
    margin-bottom: 1rem !important;
    display: inline-block !important;
    float: right !important;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2) !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
}

/* Processing Pipeline Styling */
.processing-container {
    padding: 1rem !important;
    background: rgba(255, 255, 255, 0.05) !important;
    border: 2px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2) !important;
}

.steps-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

.step-item {
    padding: 0.8rem 1rem;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    display: flex;
    align-items: flex-start;
    color: #ffffff;
    font-weight: 500;
    border-left: 3px solid rgba(59, 130, 246, 0.6);
    transition: all 0.3s ease;
}

.step-item:hover {
    background: rgba(255, 255, 255, 0.15);
    transform: translateX(5px);
}

.step-icon {
    margin-right: 0.8rem;
    font-size: 1.2rem;
    opacity: 0.9;
}

.step-content {
    flex: 1;
    color: #ffffff;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
    line-height: 1.4;
}

/* Scrollbar with white theme */
::-webkit-scrollbar { 
    width: 8px; 
}

::-webkit-scrollbar-track { 
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb { 
    background: rgba(255, 255, 255, 0.3);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover { 
    background: rgba(255, 255, 255, 0.5);
}

/* Responsive design */
@media (max-width: 768px) {
    .login-form { 
        margin: 1rem !important; 
        padding: 2rem !important; 
    }
    
    .loading-logo {
        width: 100px;
        height: 100px;
        font-size: 3rem;
    }
    
    .loading-title {
        font-size: 2rem;
    }
    
    .loading-progress {
        width: 250px;
    }
}

/* Force override any remaining dark text */
.gradio-container .gr-textbox,
.gradio-container .gr-chatbot,
.gradio-container .gr-button,
.gradio-container textarea,
.gradio-container input {
    color: #ffffff !important;
}
"""

# Create Gradio interface with enhanced UX flow
# Note: In Gradio 6.0, theme and css moved to launch() method
with gr.Blocks(title="🤖 Multi-Agent AI Chatbot") as demo:
    # Shared state variables
    current_user = gr.State(None)
    
    # Login Section
    with gr.Group(visible=True) as login_section:
        # App Logo and Branding
        gr.HTML("""
        <div class="app-logo">
            <div class="logo-icon-simple">🤖</div>
        </div>
        """)
        
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 3rem;">
            <h1 style="font-size: 2.8rem; margin-bottom: 0.5rem; color: #ffffff; font-weight: 800; text-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);">
                AI Command Center
            </h1>
            <p style="font-size: 1.2rem; margin: 0; color: #ffffff; font-weight: 500; opacity: 0.9; text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);">
                Secure Access Required • Advanced Multi-Agent Intelligence System
            </p>
        </div>
        """)
        
        with gr.Column(elem_classes=["login-form"]):
            gr.HTML("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🔐</div>
                <h3 style="margin: 0; color: #ffffff; font-weight: 700; font-size: 20px; text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);">System Authentication</h3>
            </div>
            """)
            
            username = gr.Textbox(
                label="👤 Username",
                placeholder="Enter your access credentials",
                container=True,
                elem_classes=["login-input"]
            )
            password = gr.Textbox(
                label="🔑 Password",
                placeholder="Enter your secure password",
                type="password",
                container=True,
                elem_classes=["login-input"]
            )
            login_message = gr.Markdown("", visible=False)
            
            login_btn = gr.Button("🚀 Access System", elem_classes=["login-button"])
    
    # Loading Screen
    with gr.Group(visible=False) as loading_section:
        loading_display = gr.HTML("""
        <div class="loading-screen">
            <div class="loading-logo">🤖</div>
            <h1 class="loading-title">Initializing AI System</h1>
            <p class="loading-subtitle">Activating system and preparing agents...</p>
            
            <div class="loading-progress">
                <div class="loading-bar"></div>
            </div>
            
        </div>
        """)
    
    # Main Application
    with gr.Group(visible=False) as main_app:
        # App Logo and Header - Centered
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="display: flex; align-items: center; justify-content: center; gap: 1rem; margin-bottom: 1rem;">
                <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); border-radius: 15px; display: flex; align-items: center; justify-content: center; font-size: 2rem; box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);">
                    🤖
                </div>
                <h1 style="font-size: 2.2rem; margin: 0; color: #ffffff; font-weight: 800; text-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);">
                    AI Command Center
                </h1>
            </div>
            <p style="font-size: 1.1rem; margin: 0; color: #ffffff; opacity: 0.9; font-weight: 500; text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);">
                Next-Generation Multi-Agent Intelligence • Real-time Processing • Enhanced Security
            </p>
        </div>
        """)
        
        # User info display
        user_info = gr.HTML("", elem_classes=["user-info"])
        
        with gr.Row():
            # Main chat area
            with gr.Column(scale=3):
                chatbot_interface = gr.Chatbot(
                    label="🧠 Intelligence Interface",
                    height=500,
                    show_label=True,
                    elem_classes=["chatbot"]
                )
                
                # Input area with improved alignment
                with gr.Row():
                    msg = gr.Textbox(
                        label="💬 Your Message",
                        placeholder="Ask me anything... I'm powered by advanced AI agents 🚀",
                        scale=4,
                        lines=2,
                        max_lines=4
                    )
                    with gr.Column(scale=1):
                        send_btn = gr.Button("⚡ Send")
                        
                # Action buttons
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat")
                    logout_btn = gr.Button("🚪 Logout")
                
                # File Upload Section
                with gr.Accordion("📁 Upload Documents", open=False):
                    gr.Markdown("Upload documents (PDF, DOCX, PPTX, TXT, etc.) to enhance the AI's knowledge base")
                    file_upload = gr.File(
                        label="Select Files",
                        file_count="multiple",
                        file_types=[".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt", ".md", ".csv", ".xlsx", ".xls"]
                    )
                    upload_btn = gr.Button("📤 Upload Files")
                    upload_status = gr.Markdown("")
                    
            # AI Processing Pipeline Panel
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["control-panel"]):
                    gr.HTML("""
                    <div style="text-align: center; margin-bottom: 1.5rem;">
                        <h3 style="margin: 0; display: flex; align-items: center; justify-content: center; gap: 0.5rem; font-size: 1.3rem;">
                            🧠 AI Processing Pipeline
                        </h3>
                    </div>
                    """)
                    
                    # Processing visualization
                    chain_display = gr.HTML(
                        """
                        <div class="processing-container">
                            <div class="steps-list">
                                <div class="step-item">
                                    <span class="step-icon">🔮</span>
                                    <span class="step-content">Ready for Intelligence Processing</span>
                                </div>
                            </div>
                        </div>
                        """,
                        elem_classes=["processing-container"]
                    )
    
    # Enhanced event handlers
    def handle_login(username_input, password_input):
        """Handle login attempt and show loading screen"""
        if not username_input or not password_input:
            return (
                gr.update(value="⚠️ Please provide both username and password", visible=True),
                None,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update()
            )
        
        success, user = auth_manager.authenticate(username_input, password_input)
        
        if success:
            if "role" not in user:
                user["role"] = "user"
                logger.info(f"User {username_input} assigned default role: user")
            else:
                logger.info(f"User {username_input} accessed the system with role: {user['role']}")
            
            return (
                gr.update(value="", visible=False),
                user,
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update()
            )
        else:
            logger.warning(f"Access attempt failed for user {username_input}")
            return (
                gr.update(value="❌ Invalid credentials. Please verify your username and password.", visible=True),
                None,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update()
            )
    
    def initialize_system_and_show_chat(current_user):
        """Initialize system after login and show main chat interface"""
        try:
            # Perform actual system initialization
            success, error = chatbot.initialize_system()
            
            if success:
                # Generate user info HTML
                user_info_html = f"""
                <div style="display: flex; align-items: center; gap: 12px; justify-content: flex-end;">
                    <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #ffffff; font-weight: 700; font-size: 18px; box-shadow: 0 6px 15px rgba(59, 130, 246, 0.4); text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);">
                        {current_user['username'][0].upper()}
                    </div>
                    <div style="display: flex; flex-direction: column;">
                        <span style="color: #ffffff; font-weight: 600; font-size: 16px; text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);">{current_user['username']}</span>
                        <span style="color: #ffffff; font-size: 12px; opacity: 0.8; text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);">Role: {current_user['role']}</span>
                    </div>
                </div>
                """
                
                return (
                    gr.update(visible=False),  # Hide loading
                    gr.update(visible=True),   # Show main app
                    gr.update(value=user_info_html)  # Set user info
                )
            else:
                # If initialization fails, go back to login
                return (
                    gr.update(visible=False),  # Hide loading
                    gr.update(visible=False),  # Hide main app
                    gr.update(value=f"❌ System initialization failed: {error}")
                )
                
        except Exception as e:
            logger.error(f"System initialization error: {e}")
            return (
                gr.update(visible=False),  # Hide loading
                gr.update(visible=False),  # Hide main app
                gr.update(value=f"❌ Initialization error: {str(e)}")
            )
    
    def handle_logout(current_user):
        """Handle logout"""
        if current_user:
            username = current_user.get('username', 'Unknown')
            logger.info(f"User {username} logged out")
        
        return (
            None,
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
            []
        )
    
    def send_message(message, history, current_user=None):
        """Send message with enhanced UI feedback"""
        try:
            if not message or not message.strip():
                return history, format_chain_display([]), ""
            
            new_history, chain, empty_msg = chatbot.process_message(message, history, current_user=current_user)
            formatted_chain = format_chain_display(chain)
            return new_history, formatted_chain, empty_msg
        except Exception as e:
            logger.error(f"Send message error: {e}")
            error_msg = f"⚠️ **System Error**: {str(e)}"
            error_history = history + [{"role": "assistant", "content": error_msg}] if history else [{"role": "assistant", "content": error_msg}]
            return error_history, format_chain_display([f"❌ Error: {str(e)}"]), ""
    
    def format_chain_display(chain_of_thought):
        """Format chain of thought for modern display"""
        if not chain_of_thought:
            return """
            <div class="processing-container">
                <div class="steps-list">
                    <div class="step-item">
                        <span class="step-icon">🔮</span>
                        <span class="step-content">Ready for Intelligence Processing</span>
                    </div>
                </div>
            </div>
            """
        
        steps_html = []
        icons = ["🔍", "🧠", "⚡", "🎯", "✨", "🚀"]
        
        for i, step in enumerate(chain_of_thought):
            step_str = safe_str_conversion(step)
            icon = icons[i % len(icons)]
            steps_html.append(f"""
                <div class='step-item'>
                    <span class='step-icon'>{icon}</span>
                    <span class='step-content'>{step_str}</span>
                </div>
            """)
        
        return f"""
        <div class='processing-container'>
            <div class='steps-list'>
                {''.join(steps_html)}
            </div>
        </div>
        """
    
    def clear_history():
        return []
    
    def handle_file_upload(files):
        """Handle file upload"""
        if not files:
            return "⚠️ No files selected"
        
        try:
            results = []
            success_count = 0
            error_count = 0
            
            for file in files:
                if file is None:
                    continue
                    
                file_path = file.name if hasattr(file, 'name') else str(file)
                success, saved_path, message = file_upload_manager.save_uploaded_file(file_path)
                
                if success:
                    success_count += 1
                    results.append(f"✅ {message}")
                else:
                    error_count += 1
                    results.append(f"❌ {message}")
            
            # Get upload stats
            stats = file_upload_manager.get_upload_stats()
            
            summary = f"""
### Upload Summary
- **Successful**: {success_count} files
- **Failed**: {error_count} files
- **Total files in database**: {stats['total_files']}
- **Total size**: {stats['total_size_mb']} MB

### Details
{chr(10).join(results)}

💡 **Note**: After uploading files, the system will automatically index them for RAG search.
"""
            return summary
            
        except Exception as e:
            logger.error(f"File upload error: {e}")
            return f"❌ Upload error: {str(e)}"
    
    # Wire up events
    login_btn.click(
        handle_login,
        inputs=[username, password],
        outputs=[login_message, current_user, login_section, loading_section, main_app, user_info]
    ).then(
        # After successful login, automatically initialize system
        initialize_system_and_show_chat,
        inputs=[current_user],
        outputs=[loading_section, main_app, user_info]
    )
    
    logout_btn.click(
        handle_logout,
        inputs=[current_user],
        outputs=[current_user, login_section, loading_section, main_app, username, password, login_message, chatbot_interface]
    )
    
    # Message handling
    msg.submit(
        send_message,
        inputs=[msg, chatbot_interface, current_user],
        outputs=[chatbot_interface, chain_display, msg]
    ).then(
        lambda: "",
        outputs=[msg]
    )
    
    send_btn.click(
        send_message,
        inputs=[msg, chatbot_interface, current_user],
        outputs=[chatbot_interface, chain_display, msg]
    ).then(
        lambda: "",
        outputs=[msg]
    )
    
    clear_btn.click(
        clear_history,
        outputs=[chatbot_interface]
    ).then(
        lambda: format_chain_display([]),
        outputs=[chain_display]
    )
    
    # File upload handling
    upload_btn.click(
        handle_file_upload,
        inputs=[file_upload],
        outputs=[upload_status]
    )

if __name__ == "__main__":
    import threading
    
    try:
        # Use PORT from environment (for cloud platforms like Render, Railway)
        # or fall back to configured port
        port = int(os.getenv("PORT", security_config.server_port))
        
        print(f"🚀 Starting Gradio app on 0.0.0.0:{port}")
        print("⚡ Quick start mode: Agents will initialize in background")
        
        # Initialize agents in background AFTER Gradio starts
        def init_agents_background():
            """Initialize agents after Gradio has bound to port"""
            import time
            time.sleep(3)  # Wait for Gradio to bind port
            print("🔧 Initializing AI agents in background...")
            try:
                # This will trigger lazy initialization on first use
                chatbot.initialize()
                print("✅ AI agents initialized successfully")
            except Exception as e:
                print(f"⚠️ Agent initialization warning: {e}")
                print("📝 Agents will initialize on first user request")
        
        # Start background initialization
        init_thread = threading.Thread(target=init_agents_background, daemon=True)
        init_thread.start()
        
        # Enable queue for production deployment
        demo.queue(max_size=10)  # Limit queue size to reduce memory
        
        print("🌐 Launching Gradio server...")
        
        # Launch IMMEDIATELY to bind port (critical for Render)
        demo.launch(
            server_name="0.0.0.0",  # Must be 0.0.0.0 for Render to detect
            server_port=port,
            share=False,
            # Gradio 6.0: theme and css moved here from Blocks()
            theme=gr.themes.Soft(),
            css=css
        )
        
        print(f"✅ Gradio app is running on port {port}")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR during Gradio launch: {e}")
        print(f"❌ Traceback: {traceback.format_exc()}")
        import sys
        sys.exit(1)
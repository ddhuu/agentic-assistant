"""
Cloud-optimized LLM configuration (without Ollama)
Use this for Render, Railway, Heroku, etc.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from config.security import security_config

# Only use Gemini for cloud deployment
gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    google_api_key=security_config.google_api_key
)

# Use Gemini for all LLM tasks
ollama_chat_model = gemini  # Fallback to Gemini
ollama_model = gemini  # Fallback to Gemini

DATA_DIR = security_config.data_dir

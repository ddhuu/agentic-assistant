from langchain_google_genai import ChatGoogleGenerativeAI
from config.security import security_config

# Try to import Ollama (optional - only for local development)
try:
    from langchain_ollama.chat_models import ChatOllama
    from langchain_ollama.llms import OllamaLLM
    ollama_chat_model = ChatOllama(model="mistral")
    ollama_model = OllamaLLM(model="mistral")
    print(" Ollama models loaded successfully")
except ImportError:
    print(" langchain_ollama not available - using Gemini only (expected on cloud deployment)")
    ollama_chat_model = None
    ollama_model = None

# Gemini Model (primary model for cloud deployment)
gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    google_api_key=security_config.google_api_key
)

DATA_DIR = security_config.data_dir
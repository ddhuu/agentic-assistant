import os
from langchain_google_genai import ChatGoogleGenerativeAI
from config.security import security_config

# Opik LLM Tracing - auto-traces all LangChain LLM calls
opik_tracer = None
try:
    import opik
    from opik.integrations.langchain import OpikTracer
    
    opik.configure(
        api_key=os.getenv("OPIK_API_KEY"),
        workspace=os.getenv("OPIK_WORKSPACE", "default"),
    )
    
    opik_tracer = OpikTracer(
        project_name=os.getenv("OPIK_PROJECT_NAME", "agentic-assistant"),
        tags=["agentic-assistant", "gemini-2.0-flash"],
    )
    print("✅ Opik tracer initialized successfully")
except ImportError:
    print("⚠️ opik not installed - running without LLM tracing")
except Exception as e:
    print(f"⚠️ Opik initialization failed: {e} - running without LLM tracing")

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
_callbacks = [opik_tracer] if opik_tracer else []

gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    google_api_key=security_config.google_api_key,
    callbacks=_callbacks,
)

DATA_DIR = security_config.data_dir
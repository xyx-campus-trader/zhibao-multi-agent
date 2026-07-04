"""
LLM工厂 — 统一接入 Ollama / OpenAI / DeepSeek
"""
import logging
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from config.settings import settings

logger = logging.getLogger(__name__)


def create_chat_model(temperature: float = None, streaming: bool = False) -> BaseChatModel:
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        temperature=temperature or settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        streaming=streaming,
        timeout=settings.LLM_TIMEOUT,
    )

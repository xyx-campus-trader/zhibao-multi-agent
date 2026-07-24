"""
LLM工厂 — 统一接入 Ollama / OpenAI / DeepSeek
支持超时重试（指数退避）与断路器熔断保护
"""
import logging
import time
import threading
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from config.settings import settings

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """断路器：连续失败 N 次后熔断，冷却期后进入半开状态"""

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 30.0):
        self._failure_threshold = max(failure_threshold, 1)  # 最小阈值为1，避免边界异常
        self._cooldown_seconds = cooldown_seconds
        self._failure_count = 0
        self._last_failure_time: float = time.time()  # 初始化为当前时间，避免epoch触发冷却
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._failure_count < self._failure_threshold:
                return False
            if time.time() - self._last_failure_time > self._cooldown_seconds:
                self._failure_count = 0
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()


_circuit_breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=30.0)


def get_circuit_breaker_status() -> dict:
    return {
        "is_open": _circuit_breaker.is_open,
        "failure_count": _circuit_breaker._failure_count,
    }


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


def invoke_with_retry(llm: BaseChatModel, messages: list) -> object:
    """同步调用 LLM，带指数退避重试 + 断路器"""
    import math

    if _circuit_breaker.is_open:
        raise RuntimeError("LLM 服务熔断中，请稍后再试")

    last_error: Optional[Exception] = None
    max_attempts = settings.LLM_RETRY_COUNT + 1
    for attempt in range(max_attempts):
        try:
            result = llm.invoke(messages)
            _circuit_breaker.record_success()
            return result
        except Exception as e:
            last_error = e
            _circuit_breaker.record_failure()
            if attempt < max_attempts - 1:
                wait = math.pow(2, attempt)
                logger.warning(
                    "LLM invoke failed (attempt %d/%d), retrying in %.0fs: %s",
                    attempt + 1, max_attempts, wait, str(e)[:120],
                )
                time.sleep(wait)
    raise last_error  # type: ignore[misc]


async def ainvoke_with_retry(llm: BaseChatModel, messages: list) -> object:
    """异步调用 LLM，带指数退避重试 + 断路器"""
    import math
    import asyncio

    if _circuit_breaker.is_open:
        raise RuntimeError("LLM 服务熔断中，请稍后再试")

    last_error: Optional[Exception] = None
    max_attempts = settings.LLM_RETRY_COUNT + 1
    for attempt in range(max_attempts):
        try:
            result = await llm.ainvoke(messages)
            _circuit_breaker.record_success()
            return result
        except Exception as e:
            last_error = e
            _circuit_breaker.record_failure()
            if attempt < max_attempts - 1:
                wait = math.pow(2, attempt)
                logger.warning(
                    "LLM ainvoke failed (attempt %d/%d), retrying in %.0fs: %s",
                    attempt + 1, max_attempts, wait, str(e)[:120],
                )
                await asyncio.sleep(wait)
    raise last_error  # type: ignore[misc]

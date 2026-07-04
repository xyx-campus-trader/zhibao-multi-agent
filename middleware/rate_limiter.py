"""
Redis滑动窗口限流
"""
import time
import logging
from fastapi import Request, HTTPException

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests = {}  # 内存实现，生产环境用Redis

    async def check(self, request: Request) -> None:
        client_ip = request.client.host
        now = time.time()
        window_start = now - self.window_seconds

        if client_ip not in self._requests:
            self._requests[client_ip] = []

        # 清理过期记录（滑动窗口）
        self._requests[client_ip] = [
            ts for ts in self._requests[client_ip] if ts > window_start
        ]

        if len(self._requests[client_ip]) >= self.max_requests:
            raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")

        self._requests[client_ip].append(now)

"""
综合测试：RateLimiter — 内存滑动窗口限流器边界测试 (async check)
"""
import sys, os, time, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from middleware.rate_limiter import RateLimiter


@pytest.fixture
def limiter():
    return RateLimiter(max_requests=10, window_seconds=60)


class FakeRequest:
    """模拟 FastAPI Request 对象"""
    class FakeClient:
        def __init__(self, host):
            self.host = host
    def __init__(self, host="127.0.0.1"):
        self.client = self.FakeClient(host)
        self.headers = {}


# ---- 同步测试部分 (不调用 check 的方法) ----

class TestRateLimiterInit:
    def test_default_values(self):
        rl = RateLimiter()
        assert rl.max_requests == 60
        assert rl.window_seconds == 60

    def test_custom_values(self):
        rl = RateLimiter(max_requests=100, window_seconds=30)
        assert rl.max_requests == 100
        assert rl.window_seconds == 30

    def test_zero_max_requests(self):
        rl = RateLimiter(max_requests=0, window_seconds=60)
        assert rl.max_requests == 0


# ---- 异步测试部分 ----

@pytest.mark.asyncio
class TestRateLimiterNormalFlow:
    async def test_first_request_allowed(self, limiter):
        req = FakeRequest()
        await limiter.check(req)

    async def test_requests_under_limit(self, limiter):
        req = FakeRequest()
        for _ in range(10):
            await limiter.check(req)

    async def test_exact_limit(self, limiter):
        req = FakeRequest()
        for _ in range(10):
            await limiter.check(req)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            await limiter.check(req)
        assert exc.value.status_code == 429

    async def test_over_limit(self, limiter):
        req = FakeRequest()
        for _ in range(10):
            await limiter.check(req)
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            await limiter.check(req)


@pytest.mark.asyncio
class TestRateLimiterWindow:
    async def test_window_slide_clears_old_records(self, monkeypatch):
        rl = RateLimiter(max_requests=5, window_seconds=1)
        fake_time = [time.time()]
        monkeypatch.setattr(time, 'time', lambda: fake_time[0])

        req = FakeRequest()
        for _ in range(5):
            await rl.check(req)

        fake_time[0] += 2.0  # 窗口已过期
        await rl.check(req)  # 应该被允许

    async def test_window_partial_slide(self, monkeypatch):
        rl = RateLimiter(max_requests=2, window_seconds=1)
        fake_time = [1000.0]
        monkeypatch.setattr(time, 'time', lambda: fake_time[0])

        req = FakeRequest()
        # 前2个请求填满窗口
        await rl.check(req)
        await rl.check(req)

        # 前进 1.5 秒，窗口完全过期
        fake_time[0] += 1.5
        # 旧记录被清理，可以继续
        await rl.check(req)
        await rl.check(req)

        # 第3个在同一窗口内，应该被限流
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            await rl.check(req)


@pytest.mark.asyncio
class TestRateLimiterMultipleClients:
    async def test_different_ips_independent(self, limiter):
        req1 = FakeRequest("192.168.1.1")
        req2 = FakeRequest("192.168.1.2")

        for _ in range(10):
            await limiter.check(req1)

        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            await limiter.check(req1)

        for _ in range(10):
            await limiter.check(req2)


@pytest.mark.asyncio
class TestRateLimiterEdgeCases:
    async def test_zero_limit_always_blocks(self):
        rl = RateLimiter(max_requests=0, window_seconds=60)
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            await rl.check(FakeRequest())

    async def test_clean_empty_request_list(self, limiter):
        req = FakeRequest()
        limiter._requests[req.client.host] = []
        await limiter.check(req)

    async def test_malformed_timestamps_cleaned(self, limiter):
        req = FakeRequest()
        limiter._requests[req.client.host] = [time.time() - 100000, time.time() - 99999]
        await limiter.check(req)

    async def test_large_burst(self, limiter):
        req = FakeRequest()
        from fastapi import HTTPException
        ok_count = 0
        fail_count = 0
        for _ in range(20):
            try:
                await limiter.check(req)
                ok_count += 1
            except HTTPException:
                fail_count += 1
        assert ok_count == 10
        assert fail_count == 10


@pytest.mark.asyncio
class TestRateLimiterHttpException:
    async def test_exception_has_correct_status(self, limiter):
        req = FakeRequest()
        for _ in range(10):
            await limiter.check(req)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            await limiter.check(req)
        assert exc.value.status_code == 429
        assert "频繁" in exc.value.detail

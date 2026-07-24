"""
综合测试：CircuitBreaker — 断路器熔断与冷却恢复
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from core.llm_factory import CircuitBreaker


class TestCircuitBreakerInit:
    def test_default_thresholds(self):
        cb = CircuitBreaker()
        assert cb._failure_threshold == 3
        assert cb._cooldown_seconds == 30.0

    def test_custom_thresholds(self):
        cb = CircuitBreaker(failure_threshold=5, cooldown_seconds=60.0)
        assert cb._failure_threshold == 5
        assert cb._cooldown_seconds == 60.0

    def test_initial_state_not_open(self):
        cb = CircuitBreaker()
        assert cb.is_open is False

    def test_initial_failure_count_zero(self):
        cb = CircuitBreaker()
        assert cb._failure_count == 0


class TestCircuitBreakerOpenClose:
    def test_not_open_before_threshold(self):
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(4):
            cb.record_failure()
        assert cb.is_open is False

    def test_open_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is True

    def test_open_past_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(10):
            cb.record_failure()
        assert cb.is_open is True

    def test_record_success_resets_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb._failure_count == 0
        assert cb.is_open is False

    def test_record_success_after_breach(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(5):
            cb.record_failure()
        assert cb.is_open is True
        cb.record_success()
        assert cb.is_open is False
        assert cb._failure_count == 0


class TestCircuitBreakerCooldown:
    def test_cooldown_closes_breaker(self, monkeypatch):
        """模拟时间流逝让冷却期过期"""
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=1.0)
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open is True

        # 模拟时间过去了 2 秒
        original_time = time.time
        mock_now = original_time()  # freeze current time
        fake_time = [mock_now]

        def mock_time():
            return fake_time[0]

        monkeypatch.setattr(time, 'time', mock_time)

        fake_time[0] = original_time() + 2.0  # simulate 2s later
        assert cb.is_open is False

    def test_cooldown_not_yet_expired(self):
        """冷却期未过，应仍然处于开路状态"""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=999.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is True

    def test_cooldown_half_open_resets_on_success(self):
        """冷却期过后记录成功，断路器关闭"""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is True
        # cooldown_seconds=0 会立即在下一次 is_open 检查时重置
        cb.record_success()
        assert cb.is_open is False

    def test_single_failure_after_cooldown(self, monkeypatch):
        """冷却期过后再次失败，重新计数"""
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=0.1)
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open is True

        original_time = time.time
        fake_time = [original_time()]
        monkeypatch.setattr(time, 'time', lambda: fake_time[0])
        fake_time[0] = original_time() + 1.0  # past cooldown

        # 检查 is_open 触发冷却重置
        assert cb.is_open is False  # cooldown expired, resets

        # 再次失败一次
        cb.record_failure()
        assert cb._failure_count == 1
        assert cb.is_open is False  # only 1 failure, threshold is 3


class TestCircuitBreakerEdgeCases:
    def test_threshold_one(self):
        """阈值为1时，一次失败即熔断（构造器已限制最小值为1）"""
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.is_open is True

    def test_negative_threshold_clamped(self):
        """负阈值被钳制为1"""
        cb = CircuitBreaker(failure_threshold=-1)
        assert cb._failure_threshold == 1

    def test_huge_threshold(self):
        cb = CircuitBreaker(failure_threshold=1000)
        for _ in range(999):
            cb.record_failure()
        assert cb.is_open is False

    def test_zero_cooldown(self, monkeypatch):
        """0s冷却期：is_open检查时立即重置（需要模拟时间）"""
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=0)
        original_time = time.time
        fake_time = [original_time()]
        monkeypatch.setattr(time, 'time', lambda: fake_time[0])

        for _ in range(3):
            cb.record_failure()

        # 前进1ms确保时间差
        fake_time[0] += 0.001
        assert cb.is_open is False  # cooldown expired, auto resets

    def test_thread_safety_same_count(self):
        """在单线程中验证计数器一致性"""
        cb = CircuitBreaker(failure_threshold=10)
        for _ in range(5):
            cb.record_failure()
        assert cb._failure_count == 5
        cb.record_success()
        assert cb._failure_count == 0


class TestCircuitBreakerStatus:
    def test_global_circuit_breaker_status(self):
        from core.llm_factory import get_circuit_breaker_status
        status = get_circuit_breaker_status()
        assert "is_open" in status
        assert "failure_count" in status

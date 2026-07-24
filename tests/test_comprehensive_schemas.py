"""
综合测试：Pydantic Schema Validations — 智报系统的请求/响应模型
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pydantic import ValidationError
from models.schemas import (
    ReportRequest, ReportResponse, TaskStatus,
    LoginRequest, TokenResponse, Result,
)


# ============ ReportRequest ============
class TestReportRequest:
    def test_valid_minimal(self):
        rr = ReportRequest(topic="AI行业动态")
        assert rr.topic == "AI行业动态"
        assert rr.dimensions is None

    def test_topic_min_length_1(self):
        with pytest.raises(ValidationError):
            ReportRequest(topic="")

    def test_topic_required(self):
        with pytest.raises(ValidationError):
            ReportRequest()

    def test_with_dimensions(self):
        rr = ReportRequest(topic="AI", dimensions=["政策", "公司"])
        assert rr.dimensions == ["政策", "公司"]

    def test_topic_whitespace_only(self):
        """空字符串被拒绝（min_length=1），但只有空格可以通过"""
        rr = ReportRequest(topic=" ")
        assert rr.topic == " "

    def test_long_topic(self):
        rr = ReportRequest(topic="AI" * 1000)
        assert len(rr.topic) == 2000

    def test_special_chars_in_topic(self):
        rr = ReportRequest(topic="AI & 半导体 @ 2026!")
        assert rr.topic == "AI & 半导体 @ 2026!"


# ============ ReportResponse ============
class TestReportResponse:
    def test_minimal(self):
        resp = ReportResponse(task_id="abc123", topic="AI", status="pending")
        assert resp.task_id == "abc123"
        assert resp.report is None
        assert resp.sources == []

    def test_full(self):
        resp = ReportResponse(
            task_id="x1", topic="test", status="done",
            report="this is the report", sources=[{"url": "http://a.com"}],
            review_notes="looks good", created_at="2026-07-24",
        )
        assert resp.report == "this is the report"
        assert len(resp.sources) == 1

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            ReportResponse(task_id="t", topic="t")


# ============ TaskStatus ============
class TestTaskStatus:
    def test_valid(self):
        ts = TaskStatus(task_id="abc", status="running", progress="searching")
        assert ts.error is None
        assert ts.report is None

    def test_with_error(self):
        ts = TaskStatus(task_id="abc", status="failed", progress="done", error="LLM超时")
        assert ts.error == "LLM超时"

    def test_missing_required(self):
        with pytest.raises(ValidationError):
            TaskStatus(task_id="t", status="s")


# ============ LoginRequest ============
class TestLoginRequest:
    def test_valid(self):
        lr = LoginRequest(username="admin", password="123456")
        assert lr.username == "admin"

    def test_username_min_2(self):
        with pytest.raises(ValidationError):
            LoginRequest(username="a", password="123456")

    def test_username_max_50(self):
        with pytest.raises(ValidationError):
            LoginRequest(username="a" * 51, password="123456")

    def test_password_min_6(self):
        with pytest.raises(ValidationError):
            LoginRequest(username="admin", password="12345")

    def test_password_boundary(self):
        lr = LoginRequest(username="admin", password="123456")
        assert len(lr.password) == 6

    def test_username_boundary_2(self):
        lr = LoginRequest(username="ab", password="123456")
        assert lr.username == "ab"

    def test_username_boundary_50(self):
        lr = LoginRequest(username="a" * 50, password="123456")
        assert len(lr.username) == 50


# ============ TokenResponse ============
class TestTokenResponse:
    def test_default(self):
        tr = TokenResponse(access_token="mock-token-123")
        assert tr.access_token == "mock-token-123"
        assert tr.token_type == "bearer"

    def test_custom_type(self):
        tr = TokenResponse(access_token="abc", token_type="jwt")
        assert tr.token_type == "jwt"


# ============ Result ============
class TestResult:
    def test_defaults(self):
        r = Result()
        assert r.code == 200
        assert r.message == "success"
        assert r.data is None

    def test_error(self):
        r = Result(code=500, message="error", data=None)
        assert r.code == 500

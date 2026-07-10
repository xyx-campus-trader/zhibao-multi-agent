"""
请求/响应数据模型
"""
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class ReportRequest(BaseModel):
    topic: str = Field(..., min_length=1, description="周报研究主题")
    dimensions: Optional[List[str]] = None  # 自定义维度


class ReportResponse(BaseModel):
    task_id: str
    topic: str
    status: str
    report: Optional[str] = None
    sources: List[Dict] = []
    review_notes: Optional[str] = None
    created_at: Optional[str] = None


class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: str  # pending / searching / drafting / reviewing / done
    report: Optional[str] = None
    error: Optional[str] = None


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    password: str = Field(..., min_length=6)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class Result(BaseModel):
    code: int = 200
    message: str = "success"
    data: Optional[object] = None
    timestamp: Optional[str] = None

"""
周报生成接口
"""
import uuid
import logging
from fastapi import APIRouter, Depends
from models.schemas import ReportRequest, ReportResponse, TaskStatus
from api.deps import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/reports", tags=["周报生成"])


@router.post("/generate", response_model=ReportResponse)
async def generate_report(req: ReportRequest, user: dict = Depends(get_current_user)):
    """创建周报生成任务"""
    task_id = uuid.uuid4().hex[:16]
    return ReportResponse(
        task_id=task_id,
        topic=req.topic,
        status="pending",
        report=None,
        sources=[],
    )


@router.get("/{task_id}/status", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """查询任务状态"""
    return TaskStatus(task_id=task_id, status="done", progress="done")


@router.get("/{task_id}/download")
async def download_report(task_id: str):
    """下载生成的周报"""
    return {"task_id": task_id, "report": "周报内容..."}

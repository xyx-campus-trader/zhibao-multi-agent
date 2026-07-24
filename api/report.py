"""
周报生成接口 — 对接真实工作流
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from models.schemas import ReportRequest, ReportResponse, TaskStatus
from api.deps import get_current_user
from services.report_service import get_report_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/reports", tags=["周报生成"])


@router.post("/generate", response_model=ReportResponse)
async def generate_report(req: ReportRequest, user: dict = Depends(get_current_user)):
    """创建周报生成任务，异步执行工作流"""
    service = get_report_service()
    task_id = await service.create_report(req.topic, user_id=user.get("id"))
    report = await service.get_report(task_id)

    return ReportResponse(
        task_id=task_id,
        topic=req.topic,
        status=report.get("status", "pending"),
        report=report.get("final_report") or report.get("draft"),
        sources=report.get("research_results", []),
    )


@router.get("/{task_id}/status", response_model=TaskStatus)
async def get_task_status(task_id: str):
    """查询任务状态与结果"""
    service = get_report_service()
    report = await service.get_report(task_id)

    if not report:
        raise HTTPException(status_code=404, detail="任务不存在")

    return TaskStatus(
        task_id=task_id,
        status=report.get("status", "unknown"),
        progress=report.get("status", "unknown"),
        report=report.get("final_report") or report.get("draft"),
        error=report.get("error"),
    )


@router.get("/{task_id}/download")
async def download_report(task_id: str):
    """下载生成的周报"""
    service = get_report_service()
    report = await service.get_report(task_id)

    if not report:
        raise HTTPException(status_code=404, detail="任务不存在")

    return {
        "task_id": task_id,
        "topic": report.get("topic", ""),
        "report": report.get("final_report") or report.get("draft", ""),
        "review_notes": report.get("review_notes", ""),
        "status": report.get("status", "unknown"),
        "sources": report.get("research_results", []),
    }


@router.post("/{task_id}/approve")
async def approve_report(
    task_id: str,
    approved: bool = True,
    feedback: str = "",
    user: dict = Depends(get_current_user),
):
    """人工审核：批准或驳回 HITL 中断的报告，驳回时可附修改意见"""
    service = get_report_service()
    result = await service.approve_report(task_id, approved, feedback)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return {
        "task_id": task_id,
        "approved": approved,
        "feedback": feedback,
        "status": result.get("status", "done"),
    }

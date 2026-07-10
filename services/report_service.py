"""
周报服务 — 对接工作流、持久化、恢复
"""
import logging
import uuid
from core.workflow import build_workflow
from core.search_decomposer import SearchDecomposer
from core.persistence import TaskPersistence

logger = logging.getLogger(__name__)

_service_instance = None


def get_report_service() -> "ReportService":
    global _service_instance
    if _service_instance is None:
        _service_instance = ReportService()
    return _service_instance


class ReportService:
    def __init__(self, persistence: TaskPersistence = None):
        self.decomposer = SearchDecomposer()
        self.persistence = persistence or TaskPersistence()
        self.workflow = build_workflow(self.decomposer)

    async def create_report(self, topic: str, user_id: int = None) -> str:
        task_id = uuid.uuid4().hex[:16]

        state = {
            "task_id": task_id,
            "topic": topic,
            "search_keywords": {},
            "research_results": [],
            "draft": "",
            "final_report": "",
            "review_notes": "",
            "status": "pending",
            "error": None,
            "hitl_approved": False,
            "revision_count": 0,
        }

        try:
            await self.persistence.save_task(task_id, state)
            result = await self.workflow.ainvoke(state)
            await self.persistence.save_task(task_id, result)
            return task_id
        except Exception as e:
            logger.error("Report generation failed for task %s: %s", task_id, e)
            state["status"] = "failed"
            state["error"] = str(e)
            await self.persistence.save_task(task_id, state)
            return task_id

    async def get_report(self, task_id: str) -> dict:
        return await self.persistence.load_task(task_id) or {}

    async def approve_report(self, task_id: str, approved: bool) -> dict:
        """人工审核：恢复 HITL 中断的工作流"""
        state = await self.persistence.load_task(task_id)
        if not state:
            return {"error": "任务不存在"}

        from langgraph.types import Command
        config = {"configurable": {"thread_id": task_id}}
        resume_value = Command(resume={"approved": approved})

        try:
            result = await self.workflow.ainvoke(resume_value, config)
            await self.persistence.save_task(task_id, result)
            return result
        except Exception as e:
            logger.error("HITL resume failed for task %s: %s", task_id, e)
            return {"error": str(e)}

    async def recover_tasks(self) -> dict:
        """服务重启时恢复未完成任务"""
        return await self.persistence.recover_all()

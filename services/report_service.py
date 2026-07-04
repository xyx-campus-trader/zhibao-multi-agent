"""
周报服务
"""
import logging
from core.workflow import build_workflow
from core.search_decomposer import SearchDecomposer
from core.persistence import TaskPersistence

logger = logging.getLogger(__name__)


class ReportService:
    def __init__(self, persistence: TaskPersistence = None):
        self.decomposer = SearchDecomposer()
        self.persistence = persistence or TaskPersistence()
        self.workflow = build_workflow(self.decomposer)

    async def create_report(self, topic: str, user_id: int = None) -> str:
        import uuid
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
            "hitl_approved": True,
        }

        try:
            result = await self.workflow.ainvoke(state)
            await self.persistence.save_task(task_id, result)
            return task_id
        except Exception as e:
            logger.error("Report generation failed: %s", e)
            state["status"] = "failed"
            state["error"] = str(e)
            await self.persistence.save_task(task_id, state)
            return task_id

    async def get_report(self, task_id: str) -> dict:
        return await self.persistence.load_task(task_id)

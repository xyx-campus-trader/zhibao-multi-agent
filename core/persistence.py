"""异步任务持久化 — PostgreSQL 落盘为真值 + Redis 读缓存(cache-aside)

设计原则:
- 写:任务状态同步写入 PostgreSQL,落盘成功即视为已保存;随后 best-effort 更新
  Redis 缓存,缓存失败仅记日志,不影响主流程(Redis 是加速手段,不是真值)。
- 读:先查 Redis,未命中回源 PostgreSQL 并回填缓存。
- 恢复:服务重启时从 PostgreSQL 读取未完成任务(status 非终态),数据落盘不丢。
"""
import logging
from typing import Any, Dict, Optional

from sqlalchemy import select

from models.database import TaskState

logger = logging.getLogger(__name__)

_CACHE_PREFIX = "task:"
_CACHE_TTL = 7 * 24 * 3600  # 7 天,仅缓存,过期后回源 PG
_TERMINAL_STATUS = ("done", "failed", "cancelled")


class TaskPersistence:
    """PG 为真值、Redis 为读缓存的任务状态存储。"""

    def __init__(self, db_session_factory, redis_client=None):
        if db_session_factory is None:
            raise ValueError("db_session_factory 必填:PostgreSQL 是任务状态的真值存储")
        self._db_factory = db_session_factory
        self._redis = redis_client  # 可为 None:无缓存时直接走 PG

    async def save_task(self, task_id: str, state: Dict[str, Any]) -> bool:
        """保存任务状态:PG 落盘成功即算已保存,Redis best-effort 更新。"""
        async with self._db_factory() as db:
            await db.merge(TaskState(task_id=task_id, state=state))
            await db.commit()
        await self._cache_set(task_id, state)
        return True

    async def load_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """读取任务状态:先 Redis,未命中回源 PG 并回填缓存。"""
        cached = await self._cache_get(task_id)
        if cached is not None:
            return cached

        async with self._db_factory() as db:
            obj = await db.get(TaskState, task_id)
            if obj is None:
                return None
            state = obj.state
        await self._cache_set(task_id, state)
        return state

    async def recover_all(self) -> Dict[str, Dict[str, Any]]:
        """服务重启后从 PG 恢复所有未完成任务(status 非终态)。"""
        recovered: Dict[str, Dict[str, Any]] = {}
        async with self._db_factory() as db:
            stmt = select(TaskState).where(
                TaskState.state["status"].astext.notin_(_TERMINAL_STATUS)
            )
            result = await db.execute(stmt)
            for row in result.scalars().all():
                recovered[row.task_id] = row.state
        logger.info("Recovered %d unfinished tasks from PostgreSQL", len(recovered))
        return recovered

    # ---- Redis 缓存:best-effort,任何异常都降级为"无缓存"而非报错 ----

    async def _cache_set(self, task_id: str, state: Dict[str, Any]) -> None:
        if not self._redis:
            return
        try:
            import json

            await self._redis.setex(
                f"{_CACHE_PREFIX}{task_id}",
                _CACHE_TTL,
                json.dumps(state, ensure_ascii=False),
            )
        except Exception as e:
            logger.warning("Redis cache write failed for %s (ignored): %s", task_id, e)

    async def _cache_get(self, task_id: str) -> Optional[Dict[str, Any]]:
        if not self._redis:
            return None
        try:
            import json

            raw = await self._redis.get(f"{_CACHE_PREFIX}{task_id}")
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.warning("Redis cache read failed for %s (ignored): %s", task_id, e)
            return None

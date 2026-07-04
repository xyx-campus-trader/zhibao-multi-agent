"""
持久化模块 — 异步任务三级持久化存储 (Redis → PostgreSQL → 内存)
"""
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TaskPersistence:
    """三级持久化存储管理器"""

    def __init__(self, redis_client=None, db_session_factory=None):
        self._redis = redis_client
        self._db_factory = db_session_factory
        self._memory: Dict[str, Dict[str, Any]] = {}  # 内存兜底
        self._redis_available = False

    async def save_task(self, task_id: str, state: Dict[str, Any]) -> bool:
        """保存任务状态，任一层写入成功即视为已保存"""
        # 第一层：Redis
        if self._redis:
            try:
                from redis import Redis
                if isinstance(self._redis, Redis):
                    key = f"task:{task_id}"
                    self._redis.setex(
                        key,
                        timedelta(days=7),
                        json.dumps(state, ensure_ascii=False)
                    )
                    self._redis_available = True
                    return True
            except Exception:
                self._redis_available = False
                logger.debug("Redis save failed, falling back")

        # 第二层：PostgreSQL
        if self._db_factory:
            try:
                async with self._db_factory() as db:
                    from sqlalchemy import text
                    await db.execute(
                        text(
                            "INSERT INTO task_states (task_id, state, updated_at) "
                            "VALUES (:id, :state, :ts) "
                            "ON CONFLICT (task_id) DO UPDATE "
                            "SET state = :state, updated_at = :ts"
                        ),
                        {
                            "id": task_id,
                            "state": json.dumps(state, ensure_ascii=False),
                            "ts": datetime.utcnow(),
                        }
                    )
                    await db.commit()
                    return True
            except Exception:
                logger.debug("PostgreSQL save failed, falling back to memory")

        # 第三层：内存（最终兜底）
        self._memory[task_id] = state
        return True

    async def load_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """恢复任务状态，按 Redis → PostgreSQL → 内存 优先级查找"""
        # Redis
        if self._redis and self._redis_available:
            try:
                raw = self._redis.get(f"task:{task_id}")
                if raw:
                    return json.loads(raw)
            except Exception:
                pass

        # PostgreSQL
        if self._db_factory:
            try:
                async with self._db_factory() as db:
                    from sqlalchemy import text
                    result = await db.execute(
                        text("SELECT state FROM task_states WHERE task_id = :id"),
                        {"id": task_id}
                    )
                    row = result.fetchone()
                    if row:
                        return json.loads(row[0])
            except Exception:
                pass

        # 内存
        return self._memory.get(task_id)

    async def recover_all(self) -> Dict[str, Dict[str, Any]]:
        """服务重启后恢复所有未完成任务"""
        recovered = {}
        # 从Redis恢复
        if self._redis and self._redis_available:
            try:
                keys = self._redis.keys("task:*")
                for key in keys:
                    raw = self._redis.get(key)
                    if raw:
                        task_id = key.decode().replace("task:", "")
                        recovered[task_id] = json.loads(raw)
            except Exception:
                pass
        # 从内存补充
        for tid, state in self._memory.items():
            if tid not in recovered:
                recovered[tid] = state
        logger.info("Recovered %d tasks on startup", len(recovered))
        return recovered

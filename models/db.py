"""数据库与缓存连接管理:async SQLAlchemy 引擎 + session 工厂 + async Redis"""
import logging

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config.settings import settings
from models.database import Base

logger = logging.getLogger(__name__)

_engine = None
_session_factory = None
_redis = None


def get_session_factory() -> async_sessionmaker:
    """获取(懒加载)异步 session 工厂。PG 是任务状态的真值存储。"""
    global _engine, _session_factory
    if _session_factory is None:
        _engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
        _session_factory = async_sessionmaker(
            _engine, class_=AsyncSession, expire_on_commit=False
        )
    return _session_factory


async def init_db() -> async_sessionmaker:
    """建表并返回 session 工厂,服务启动时调用。"""
    factory = get_session_factory()
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("PostgreSQL connected, tables ensured")
    return factory


async def get_redis():
    """获取 async Redis 客户端。连接失败返回 None——Redis 仅作读缓存,不可用不影响主流程。"""
    global _redis
    if _redis is None:
        try:
            from redis.asyncio import Redis

            client = Redis.from_url(settings.REDIS_URL, decode_responses=True)
            await client.ping()
            _redis = client
            logger.info("Redis connected (read cache enabled)")
        except Exception as e:
            logger.warning("Redis unavailable, running without cache: %s", e)
            _redis = None
    return _redis


async def close_connections() -> None:
    """服务关闭时释放连接。"""
    global _engine, _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None
    if _engine is not None:
        await _engine.dispose()
        _engine = None

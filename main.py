"""
智报多Agent自动生成系统 — FastAPI 主入口
基于 Multi-Agent 协作的自动采编平台

工作流: Orchestrator → Researcher → Editor → Reviewer
"""
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from config.settings import settings
from api.auth import router as auth_router
from api.report import router as report_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 服务启动:初始化 PG/Redis 连接,并从数据库恢复未完成任务
    try:
        from services.report_service import init_report_service
        service = await init_report_service()
        recovered = await service.recover_tasks()
        if recovered:
            logger.info("Recovered %d unfinished tasks on startup", len(recovered))
    except Exception as e:
        logger.warning("Startup init/recovery failed: %s", e)

    logger.info(
        "智报系统启动 | provider=%s model=%s",
        settings.LLM_PROVIDER, settings.LLM_MODEL,
    )
    yield

    # 服务关闭:释放数据库与缓存连接
    try:
        from models.db import close_connections
        await close_connections()
    except Exception as e:
        logger.warning("Connection cleanup failed: %s", e)


app = FastAPI(
    title="智报多Agent自动生成系统",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(report_router)

# 静态文件 & 前端页面
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(_STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

@app.get("/")
async def index():
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.get("/health")
async def health():
    from core.llm_factory import get_circuit_breaker_status
    return {
        "status": "ok",
        "service": "智报多Agent系统",
        "circuit_breaker": get_circuit_breaker_status(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)

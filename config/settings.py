"""
全局配置管理
"""
from pydantic_settings import BaseSettings
from typing import Literal, List


class Settings(BaseSettings):
    # ===== 应用 =====
    APP_NAME: str = "智报多Agent自动生成系统"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8002

    # ===== LLM =====
    LLM_PROVIDER: Literal["openai", "ollama", "deepseek"] = "deepseek"
    LLM_MODEL: str = "deepseek-chat"
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"
    LLM_API_KEY: str = ""
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 4096
    LLM_TIMEOUT: int = 120
    LLM_RETRY_COUNT: int = 2

    # ===== 数据库 =====
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "postgres"
    DB_PASSWORD: str = ""
    DB_NAME: str = "zhibao"
    SQLITE_URL: str = "sqlite+aiosqlite:///./zhibao.db"

    @property
    def DATABASE_URL(self) -> str:
        if self.SQLITE_URL:
            return self.SQLITE_URL
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # ===== Redis =====
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # ===== JWT =====
    JWT_SECRET: str = "change-me"
    JWT_EXPIRE_MINUTES: int = 1440

    # ===== Agent =====
    AGENT_MAX_RETRIES: int = 3
    AGENT_SEARCH_TIMEOUT: int = 30
    AGENT_SEARCH_DIMENSIONS: List[str] = ["政策动向", "公司动态", "投融资", "行业趋势"]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }


settings = Settings()

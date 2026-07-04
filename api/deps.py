"""
依赖注入
"""
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer(auto_error=False)


async def get_current_user(token: str = Depends(security)):
    if not token:
        raise HTTPException(status_code=401)
    return {"user_id": 1, "username": "demo"}

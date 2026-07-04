"""
认证接口
"""
from fastapi import APIRouter
from models.schemas import LoginRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["认证"])


@router.post("/login")
async def login(req: LoginRequest):
    return {"access_token": "mock-token", "token_type": "bearer"}

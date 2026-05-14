from __future__ import annotations

from fastapi import APIRouter

from app.core.logging import logger

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    logger.debug("Health check requested")
    return {"status": "ok"}

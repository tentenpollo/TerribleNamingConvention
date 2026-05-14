from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.dependencies import require_role
from app.core.roles import Role
from app.models.user import User

router = APIRouter(prefix="/test")


@router.get("/member-only")
async def member_only(
    current_user: User = Depends(require_role(Role.MEMBER)),
) -> dict[str, str]:
    return {"user_id": str(current_user.id), "role": current_user.role}


@router.get("/admin-only")
async def admin_only(
    current_user: User = Depends(require_role(Role.ADMIN)),
) -> dict[str, str]:
    return {"user_id": str(current_user.id), "role": current_user.role}


@router.get("/superadmin-only")
async def superadmin_only(
    current_user: User = Depends(require_role(Role.SUPER_ADMIN)),
) -> dict[str, str]:
    return {"user_id": str(current_user.id), "role": current_user.role}

from __future__ import annotations

import enum


class Role(enum.StrEnum):
    MEMBER = "member"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


ROLE_HIERARCHY: dict[Role, int] = {
    Role.MEMBER: 0,
    Role.ADMIN: 1,
    Role.SUPER_ADMIN: 2,
}


def has_permission(user_role: Role, required_role: Role) -> bool:
    return ROLE_HIERARCHY[user_role] >= ROLE_HIERARCHY[required_role]

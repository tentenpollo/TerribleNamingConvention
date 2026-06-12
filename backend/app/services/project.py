from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    AccessDeniedError,
    ProjectNotFoundError,
    QdrantError,
    TeamNotFoundError,
)
from app.core.logging import logger
from app.ingestion.vector_store import VectorStore
from app.models.project import Project
from app.models.team import Team
from app.schemas.project import ProjectCreate, ProjectUpdate


class ProjectService:
    def __init__(self, session: AsyncSession, vector_store: VectorStore) -> None:
        self.session = session
        self.vector_store = vector_store

    async def create(self, data: ProjectCreate) -> Project:
        team = await self.session.get(Team, data.team_id)
        if team is None:
            raise TeamNotFoundError(f"Team {data.team_id} does not exist")

        project = Project(
            name=data.name,
            description=data.description,
            team_id=data.team_id,
            config=data.config,
        )
        self.session.add(project)
        await self.session.commit()
        await self.session.refresh(project)
        return project

    async def delete(self, project_id: UUID, accessible_ids: list[UUID]) -> None:
        project = await self.get_by_id(project_id, accessible_ids)
        await self.session.delete(project)
        await self.session.commit()

        # A concurrent ingest job can recreate this collection after deletion; v1 accepts
        # that race and lets the worker fail when it observes the missing project row.
        try:
            await self.vector_store.delete_collection(project_id)
        except QdrantError as exc:
            logger.error(
                "Failed to delete Qdrant collection after project delete",
                project_id=str(project_id),
                error=str(exc),
            )

    async def get_by_id(self, project_id: UUID, accessible_ids: list[UUID]) -> Project:
        project = await self.session.get(Project, project_id)
        if project is None:
            raise ProjectNotFoundError(f"Project {project_id} does not exist")
        if project_id not in accessible_ids:
            raise AccessDeniedError(f"Project {project_id} is outside the user's access scope")
        return project

    async def list_for_user(self, accessible_ids: list[UUID]) -> list[Project]:
        if not accessible_ids:
            return []

        result = await self.session.execute(
            select(Project)
            .where(Project.id.in_(accessible_ids))
            .order_by(Project.created_at, Project.name),
        )
        return list(result.scalars().all())

    async def update(
        self,
        project_id: UUID,
        data: ProjectUpdate,
        accessible_ids: list[UUID],
    ) -> Project:
        project = await self.get_by_id(project_id, accessible_ids)
        if data.name is not None:
            project.name = data.name
        if data.description is not None:
            project.description = data.description

        await self.session.commit()
        await self.session.refresh(project)
        return project

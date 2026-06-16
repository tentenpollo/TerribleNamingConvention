from __future__ import annotations

from app.workers.settings import WorkerSettings


def test_worker_settings_includes_reindex_project_function() -> None:
    function_names = [fn.__name__ for fn in WorkerSettings.functions]
    assert "reindex_project" in function_names


def test_worker_settings_includes_weekly_cag_rebuild_cron() -> None:
    cron_job_names = [cj.name for cj in WorkerSettings.cron_jobs]
    assert "cron:cag_weekly_rebuild" in cron_job_names

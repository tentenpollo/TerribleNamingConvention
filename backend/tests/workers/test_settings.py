from __future__ import annotations

from app.workers.settings import WorkerSettings


def test_worker_settings_includes_weekly_cag_rebuild_cron() -> None:
    cron_job_names = [cj.name for cj in WorkerSettings.cron_jobs]
    assert "cron:cag_weekly_rebuild" in cron_job_names

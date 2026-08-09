"""Ownership-checked HTTP delivery for generated artifacts."""

from __future__ import annotations

from flask import Flask, abort, send_file

from ultrasongs.storage import ArtifactRepository, StorageError


def register_artifact_downloads(server: Flask, artifacts: ArtifactRepository) -> None:
    """Register an opaque-ID route that never accepts a filesystem path."""

    @server.get("/artifacts/<project_id>/<run_id>/<artifact_id>")
    def download_artifact(project_id: str, run_id: str, artifact_id: str):
        try:
            manifest = artifacts.get_manifest(project_id, run_id)
            record = next(
                (item for item in manifest.artifacts if item.artifact_id == artifact_id),
                None,
            )
            if record is None:
                abort(404)
            path = artifacts.get_artifact_path(
                project_id,
                run_id,
                artifact_id,
                verify=True,
            )
        except (StorageError, ValueError):
            abort(404)
        return send_file(
            path,
            as_attachment=True,
            download_name=record.original_name or path.name,
            mimetype=record.media_type,
        )


__all__ = ["register_artifact_downloads"]

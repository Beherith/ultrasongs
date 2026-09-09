"""Reference UltraStar upload inspection and form prefilling."""

from __future__ import annotations

import base64
import binascii
from pathlib import Path

from dash import Dash, Input, Output, no_update

from ultrasongs.config import AppSettings
from ultrasongs.domain.validation import ReferenceSongInspection, inspect_reference_bytes


def decode_upload(contents: str) -> bytes:
    """Decode a Dash upload data URL with a useful error for malformed input."""

    if not contents or "," not in contents:
        raise ValueError("Upload data is missing or malformed")
    metadata, encoded = contents.split(",", 1)
    if not metadata.startswith("data:") or ";base64" not in metadata.lower():
        raise ValueError("Upload must be a base64 data URL")
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Upload data is not valid base64") from exc


def inspect_reference_upload(
    contents: str,
    filename: str | None,
    *,
    max_bytes: int | None = None,
) -> ReferenceSongInspection:
    if filename and Path(filename).suffix.lower() != ".txt":
        raise ValueError("Reference song must be an UltraStar .txt file")
    payload = decode_upload(contents)
    if max_bytes is not None and len(payload) > max_bytes:
        raise ValueError("Reference upload exceeds the configured size limit")
    return inspect_reference_bytes(payload)


def reference_summary(
    inspection: ReferenceSongInspection, filename: str | None
) -> dict[str, object]:
    """Return only small, reviewable metadata suitable for a browser store."""

    return {
        "filename": filename,
        "title": inspection.title,
        "artist": inspection.artist,
        "bpm": inspection.bpm,
        "gap_ms": inspection.gap_ms,
        "note_count": inspection.note_count,
        "duration_ms": inspection.duration_ms,
    }


def reference_status(inspection: ReferenceSongInspection, filename: str | None) -> str:
    name = filename or "Reference song"
    return (
        f"{name}: {inspection.note_count} notes · {inspection.bpm:g} BPM · "
        f"{inspection.duration_ms / 1000:.1f}s"
    )


def register_reference_callbacks(app: Dash, settings: AppSettings) -> None:
    @app.callback(
        Output("reference-summary-store", "data"),
        Output("title-input", "value"),
        Output("artist-input", "value"),
        Output("lyrics-input", "value"),
        Output("reference-status", "children"),
        Input("reference-upload", "contents"),
        Input("reference-upload", "filename"),
        prevent_initial_call=True,
    )
    def parse_reference(contents: str | None, filename: str | None):
        if not contents:
            return None, no_update, no_update, no_update, "No reference selected."
        try:
            inspection = inspect_reference_upload(
                contents,
                filename,
                max_bytes=settings.security.max_upload_megabytes * 1024 * 1024,
            )
        except ValueError as exc:
            return None, no_update, no_update, no_update, f"Could not read reference: {exc}"
        return (
            reference_summary(inspection, filename),
            inspection.title,
            inspection.artist,
            inspection.reconstructed_lyrics,
            reference_status(inspection, filename),
        )


__all__ = [
    "decode_upload",
    "inspect_reference_upload",
    "reference_status",
    "reference_summary",
    "register_reference_callbacks",
]

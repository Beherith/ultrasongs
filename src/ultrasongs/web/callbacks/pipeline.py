"""Thin callback seam between Dash inputs and a pipeline runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from dash import Dash, Input, Output, State, html, no_update

from ultrasongs.config import AppSettings, EffectiveSettingsSnapshot


@dataclass(frozen=True, slots=True)
class BrowserUpload:
    filename: str | None
    contents: str


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    mode: str
    title: str
    artist: str
    lyrics: str
    audio: BrowserUpload | None
    video: BrowserUpload | None
    reference: BrowserUpload | None
    settings: EffectiveSettingsSnapshot


@dataclass(frozen=True, slots=True)
class PipelineSubmission:
    job_id: str
    project_id: str
    message: str = "Pipeline run accepted."


class SubmissionAdapter(Protocol):
    """Minimal local job interface injected into the Dash callback layer."""

    def submit(self, request: PipelineRequest) -> PipelineSubmission: ...

    def status(self, job_id: str) -> Any: ...


def build_pipeline_request(
    settings: AppSettings,
    *,
    mode: str,
    title: str | None,
    artist: str | None,
    lyrics: str | None,
    audio_contents: str | None,
    audio_filename: str | None,
    video_contents: str | None,
    video_filename: str | None,
    reference_contents: str | None,
    reference_filename: str | None,
    overrides: dict[str, Any] | None,
) -> PipelineRequest:
    """Validate lightweight form invariants before handing work to the runner."""

    if mode not in {"generate", "validate"}:
        raise ValueError("Unknown processing mode")
    if not audio_contents and not video_contents:
        raise ValueError("Upload an audio track or video")
    if mode == "validate" and not reference_contents:
        raise ValueError("Upload a reference UltraStar TXT for validation")
    if not (lyrics or "").strip():
        raise ValueError("Lyrics are required")
    return PipelineRequest(
        mode=mode,
        title=(title or "").strip(),
        artist=(artist or "").strip(),
        lyrics=(lyrics or "").strip(),
        audio=_browser_upload(audio_filename, audio_contents),
        video=_browser_upload(video_filename, video_contents),
        reference=_browser_upload(reference_filename, reference_contents),
        settings=settings.effective_snapshot(overrides),
    )


def register_pipeline_callbacks(
    app: Dash,
    settings: AppSettings,
    submission_adapter: SubmissionAdapter,
) -> None:
    @app.callback(
        Output("project-store", "data"),
        Output("run-store", "data"),
        Output("progress-placeholder", "children"),
        Output("result-placeholder", "children"),
        Output("run-poll", "disabled"),
        Input("run-button", "n_clicks"),
        State("mode-selector", "value"),
        State("title-input", "value"),
        State("artist-input", "value"),
        State("lyrics-input", "value"),
        State("audio-upload", "contents"),
        State("audio-upload", "filename"),
        State("video-upload", "contents"),
        State("video-upload", "filename"),
        State("reference-upload", "contents"),
        State("reference-upload", "filename"),
        State("settings-overrides-store", "data"),
        prevent_initial_call=True,
    )
    def submit_run(
        _clicks: int,
        mode: str,
        title: str | None,
        artist: str | None,
        lyrics: str | None,
        audio_contents: str | None,
        audio_filename: str | None,
        video_contents: str | None,
        video_filename: str | None,
        reference_contents: str | None,
        reference_filename: str | None,
        overrides: dict[str, Any] | None,
    ):
        try:
            request = build_pipeline_request(
                settings,
                mode=mode,
                title=title,
                artist=artist,
                lyrics=lyrics,
                audio_contents=audio_contents,
                audio_filename=audio_filename,
                video_contents=video_contents,
                video_filename=video_filename,
                reference_contents=reference_contents,
                reference_filename=reference_filename,
                overrides=overrides,
            )
        except ValueError as exc:
            return no_update, no_update, f"Cannot start: {exc}", no_update, True

        try:
            submission = submission_adapter.submit(request)
        except Exception as exc:  # the runner boundary must surface failures in the UI
            return no_update, no_update, f"Could not start pipeline: {exc}", no_update, True
        return (
            {"project_id": submission.project_id},
            {
                "job_id": submission.job_id,
                "project_id": submission.project_id,
                "run_id": None,
                "status": "queued",
            },
            submission.message,
            "Run submitted. Results will appear when processing completes.",
            False,
        )

    @app.callback(
        Output("run-store", "data", allow_duplicate=True),
        Output("progress-placeholder", "children", allow_duplicate=True),
        Output("result-placeholder", "children", allow_duplicate=True),
        Output("run-poll", "disabled", allow_duplicate=True),
        Input("run-poll", "n_intervals"),
        State("run-store", "data"),
        prevent_initial_call=True,
    )
    def poll_run(_intervals: int, run_store: dict[str, Any] | None):
        job_id = (run_store or {}).get("job_id")
        if not job_id:
            return no_update, no_update, no_update, True
        try:
            status = submission_adapter.status(str(job_id))
        except (KeyError, ValueError) as exc:
            return no_update, f"Cannot poll pipeline: {exc}", no_update, True
        result = _result_text(status)
        progress = status.error if status.state == "failed" else status.message
        return status.to_store(), progress, result, status.terminal


def _browser_upload(filename: str | None, contents: str | None) -> BrowserUpload | None:
    if contents is None:
        return None
    return BrowserUpload(filename=filename, contents=contents)


def _result_text(status: Any) -> Any:
    if status.state == "failed":
        return "No result artifacts were produced."
    if status.state != "succeeded":
        return "Processing is underway; artifact IDs will appear here on completion."
    links = html.Ul(
        [
            html.Li(
                html.A(
                    f"Download {kind}",
                    href=(
                        f"/artifacts/{status.project_id}/{status.run_id}/{artifact_id}"
                    ),
                )
            )
            for kind, artifact_id in status.artifact_ids.items()
        ],
        className="artifact-links",
    )
    return html.Div([html.Strong(f"Completed run {status.run_id}."), links])


__all__ = [
    "BrowserUpload",
    "PipelineRequest",
    "PipelineSubmission",
    "SubmissionAdapter",
    "build_pipeline_request",
    "register_pipeline_callbacks",
]

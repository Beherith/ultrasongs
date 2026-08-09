"""Declarative layout for the non-editor UltraSongs workflow."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from dash import dcc, html

from ultrasongs.config import AppSettings


def build_layout(settings: AppSettings) -> html.Div:
    """Build a fresh layout so sessions do not share mutable component state."""

    audio_accept = ",".join(settings.security.allowed_audio_extensions)
    video_accept = ",".join(settings.security.allowed_video_extensions)
    return html.Div(
        className="app-shell",
        children=[
            dcc.Store(id="project-store", storage_type="memory", data={"project_id": None}),
            dcc.Store(
                id="run-store",
                storage_type="memory",
                data={"job_id": None, "run_id": None, "status": "idle"},
            ),
            dcc.Interval(id="run-poll", interval=1_000, n_intervals=0, disabled=True),
            dcc.Store(id="settings-overrides-store", storage_type="memory", data={}),
            dcc.Store(id="reference-summary-store", storage_type="memory", data=None),
            html.Header(
                className="hero",
                children=[
                    html.Div("ULTRA", className="eyebrow"),
                    html.H1(["Turn recordings into ", html.Em("singable"), " songs."]),
                    html.P(
                        "A focused Python pipeline for generating UltraStar files and "
                        "checking them against a known-good song."
                    ),
                ],
            ),
            html.Main(
                className="workspace",
                children=[
                    html.Section(
                        className="mode-card",
                        children=[
                            html.Div(
                                [
                                    html.Span("01", className="section-number"),
                                    html.Div(
                                        [
                                            html.H2("Choose a workflow"),
                                            html.P(id="mode-description"),
                                        ]
                                    ),
                                ],
                                className="section-heading",
                            ),
                            dcc.RadioItems(
                                id="mode-selector",
                                value="generate",
                                options=[
                                    {
                                        "label": "Generate a song",
                                        "value": "generate",
                                    },
                                    {
                                        "label": "Validate against a reference",
                                        "value": "validate",
                                    },
                                ],
                                className="mode-selector",
                                inputClassName="mode-radio",
                                labelClassName="mode-option",
                            ),
                        ],
                    ),
                    html.Div(
                        className="content-grid",
                        children=[
                            html.Section(
                                className="panel input-panel",
                                children=[
                                    _section_heading("02", "Add source files"),
                                    html.Div(
                                        className="upload-grid",
                                        children=[
                                            _upload(
                                                "audio-upload",
                                                "Audio track",
                                                "MP3, WAV, FLAC, M4A or OGG",
                                                audio_accept,
                                            ),
                                            _upload(
                                                "video-upload",
                                                "Optional video",
                                                "MP4, MKV, WEBM or MOV",
                                                video_accept,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        id="reference-upload-section",
                                        className="reference-upload",
                                        style={"display": "none"},
                                        children=[
                                            html.Div(
                                                [
                                                    html.Strong("Reference UltraStar TXT"),
                                                    html.Span(
                                                        "Used for end-to-end similarity scoring.",
                                                        className="field-hint",
                                                    ),
                                                ],
                                                className="field-label",
                                            ),
                                            dcc.Upload(
                                                id="reference-upload",
                                                accept=".txt,text/plain",
                                                multiple=False,
                                                className="upload-zone reference-zone",
                                                children=html.Div(
                                                    [
                                                        html.Span(
                                                            "Drop a song TXT here",
                                                            className="upload-title",
                                                        ),
                                                        html.Span(
                                                            "or choose a file",
                                                            className="upload-copy",
                                                        ),
                                                    ]
                                                ),
                                            ),
                                            html.Div(
                                                id="reference-status",
                                                className="inline-status",
                                                role="status",
                                            ),
                                        ],
                                    ),
                                    html.Hr(),
                                    _section_heading("03", "Describe the song"),
                                    html.Div(
                                        className="metadata-grid",
                                        children=[
                                            _text_field("title-input", "Title", "Song title"),
                                            _text_field("artist-input", "Artist", "Artist name"),
                                        ],
                                    ),
                                    html.Label(
                                        [
                                            html.Span("Lyrics"),
                                            html.Span(
                                                        "Reference lyrics remain editable "
                                                        "before processing.",
                                                className="field-hint",
                                            ),
                                            dcc.Textarea(
                                                id="lyrics-input",
                                                placeholder="Paste the complete lyrics…",
                                                className="lyrics-input",
                                            ),
                                        ],
                                        className="field lyrics-field",
                                    ),
                                ],
                            ),
                            html.Aside(
                                className="panel settings-panel",
                                children=[
                                    _section_heading("04", "Tune the pipeline"),
                                    html.P(
                                        "Only run-safe options from the central configuration are "
                                        "shown here.",
                                        className="panel-intro",
                                    ),
                                    html.Details(
                                        className="advanced-settings",
                                        children=[
                                            html.Summary("Advanced settings"),
                                            html.Div(
                                                build_advanced_settings(settings),
                                                className="settings-groups",
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "Apply settings",
                                                id="apply-settings-button",
                                                n_clicks=0,
                                                className="button button-secondary",
                                            ),
                                            html.Button(
                                                "Reset",
                                                id="reset-settings-button",
                                                n_clicks=0,
                                                className="button button-quiet",
                                            ),
                                        ],
                                        className="settings-actions",
                                    ),
                                    html.Div(
                                        "Using central defaults",
                                        id="settings-status",
                                        className="inline-status",
                                        role="status",
                                    ),
                                    html.Div(className="run-divider"),
                                    html.Button(
                                        "Generate UltraStar song",
                                        id="run-button",
                                        n_clicks=0,
                                        className="button button-primary",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Section(
                        className="panel run-panel",
                        children=[
                            _section_heading("05", "Processing run"),
                            html.Div(
                                id="progress-placeholder",
                                className="progress-placeholder",
                                role="status",
                                **{"aria-live": "polite"},
                                children="Ready when you are.",
                            ),
                            html.Div(
                                id="result-placeholder",
                                className="result-placeholder",
                                children=(
                                    "Downloads, scores, and the pipeline report will appear here."
                                ),
                            ),
                        ],
                    ),
                ],
            ),
            html.Footer("UltraSongs · Python + Dash", className="footer"),
        ],
    )


def build_advanced_settings(settings: AppSettings) -> list[html.Div]:
    """Generate controls directly from the central UI-safe settings schema."""

    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for path, metadata in settings.ui_override_schema().items():
        group, _, _ = path.partition(".")
        grouped[group].append((path, metadata))

    groups: list[html.Div] = []
    for group, options in grouped.items():
        fields = [_setting_field(path, metadata) for path, metadata in options]
        groups.append(
            html.Div(
                [html.H3(group.replace("_", " ").title()), *fields],
                className="settings-group",
            )
        )
    return groups


def _setting_field(path: str, metadata: dict[str, Any]) -> html.Div:
    title = str(metadata.get("title", path))
    description = str(metadata.get("description", ""))
    component_id = {"type": "setting-input", "path": path}
    default = metadata.get("default")

    if "enum" in metadata:
        control: Any = dcc.Dropdown(
            id=component_id,
            options=[{"label": str(value), "value": value} for value in metadata["enum"]],
            value=default,
            clearable=False,
        )
    elif metadata.get("type") == "boolean":
        control = dcc.Dropdown(
            id=component_id,
            options=[
                {"label": "Enabled", "value": True},
                {"label": "Disabled", "value": False},
            ],
            value=default,
            clearable=False,
        )
    elif metadata.get("type") in {"integer", "number"}:
        control = dcc.Input(
            id=component_id,
            type="number",
            value=default,
            min=metadata.get("minimum", metadata.get("exclusiveMinimum")),
            max=metadata.get("maximum", metadata.get("exclusiveMaximum")),
            step=1 if metadata.get("type") == "integer" else "any",
        )
    elif metadata.get("type") == "array":
        control = dcc.Input(
            id=component_id,
            type="text",
            value=json.dumps(default),
            debounce=True,
        )
    else:
        control = dcc.Input(
            id=component_id,
            type="text",
            value="" if default is None else str(default),
            disabled="const" in metadata,
            debounce=True,
        )

    return html.Div(
        [
            html.Label(title, htmlFor=str(component_id)),
            control,
            html.Small(description),
        ],
        className="setting-field",
    )


def _upload(component_id: str, title: str, hint: str, accept: str) -> html.Div:
    return html.Div(
        [
            html.Div(
                [html.Strong(title), html.Span(hint, className="field-hint")],
                className="field-label",
            ),
            dcc.Upload(
                id=component_id,
                accept=accept,
                multiple=False,
                className="upload-zone",
                children=html.Div(
                    [
                        html.Span("Drop a file here", className="upload-title"),
                        html.Span("or choose a file", className="upload-copy"),
                    ]
                ),
            ),
        ]
    )


def _text_field(component_id: str, label: str, placeholder: str) -> html.Label:
    return html.Label(
        [html.Span(label), dcc.Input(id=component_id, type="text", placeholder=placeholder)],
        className="field",
    )


def _section_heading(number: str, title: str) -> html.Div:
    return html.Div(
        [html.Span(number, className="section-number"), html.H2(title)],
        className="section-heading compact",
    )


__all__ = ["build_advanced_settings", "build_layout"]

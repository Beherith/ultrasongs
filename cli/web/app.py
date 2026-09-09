"""Dash web frontend for the Ultrasongs pipeline.

Single worker, FIFO queue, one job at a time. See docs/web_ui_plan.md.
No TLS — bind to localhost or use a reverse proxy for LAN access.
"""

import base64
import dataclasses
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import dash
from dash import dcc, html
from flask import abort, jsonify, send_file, request as flask_request

from cli.config import Config, config_from_dict, load_jsonc
from cli.logging_setup import get_logger
from cli.pipeline import ProcessRequest, prepare_lyrics_input
from cli.web import auth, settings_meta
from cli.web.jobs import ALLOWED_UPLOAD_EXTENSIONS, COVER_UPLOAD_EXTENSIONS, JobManager

logger = get_logger("cli.web.app")

DEFAULT_WEB_CONFIG_PATH = Path(__file__).parent.parent / "web_config.jsonc"
STAGING_DIRNAME = "staging"
SETTING_ID = "setting-{key}"
SOURCE_URL = "https://github.com/Beherith/ultrasongs"

# Settings rendered in the main form instead of the collapsed settings section.
FORM_SETTING_KEYS = frozenset({"whisper_language"})


@dataclass(frozen=True)
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8030
    web_dir: str = "./web_jobs"
    job_retention_days: float = 7.0
    max_upload_mb: float = 2048.0
    poll_interval_s: float = 1.0


def load_web_config(path: str | Path | None = None) -> WebConfig:
    """Load web_config.jsonc, falling back to defaults for missing/invalid keys."""
    cfg_path = Path(path) if path else DEFAULT_WEB_CONFIG_PATH
    data: dict = load_jsonc(cfg_path) if cfg_path.exists() else {}

    def pick(name: str, typ, default):
        value = data.get(name, default)
        try:
            return typ(value)
        except (TypeError, ValueError):
            logger.warning(f"Invalid web config value for {name}: {value!r} - using default")
            return default

    return WebConfig(
        host=pick("host", str, "127.0.0.1"),
        port=pick("port", int, 8030),
        web_dir=pick("web_dir", str, "./web_jobs"),
        job_retention_days=pick("job_retention_days", float, 7.0),
        max_upload_mb=pick("max_upload_mb", float, 2048.0),
        poll_interval_s=pick("poll_interval_s", float, 1.0),
    )


# ── pure helpers (also used by tests) ────────────────────────────────────────


def parse_setting_value(meta: settings_meta.SettingMeta, raw) -> object:
    """Convert a raw form value into the typed config value."""
    if meta.kind == "checkbox":
        return bool(raw)
    if meta.kind == "number":
        value = int(float(raw))
        if meta.min_value is not None and value < meta.min_value:
            raise ValueError(f"{meta.key} must be at least {meta.min_value:g}")
        if meta.max_value is not None and value > meta.max_value:
            raise ValueError(f"{meta.key} must be at most {meta.max_value:g}")
        return value
    if meta.kind == "float":
        value = float(raw)
        if meta.min_value is not None and value < meta.min_value:
            raise ValueError(f"{meta.key} must be at least {meta.min_value:g}")
        if meta.max_value is not None and value > meta.max_value:
            raise ValueError(f"{meta.key} must be at most {meta.max_value:g}")
        return value
    return str(raw)


def build_config_overrides(values: dict[str, object]) -> dict:
    """Map raw form values onto config keys per SETTINGS metadata."""
    overrides: dict = {}
    for key, raw in values.items():
        meta = settings_meta.SETTINGS_BY_KEY.get(key)
        if meta is None:
            continue
        overrides[key] = parse_setting_value(meta, raw)
    return overrides


def validate_process_inputs(upload_info, title: str, artist: str, lyrics: str) -> str | None:
    """Return an error message, or None if the inputs are acceptable."""
    if not upload_info:
        return "Please upload an audio or video file first."
    if not (title or "").strip():
        return "Please enter a song title."
    if not (artist or "").strip():
        return "Please enter an artist name."
    if not (lyrics or "").strip():
        return "Please paste the lyrics."
    return None


def lyrics_prefill(lyrics: str, current_title: str, current_artist: str) -> tuple[str, str]:
    """Prefill title/artist from pasted Ultrastar headers (only when empty)."""
    if not (lyrics or "").strip():
        return current_title, current_artist
    _, meta = prepare_lyrics_input(lyrics)
    if meta is None:
        return current_title, current_artist
    title = current_title if (current_title or "").strip() else (meta.title or "")
    artist = current_artist if (current_artist or "").strip() else (meta.artist or "")
    return title, artist


LYRICS_UPLOAD_EXTENSIONS = {".txt"}
LYRICS_MAX_BYTES = 10 * 1024 * 1024

COVER_MAX_BYTES = 10 * 1024 * 1024


def split_filename_artist_title(filename: str) -> tuple[str, str]:
    """Parse an 'artist - title' file name, splitting on the first dash."""
    stem = Path(filename).stem
    for sep in (" - ", "-"):
        if sep not in stem:
            continue
        artist, _, title = stem.partition(sep)
        artist, title = artist.strip(), title.strip()
        if artist and title:
            return artist, title
    return "", ""


# ── layout ───────────────────────────────────────────────────────────────────


def _setting_control(meta: settings_meta.SettingMeta, value) -> html.Div:
    cid = SETTING_ID.format(key=meta.key)
    if meta.kind == "select":
        options = [{"label": "(auto-detect)" if v == "" else v, "value": v} for v in meta.choices]
        control = dcc.Dropdown(id=cid, options=options, value=value, clearable=False)
    elif meta.kind == "checkbox":
        control = dcc.Input(
            id=cid, type="checkbox", value=bool(value),
            style={"width": "18px", "height": "18px", "margin": "4px 0"},
        )
    elif meta.kind == "number":
        control = dcc.Input(
            id=cid, type="number", value=value,
            min=meta.min_value, max=meta.max_value, step=meta.step,
            style=CSS["input"],
        )
    elif meta.kind == "float":
        control = dcc.Input(
            id=cid, type="number", value=value,
            min=meta.min_value, max=meta.max_value, step=meta.step,
            style=CSS["input"],
        )
    else:
        control = dcc.Input(id=cid, type="text", value=value, style=CSS["input"])
    label_text = meta.key
    if meta.kind == "select" and value == "":
        label_text = "(auto-detect)"
    return html.Div(
        [
            html.Div(
                [html.Label(label_text, htmlFor=cid, style=CSS["setting-label"]), control],
                style=CSS["setting-row"],
            ),
            html.Small(meta.help, style=CSS["setting-help"]),
        ]
    )


def build_settings_section(config: Config) -> html.Details:
    """Collapsible, grouped settings section (one control per config key)."""
    groups: "OrderedDict[str, list[html.Div]]" = OrderedDict()
    for meta in settings_meta.SETTINGS:
        if meta.key in FORM_SETTING_KEYS:
            continue
        groups.setdefault(meta.group, []).append(_setting_control(meta, getattr(config, meta.key)))
    children = []
    for group_name, controls in groups.items():
        children.append(html.H4(group_name, style=CSS["group-title"]))
        children.extend(controls)
    return html.Details(
        [html.Summary("Settings (all pipeline options, default collapsed)", style=CSS["summary"]),
         html.Div(children, style=CSS["settings-body"])],
        style=CSS["details"],
    )


CSS = {
    "page": {
        "maxWidth": "880px", "margin": "0 auto", "padding": "24px 16px",
        "fontFamily": "system-ui, sans-serif", "color": "#e5e7eb",
    },
    "card": {
        "background": "#1f2430", "borderRadius": "10px", "padding": "20px",
        "marginBottom": "18px", "border": "1px solid #2c3345",
    },
    "label": {"display": "block", "marginBottom": "6px", "fontSize": "14px", "color": "#9ca3af"},
    "input": {
        "background": "#11141b", "color": "#e5e7eb",
        "border": "1px solid #3b4358", "borderRadius": "6px",
        "padding": "6px 8px", "fontSize": "14px",
        "width": "100%", "boxSizing": "border-box",
    },
    "row": {"display": "flex", "gap": "16px"},
    "col": {"flex": "1"},
    "upload": {
        "border": "2px dashed #3b4358", "borderRadius": "8px", "padding": "18px",
        "textAlign": "center", "color": "#9ca3af", "marginBottom": "14px",
    },
    "group-title": {"margin": "14px 0 6px", "color": "#7dd3fc", "fontSize": "15px"},
    "summary": {"cursor": "pointer", "color": "#7dd3fc", "fontSize": "15px", "userSelect": "none"},
    "details": {"marginBottom": "18px"},
    "settings-body": {"paddingLeft": "8px"},
    "setting-row": {"display": "flex", "alignItems": "center", "gap": "10px", "marginBottom": "2px"},
    "setting-label": {"width": "230px", "fontSize": "13px", "color": "#d1d5db", "flexShrink": 0},
    "setting-help": {"display": "block", "fontSize": "12px", "color": "#6b7280",
                     "margin": "0 0 8px 240px"},
    "pill": {
        "padding": "3px 12px", "borderRadius": "999px", "fontSize": "13px",
        "fontWeight": 600, "background": "#3b4358",
    },
    "log": {
        "background": "#11141b", "borderRadius": "6px", "padding": "10px",
        "fontFamily": "ui-monospace, monospace", "fontSize": "12px",
        "maxHeight": "320px", "overflowY": "auto", "whiteSpace": "pre-wrap",
        "color": "#a5b4c4",
    },
    "error": {"color": "#f87171", "background": "#2a1a1e", "borderRadius": "6px",
              "padding": "10px", "marginBottom": "10px"},
    "ok": {"color": "#34d399", "background": "#12251c", "borderRadius": "6px",
           "padding": "10px", "marginBottom": "10px"},
    "btn": {"marginRight": "8px"},
}

PILL_COLORS = {
    "queued": ("Queued", "#eab308"),
    "running": ("Running", "#38bdf8"),
    "succeeded": ("Succeeded", "#34d399"),
    "failed": ("Failed", "#f87171"),
}


def build_layout(web_cfg: WebConfig, pipeline_config: Config) -> html.Div:
    song_form = html.Div(
        [
            html.Div([
                html.Label("Convert any mp3 or video file to Ultrastar format. The uploaded files should be complete and of good quality. You can download MP3's off of YouTube with "),
                html.A("https://cnvmp3.com/", href="https://cnvmp3.com/", style={"color": "#7dd3fc"}),
            ]),

            dcc.Upload(
                id="upload",
                children=html.Div(
                    ["Drag & drop an audio/video file here, or ",
                     html.A("browse", style={"color": "#7dd3fc"})],
                    style=CSS["upload"],
                ),
                multiple=False,
            ),
            html.Div(id="upload-info"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Artist *", style=CSS["label"]),
                            dcc.Input(id="artist", type="text", placeholder="Artist name",
                                      style=CSS["input"]),
                        ],
                        style=CSS["col"],
                    ),
                    html.Div(
                        [
                            html.Label("Title *", style=CSS["label"]),
                            dcc.Input(id="title", type="text", placeholder="Song title",
                                      style=CSS["input"]),
                        ],
                        style=CSS["col"],
                    ),
                ],
                style=CSS["row"],
            ),
            html.Div([
                html.Div([
                    html.Label("Lyrics *",
                               style={**CSS["label"], "display": "inline",
                                      "marginRight": "12px", "marginBottom": "0"}),
                    dcc.Upload(
                        id="lyrics-upload",
                        children=html.Span(
                            "or upload a lyrics .txt file",
                            style={"color": "#7dd3fc", "fontSize": "13px",
                                   "textDecoration": "underline", "cursor": "pointer"},
                        ),
                        style={"display": "inline-block", "marginBottom": "10px"},
                        multiple=False,
                    ),
                ]),
                html.Div(id="lyrics-info"),
                dcc.Textarea(id="lyrics", rows=14,
                             placeholder="Paste the song lyrics here (plain text or an Ultrastar .txt). The pasted lyrics are the gold standard, and should be complete and accurate.",
                             style={**CSS["input"],
                                     "fontFamily": "ui-monospace, monospace",
                                     "fontSize": "13px"}),
            ]),
            html.Div([
                html.Div([
                    html.Label("Cover art (optional)",
                               style={**CSS["label"], "display": "inline",
                                      "marginRight": "12px", "marginBottom": "0"}),
                    dcc.Upload(
                        id="cover-upload",
                        children=html.Span(
                            "or upload a JPEG cover image",
                            style={"color": "#7dd3fc", "fontSize": "13px",
                                   "textDecoration": "underline", "cursor": "pointer"},
                        ),
                        style={"display": "inline-block", "marginBottom": "10px"},
                        multiple=False,
                    ),
                ]),
                html.Div(id="cover-info"),
            ]),
            html.Div(
                [_setting_control(
                    settings_meta.SETTINGS_BY_KEY["whisper_language"],
                    pipeline_config.whisper_language,
                )],
                style={"marginBottom": "14px"},
            ),
            html.Div([
                html.Button("Process", id="process-btn", n_clicks=0,
                            style={"background": "#2563eb", "color": "white",
                                   "border": "0", "borderRadius": "6px",
                                   "padding": "10px 22px", "fontSize": "15px",
                                   "fontWeight": 600, "cursor": "pointer",
                                   "marginRight": "10px"}),
                html.Button("Reset settings to file defaults", id="reset-btn",
                            style={"background": "#3b4358", "color": "#e5e7eb",
                                   "border": "0", "borderRadius": "6px",
                                   "padding": "10px 16px", "cursor": "pointer"}),
            ]),
            html.Div(id="form-error", style={"marginTop": "10px"}),
        ],
        style=CSS["card"],
    )

    job_card = html.Div(
        [
            html.Div(
                [
                    html.Span("Idle", id="job-status", style=CSS["pill"]),
                    html.Span(id="job-elapsed", style={"fontSize": "13px", "color": "#9ca3af"}),
                    html.Span(id="job-title", style={"fontSize": "13px", "color": "#9ca3af"}),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "12px"},
            ),
            html.Div(id="job-banner", style={"marginTop": "10px"}),
            html.Pre(id="job-log", style={**CSS["log"], "display": "none", "marginTop": "10px"}),
            html.Div(id="job-result", style={"marginTop": "12px"}),
        ],
        style=CSS["card"],
    )

    page = html.Div(
        [
            song_form,
            build_settings_section(pipeline_config),
            job_card,
            html.A("Source on GitHub", href=SOURCE_URL, target="_blank", rel="noopener",
                   style={"fontSize": "13px", "color": "#7dd3fc", "textDecoration": "none"}),
        ],
        style=CSS["page"],
    )

    return html.Div(
        [
            page,
            dcc.Store(id="upload-store"),
            dcc.Store(id="cover-store"),
            dcc.Store(id="active-job-store"),
            dcc.Interval(id="poll-interval", interval=web_cfg.poll_interval_s * 1000),
        ],
        style={"background": "#0f1117", "minHeight": "100vh"},
    )


# ── app factory ──────────────────────────────────────────────────────────────


def create_app(web_cfg: WebConfig, pipeline_config: Config, manager: JobManager,
               password: str | None) -> dash.Dash:
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.title = "Ultrasongs"
    app.layout = build_layout(web_cfg, pipeline_config)
    auth.install_auth(app.server, password)

    @app.server.route("/download/<job_id>/<path:filename>")
    def download(job_id: str, filename: str):
        job = manager.job(job_id)
        if job is None:
            abort(404)
        base = job.output_dir.resolve()
        target = (base / filename).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            abort(404)
        return send_file(target, as_attachment=True, download_name=target.name)

    @app.server.route("/stage-upload", methods=["POST"])
    def stage_upload():
        f = flask_request.files.get("file")
        if f is None or not f.filename:
            return jsonify({"error": "no file"}), 400
        name = Path(f.filename).name
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
            return jsonify({"error": f"Unsupported file type: '{suffix or name}'"}), 400
        data = f.read()
        max_bytes = manager.max_upload_bytes
        if len(data) > max_bytes:
            return jsonify({"error": f"Upload too large (limit {max_bytes // (1024 * 1024)} MB)"}), 400
        staging = manager.web_dir / STAGING_DIRNAME
        staging.mkdir(parents=True, exist_ok=True)
        staged = staging / f"{uuid.uuid4().hex[:12]}{suffix}"
        staged.write_bytes(data)
        return jsonify({"path": str(staged), "name": name, "size": len(data)})

    _register_callbacks(app, web_cfg, pipeline_config, manager)
    return app


# ── callbacks ────────────────────────────────────────────────────────────────


def _register_callbacks(app: dash.Dash, web_cfg: WebConfig, pipeline_config: Config,
                        manager: JobManager) -> None:
    from dash import Input, Output, State, no_update
    from dash.exceptions import PreventUpdate

    setting_ids = [SETTING_ID.format(key=m.key) for m in settings_meta.SETTINGS]

    @app.callback(
        Output("upload-store", "data"),
        Output("upload-info", "children"),
        Output("title", "value"),
        Output("artist", "value"),
        Input("upload", "contents"),
        State("upload", "filename"),
        State("title", "value"),
        State("artist", "value"),
    )
    def upload_picked(contents: str | None, filename: str | None,
                      title: str | None, artist: str | None):
        if not contents or not filename:
            return None, None, no_update, no_update
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
            return no_update, html.Div(
                f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}",
                style=CSS["error"]), no_update, no_update
        b64 = contents.split(",", 1)[1]
        data = base64.b64decode(b64)
        if len(data) > manager.max_upload_bytes:
            return no_update, html.Div(
                f"File too large ({len(data) // (1024 * 1024)} MB, "
                f"limit {manager.max_upload_bytes // (1024 * 1024)} MB)", style=CSS["error"]), no_update, no_update
        staging = manager.web_dir / STAGING_DIRNAME
        staging.mkdir(parents=True, exist_ok=True)
        staged = staging / f"{uuid.uuid4().hex[:12]}{suffix}"
        staged.write_bytes(data)
        name_artist, name_title = split_filename_artist_title(filename)
        if not (title or "").strip() and name_title:
            title = name_title
        if not (artist or "").strip() and name_artist:
            artist = name_artist
        size_mb = len(data) / (1024 * 1024)
        return {"path": str(staged), "name": filename, "size": len(data)}, html.Div(
            [html.Span(f"{filename} ({size_mb:.1f} MB)",
                       style={"color": "#34d399", "fontSize": "14px"}),
              html.A("remove", href="#", id="upload-clear",
                     style={"marginLeft": "10px", "color": "#9ca3af",
                            "fontSize": "13px", "textDecoration": "underline"})],
        ), title, artist

    @app.callback(
        Output("lyrics", "value"),
        Output("lyrics-info", "children"),
        Input("lyrics-upload", "contents"),
        State("lyrics-upload", "filename"),
        prevent_initial_call=True,
    )
    def lyrics_file_picked(contents: str | None, filename: str | None):
        if not contents or not filename:
            raise PreventUpdate
        suffix = Path(filename).suffix.lower()
        if suffix not in LYRICS_UPLOAD_EXTENSIONS:
            return no_update, html.Div(
                f"Unsupported lyrics file type '{suffix}'. Use a .txt file.",
                style=CSS["error"])
        data = base64.b64decode(contents.split(",", 1)[1])
        if len(data) > LYRICS_MAX_BYTES:
            return no_update, html.Div(
                f"Lyrics file too large (limit {LYRICS_MAX_BYTES // (1024 * 1024)} MB).",
                style=CSS["error"])
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("windows-1252", errors="replace")
        return text, html.Div(
            f"Loaded {filename} ({len(data) // 1024} KB)",
            style={"color": "#34d399", "fontSize": "13px", "marginBottom": "8px"},
        )

    @app.callback(
        Output("upload-store", "data", allow_duplicate=True),
        Input("upload-clear", "n_clicks"),
        State("upload-store", "data"),
        prevent_initial_call=True,
    )
    def upload_cleared(n_clicks, store):
        if not n_clicks or store is None:
            raise PreventUpdate
        _remove_staged(store)
        return None

    @app.callback(
        Output("cover-store", "data"),
        Output("cover-info", "children"),
        Input("cover-upload", "contents"),
        State("cover-upload", "filename"),
        prevent_initial_call=True,
    )
    def cover_picked(contents: str | None, filename: str | None):
        if not contents or not filename:
            raise PreventUpdate
        suffix = Path(filename).suffix.lower()
        if suffix not in COVER_UPLOAD_EXTENSIONS:
            return no_update, html.Div(
                f"Unsupported cover image type '{suffix or filename}'. Use a .jpg or .jpeg file.",
                style=CSS["error"])
        data = base64.b64decode(contents.split(",", 1)[1])
        if len(data) > COVER_MAX_BYTES:
            return no_update, html.Div(
                f"Cover image too large (limit {COVER_MAX_BYTES // (1024 * 1024)} MB).",
                style=CSS["error"])
        staging = manager.web_dir / STAGING_DIRNAME
        staging.mkdir(parents=True, exist_ok=True)
        staged = staging / f"{uuid.uuid4().hex[:12]}{suffix}"
        staged.write_bytes(data)
        return {"path": str(staged), "name": filename, "size": len(data)}, html.Div(
            [html.Span(f"{filename} ({len(data) // 1024} KB)",
                       style={"color": "#34d399", "fontSize": "13px"}),
              html.A("remove", href="#", id="cover-clear",
                     style={"marginLeft": "10px", "color": "#9ca3af",
                            "fontSize": "13px", "textDecoration": "underline"})],
        )

    @app.callback(
        Output("cover-store", "data", allow_duplicate=True),
        Input("cover-clear", "n_clicks"),
        State("cover-store", "data"),
        prevent_initial_call=True,
    )
    def cover_cleared(n_clicks, store):
        if not n_clicks or store is None:
            raise PreventUpdate
        _remove_staged(store)
        return None

    @app.callback(
        Output("title", "value", allow_duplicate=True),
        Output("artist", "value", allow_duplicate=True),
        Input("lyrics", "value"),
        State("title", "value"),
        State("artist", "value"),
        prevent_initial_call=True,
    )
    def lyrics_pasted(lyrics, title, artist):
        new_title, new_artist = lyrics_prefill(lyrics or "", title or "", artist or "")
        if new_title == (title or "") and new_artist == (artist or ""):
            raise PreventUpdate
        return new_title, new_artist

    @app.callback(
        [Output(f"setting-{m.key}", "value") for m in settings_meta.SETTINGS],
        Input("reset-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_settings(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        return [
            bool(getattr(pipeline_config, m.key)) if m.kind == "checkbox"
            else getattr(pipeline_config, m.key)
            for m in settings_meta.SETTINGS
        ]

    @app.callback(
        Output("active-job-store", "data"),
        Output("form-error", "children"),
        Input("process-btn", "n_clicks"),
        State("upload-store", "data"),
        State("title", "value"),
        State("artist", "value"),
        State("lyrics", "value"),
        State("cover-store", "data"),
        *[State(SETTING_ID.format(key=m.key), "value") for m in settings_meta.SETTINGS],
        prevent_initial_call=True,
    )
    def process_click(n_clicks, upload_info, title, artist, lyrics, cover_info, *setting_values):
        if not n_clicks:
            raise PreventUpdate
        title = (title or "").strip()
        artist = (artist or "").strip()
        lyrics = lyrics or ""
        error = validate_process_inputs(upload_info, title, artist, lyrics)
        if error:
            return no_update, html.Div(error, style=CSS["error"])

        values = {m.key: v for m, v in zip(settings_meta.SETTINGS, setting_values)}
        try:
            overrides = build_config_overrides(values)
            job_config = config_from_dict({**dataclasses.asdict(pipeline_config), **overrides},
                                          pipeline_config._config_path)
        except (ValueError, TypeError) as exc:
            return no_update, html.Div(f"Invalid setting value: {exc}", style=CSS["error"])

        lyrics_text, _ = prepare_lyrics_input(lyrics)
        base_dir = Path(upload_info["path"])
        cover_bytes = None
        cover_name = None
        if cover_info:
            cover_file = Path(cover_info["path"])
            if not cover_file.is_file():
                return no_update, html.Div(
                    "Cover image is missing. Please upload it again.", style=CSS["error"])
            cover_bytes = cover_file.read_bytes()
            cover_name = cover_info["name"]
        try:
            upload_bytes = base_dir.read_bytes()
            req = ProcessRequest(
                title=title,
                artist=artist,
                lyrics_text=lyrics_text,
                input_path=base_dir,
                video_path=None,
                cover_path=None,
                config=job_config,
                output_dir=Path(job_config.output_dir),
                include_intermediates=True,
            )
            job = manager.submit(req, upload_bytes, Path(upload_info["name"]).name,
                                 cover_bytes=cover_bytes, cover_name=cover_name)
        except ValueError as exc:
            return no_update, html.Div(str(exc), style=CSS["error"])
        finally:
            _remove_staged(upload_info)
            _remove_staged(cover_info)
        logger.info(f"Job {job.id} submitted from web: {title!r}")
        return job.id, None

    @app.callback(
        Output("job-status", "children"),
        Output("job-status", "style"),
        Output("job-elapsed", "children"),
        Output("job-title", "children"),
        Output("job-banner", "children"),
        Output("job-log", "children"),
        Output("job-log", "style"),
        Output("job-result", "children"),
        Output("process-btn", "disabled"),
        Output("process-btn", "children"),
        Output("active-job-store", "data", allow_duplicate=True),
        Input("poll-interval", "n_intervals"),
        State("active-job-store", "data"),
        prevent_initial_call=True,
    )
    def poll(n_intervals, active_job_id):
        if not active_job_id:
            raise PreventUpdate
        snap = manager.snapshot(active_job_id)
        if snap is None:
            return (None, CSS["pill"], "", "", None,
                    "", {**CSS["log"], "display": "none"}, None,
                    False, "Process", None)
        label, color = PILL_COLORS.get(snap["status"], (snap["status"], "#e5e7eb"))
        position = snap["position"]
        if position is not None:
            label = f"Queued (#{position + 1})"
        elapsed = snap["elapsed_s"]
        mm, ss = divmod(int(elapsed), 60)
        elapsed_text = f"{mm:02d}:{ss:02d}"
        log_style = CSS["log"]
        if not snap["logs_tail"]:
            log_style = {**log_style, "display": "none"}

        banner_children = None
        if snap["status"] == "failed":
            banner_children = html.Div(snap["error"] or "Job failed", style=CSS["error"])
        result_children = None
        if snap["status"] == "succeeded":
            def download_href(filename: str) -> str:
                return f"/download/{snap['id']}/{quote(snap['run_dir'])}/{quote(filename)}"

            links = []
            if snap["zip_name"]:
                links.append(html.A(
                    "Download ZIP",
                    href=download_href(snap["zip_name"]),
                    style={"background": "#2563eb", "color": "white", "padding": "8px 16px",
                           "borderRadius": "6px", "textDecoration": "none",
                           "fontWeight": 600, "marginRight": "10px"}))
            if snap["editor_name"]:
                links.append(html.A(
                    "Open editor",
                    href=download_href(snap["editor_name"]),
                    style={"background": "#3b4358", "color": "#e5e7eb", "padding": "8px 16px",
                           "borderRadius": "6px", "textDecoration": "none",
                           "marginRight": "10px"}))
            if snap["html_name"]:
                links.append(html.A(
                    "Open preview",
                    href=download_href(snap["html_name"]),
                    style={"background": "#3b4358", "color": "#e5e7eb", "padding": "8px 16px",
                           "borderRadius": "6px", "textDecoration": "none",
                           "marginRight": "10px"}))
            for f in snap["files"]:
                if f["name"] in (snap["zip_name"], snap["html_name"], snap["editor_name"]):
                    continue
                links.append(html.A(
                    f"{f['name']} ({f['size'] // 1024} KB)",
                    href=download_href(f["name"]),
                    style={"display": "inline-block", "margin": "6px 10px 0 0",
                           "color": "#7dd3fc", "fontSize": "13px"}))
            result_children = html.Div([html.Div("Done!", style=CSS["ok"]),
                                        html.Div(links)])

        busy = snap["status"] in ("queued", "running")
        btn_label = "Processing…" if snap["status"] == "running" else (
            f"Queued (#{position + 1})" if position is not None else "Processing…")
        return (label, {**CSS["pill"], "color": color}, elapsed_text,
                f" {snap['title']}", banner_children,
                "\n".join(snap["logs_tail"]), log_style, result_children,
                busy, btn_label, active_job_id)


def _remove_staged(store: dict | None) -> None:
    if not store:
        return
    try:
        Path(store["path"]).unlink(missing_ok=True)
    except OSError:
        pass


# ── entry point ──────────────────────────────────────────────────────────────


def run_server(web_cfg: WebConfig, pipeline_config: Config,
               host: str | None = None, port: int | None = None,
               password: str | None = None) -> int:
    manager = JobManager(
        web_dir=web_cfg.web_dir,
        retention_days=web_cfg.job_retention_days,
        max_upload_mb=web_cfg.max_upload_mb,
    )
    manager.start()
    app = create_app(web_cfg, pipeline_config, manager, password=password)
    host = host or web_cfg.host
    port = port or web_cfg.port
    logger.info(f"Web service serving on http://{host}:{port} (jobs in {web_cfg.web_dir})")
    try:
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
    finally:
        manager.stop()
    return 0

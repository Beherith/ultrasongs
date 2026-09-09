"""Tests for cli/web/auth.py using the Flask test client."""

import pytest
from flask import Flask

from cli.web.auth import install_auth, load_web_password, parse_env_file, resolve_auth


def _app(password: str | None) -> Flask:
    app = Flask(__name__)
    install_auth(app, password)

    @app.route("/")
    def index():
        return "ok"

    return app


class TestAuthGuard:
    def test_anonymous_redirected_to_login(self):
        client = _app("secret").test_client()
        resp = client.get("/")
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_login_page_shown(self):
        client = _app("secret").test_client()
        resp = client.get("/login")
        assert resp.status_code == 200
        assert b"Ultrasongs" in resp.data

    def test_wrong_password_rejected(self):
        client = _app("secret").test_client()
        resp = client.post("/login", data={"password": "wrong"})
        assert resp.status_code == 401
        # Still anonymous
        assert client.get("/").status_code == 302

    def test_right_password_authenticates(self):
        client = _app("secret").test_client()
        resp = client.post("/login", data={"password": "secret"})
        assert resp.status_code == 302
        assert resp.headers["Location"] == "/"
        assert client.get("/").status_code == 200

    def test_logout_clears_session(self):
        client = _app("secret").test_client()
        client.post("/login", data={"password": "secret"})
        assert client.get("/").status_code == 200
        resp = client.get("/logout")
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]
        assert client.get("/").status_code == 302


class TestNoAuth:
    def test_no_password_allows_anonymous(self):
        client = _app(None).test_client()
        assert client.get("/").status_code == 200

    def test_login_redirects_to_index_when_disabled(self):
        client = _app(None).test_client()
        resp = client.get("/login")
        assert resp.status_code == 302
        assert resp.headers["Location"] == "/"

    def test_logout_redirects_to_index_when_disabled(self):
        client = _app(None).test_client()
        resp = client.get("/logout")
        assert resp.status_code == 302
        assert resp.headers["Location"] == "/"


class TestResolveAuth:
    def test_no_password_no_flag_is_error(self):
        password, error = resolve_auth(None, False, "127.0.0.1")
        assert password is None
        assert "ULTRASONGS_WEB_PASSWORD" in (error or "")

    def test_password_env_used(self):
        password, error = resolve_auth("s3cret", False, "0.0.0.0")
        assert password == "s3cret"
        assert error is None

    def test_no_auth_refused_off_localhost(self):
        password, error = resolve_auth(None, True, "0.0.0.0")
        assert password is None
        assert error is not None
        assert "localhost" in error

    def test_no_auth_refused_on_lan_host(self):
        _, error = resolve_auth(None, True, "192.168.1.10")
        assert error is not None

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
    def test_no_auth_allowed_on_localhost(self, host):
        password, error = resolve_auth(None, True, host)
        assert password is None
        assert error is None


class TestParseEnvFile:
    def test_parses_key_value_lines(self, tmp_path):
        p = tmp_path / ".env.local"
        p.write_text("# comment\n\nULTRASONGS_WEB_PASSWORD=secret\nOTHER=value\n", encoding="utf-8")
        values = parse_env_file(p)
        assert values["ULTRASONGS_WEB_PASSWORD"] == "secret"
        assert values["OTHER"] == "value"

    def test_strips_quotes(self, tmp_path):
        p = tmp_path / ".env.local"
        p.write_text('ULTRASONGS_WEB_PASSWORD="quoted pass"\n', encoding="utf-8")
        assert parse_env_file(p)["ULTRASONGS_WEB_PASSWORD"] == "quoted pass"

    def test_ignores_malformed_lines(self, tmp_path):
        p = tmp_path / ".env.local"
        p.write_text("no-equals-here\n=orphan\n#ULTRASONGS_WEB_PASSWORD=commented\n", encoding="utf-8")
        assert parse_env_file(p) == {}


class TestLoadWebPassword:
    def test_env_var_wins(self, tmp_path, monkeypatch):
        p = tmp_path / ".env.local"
        p.write_text("ULTRASONGS_WEB_PASSWORD=file-pass\n", encoding="utf-8")
        monkeypatch.setenv("ULTRASONGS_WEB_PASSWORD", "env-pass")
        assert load_web_password(env_file=p) == "env-pass"

    def test_falls_back_to_env_file(self, tmp_path, monkeypatch):
        p = tmp_path / ".env.local"
        p.write_text("ULTRASONGS_WEB_PASSWORD=file-pass\n", encoding="utf-8")
        monkeypatch.delenv("ULTRASONGS_WEB_PASSWORD", raising=False)
        assert load_web_password(env_file=p) == "file-pass"

    def test_missing_file_and_env_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ULTRASONGS_WEB_PASSWORD", raising=False)
        assert load_web_password(env_file=tmp_path / "nope.env.local") is None

    def test_empty_password_returns_none(self, tmp_path, monkeypatch):
        p = tmp_path / ".env.local"
        p.write_text("ULTRASONGS_WEB_PASSWORD=\n", encoding="utf-8")
        monkeypatch.delenv("ULTRASONGS_WEB_PASSWORD", raising=False)
        assert load_web_password(env_file=p) is None

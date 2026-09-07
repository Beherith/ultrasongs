"""Session authentication for the web service.

Password comes from the ULTRASONGS_WEB_PASSWORD environment variable.
No TLS is provided — bind to localhost or use a reverse proxy for LAN access.
"""

import hmac
import secrets
from typing import TYPE_CHECKING

from flask import redirect, request, session

if TYPE_CHECKING:  # pragma: no cover
    from flask import Flask

AUTH_SESSION_KEY = "us_auth"

LOGIN_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Ultrasongs - Login</title>
<style>
body {{ font-family: system-ui, sans-serif; display: flex; align-items: center;
       justify-content: center; min-height: 100vh; margin: 0; background: #1e1e2e; }}
form {{ background: #2a2a3c; padding: 2rem 2.5rem; border-radius: 8px; width: 260px; }}
h1 {{ font-size: 1.2rem; margin: 0 0 1rem; color: #cdd6f4; }}
input[type=password] {{ width: 100%; box-sizing: border-box; padding: .5rem;
       border: 1px solid #45475a; border-radius: 4px; background: #1e1e2e;
       color: #cdd6f4; margin-bottom: 1rem; }}
button {{ width: 100%; padding: .5rem; border: 0; border-radius: 4px;
       background: #89b4fa; color: #1e1e2e; font-weight: 600; cursor: pointer; }}
.error {{ color: #f38ba8; font-size: .85rem; margin-bottom: .75rem; }}
</style>
</head>
<body>
<form method="post" action="/login">
<h1>Ultrasongs</h1>
{error}
<input type="password" name="password" placeholder="Password" autofocus>
<button type="submit">Sign in</button>
</form>
</body>
</html>
"""


def resolve_auth(password_env: str | None, no_auth: bool, host: str) -> tuple[str | None, str | None]:
    """Decide the effective auth mode. Returns (password, error).

    password None + error None means auth disabled (localhost only).
    """
    if no_auth:
        if host not in ("127.0.0.1", "localhost"):
            return None, "--no-auth is only allowed when binding to 127.0.0.1 or localhost"
        return None, None
    if password_env:
        return password_env, None
    return None, "Set the ULTRASONGS_WEB_PASSWORD environment variable (or use --no-auth on localhost)"


def install_auth(server: "Flask", password: str | None) -> None:
    """Install /login, /logout routes and a before_request session guard.

    password=None disables authentication (caller must have validated the host).
    """
    server.secret_key = secrets.token_hex(32)
    server.permanent_session_lifetime = 3600 * 24 * 7

    def _is_public(path: str) -> bool:
        if path in ("/login", "/logout"):
            return True
        # Dash internal endpoints and assets
        return path.startswith(("/_dash", "/assets/", "/favicon", "/_favicon"))

    if password is not None:

        @server.before_request
        def _auth_guard():
            if _is_public(request.path):
                return None
            if session.get(AUTH_SESSION_KEY):
                return None
            return redirect(f"/login?next={request.path}")

    @server.route("/login", methods=["GET", "POST"])
    def login():
        if password is None:
            return redirect("/")
        error = ""
        if request.method == "POST":
            provided = request.form.get("password", "")
            if hmac.compare_digest(provided.encode("utf-8"), password.encode("utf-8")):
                session[AUTH_SESSION_KEY] = True
                session.permanent = True
                return redirect("/")
            error = '<div class="error">Wrong password</div>'
        return LOGIN_PAGE.format(error=error), 200 if request.method == "GET" else 401

    @server.route("/logout")
    def logout():
        session.clear()
        return redirect("/login" if password is not None else "/")

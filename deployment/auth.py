"""Server-side login identity for the pinned Gradio request contract."""
import hashlib
import importlib.metadata

from starlette.requests import Request


def login_owner(request):
    """Return a login-specific digest; never accept body/header ownership claims."""
    if not isinstance(request, Request):
        return None
    try:
        if importlib.metadata.version("gradio") != "6.30.0":
            return None
    except importlib.metadata.PackageNotFoundError:
        return None
    app = request.app
    if not getattr(app, "auth", None) or getattr(app, "auth_dependency", None) is not None:
        return None
    registry, cookie_id = getattr(app, "tokens", None), getattr(app, "cookie_id", None)
    if not isinstance(registry, dict) or not isinstance(cookie_id, str):
        return None
    token = request.cookies.get(f"access-token-{cookie_id}") or request.cookies.get(
        f"access-token-unsecure-{cookie_id}")
    username = registry.get(token) if isinstance(token, str) else None
    if not isinstance(username, str) or not username:
        return None
    return owner_digest(username, token)


def owner_digest(username, token):
    return hashlib.sha256((username + "\0" + token).encode("utf-8")).hexdigest()

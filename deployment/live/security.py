"""Pinned Gradio 6.30 login bridge and bounds before component JSON parsing.

The component-server route supplies a Starlette Request, not gr.Request. Resolve
its cookie through the *server's* login-token registry, as Gradio's /user route
does. Never use headers, body usernames, or body session_hash as identity.
Unsupported auth modes or a missing transport guard disable live capture.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
from contextlib import asynccontextmanager
from threading import BoundedSemaphore

from starlette.requests import Request

from .buffer import LiveBufferError

MAX_LIVE_BODY_BYTES = 3 * 1024 * 1024
MAX_LIVE_REQUESTS = 4
BODY_TIMEOUT_S = 5.0


def live_app_kwargs(service):
    """Use for launch or mount; mounted Gradio apps compose router lifespans directly."""
    from starlette.middleware import Middleware

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            service.close()

    return {"middleware": [Middleware(LiveTransportMiddleware, service=service)], "lifespan": lifespan}


def request_owner(request) -> str:
    if not isinstance(request, Request) or not request.scope.get("aval.live_bounded"):
        raise LiveBufferError("LIVE_AUTHENTICATION_REQUIRED")
    try:
        supported = importlib.metadata.version("gradio") == "6.30.0"
    except importlib.metadata.PackageNotFoundError:
        supported = False
    if not supported:
        raise LiveBufferError("LIVE_AUTHENTICATION_REQUIRED")
    app = request.app
    # This bridge deliberately supports the pinned built-in Gradio login only.
    if not getattr(app, "auth", None) or getattr(app, "auth_dependency", None) is not None:
        raise LiveBufferError("LIVE_AUTHENTICATION_REQUIRED")
    registry = getattr(app, "tokens", None)
    cookie_id = getattr(app, "cookie_id", None)
    if not isinstance(registry, dict) or not isinstance(cookie_id, str):
        raise LiveBufferError("LIVE_AUTHENTICATION_REQUIRED")
    token = request.cookies.get(f"access-token-{cookie_id}") or request.cookies.get(
        f"access-token-unsecure-{cookie_id}")
    username = registry.get(token) if isinstance(token, str) else None
    if not isinstance(username, str) or not username:
        raise LiveBufferError("LIVE_AUTHENTICATION_REQUIRED")
    # Different logins for the same account also remain separate. No cookie is
    # retained in a capture session, diagnostic, response, or log.
    return hashlib.sha256((username + "\0" + token).encode("utf-8")).hexdigest()


class LiveTransportMiddleware:
    """Bound body bytes, body time, and in-flight work before Gradio parses JSON.

    A semaphore covers parsing *and* handler execution; cancelled receive tasks
    never leave a background reader. HTTP requests do not wait for a free slot.
    Shutdown closes the service's expiry worker and clears retained media.
    """

    def __init__(self, app, service, *, body_limit=MAX_LIVE_BODY_BYTES,
                 body_timeout=BODY_TIMEOUT_S, max_requests=MAX_LIVE_REQUESTS):
        self.app = app
        self.service = service
        self.body_limit = min(body_limit, MAX_LIVE_BODY_BYTES)
        self.body_timeout = min(body_timeout, BODY_TIMEOUT_S)
        self.slots = BoundedSemaphore(min(max_requests, MAX_LIVE_REQUESTS))

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            async def lifecycle():
                message = await receive()
                if message["type"] == "lifespan.shutdown":
                    self.service.close()
                return message
            return await self.app(scope, lifecycle, send)
        if scope["type"] != "http" or not scope.get("path", "").rstrip("/").endswith(
                "/gradio_api/component_server"):
            return await self.app(scope, receive, send)

        async def reject(status, code):
            body = json.dumps({"status": "REJECTED", "error_code": code,
                               "predictions": None}).encode("ascii")
            await send({"type": "http.response.start", "status": status,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"cache-control", b"no-store")]})
            await send({"type": "http.response.body", "body": body})

        # Preserve Gradio's authentication-before-body boundary. Anonymous or
        # unsupported callers must not occupy a body-reader slot at all.
        scope["aval.live_bounded"] = True
        try:
            request_owner(Request(scope))
        except LiveBufferError:
            return await reject(401, "LIVE_AUTHENTICATION_REQUIRED")
        if not self.slots.acquire(blocking=False):
            return await reject(429, "LIVE_RESOURCE_BUSY")
        try:
            headers = {k.lower(): v for k, v in scope.get("headers", [])}
            if headers.get(b"content-encoding", b"identity").lower() != b"identity":
                return await reject(415, "LIVE_UNSUPPORTED_ENCODING")
            # No multipart/file route is needed by this widget.
            if headers.get(b"content-type", b"").split(b";")[0].strip().lower() != b"application/json":
                return await reject(415, "LIVE_JSON_REQUIRED")
            length = headers.get(b"content-length")
            if length is not None:
                try:
                    declared = int(length)
                    if declared < 0:
                        raise ValueError
                except ValueError:
                    return await reject(400, "LIVE_INVALID_BODY")
                if declared > self.body_limit:
                    return await reject(413, "LIVE_BODY_TOO_LARGE")
            parts = bytearray()
            try:
                async with asyncio.timeout(self.body_timeout):
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            return
                        chunk = message.get("body", b"")
                        if len(parts) + len(chunk) > self.body_limit:
                            return await reject(413, "LIVE_BODY_TOO_LARGE")
                        parts.extend(chunk)
                        if not message.get("more_body", False):
                            break
            except TimeoutError:
                return await reject(408, "LIVE_BODY_TIMEOUT")
            body = bytes(parts)
            parts.clear()
            try:
                envelope = json.loads(body)
                # Gradio indexes retained session state before calling a server
                # function. Bound that client claim even though it is never used
                # for ownership, or large keys could accumulate outside our service.
                if (type(envelope) is not dict
                        or type(envelope.get("session_hash")) is not str
                        or not 1 <= len(envelope["session_hash"]) <= 128
                        or not envelope["session_hash"].isascii()
                        or type(envelope.get("component_id")) is not int
                        or not 0 <= envelope["component_id"] < 2**31
                        or type(envelope.get("fn_name")) is not str
                        or not 1 <= len(envelope["fn_name"]) <= 64
                        or not envelope["fn_name"].isascii()
                        or type(envelope.get("data")) is not dict):
                    return await reject(400, "LIVE_INVALID_BODY")
                del envelope
            except (ValueError, UnicodeError, RecursionError):
                return await reject(400, "LIVE_INVALID_BODY")
            delivered = False

            async def bounded_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            async def private_send(message):
                if message["type"] == "http.response.start":
                    message = dict(message)
                    message["headers"] = [(k, v) for k, v in message.get("headers", [])
                                          if k.lower() != b"cache-control"] + [(b"cache-control", b"no-store")]
                await send(message)

            return await self.app(scope, bounded_receive, private_send)
        finally:
            self.slots.release()

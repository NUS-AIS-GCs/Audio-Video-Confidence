"""Bounded single-file transport and login-owned, expiring upload handles.

Never delegate multipart storage or media serving to Gradio's shared cache.
Only authenticated consumers open leases. Public uploads remain admission-only;
the separate internal adapter additionally restricts inference to permitted TRAIN bytes.
"""
from __future__ import annotations

import asyncio
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import secrets
from threading import BoundedSemaphore, Event, RLock, Thread
import time

from filelock import FileLock
from python_multipart.multipart import MultipartParser, parse_options_header
from starlette.requests import Request

from deployment.auth import login_owner, owner_digest

MAX_FILES = 4                 # includes uploads in transit and active verification
MAX_OWNER_FILES = 2
MAX_INFLIGHT = 2
MAX_REQUESTS_PER_MINUTE = 16   # global; no unbounded per-client rate table
UPLOAD_TTL_S = 300.0
RECEIVE_TIMEOUT_S = 30.0
MULTIPART_OVERHEAD = 16 * 1024
MAX_CONTROL_BODY = 16 * 1024
_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}


class UploadError(Exception):
    def __init__(self, code, status=400):
        self.code, self.status = code, status
        super().__init__(code)


@dataclass
class _Entry:
    owner: str
    path: Path
    created: float
    state: str = "receiving"
    filename: str = ""


class UploadStore:
    def __init__(self, root, max_bytes, *, ttl=UPLOAD_TTL_S, interval=1.0, clock=time.monotonic):
        if not 1 <= max_bytes <= 50 * 1024 * 1024 or not 0 < ttl <= UPLOAD_TTL_S:
            raise ValueError("invalid upload limits")
        self.root = Path(root) / "aval-owned-uploads"
        self.max_bytes, self.ttl, self.clock = max_bytes, ttl, clock
        self.interval = min(max(interval, 0.01), 1.0)
        self._entries = {}
        self._requests = deque(maxlen=MAX_REQUESTS_PER_MINUTE)
        self._lock, self._stop = RLock(), Event()
        self._worker, self._directory_lock, self._auth_app = None, None, None
        self._started, self._closed = False, False

    def start(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("UPLOAD_STORE_CLOSED")
            if self._started:
                return
            if self.root.is_symlink():
                raise RuntimeError("UNSAFE_UPLOAD_DIRECTORY")
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.root, 0o700)
            # Verification finalizers run on Gradio worker threads, including
            # the last lease after ASGI shutdown; the process lock is shared.
            guard = FileLock(str(self.root / ".lock"), timeout=0, thread_local=False)
            guard.acquire()  # exclusive process, including cleanup after an unclean exit
            try:
                for path in self.root.iterdir():
                    if path.name == ".lock":
                        continue
                    if path.is_symlink() or not path.is_file() or not re.fullmatch(
                            r"payload-[0-9a-f]{64}\.(part|mp4|mov|mkv|webm)", path.name):
                        raise RuntimeError("UNSAFE_UPLOAD_DIRECTORY")
                    path.unlink()
            except BaseException:
                guard.release()
                raise
            self._directory_lock, self._started = guard, True
            self._worker = Thread(target=self._expire_loop, name="aval-upload-expiry", daemon=True)
            self._worker.start()

    def bind_auth(self, app):
        with self._lock:
            if self._auth_app is not None and self._auth_app is not app:
                raise UploadError("UPLOAD_UNAVAILABLE", 503)
            self._auth_app = app

    def _owner_active(self, owner):
        if self._auth_app is None:
            return True
        registry = getattr(self._auth_app, "tokens", {})
        if not isinstance(registry, dict) or not getattr(self._auth_app, "auth", None):
            return False
        return any(isinstance(name, str) and isinstance(token, str)
                   and owner_digest(name, token) == owner for token, name in registry.copy().items())

    def _delete(self, handle):
        entry = self._entries[handle]
        try:
            entry.path.unlink(missing_ok=True)
        except OSError:
            # Retain the reservation and retry; never silently reuse its disk budget.
            entry.state = "deleting"
            return
        del self._entries[handle]

    def expire(self):
        with self._lock:
            now = self.clock()
            for handle, entry in list(self._entries.items()):
                if entry.state in {"ready", "deleting"} and (
                        self._closed or entry.state == "deleting" or now-entry.created >= self.ttl
                        or not self._owner_active(entry.owner)):
                    self._delete(handle)

    def _expire_loop(self):
        while not self._stop.wait(self.interval):
            self.expire()

    def reserve(self, owner):
        with self._lock:
            if not self._started or self._closed:
                raise UploadError("UPLOAD_UNAVAILABLE", 503)
            self.expire()
            now = self.clock()
            while self._requests and now-self._requests[0] >= 60:
                self._requests.popleft()
            if len(self._requests) >= MAX_REQUESTS_PER_MINUTE:
                raise UploadError("UPLOAD_RATE_LIMIT", 429)
            self._requests.append(now)
            if len(self._entries) >= MAX_FILES or sum(e.owner == owner for e in self._entries.values()) >= MAX_OWNER_FILES:
                raise UploadError("UPLOAD_CAPACITY_REACHED", 429)
            handle = secrets.token_hex(32)
            self._entries[handle] = _Entry(owner, self.root/f"payload-{handle}.part", now)
            return handle, self._entries[handle]

    def publish(self, handle, filename):
        with self._lock:
            entry = self._entries[handle]
            if self._closed or not self._owner_active(entry.owner):
                raise UploadError("UPLOAD_NOT_FOUND", 404)
            final = entry.path.with_suffix(Path(filename).suffix.lower())
            entry.path.replace(final)
            entry.path, entry.filename, entry.state = final, filename, "ready"

    def discard(self, handle):
        with self._lock:
            if handle in self._entries:
                self._delete(handle)
            self._release_if_closed()

    @contextmanager
    def lease(self, handle, owner):
        with self._lock:
            self.expire()
            entry = self._entries.get(handle) if isinstance(handle, str) else None
            if (self._closed or entry is None or entry.owner != owner or entry.state != "ready"
                    or not self._owner_active(owner)):
                raise UploadError("UPLOAD_NOT_FOUND", 404)
            entry.state = "verifying"
        try:
            yield entry.path, entry.filename
        finally:
            self.discard(handle)

    def _release_if_closed(self):
        if self._closed and not self._entries and self._directory_lock is not None:
            self._directory_lock.release()
            self._directory_lock = None

    def close(self):
        with self._lock:
            self._closed = True
            self.expire()  # active receiving/verifying leases retain their files until finally
            self._release_if_closed()
            self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2)


class _SingleFileParser:
    """Streaming, strict form grammar: exactly one `files` part, no fields."""
    def __init__(self, content_type, entry, max_bytes):
        kind, options = parse_options_header(content_type)
        boundary = options.get(b"boundary", b"")
        if kind != b"multipart/form-data" or not 1 <= len(boundary) <= 200 or not boundary.isascii():
            raise UploadError("UPLOAD_MULTIPART_REQUIRED", 415)
        self.entry, self.max_bytes = entry, max_bytes
        self.parts, self.size, self.header_bytes = 0, 0, 0
        self.headers, self.field, self.value = {}, bytearray(), bytearray()
        self.file, self.filename, self.complete = None, None, False
        self.parser = MultipartParser(boundary, {name: getattr(self, name) for name in (
            "on_part_begin", "on_header_field", "on_header_value", "on_header_end",
            "on_headers_finished", "on_part_data", "on_part_end", "on_end")})

    def on_part_begin(self):
        self.parts += 1
        if self.parts != 1:
            raise UploadError("UPLOAD_SINGLE_FILE_REQUIRED")

    def _header(self, target, data, start, end):
        self.header_bytes += end-start
        if self.header_bytes > 8192:
            raise UploadError("UPLOAD_INVALID_MULTIPART")
        target.extend(data[start:end])

    def on_header_field(self, data, start, end):
        self._header(self.field, data, start, end)

    def on_header_value(self, data, start, end):
        self._header(self.value, data, start, end)

    def on_header_end(self):
        key = bytes(self.field).lower()
        if key in self.headers or len(self.headers) >= 8:
            raise UploadError("UPLOAD_INVALID_MULTIPART")
        self.headers[key] = bytes(self.value)
        self.field.clear(); self.value.clear()

    def on_headers_finished(self):
        kind, options = parse_options_header(self.headers.get(b"content-disposition", b""))
        filename = options.get(b"filename", b"")
        if kind != b"form-data" or options.get(b"name") != b"files" or not 1 <= len(filename) <= 255:
            raise UploadError("UPLOAD_SINGLE_FILE_REQUIRED")
        try:
            filename = filename.decode("utf-8")
        except UnicodeError:
            raise UploadError("UNSUPPORTED_FILENAME") from None
        if any(c in filename for c in ("/", "\\", "\0")) or Path(filename).suffix.lower() not in _EXTENSIONS:
            raise UploadError("UNSUPPORTED_FILENAME")
        self.filename = filename
        self.file = self.entry.path.open("xb")
        os.chmod(self.entry.path, 0o600)

    def on_part_data(self, data, start, end):
        self.size += end-start
        if self.size > self.max_bytes:
            raise UploadError("UPLOAD_TOO_LARGE", 413)
        self.file.write(data[start:end])

    def on_part_end(self):
        if self.file is not None:
            self.file.close()

    def on_end(self):
        self.complete = True

    def finish(self):
        self.parser.finalize()
        if not self.complete or self.parts != 1 or not self.size or self.filename is None:
            raise UploadError("UPLOAD_INVALID_MULTIPART")

    def close(self):
        if self.file is not None:
            self.file.close()


class UploadTransportMiddleware:
    def __init__(self, app, store, *, timeout=RECEIVE_TIMEOUT_S):
        self.app, self.store = app, store
        self.timeout = min(timeout, RECEIVE_TIMEOUT_S)
        self.slots = BoundedSemaphore(MAX_INFLIGHT)
        self.control_slots = BoundedSemaphore(4)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "").rstrip("/")
        # Disable every built-in media download/proxy/stream variant; preview is local.
        media_route = any(marker in path for marker in (
            "/gradio_api/file=", "/gradio_api/file/", "/gradio_api/proxy=", "/gradio_api/stream/"))
        upload_route = path.endswith("/gradio_api/upload")
        unused_route = path.endswith("/gradio_api/process_recording")
        control_route = scope["method"] == "POST" and (
            path.endswith(("/gradio_api/queue/join", "/gradio_api/cancel", "/gradio_api/reset"))
            or any(p in path for p in ("/gradio_api/run/", "/gradio_api/api/", "/gradio_api/call/")))

        async def reply(status, body):
            data = json.dumps(body).encode("utf-8")
            await send({"type": "http.response.start", "status": status, "headers": [
                (b"content-type", b"application/json"), (b"cache-control", b"no-store")]})
            await send({"type": "http.response.body", "body": data})

        async def error(code, status=400):
            return await reply(status, {"status": "REJECTED", "predictions": None, "error_code": code})

        scope["aval.upload_guard"] = self.store
        if media_route or unused_route or upload_route or control_route:
            request = Request(scope)
            owner = login_owner(request)
            if owner is None:
                return await error("UPLOAD_AUTHENTICATION_REQUIRED", 401)
            if media_route or unused_route:
                return await error("MEDIA_ROUTE_DISABLED", 403)
            if control_route:
                # Gradio traverses FileData BEFORE component.preprocess(), even
                # for scalar inputs. Reject file objects before that traversal.
                if not self.control_slots.acquire(blocking=False):
                    return await error("UPLOAD_RESOURCE_BUSY", 429)
                try:
                    if request.headers.get("content-type", "").split(";")[0].lower() != "application/json":
                        raise UploadError("UPLOAD_INVALID_REQUEST")
                    if request.headers.get("content-encoding", "identity").lower() != "identity":
                        raise UploadError("UPLOAD_UNSUPPORTED_ENCODING", 415)
                    length = request.headers.get("content-length")
                    if length is not None:
                        try:
                            declared = int(length)
                        except ValueError:
                            raise UploadError("UPLOAD_INVALID_LENGTH") from None
                        if declared < 0 or declared > MAX_CONTROL_BODY:
                            raise UploadError("UPLOAD_TOO_LARGE", 413)
                    body = bytearray()
                    async with asyncio.timeout(min(5.0, self.timeout)):
                        while True:
                            message = await receive()
                            if message["type"] == "http.disconnect":
                                return
                            chunk = message.get("body", b"")
                            if len(body)+len(chunk) > MAX_CONTROL_BODY:
                                raise UploadError("UPLOAD_TOO_LARGE", 413)
                            body.extend(chunk)
                            if not message.get("more_body", False):
                                break
                    envelope = json.loads(body)
                    if (type(envelope) is not dict or type(envelope.get("session_hash")) is not str
                            or not 1 <= len(envelope["session_hash"]) <= 128
                            or not envelope["session_hash"].isascii()
                            or envelope.get("event_data") is not None):
                        raise UploadError("UPLOAD_INVALID_REQUEST")
                    data = envelope.get("data", [])
                    if type(data) is not list or len(data) > 1 or any(
                            v is not None and (type(v) is not str or len(v) > 128) for v in data):
                        raise UploadError("UPLOAD_INVALID_REQUEST")
                    delivered = False

                    async def bounded_receive():
                        nonlocal delivered
                        if not delivered:
                            delivered = True
                            return {"type": "http.request", "body": bytes(body), "more_body": False}
                        return await receive()

                    return await self.app(scope, bounded_receive, send)
                except TimeoutError:
                    return await error("UPLOAD_TIMEOUT", 408)
                except UploadError as exc:
                    return await error(exc.code, exc.status)
                except (ValueError, UnicodeError, RecursionError):
                    return await error("UPLOAD_INVALID_REQUEST")
                finally:
                    self.control_slots.release()
            if scope["method"] != "POST":
                return await error("UPLOAD_METHOD_NOT_ALLOWED", 405)
            self.store.bind_auth(request.app)
            headers = request.headers
            if headers.get("content-encoding", "identity").lower() != "identity":
                return await error("UPLOAD_UNSUPPORTED_ENCODING", 415)
            limit = self.store.max_bytes + MULTIPART_OVERHEAD
            length = headers.get("content-length")
            try:
                if length is not None and (int(length) < 0 or int(length) > limit):
                    return await error("UPLOAD_TOO_LARGE", 413)
            except ValueError:
                return await error("UPLOAD_INVALID_LENGTH")
            if not self.slots.acquire(blocking=False):
                return await error("UPLOAD_RESOURCE_BUSY", 429)
            handle, parser, published = None, None, False
            try:
                handle, entry = self.store.reserve(owner)
                parser = _SingleFileParser(headers.get("content-type", ""), entry, self.store.max_bytes)
                total = 0
                async with asyncio.timeout(self.timeout):
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            return
                        body = message.get("body", b"")
                        total += len(body)
                        if total > limit:
                            raise UploadError("UPLOAD_TOO_LARGE", 413)
                        for offset in range(0, len(body), 65536):
                            parser.parser.write(body[offset:offset+65536])
                        if not message.get("more_body", False):
                            break
                parser.finish()
                self.store.publish(handle, parser.filename)
                # Opaque handle, never a filesystem path. Compatible with FileData.path.
                await reply(200, [handle])
                published = True
            except TimeoutError:
                await error("UPLOAD_TIMEOUT", 408)
            except UploadError as exc:
                await error(exc.code, exc.status)
            except Exception:
                await error("UPLOAD_VALIDATION_FAILED")
            finally:
                if parser is not None:
                    parser.close()
                if handle is not None and not published:
                    self.store.discard(handle)
                self.slots.release()
            return
        return await self.app(scope, receive, send)

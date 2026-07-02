"""HTTP/SSE server for MiniBot agent runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from queue import Empty, Queue
import threading
from typing import Any

from pydantic import BaseModel, Field
from starlette.requests import Request

from .bootstrap import MiniBotRuntime, build_runtime
from .config import Config, load_env
from .runtime.agent_session import RunCancelled
from .runtime.approvals import ApprovalBroker
from .runtime.events import RuntimeEvent
from .runtime.events import RuntimeEventEmitter
from .run_log import make_run_id
from .session import MessageEvent, Session
from .session.models import utc_now


_SENTINEL = object()
_WEB_DIR = Path(__file__).resolve().parent / "web"


class RunRequest(BaseModel):
    input: str = Field(min_length=1)
    session_id: str | None = None
    mode: str = "default"


class ApprovalResolution(BaseModel):
    approved: bool


class SessionCreateRequest(BaseModel):
    session_id: str | None = None
    title: str | None = None


class SessionUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


def _json_default(value: Any) -> str:
    return str(value)


def _to_sse(event: RuntimeEvent) -> str:
    data = json.dumps(event.to_dict(), ensure_ascii=False, default=_json_default)
    return f"id: {event.seq}\nevent: {event.type}\ndata: {data}\n\n"


def _error_event(
    *,
    run_id: str | None = None,
    session_id: str | None,
    exc: Exception,
) -> RuntimeEvent:
    emitter = RuntimeEventEmitter(
        run_id=run_id or make_run_id(),
        session_id=session_id or "",
    )
    return emitter.emit(
        "run.failed",
        {
            "session_id": session_id,
            "error_type": type(exc).__name__,
            "message": str(exc),
        },
    )


def _last_event_seq(raw: str | None) -> int | None:
    if not raw:
        return None
    candidate = raw.rsplit(":", 1)[-1]
    try:
        return int(candidate)
    except ValueError:
        return None


class RunEventStore:
    """In-process event backlog and fan-out for active HTTP subscribers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, list[RuntimeEvent]] = {}
        self._subscribers: dict[str, list[Queue[RuntimeEvent | object]]] = {}
        self._status: dict[str, str] = {}

    def create(self, run_id: str) -> None:
        with self._lock:
            self._events.setdefault(run_id, [])
            self._status[run_id] = "running"

    def is_terminal(self, run_id: str) -> bool:
        with self._lock:
            return self._status.get(run_id) in {
                "completed",
                "failed",
                "cancelled",
            }

    def append(self, event: RuntimeEvent) -> None:
        terminal_status = _terminal_status(event.type)
        with self._lock:
            self._events.setdefault(event.run_id, []).append(event)
            self._status.setdefault(event.run_id, "running")
            if terminal_status is not None:
                self._status[event.run_id] = terminal_status
            subscribers = list(self._subscribers.get(event.run_id, []))
            if terminal_status is not None:
                self._subscribers.pop(event.run_id, None)

            for queue in subscribers:
                queue.put(event)
                if terminal_status is not None:
                    queue.put(_SENTINEL)

    def subscribe(
        self,
        run_id: str,
        *,
        last_seq: int | None = None,
    ) -> Queue[RuntimeEvent | object] | None:
        queue: Queue[RuntimeEvent | object] = Queue()
        with self._lock:
            if run_id not in self._events:
                return None

            backlog = self._events[run_id]
            if last_seq is not None:
                backlog = [event for event in backlog if event.seq > last_seq]
            for event in backlog:
                queue.put(event)

            status = self._status.get(run_id)
            if status in {"completed", "failed", "cancelled"}:
                queue.put(_SENTINEL)
            else:
                self._subscribers.setdefault(run_id, []).append(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: Queue[RuntimeEvent | object]) -> None:
        with self._lock:
            subscribers = self._subscribers.get(run_id)
            if not subscribers:
                return
            self._subscribers[run_id] = [
                subscriber for subscriber in subscribers if subscriber is not queue
            ]
            if not self._subscribers[run_id]:
                self._subscribers.pop(run_id, None)


def _terminal_status(event_type: str) -> str | None:
    if event_type == "run.completed":
        return "completed"
    if event_type == "run.failed":
        return "failed"
    if event_type == "run.cancelled":
        return "cancelled"
    return None


def _session_payload(session: Session) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "title": session.title,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "message_count": session.message_count,
        "turn_count": session.turn_count() if session.messages else None,
    }


def _message_payload(message: MessageEvent) -> dict[str, Any]:
    return message.to_dict()


def _load_or_create_current_session(runtime: MiniBotRuntime) -> Session:
    session, _ = runtime.manager.startup_session()
    return session


def _resolve_session(runtime: MiniBotRuntime, session_id: str) -> Session | None:
    if session_id == "current":
        return _load_or_create_current_session(runtime)
    return runtime.manager.load(session_id)


def create_app(runtime: MiniBotRuntime):
    from fastapi import Body, FastAPI, HTTPException
    from starlette.responses import FileResponse, JSONResponse, StreamingResponse
    from starlette.staticfiles import StaticFiles

    app = FastAPI(title="MiniBot", version="0.1.0")
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")
    event_store = RunEventStore()

    def start_run(request: RunRequest) -> str:
        run_id = make_run_id()
        event_store.create(run_id)

        def sink(event: RuntimeEvent) -> None:
            event_store.append(event)

        def worker() -> None:
            try:
                runtime.agent_session.prompt(
                    request.session_id,
                    request.input,
                    run_id=run_id,
                    event_handler=sink,
                )
            except RunCancelled:
                pass
            except Exception as exc:
                if not event_store.is_terminal(run_id):
                    event_store.append(
                        _error_event(
                            run_id=run_id,
                            session_id=request.session_id,
                            exc=exc,
                        )
                    )

        threading.Thread(target=worker, daemon=True).start()
        return run_id

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_WEB_DIR / "index.html", media_type="text/html")

    @app.get("/sessions")
    def list_sessions() -> JSONResponse:
        return JSONResponse(
            {
                "current_session_id": runtime.manager.get_current_session_id(),
                "sessions": [
                    _session_payload(session)
                    for session in runtime.manager.list_sessions()
                ],
            }
        )

    @app.get("/sessions/current")
    def current_session() -> JSONResponse:
        session = _load_or_create_current_session(runtime)
        return JSONResponse({"session": _session_payload(session)})

    @app.post("/sessions")
    def create_session_endpoint(
        request: SessionCreateRequest = Body(default_factory=SessionCreateRequest),
    ) -> JSONResponse:
        try:
            session = runtime.manager.create_current_session(
                session_id=request.session_id,
                title=request.title,
            )
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(
            {"session": _session_payload(session)}, status_code=201
        )

    @app.patch("/sessions/{session_id}")
    def update_session(
        session_id: str,
        request: SessionUpdateRequest = Body(...),
    ) -> JSONResponse:
        if session_id == "current":
            raise HTTPException(
                status_code=400, detail="cannot update the 'current' alias"
            )
        session = runtime.manager.load(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        session.title = request.title.strip()
        session.updated_at = utc_now()
        runtime.manager.update_metadata(session)
        return JSONResponse({"session": _session_payload(session)})

    @app.delete("/sessions/{session_id}")
    def delete_session_endpoint(session_id: str) -> JSONResponse:
        if session_id == "current":
            raise HTTPException(
                status_code=400, detail="cannot delete the 'current' alias"
            )
        removed = runtime.manager.delete_session(session_id)
        if not removed:
            raise HTTPException(status_code=404, detail="session not found")
        runtime.budget.forget(session_id)
        return JSONResponse({"ok": True, "session_id": session_id})

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str) -> JSONResponse:
        session = _resolve_session(runtime, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        return JSONResponse({"session": _session_payload(session)})

    @app.get("/sessions/{session_id}/messages")
    def get_session_messages(session_id: str) -> JSONResponse:
        session = _resolve_session(runtime, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        return JSONResponse(
            {
                "session": _session_payload(session),
                "messages": [_message_payload(message) for message in session.messages],
            }
        )

    @app.post("/runs")
    def create_run(request: RunRequest = Body(...)) -> JSONResponse:
        run_id = start_run(request)
        return JSONResponse(
            {
                "run_id": run_id,
                "session_id": request.session_id,
                "mode": request.mode,
                "status": "running",
            },
            status_code=202,
        )

    @app.get("/runs/{run_id}/events")
    def stream_run_events(run_id: str, request: Request) -> StreamingResponse:
        queue = event_store.subscribe(
            run_id,
            last_seq=_last_event_seq(request.headers.get("last-event-id")),
        )
        if queue is None:
            raise HTTPException(status_code=404, detail="run not found")

        def events():
            try:
                while True:
                    try:
                        item = queue.get(timeout=10)
                    except Empty:
                        yield ": keepalive\n\n"
                        continue
                    if item is _SENTINEL:
                        break
                    assert isinstance(item, RuntimeEvent)
                    yield _to_sse(item)
            finally:
                event_store.unsubscribe(run_id, queue)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/runs/{run_id}/cancel")
    def cancel_run_endpoint(run_id: str) -> JSONResponse:
        cancelled = runtime.agent_session.abort(run_id)
        if not cancelled:
            raise HTTPException(
                status_code=404, detail="run not found or already finished"
            )
        return JSONResponse({"ok": True, "run_id": run_id})

    @app.post("/runs/{run_id}/approvals/{approval_id}")
    def resolve_approval(
        run_id: str,
        approval_id: str,
        request: ApprovalResolution = Body(...),
    ) -> JSONResponse:
        if runtime.approval_broker is None:
            raise HTTPException(
                status_code=500,
                detail="approval broker is not configured",
            )
        matched = runtime.approval_broker.resolve(
            run_id=run_id,
            approval_id=approval_id,
            approved=request.approved,
        )
        return JSONResponse({"ok": True, "matched_pending": matched})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MiniBot SSE server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    load_env()
    config = Config.from_env()
    broker = ApprovalBroker()
    try:
        runtime = build_runtime(
            config=config,
            workspace=Path.cwd(),
            approval_broker=broker,
            approval_handler=broker.wait,
        )
    except RuntimeError as exc:
        print(f"配置错误: {exc}")
        return
    app = create_app(runtime)

    import uvicorn

    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()

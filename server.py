"""HTTP/SSE server for MiniBot agent runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from queue import Empty, Queue
import threading
from typing import Any

from pydantic import BaseModel, Field

from .bootstrap import MiniBotRuntime, build_runtime
from .config import Config, load_env
from .runtime import ApprovalBroker, RuntimeEvent
from .runtime.events import RuntimeEventEmitter
from .run_log import make_run_id
from .session import MessageEvent, Session


_SENTINEL = object()
_WEB_DIR = Path(__file__).resolve().parent / "web"


class RunStreamRequest(BaseModel):
    input: str = Field(min_length=1)
    session_id: str | None = None


class ApprovalResolution(BaseModel):
    approved: bool


def _json_default(value: Any) -> str:
    return str(value)


def _to_sse(event: RuntimeEvent) -> str:
    data = json.dumps(event.to_dict(), ensure_ascii=False, default=_json_default)
    return f"id: {event.id}\nevent: {event.type}\ndata: {data}\n\n"


def _error_event(
    *,
    session_id: str | None,
    exc: Exception,
) -> RuntimeEvent:
    emitter = RuntimeEventEmitter(
        run_id=make_run_id(),
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
    session = runtime.manager.load_current_session()
    if session is not None:
        return session
    latest = runtime.manager.latest_session(prefer_non_empty=True)
    if latest is not None:
        runtime.manager.set_current_session(latest.session_id)
        return latest
    session = runtime.manager.create_session()
    runtime.manager.set_current_session(session.session_id)
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

    @app.post("/runs/stream")
    def stream_run(request: RunStreamRequest = Body(...)) -> StreamingResponse:
        queue: Queue[RuntimeEvent | object] = Queue()
        emitted_types: list[str] = []

        def sink(event: RuntimeEvent) -> None:
            emitted_types.append(event.type)
            queue.put(event)

        def worker() -> None:
            try:
                runtime.controller.run_turn(
                    session_id=request.session_id,
                    user_input=request.input,
                    event_handler=sink,
                )
            except Exception as exc:
                if "run.failed" not in emitted_types:
                    queue.put(_error_event(session_id=request.session_id, exc=exc))
            finally:
                queue.put(_SENTINEL)

        threading.Thread(target=worker, daemon=True).start()

        def events():
            while True:
                try:
                    item = queue.get(timeout=15)
                except Empty:
                    yield ": keepalive\n\n"
                    continue
                if item is _SENTINEL:
                    break
                assert isinstance(item, RuntimeEvent)
                yield _to_sse(item)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/runs/{run_id}/approvals/{approval_id}")
    def resolve_approval(
        run_id: str,
        approval_id: str,
        request: ApprovalResolution = Body(...),
    ) -> JSONResponse:
        matched = runtime.controller.resolve_approval(
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
            approval_handler=None if config.auto_approve else broker.wait,
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

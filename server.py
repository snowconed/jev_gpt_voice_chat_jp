"""Local-only voice prototype. The API key is read solely in the server process."""
from __future__ import annotations
import argparse
import asyncio
import json
import os
from pathlib import Path
import secrets

from aiohttp import web, WSMsgType, ClientConnectionError
from chat_core import ChatCore
from chat_config import load_settings
from transcription_client import MODEL
from speech_client import MODEL as SPEECH_MODEL, VOICE
from jev_runtime import JevSpeechRuntime
from microphone_forwarder import MicrophoneForwarder
from filler_cache import load_filler_clips

ROOT = Path(__file__).resolve().parent


def create_app(data_dir=None, port=8783, core=None, turn_mode="jev", service_tier=None, config_path=None,
               tts="gpt", filler_cache_dir=None):
    if turn_mode != "jev":
        raise ValueError("Only Jev turn-taking is supported")
    if service_tier not in {None, "default", "ultrafast"}:
        raise ValueError("Unknown service tier")
    if tts != "gpt":
        raise ValueError("Only GPT speech is supported")
    speech_metadata = {
        "voice": VOICE,
        "model": SPEECH_MODEL,
    }
    filler_supported = True
    filler_catalog = {}
    session_fillers = {}
    if core is None:
        from dataclasses import replace
        settings = load_settings(config_path)
        if service_tier is not None:
            settings = replace(settings, service_tier=service_tier)
        core = ChatCore(Path(data_dir or ROOT / "data_jev"), settings=settings)
    service_tier = service_tier or getattr(getattr(core, "settings", None), "service_tier", "ultrafast")
    token = secrets.token_urlsafe(32)
    allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
    active = {}
    connections = {}
    runtime_finalizers = set()
    used = set()
    connection_lock = asyncio.Lock()
    closing = False
    shutdown_started = asyncio.Event()

    async def until_shutdown(operation):
        """Stop a reservation/handshake before it can create a paid session.

        Cancelling a to_thread wait cannot kill its underlying synchronous
        operation. These reservation operations only read/create local state.
        """
        if closing:
            operation.close()
            raise web.HTTPServiceUnavailable(text="Server is stopping")
        task = asyncio.create_task(operation)
        stopped = asyncio.create_task(shutdown_started.wait())
        try:
            done, _ = await asyncio.wait((task, stopped), return_when=asyncio.FIRST_COMPLETED)
            if closing or stopped in done:
                raise web.HTTPServiceUnavailable(text="Server is stopping")
            return await task
        finally:
            for pending in (task, stopped):
                if not pending.done():
                    pending.cancel()
            await asyncio.gather(task, stopped, return_exceptions=True)

    @web.middleware
    async def local_only(request, handler):
        if request.host not in allowed:
            raise web.HTTPForbidden(text="Local access only")
        origin = request.headers.get("Origin")
        if origin and origin not in {"http://"+h for h in allowed}:
            raise web.HTTPForbidden(text="Origin rejected")
        if request.method != "GET" and not secrets.compare_digest(request.headers.get("X-CSRF-Token", ""), token):
            raise web.HTTPForbidden(text="Token required")
        if closing and request.path.startswith("/api/"):
            raise web.HTTPServiceUnavailable(text="Server is stopping")
        response = await handler(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "microphone=(self)"
        response.headers["Content-Security-Policy"] = "default-src 'self'; connect-src 'self' ws://127.0.0.1:* ws://localhost:*; style-src 'self'; script-src 'self'; frame-ancestors 'none'"
        return response

    app = web.Application(middlewares=[local_only], client_max_size=16384)

    async def config(request):
        from jev_client import has_key
        roles = [{"id": "chat", "label": "雑談"}]
        return web.json_response({"csrf_token":token,"roles":roles,"voice":speech_metadata["voice"],
                                  "model":MODEL,
                                  "speech_model":speech_metadata["model"],
                                  "speech_voice_label":speech_metadata.get("label", speech_metadata["voice"]),
                                  "tts_backend":tts, "version":"1.0.0",
                                  "filler_supported":filler_supported,
                                  "filler_clips":[{**clip.metadata(), "url":f"/api/fillers/{clip.id}.pcm"}
                                                  for clip in filler_catalog.values()],
                                  "chat_model":getattr(getattr(core,"settings",None),"model","gpt-6.1-sol"),
                                  "turn_mode":turn_mode, "service_tier":service_tier,
                                  "interruption_policy":"semantic",
                                  "turn_key_configured":has_key(),
                                  "key_configured":bool(os.environ.get("OPENAI_API_KEY"))})

    async def create(request):
        if not os.environ.get("OPENAI_API_KEY"):
            return web.json_response({"error":"OPENAI_API_KEY が設定されていません。"},status=503)
        from jev_client import has_key
        if not has_key():
            return web.json_response({"error":"TYPESAFE_API_KEY または JEV_API_KEY が設定されていません。"},status=503)
        try:
            body = await request.json()
            if not isinstance(body, dict) or body.get("role_id", "chat") != "chat":
                raise ValueError("invalid role")
        except (ValueError, TypeError):
            return web.json_response({"error":"役割の指定を確認してください。"},status=400)
        if type(body.get("filler_enabled", False)) is not bool:
            return web.json_response({"error":"フィラー設定を確認してください。"}, status=400)
        try:
            state = await until_shutdown(asyncio.to_thread(core.create, body.get("role_id", "chat")))
        except (ValueError, KeyError):
            return web.json_response({"error":"役割の指定を確認してください。"},status=400)
        selected = bool(body.get("filler_enabled", False) and filler_supported and filler_catalog)
        # Bind the pre-conversation choice to this session; WebSocket messages
        # cannot turn it on later or switch to another voice's cached clips.
        session_fillers[state["id"]] = selected
        return web.json_response({**state, "filler_enabled":selected})

    async def filler_audio(request):
        clip = filler_catalog.get(request.match_info["clip_id"])
        if clip is None:
            raise web.HTTPNotFound()
        return web.Response(body=clip.pcm, content_type="application/octet-stream")

    async def voice(request):
        if not secrets.compare_digest(request.query.get("token", ""),token):
            raise web.HTTPForbidden()
        sid = request.match_info["sid"]
        async with connection_lock:
            if closing:
                raise web.HTTPServiceUnavailable(text="Server is stopping")
            if sid in used or active:
                raise web.HTTPConflict(text="Start a new session; only one voice connection is supported.")
            # Reserve before core.view/prepare yields; simultaneous handshakes
            # must not create two paid transcription sessions.
            active[sid] = None
        ws = web.WebSocketResponse(heartbeat=20,max_msg_size=65536)
        connection = {"ws": ws, "runtime": None, "close_task": None, "done": asyncio.Event()}
        connections[sid] = connection
        runtime = None
        receiver = starter = completed = microphone_task = progress_task = None
        microphone = None

        async def send_json(data):
            if not ws.closed:
                try:
                    await ws.send_json(data)
                except (ClientConnectionError, ConnectionResetError):
                    # The browser can close its transport just before ws.closed
                    # changes. Cleanup must still finalize the voice session.
                    pass
        async def send_bytes(data):
            if not ws.closed:
                try:
                    await ws.send_bytes(data)
                except (ClientConnectionError, ConnectionResetError):
                    pass
        async def receive_browser():
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    microphone.push(msg.data)
                elif msg.type == WSMsgType.TEXT:
                    event = json.loads(msg.data)
                    if not isinstance(event, dict):
                        raise ValueError("event must be an object")
                    typ = event.get("type")
                    if typ == "stop":
                        break
                    if typ == "played":
                        if not isinstance(event.get("id"), str):
                            raise ValueError("invalid playback id")
                        await runtime.played(event["id"])
                    elif typ == "filler_skipped":
                        if not isinstance(event.get("id"), str) or len(event["id"]) > 100:
                            raise ValueError("invalid filler id")
                        if callable(getattr(runtime, "filler_skipped", None)):
                            await runtime.filler_skipped(event["id"])
                    elif typ == "text":
                        if not isinstance(event.get("text"), str):
                            raise ValueError("invalid input text")
                        input_id = event.get("client_input_id")
                        if input_id is not None and (not isinstance(input_id, str) or not 1 <= len(input_id) <= 100):
                            raise ValueError("invalid input id")
                        await runtime.enqueue(event["text"])
                        if input_id is not None:
                            await send_json({"type":"input_accepted", "client_input_id":input_id})
                    elif typ == "user_activity":
                        # Older tabs report microphone energy as activity. Only
                        # recognized speech may interrupt; PCM still arrives on
                        # the binary path and explicit stop/text remain active.
                        continue
                    elif typ == "audio_diagnostic":
                        allowed_stages = {"context_state", "playback_received", "playback_buffered", "playing",
                                          "rendered", "rendered_silent", "processor_error", "capture_stats", "buffering",
                                          "send_backpressure", "played", "local_interrupt", "connected", "client_error"}
                        stage = event.get("stage")
                        if stage in allowed_stages:
                            diagnostic = {"stage": stage}
                            if isinstance(event.get("id"), str):
                                diagnostic["id"] = event["id"][:100]
                            context_state = event.get("context_state")
                            if context_state in {"running", "suspended", "closed", "interrupted"}:
                                diagnostic["context_state"] = context_state
                            for field in ("sample_rate", "received_bytes", "capture_ms", "elapsed_ms",
                                          "capture_rms", "capture_peak", "buffered_bytes"):
                                value = event.get(field)
                                if type(value) in (int, float) and 0 <= value <= 100000000:
                                    diagnostic[field] = value
                            runtime.log("browser_audio", **diagnostic)
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break

        try:
            try:
                await until_shutdown(asyncio.to_thread(core.view, sid))
            except (KeyError, ValueError, FileNotFoundError):
                raise web.HTTPNotFound()
            await until_shutdown(ws.prepare(request))
            speech_options = {}
            if session_fillers.get(sid, False):
                speech_options["filler_clips"] = [clip.metadata() for clip in filler_catalog.values()]
            runtime = JevSpeechRuntime(core,sid,send_json,send_bytes,
                                       Path(data_dir or ROOT / "data_jev"),
                                       **speech_options)
            runtime.log("voice_configuration", turn_mode=turn_mode, service_tier=service_tier, tts_backend=tts)
            runtime.log("filler_configuration", enabled=session_fillers.pop(sid, False))
            active[sid] = runtime
            connection["runtime"] = runtime
            used.add(sid)
            # UI observation runs independently of audio transport, core session
            # locks and API response times. It never participates in turn decisions.
            if callable(getattr(runtime, "processing_snapshot", None)):
                from processing_state import ProcessingStatePublisher
                publisher = ProcessingStatePublisher(runtime, core, sid, turn_mode, send_json)
                progress_task = asyncio.create_task(publisher.run(), name="voice_processing_state")
            microphone = MicrophoneForwarder(runtime)
            microphone_task = asyncio.create_task(microphone.run())
            # Receive stop/disconnect even while the remote ASR handshake runs.
            # Runtime ignores microphone frames until that handshake is ready.
            receiver = asyncio.create_task(receive_browser())
            async def start_runtime():
                if closing:
                    runtime.finished.set()
                    return
                await runtime.start()
            starter = asyncio.create_task(start_runtime())
            completed = asyncio.create_task(runtime.finished.wait())
            done, _ = await asyncio.wait((receiver, starter, completed, microphone_task), return_when=asyncio.FIRST_COMPLETED)
            if starter in done:
                await starter
                done, _ = await asyncio.wait((receiver, completed, microphone_task), return_when=asyncio.FIRST_COMPLETED)
            if microphone_task in done:
                await microphone_task
            if receiver in done:
                await receiver
            if runtime.failure_message:
                await send_json({"type":"error", "message":runtime.failure_message})
        except web.HTTPException:
            if closing and ws.prepared:
                return ws
            raise
        except Exception as exc:
            if runtime:
                runtime.log("connection_error",error_type=type(exc).__name__)
            if ws.prepared:
                try:
                    await send_json({"type":"error","message":runtime.failure_message if runtime and runtime.failure_message else
                                     "音声接続に失敗しました。APIの利用状況と通信を確認し、新しい会話を開始してください。"})
                except Exception:
                    pass
            else:
                raise web.HTTPInternalServerError(text="Unable to start voice session") from None
        finally:
            for task in (receiver, starter, completed, microphone_task, progress_task):
                if task and not task.done():
                    task.cancel()
            await asyncio.gather(*(t for t in (receiver, starter, completed, microphone_task, progress_task) if t), return_exceptions=True)
            try:
                if runtime:
                    if connection["close_task"] is None:
                        connection["close_task"] = asyncio.create_task(runtime.close())
                        runtime_finalizers.add(connection["close_task"])

                        def finalized(task):
                            runtime_finalizers.discard(task)
                            if not task.cancelled():
                                task.exception()
                        connection["close_task"].add_done_callback(finalized)
                    await asyncio.shield(connection["close_task"])
            finally:
                active.pop(sid,None)
                session_fillers.pop(sid, None)
                try:
                    if ws.prepared:
                        try:
                            await send_json({"type":"closed"})
                        finally:
                            await ws.close()
                finally:
                    connection["done"].set()
                    connections.pop(sid, None)
        return ws

    async def static(request):
        filename = request.match_info.get("name", "index.html")
        if filename not in {"index.html","app.js","style.css","pcm-worklet.js"}:
            raise web.HTTPNotFound()
        return web.FileResponse(ROOT / "web" / filename)

    async def shutdown(app):
        nonlocal closing
        # aiohttp waits for handlers before on_cleanup. An open browser socket
        # must be released here, otherwise Ctrl+C waits for shutdown_timeout.
        closing = True
        shutdown_started.set()
        current = list(connections.values())
        for connection in current:
            runtime = connection["runtime"]
            if runtime is not None:
                runtime.finished.set()

        async def stop_browser(connection):
            ws = connection["ws"]
            if not ws.prepared or ws.closed:
                return
            try:
                # Stop microphone capture immediately, without waiting for
                # remote finalization or the core session's lock.
                await asyncio.wait_for(ws.send_json({"type": "closed"}), .5)
            except (TimeoutError, ClientConnectionError, ConnectionResetError, RuntimeError):
                pass
            try:
                await asyncio.wait_for(ws.close(code=1001, message=b"Server shutdown"), 1)
            except (TimeoutError, ClientConnectionError, ConnectionResetError, RuntimeError):
                pass

        await asyncio.gather(*(stop_browser(connection) for connection in current))

    async def cleanup(app):
        # Voice handlers own runtime.close. Await any shielded finalization left
        # after aiohttp cancelled a handler; never run a second close in parallel.
        await asyncio.gather(*list(runtime_finalizers), return_exceptions=True)
        await asyncio.to_thread(core.close)

    async def start_speech(app):
        if filler_supported:
            clips = load_filler_clips(filler_cache_dir if filler_cache_dir is not None else ROOT / "assets" / "fillers",
                                      tts, model=speech_metadata["model"], voice=speech_metadata["voice"])
            filler_catalog.update((clip.id, clip) for clip in clips)
    app.router.add_get("/api/config",config)
    app.router.add_get("/api/fillers/{clip_id:[a-f0-9]{64}}.pcm",filler_audio)
    app.router.add_post("/api/sessions",create)
    app.router.add_get("/api/voice/{sid:[a-f0-9]{32}}",voice)
    app.router.add_get("/",static)
    app.router.add_get("/{name}",static)
    app.on_shutdown.append(shutdown)
    app.on_cleanup.append(cleanup)
    app.on_startup.append(start_speech)
    return app


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Jev voice chat with GPT speech")
    parser.add_argument("--port",type=int,default=None)
    parser.add_argument("--data-dir",type=Path,default=None)
    parser.add_argument("--turn-mode",choices=["jev"],default="jev")
    parser.add_argument("--service-tier",choices=["default", "ultrafast"],default=None)
    parser.add_argument("--config",type=Path,default=ROOT/"config.json")
    parser.add_argument("--tts",choices=["gpt"],default="gpt")
    args=parser.parse_args()
    port = args.port or 8783
    web.run_app(create_app(args.data_dir,port,turn_mode=args.turn_mode,service_tier=args.service_tier,config_path=args.config,
                          tts=args.tts),
                host="127.0.0.1",port=port,access_log=None)

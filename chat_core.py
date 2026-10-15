"""Independent, synchronous casual-chat core; no experience-system imports.

ChatCore(data_dir, settings=None, responder=None) exposes:
  create(role_id='chat') -> public session view (a local greeting; no API call)
  view(sid) -> {id, role_id, phase, messages, busy, error, ...}
  view_if_idle(sid) -> consistent public view, or None while its lock is busy
  pending(sid) -> [{id, text, kind}] in delivery order
  submit(sid, text, request_id) -> public view, serialized per session
  submit_cancellable(sid, text, request_id, cancel_event) -> same, with caller cancellation
  cancel(sid) -> signal the current request without waiting for its session lock
  delivery(sid, message_id, status, heard_text='') -> public view
  close() -> stop accepting work and close the replaceable responder

The responder implements respond(history, settings) -> str and optional close().
An optional respond_cancellable(history, settings, cancel_event) can stop a
superseded request; legacy responders remain supported with late-output discard.
History includes user input and only assistant text confirmed as heard. A browser
completion acknowledges the whole text; an interruption may supply an exact
prefix confirmed by completed audio segments, without guessing word alignment.
Delivery transitions are terminal and idempotent. Reusing a request ID never
calls the model again. Model failures preserve the input and allow a later turn
in the same session. JSON is saved atomically under data_dir/sessions/<id>/;
startup cancels pending output and marks in-flight requests interrupted, without
calling an API or assuming old audio was heard. Methods are intended for a
single server process; use separate data directories for independent servers.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid

from chat_config import ChatSettings, load_settings
from chat_model import APIResponder, ChatGenerationCancelled, MAX_REPLY_CHARS

MAX_USER_CHARS = 2000
MAX_MESSAGES = 100
SAFE_MODEL_ERROR = "返答を生成できませんでした。このまま続けてお話しください。"
SAFE_RECOVERY_ERROR = "前回の応答処理は中断されました。新しい会話を開始してください。"
_ID = re.compile(r"[0-9a-f]{32}\Z")
_REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{8,100}\Z")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class ChatCore:
    def __init__(self, data_dir, *, settings=None, responder=None):
        self.data_dir = Path(data_dir).resolve()
        self.settings = settings if settings is not None else load_settings()
        if not isinstance(self.settings, ChatSettings):
            raise ValueError("会話設定の形式が不正です。")
        self.responder = responder if responder is not None else APIResponder()
        self._root = self.data_dir / "sessions"
        self._root.mkdir(parents=True, exist_ok=True)
        self._guard = threading.RLock()
        self._locks = {}
        self._sessions = {}
        self._active_requests = {}
        self._closed = False
        self.recovery_errors = 0
        self._recover()

    def _path(self, sid):
        if not isinstance(sid, str) or not _ID.fullmatch(sid):
            raise ValueError("会話IDが不正です。")
        path = (self._root / sid / "session.json").resolve()
        if not path.is_relative_to(self._root):
            raise ValueError("会話の保存先が不正です。")
        return path

    def _save(self, state):
        path = self._path(state["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(".session-" + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(state, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            # Windows virus scanners can briefly hold a file. Retrying only the
            # local rename does not resend a paid model request.
            for attempt in range(3):
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError:
                    if attempt == 2:
                        raise
                    time.sleep(.02)
        except OSError:
            raise RuntimeError("会話の記録を保存できません。") from None
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _valid_saved(state, sid):
        if (not isinstance(state, dict) or state.get("version") != 1
                or state.get("id") != sid or state.get("role_id") != "chat"
                or not isinstance(state.get("messages"), list)
                or len(state["messages"]) > MAX_MESSAGES
                or not isinstance(state.get("requests"), dict)):
            return False
        ids = set()
        for row in state["messages"]:
            if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                    or not _ID.fullmatch(row["id"]) or row["id"] in ids
                    or row.get("role") not in ("user", "assistant")
                    or not isinstance(row.get("content"), str)
                    or not row["content"].strip() or len(row["content"]) > 2000):
                return False
            ids.add(row["id"])
            if row["role"] == "assistant":
                delivery = row.get("delivery")
                if (not isinstance(delivery, dict)
                        or delivery.get("status") not in ("pending", "completed", "interrupted", "cancelled")
                        or not isinstance(delivery.get("heard_text"), str)
                        or not row["content"].startswith(delivery["heard_text"])):
                    return False
                if delivery["status"] == "completed" and delivery["heard_text"] != row["content"]:
                    return False
                if delivery["status"] in ("pending", "cancelled") and delivery["heard_text"]:
                    return False
        for rid, request in state["requests"].items():
            if (not isinstance(rid, str) or not _REQUEST_ID.fullmatch(rid)
                    or not isinstance(request, dict)
                    or request.get("status") not in ("running", "completed", "failed", "interrupted")
                    or not isinstance(request.get("text"), str)
                    or not request["text"].strip() or len(request["text"]) > MAX_USER_CHARS):
                return False
        return True

    def _recover(self):
        for folder in self._root.iterdir():
            if not _ID.fullmatch(folder.name) or not folder.is_dir():
                continue
            try:
                path = self._path(folder.name)
                if path.stat().st_size > 1_000_000:
                    raise ValueError("invalid stored session")
                state = json.loads(path.read_text(encoding="utf-8"))
                if not self._valid_saved(state, folder.name):
                    raise ValueError("invalid stored session")
                interrupted = bool(state.get("busy"))
                changed = interrupted
                state["busy"] = False
                for request in state["requests"].values():
                    if request["status"] == "running":
                        request["status"] = "interrupted"
                        interrupted = changed = True
                for message in state["messages"]:
                    if message["role"] == "assistant" and message["delivery"]["status"] == "pending":
                        message["delivery"] = {"status": "cancelled", "heard_text": "", "at": _now()}
                        changed = True
                if interrupted:
                    state["error"] = SAFE_RECOVERY_ERROR
                elif state.get("error"):
                    state["error"] = SAFE_MODEL_ERROR
                if changed:
                    state["updated_at"] = _now()
                    self._save(state)
                self._sessions[state["id"]] = state
                self._locks[state["id"]] = threading.RLock()
            except (OSError, ValueError, UnicodeError, RuntimeError):
                # Preserve corrupt records for inspection; do not expose their
                # contents or overwrite them with a fabricated conversation.
                self.recovery_errors += 1

    def _session(self, sid):
        self._path(sid)
        with self._guard:
            if sid not in self._sessions:
                raise KeyError("会話が見つかりません。")
            return self._sessions[sid], self._locks[sid]

    @staticmethod
    def _view(state):
        return deepcopy({key: state.get(key) for key in
                         ("id", "role_id", "phase", "messages", "busy", "error", "created_at", "updated_at")})

    @staticmethod
    def _assistant(text, kind="reply"):
        return {"id": uuid.uuid4().hex, "role": "assistant", "content": text,
                "kind": kind, "created_at": _now(),
                "delivery": {"status": "pending", "heard_text": ""}}

    def create(self, role_id="chat"):
        if role_id != "chat":
            raise ValueError("会話の種類が不正です。")
        with self._guard:
            if self._closed:
                raise RuntimeError("会話サービスは終了しています。")
            sid, now = uuid.uuid4().hex, _now()
            state = {"version": 1, "id": sid, "role_id": "chat", "phase": "chat",
                     "created_at": now, "updated_at": now, "busy": False, "error": None,
                     "messages": [self._assistant(self.settings.greeting.strip(), "greeting")],
                     "requests": {}}
            self._save(state)
            self._sessions[sid] = state
            self._locks[sid] = threading.RLock()
            return self._view(state)

    def view(self, sid):
        state, lock = self._session(sid)
        with lock:
            return self._view(state)

    def view_if_idle(self, sid):
        """Return a consistent snapshot, or None while a request owns the lock.

        Voice turn detection must keep running if a model request starts just
        before a periodic UI refresh acquires this session's lock.
        """
        state, lock = self._session(sid)
        if not lock.acquire(blocking=False):
            return None
        try:
            return self._view(state)
        finally:
            lock.release()

    def pending(self, sid):
        state, lock = self._session(sid)
        with lock:
            return [{"id": row["id"], "text": row["content"], "kind": row["kind"]}
                    for row in state["messages"] if row["role"] == "assistant"
                    and row["delivery"]["status"] == "pending"]

    @staticmethod
    def _history(state):
        history = []
        for row in state["messages"]:
            text = row["content"] if row["role"] == "user" else row["delivery"]["heard_text"]
            if text.strip():
                history.append({"role": row["role"], "content": text})
        return history

    def submit(self, sid, text, request_id):
        return self._submit(sid, text, request_id, threading.Event())

    def submit_cancellable(self, sid, text, request_id, cancel_event):
        """Persist input even if superseded before its worker starts."""
        if not isinstance(cancel_event, threading.Event):
            raise ValueError("返答の中断通知が不正です。")
        return self._submit(sid, text, request_id, cancel_event)

    def cancel(self, sid):
        # Do not acquire the per-session lock: submit owns it while the model
        # runs. Signalling an Event never closes another session's transport.
        self._path(sid)
        with self._guard:
            if sid not in self._sessions:
                raise KeyError("会話が見つかりません。")
            event = self._active_requests.get(sid)
            if event is None:
                return False
            event.set()
            return True

    def _submission_view(self, state, request_id):
        view = self._view(state)
        view.update(request_id=request_id, request_status=state["requests"][request_id]["status"])
        return view

    def _submit(self, sid, text, request_id, cancel_event):
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_USER_CHARS:
            raise ValueError("入力は1〜2000文字で指定してください。")
        if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
            raise ValueError("リクエストIDが不正です。")
        text = text.strip()
        state, lock = self._session(sid)
        with lock:
            previous = state["requests"].get(request_id)
            if previous is not None:
                if previous["text"] != text:
                    raise ValueError("同じリクエストIDに異なる入力は指定できません。")
                return self._submission_view(state, request_id)
            with self._guard:
                if self._closed:
                    raise RuntimeError("会話サービスは終了しています。")
            if len(state["messages"]) + 2 > MAX_MESSAGES:
                raise ValueError("会話が長くなりました。新しい会話を開始してください。")
            request = {"text": text, "status": "running", "started_at": _now(),
                       "model": self.settings.model, "reasoning_effort": self.settings.reasoning_effort,
                       "service_tier": self.settings.service_tier}
            state["requests"][request_id] = request
            state["messages"].append({"id": uuid.uuid4().hex, "role": "user", "content": text,
                                      "request_id": request_id, "created_at": _now()})
            state["busy"], state["error"], state["updated_at"] = True, None, _now()
            self._save(state)
            with self._guard:
                self._active_requests[sid] = cancel_event
                if self._closed:
                    cancel_event.set()
            started = time.monotonic()
            responder_invoked = False
            try:
                if cancel_event.is_set():
                    raise ChatGenerationCancelled("返答の生成を中断しました。")
                respond = getattr(self.responder, "respond_cancellable", None)
                responder_invoked = True
                answer = (respond(self._history(state), self.settings, cancel_event)
                          if callable(respond) else self.responder.respond(self._history(state), self.settings))
                if cancel_event.is_set():
                    raise ChatGenerationCancelled("返答の生成を中断しました。")
                if not isinstance(answer, str) or not answer.strip() or len(answer) > MAX_REPLY_CHARS:
                    raise ValueError("invalid model response")
                with self._guard:
                    if self._closed or cancel_event.is_set():
                        request["status"] = "interrupted"
                        state["error"] = SAFE_RECOVERY_ERROR if self._closed else None
                    else:
                        state["messages"].append(self._assistant(answer.strip()))
                        request["status"] = "completed"
            except Exception as error:
                # A provider error can contain authorization headers or user
                # text. Only a fixed public message is retained.
                with self._guard:
                    interrupted = (cancel_event.is_set() or self._closed
                                   or isinstance(error, ChatGenerationCancelled))
                    request["status"] = "interrupted" if interrupted else "failed"
                    state["error"] = (SAFE_RECOVERY_ERROR if self._closed else
                                      None if interrupted else SAFE_MODEL_ERROR)
            finally:
                # The built-in responder returns only safe, thread-local
                # diagnostics. An optional adapter must not break persistence
                # if its diagnostic hook is unavailable or malformed.
                if responder_invoked:
                    try:
                        diagnostics = getattr(self.responder, "get_last_call", None)
                        if callable(diagnostics):
                            detail = diagnostics()
                            if isinstance(detail, dict):
                                json.dumps(detail, allow_nan=False)
                                request["diagnostics"] = deepcopy(detail)
                    except Exception:
                        pass
                with self._guard:
                    if self._active_requests.get(sid) is cancel_event:
                        self._active_requests.pop(sid)
                request["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
                request["finished_at"] = _now()
                state["busy"], state["updated_at"] = False, _now()
                self._save(state)
            return self._submission_view(state, request_id)

    def delivery(self, sid, message_id, status, heard_text=""):
        if (not isinstance(message_id, str) or not _ID.fullmatch(message_id)
                or status not in ("completed", "interrupted", "cancelled")
                or not isinstance(heard_text, str)):
            raise ValueError("音声再生結果の形式が不正です。")
        state, lock = self._session(sid)
        with lock:
            row = next((item for item in state["messages"] if item["id"] == message_id
                        and item["role"] == "assistant"), None)
            if row is None:
                raise KeyError("対象の返答が見つかりません。")
            if status == "completed":
                if heard_text not in ("", row["content"]):
                    raise ValueError("再生済みの本文が一致しません。")
                heard_text = row["content"]
            elif status == "cancelled" and heard_text:
                raise ValueError("未再生の返答に再生済み本文は指定できません。")
            elif not row["content"].startswith(heard_text):
                raise ValueError("再生済み本文は返答の先頭と完全に一致する必要があります。")
            current = row["delivery"]
            if current["status"] != "pending":
                if current["status"] == status and current["heard_text"] == heard_text:
                    return self._view(state)
                raise ValueError("確定した音声再生結果は変更できません。")
            row["delivery"] = {"status": status, "heard_text": heard_text, "at": _now()}
            state["updated_at"] = _now()
            self._save(state)
            return self._view(state)

    def close(self):
        with self._guard:
            if self._closed:
                return
            self._closed = True
            for event in self._active_requests.values():
                event.set()
        close = getattr(self.responder, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                # Shutdown must not leak provider exceptions or prevent the
                # server's remaining transports from closing.
                pass

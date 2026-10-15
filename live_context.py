"""Ordered, bounded publication of external playback facts to muted GPT-Live.

Only confirmed playback belongs in progress/completed/interrupted text. The
caller owns that evidence; this module never estimates a prefix from time.
Acknowledgments confirm context acceptance, never that anyone heard speech.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
import json
import uuid


MAX_CONTEXT_BYTES = 450
MAX_TEXT_CHARS = 12000
MAX_PENDING_ACKS = 512
STATUSES = {"started", "progress", "completed", "interrupted", "cancelled"}
TEXT_ROLES = {"started": "unconfirmed_playback_text", "progress": "confirmed_played_prefix",
              "completed": "confirmed_played_text", "interrupted": "confirmed_played_prefix",
              "cancelled": "no_played_text"}


def playback_context_chunks(playback_id, status, text, update_id):
    """Keep each complete JSON string safely below the 500-token append limit.

    The 450-byte UTF-8 cap includes JSON escaping and metadata. This is a
    deliberately conservative byte budget for byte-based tokenization; it
    avoids a tokenizer dependency and also bounds unusual scripts/emoji.
    All original text, including whitespace, survives chunk concatenation.
    """
    if (not isinstance(playback_id, str) or not playback_id
            or len(playback_id.encode("utf-8")) > 96):
        raise ValueError("invalid playback ID")
    if status not in STATUSES:
        raise ValueError("invalid playback status")
    if not isinstance(text, str) or len(text) > MAX_TEXT_CHARS:
        raise ValueError("invalid playback text")
    if status == "cancelled" and text:
        raise ValueError("cancelled playback cannot have confirmed played text")
    if not isinstance(update_id, str) or len(update_id) > 32 or not update_id:
        raise ValueError("invalid context update ID")
    metadata = {"kind": "external_playback", "playback_id": playback_id,
                "status": status, "text_role": TEXT_ROLES[status], "update_id": update_id}

    def encode(fragment, part, last):
        return json.dumps({**metadata, "part": part, "last": last, "text": fragment},
                          ensure_ascii=False, separators=(",", ":"))

    # false takes one more byte than true; reserve enough index space up front.
    if len(encode("", MAX_TEXT_CHARS, False).encode("utf-8")) > MAX_CONTEXT_BYTES - 12:
        raise ValueError("playback metadata exceeds context budget")
    fragments, fragment = [], ""
    for character in text:
        candidate = fragment + character
        if len(encode(candidate, MAX_TEXT_CHARS, False).encode("utf-8")) > MAX_CONTEXT_BYTES:
            fragments.append(fragment)
            fragment = character
        else:
            fragment = candidate
    fragments.append(fragment)
    return [encode(fragment, index + 1, index == len(fragments) - 1)
            for index, fragment in enumerate(fragments)]


class LiveContextPublisher:
    """Nonblocking playback notifications with one ordered sender.

    on_diagnostic receives a dict with no transcript or provider error message.
    Queue overflow and rejected/unconfirmed commands are explicit diagnostics.
    No retry or acknowledgment wait can delay microphone or playback handling.
    """

    def __init__(self, client, on_diagnostic=None, *, max_queue=32):
        if not isinstance(max_queue, int) or max_queue < 1:
            raise ValueError("context queue size must be positive")
        self.client = client
        self.on_diagnostic = on_diagnostic
        self.queue = asyncio.Queue(maxsize=max_queue)
        self.pending = OrderedDict()
        self.closed = False
        self.task = None
        self.prefix = "live_context_" + uuid.uuid4().hex[:12] + "_"

    def _diagnostic(self, status, **fields):
        if self.on_diagnostic is not None:
            self.on_diagnostic({"status": status, **fields})

    def publish(self, playback_id, status, text="", delegation_id=None):
        if delegation_id is not None and (not isinstance(delegation_id, str) or not delegation_id):
            raise ValueError("invalid client delegation ID")
        update_id = uuid.uuid4().hex[:16]
        chunks = playback_context_chunks(playback_id, status, text, update_id)
        fields = {"playback_id": playback_id, "playback_status": status, "update_id": update_id}
        if self.closed or self.queue.full():
            self._diagnostic("dropped", **fields, reason="closed" if self.closed else "queue_full")
            return False
        self.queue.put_nowait((chunks, delegation_id, fields))
        if self.task is None:
            self.task = asyncio.create_task(self._run(), name="live_context_sender")
        return True

    async def _run(self):
        while True:
            chunks, delegation_id, fields = await self.queue.get()
            try:
                for index, content in enumerate(chunks):
                    if self.closed:
                        return
                    event_id = self.prefix + uuid.uuid4().hex
                    metadata = {**fields, "event_id": event_id, "part": index + 1,
                                "parts": len(chunks)}
                    if len(self.pending) >= MAX_PENDING_ACKS:
                        _, oldest = self.pending.popitem(last=False)
                        self._diagnostic("unconfirmed", **oldest, reason="tracking_limit")
                    # Register before awaiting the network: an acknowledgment
                    # can arrive while send() is still returning.
                    self.pending[event_id] = metadata
                    try:
                        await self.client.think(content, delegation_id, event_id=event_id)
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        self.pending.pop(event_id, None)
                        self._diagnostic("send_failed", **metadata, error_type=type(error).__name__)
                        break  # Do not publish a tail fragment without its prefix.
                    self._diagnostic("sent", **metadata)
            finally:
                self.queue.task_done()

    def handle_event(self, event):
        typ = event.get("type")
        if typ not in {"session.thinking.appended", "error", "session.error"}:
            return False
        error = event.get("error") or {}
        event_id = error.get("client_event_id") or event.get("client_event_id")
        if not isinstance(event_id, str) or not event_id.startswith(self.prefix):
            return False
        metadata = self.pending.pop(event_id, {"event_id": event_id})
        if typ == "session.thinking.appended":
            self._diagnostic("accepted", **metadata)
        else:
            self._diagnostic("rejected", **metadata, code=error.get("code"))
        return True

    async def flush(self):
        """Wait for local sends only; useful for tests, never waits for ACKs."""
        await self.queue.join()

    async def close(self):
        if self.closed:
            return
        self.closed = True
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        dropped = 0
        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()
            dropped += 1
        if dropped:
            self._diagnostic("dropped", reason="close", updates=dropped)
        if self.pending:
            self._diagnostic("unconfirmed", reason="close", commands=len(self.pending))
            self.pending.clear()

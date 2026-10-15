"""Best-effort chat status; observing never schedules a turn or calls the model.

The event loop already owns the runtime state. No core lock, saved conversation
read, background worker, or extra API request is needed to publish this view.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy


class ProcessingStatePublisher:
    def __init__(self, runtime, core, sid, mode, send_json, *, interval=.5):
        # Keep the server's constructor contract; core and sid are deliberately
        # unused so UI updates cannot wait behind a model request's core lock.
        self.runtime = runtime
        self.mode = mode
        self.send_json = send_json
        self.interval = interval
        self.last_error = None

    def snapshot(self):
        state = self.runtime.processing_snapshot()
        if not isinstance(state, dict):
            raise TypeError("invalid runtime progress snapshot")
        return {"type": "processing_state", "mode": self.mode,
                "turn": deepcopy(state.get("turn", {})),
                "reply": deepcopy(state.get("reply", {})),
                "recent": deepcopy(state.get("recent", []))}

    def _report_error(self, error):
        signature = type(error).__name__
        if signature != self.last_error:
            self.runtime.log("processing_state_error", phase="publish", error_type=signature)
            self.last_error = signature

    async def run(self):
        while True:
            try:
                await self.send_json(self.snapshot())
                self.last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # Display failure must not end microphone capture or dialogue.
                self._report_error(error)
            await asyncio.sleep(self.interval)

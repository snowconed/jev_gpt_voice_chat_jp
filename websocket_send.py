"""Bounded, serialized sends with diagnostics for pending network writes.

Only timing and transport state are retained here, never event payloads or
credentials. Successful send_json means locally buffered, not server receipt.
"""
from __future__ import annotations

import asyncio
import time

from websocket_diagnostics import websocket_snapshot

SEND_TIMEOUT_SECONDS = 5.0


def send_snapshot(client):
    snapshot = websocket_snapshot(client.ws)
    pending = getattr(client, "_pending_sends", {})
    active = next((row for row in pending.values() if row["phase"] == "socket"), None)
    row = active or next(iter(pending.values()), None)
    snapshot.update(
        send_phase=row["phase"] if row else "idle",
        send_pending_ms=round((time.perf_counter() - row["started"]) * 1000, 3) if row else 0.0,
        send_waiters=sum(item["phase"] == "wait_lock" for item in pending.values()),
    )
    return snapshot


async def send_serialized(client, event, timeout_message):
    token = object()
    row = {"started": time.perf_counter(), "phase": "wait_lock"}
    client._pending_sends[token] = row
    try:
        async with asyncio.timeout(SEND_TIMEOUT_SECONDS):
            async with client._send_lock:
                row["phase"] = "socket"
                await client.ws.send_json(event)
    except TimeoutError:
        client.transport_metrics["send_timeouts"] += 1
        client.transport_metrics["last_send_timeout"] = send_snapshot(client)
        raise RuntimeError(timeout_message) from None
    finally:
        elapsed = (time.perf_counter() - row["started"]) * 1000
        client.transport_metrics["max_event_send_ms"] = max(
            client.transport_metrics.get("max_event_send_ms", 0.0), round(elapsed, 3))
        del client._pending_sends[token]

"""Best-effort WebSocket write diagnostics containing only fixed scalar fields.

No headers, URLs, exceptions, audio, or object representations are returned.
Private attribute paths below are isolated because they may change between
aiohttp/Python versions; unavailable measurements remain None.
"""
from __future__ import annotations

from typing import Any


_MISSING = object()


def _attribute(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _call(obj: Any, name: str, *args: Any) -> Any:
    method = _attribute(obj, name)
    if not callable(method):
        return None
    try:
        return method(*args)
    except Exception:
        return None


def _boolean(value: Any) -> bool | None:
    # Do not coerce objects or strings: that could turn unknown state into False.
    return value if type(value) is bool else None


def _number(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _boolean_attribute(obj: Any, public: str, private: str) -> bool | None:
    value = _boolean(_attribute(obj, public))
    return value if value is not None else _boolean(_attribute(obj, private))


def websocket_snapshot(ws: Any) -> dict[str, bool | int | None]:
    """Return safe, non-blocking state; a missing measurement is never zero.

    aiohttp 3.14.4 exposes closed/compress publicly. Its underlying transport
    is reachable through the private response or writer, then public transport
    methods provide buffer sizes and closing state. In CPython 3.12's TLS
    implementation, _SSLProtocolTransport.get_write_buffer_size() covers TLS
    outgoing/backlog data only. The separate TCP buffer is on
    transport._ssl_protocol._transport, so these values must not be conflated.
    Plain WebSockets have only a TCP buffer; their TLS measurement is None.
    """
    closed = _boolean_attribute(ws, "closed", "_closed")
    compression = _number(_attribute(ws, "compress"))
    if compression is None:
        compression = _number(_attribute(ws, "_compress"))

    writer = _attribute(ws, "_writer")
    response = _attribute(ws, "_response")
    connection = _attribute(response, "connection")
    transport = _attribute(connection, "transport")
    if transport is None:
        transport = _attribute(writer, "transport")

    protocol = _call(transport, "get_protocol")
    if protocol is None:
        protocol = _attribute(writer, "protocol")
    paused = _boolean_attribute(protocol, "writing_paused", "_paused")

    ssl_protocol = _attribute(transport, "_ssl_protocol", _MISSING)
    is_tls = ssl_protocol is not _MISSING
    if not is_tls:
        is_tls = _call(transport, "get_extra_info", "ssl_object") is not None
    tcp_transport = _attribute(ssl_protocol, "_transport") if is_tls else transport

    closing = _boolean(_call(transport, "is_closing"))
    if closing is None:
        closing = _boolean(_call(tcp_transport, "is_closing"))

    return {
        "ws_closed": closed,
        "ws_compression": compression,
        "protocol_paused": paused,
        "tls_write_buffer_bytes": (
            _number(_call(transport, "get_write_buffer_size")) if is_tls else None
        ),
        "tcp_write_buffer_bytes": _number(_call(tcp_transport, "get_write_buffer_size")),
        "transport_closing": closing,
    }

"""Offline diagnostic checks; no event loop, socket, or API is created."""
import json
from types import SimpleNamespace as NS
import unittest

from websocket_diagnostics import websocket_snapshot


KEYS = {
    "ws_closed", "ws_compression", "protocol_paused", "tls_write_buffer_bytes",
    "tcp_write_buffer_bytes", "transport_closing",
}


class Transport:
    def __init__(self, size=0, closing=False, protocol=None):
        self.size = size
        self.closing = closing
        self.protocol = protocol

    def get_write_buffer_size(self):
        return self.size

    def is_closing(self):
        return self.closing

    def get_protocol(self):
        return self.protocol

    def get_extra_info(self, name):
        return None


class RaisingAttributes:
    def __getattribute__(self, name):
        raise RuntimeError("secret-do-not-serialize")


class SnapshotTests(unittest.TestCase):
    def assert_safe(self, result):
        self.assertEqual(set(result), KEYS)
        self.assertTrue(all(value is None or type(value) in (bool, int)
                            for value in result.values()))
        self.assertNotIn("secret-do-not-serialize", json.dumps(result))

    def test_missing_websocket_and_attributes_are_unknown(self):
        for ws in (None, object(), NS()):
            with self.subTest(ws=type(ws).__name__):
                result = websocket_snapshot(ws)
                self.assert_safe(result)
                self.assertTrue(all(value is None for value in result.values()))

    def test_tls_and_tcp_buffers_are_separate_and_zero_is_valid(self):
        tcp = Transport(16384)
        tls = Transport(0, protocol=NS(writing_paused=True))
        tls._ssl_protocol = NS(_transport=tcp)
        ws = NS(closed=False, compress=15, _writer=NS(transport=tls))
        result = websocket_snapshot(ws)
        self.assert_safe(result)
        self.assertEqual(result, {
            "ws_closed": False, "ws_compression": 15, "protocol_paused": True,
            "tls_write_buffer_bytes": 0, "tcp_write_buffer_bytes": 16384,
            "transport_closing": False,
        })

    def test_plain_transport_has_no_tls_buffer(self):
        tcp = Transport(27, closing=True, protocol=NS(writing_paused=False))
        result = websocket_snapshot(NS(closed=True, compress=0,
                                       _writer=NS(transport=tcp)))
        self.assert_safe(result)
        self.assertIsNone(result["tls_write_buffer_bytes"])
        self.assertEqual(result["tcp_write_buffer_bytes"], 27)
        self.assertTrue(result["transport_closing"])
        self.assertFalse(result["protocol_paused"])

    def test_public_values_and_response_connection_are_preferred(self):
        transport = Transport(13, protocol=NS(writing_paused=False, _paused=True))
        ws = NS(closed=False, _closed=True, compress=15, _compress=0,
                _response=NS(connection=NS(transport=transport)),
                _writer=NS(transport=Transport(99), protocol=NS(writing_paused=True)))
        result = websocket_snapshot(ws)
        self.assert_safe(result)
        self.assertFalse(result["ws_closed"])
        self.assertEqual(result["ws_compression"], 15)
        self.assertFalse(result["protocol_paused"])
        self.assertEqual(result["tcp_write_buffer_bytes"], 13)

    def test_private_fallbacks_after_public_getter_errors(self):
        class WS:
            _closed = True
            _compress = 0
            _response = RaisingAttributes()
            _writer = NS(transport=Transport(5), protocol=NS(_paused=True))

            @property
            def closed(self):
                raise RuntimeError("secret-do-not-serialize")

            @property
            def compress(self):
                raise RuntimeError("secret-do-not-serialize")

        result = websocket_snapshot(WS())
        self.assert_safe(result)
        self.assertTrue(result["ws_closed"])
        self.assertEqual(result["ws_compression"], 0)
        self.assertTrue(result["protocol_paused"])
        self.assertEqual(result["tcp_write_buffer_bytes"], 5)

    def test_all_attributes_can_raise_without_leaking_error_text(self):
        result = websocket_snapshot(RaisingAttributes())
        self.assert_safe(result)
        self.assertTrue(all(value is None for value in result.values()))

    def test_transport_methods_can_raise_and_fall_back_independently(self):
        def fail(*args):
            raise RuntimeError("secret-do-not-serialize")

        tcp = Transport(43, closing=True)
        tls = NS(_ssl_protocol=NS(_transport=tcp), get_protocol=fail,
                 get_write_buffer_size=fail, is_closing=fail)
        result = websocket_snapshot(NS(_writer=NS(transport=tls, protocol=NS(_paused=False))))
        self.assert_safe(result)
        self.assertIsNone(result["tls_write_buffer_bytes"])
        self.assertEqual(result["tcp_write_buffer_bytes"], 43)
        self.assertTrue(result["transport_closing"])
        self.assertFalse(result["protocol_paused"])

    def test_invalid_types_are_not_coerced_or_serialized(self):
        transport = Transport("secret-do-not-serialize", closing="secret-do-not-serialize",
                              protocol=NS(writing_paused=1))
        ws = NS(closed=1, compress=True, _writer=NS(transport=transport))
        result = websocket_snapshot(ws)
        self.assert_safe(result)
        self.assertTrue(all(value is None for value in result.values()))
        for invalid in (-1, 1.5, float("inf")):
            transport.size = invalid
            self.assertIsNone(websocket_snapshot(ws)["tcp_write_buffer_bytes"])

    def test_detached_tls_layer_does_not_mislabel_tls_as_tcp(self):
        tls = Transport(17)
        tls._ssl_protocol = None
        result = websocket_snapshot(NS(_writer=NS(transport=tls)))
        self.assert_safe(result)
        self.assertEqual(result["tls_write_buffer_bytes"], 17)
        self.assertIsNone(result["tcp_write_buffer_bytes"])

    def test_public_ssl_detection_when_private_tls_path_is_missing(self):
        tls = Transport(18)
        tls.get_extra_info = lambda name: object() if name == "ssl_object" else None
        result = websocket_snapshot(NS(_writer=NS(transport=tls)))
        self.assert_safe(result)
        self.assertEqual(result["tls_write_buffer_bytes"], 18)
        self.assertIsNone(result["tcp_write_buffer_bytes"])


if __name__ == "__main__":
    unittest.main()

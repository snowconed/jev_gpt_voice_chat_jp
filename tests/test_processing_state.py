"""Status stays independent of model/core I/O and tolerates a failed publish."""
import asyncio
from copy import deepcopy
import unittest

from processing_state import ProcessingStatePublisher


class NoCoreAccess:
    def __getattr__(self, name):
        raise AssertionError("Status must not access the core: " + name)


class Runtime:
    def __init__(self):
        self.state = {"turn": {"code": "jev_judging", "label": "判定中"},
                      "reply": {"code": "core_processing", "label": "返答を考えています"},
                      "recent": [{"label": "入力確定", "detail": "確認しました"}]}
        self.logs = []
        self.failure_message = None

    def processing_snapshot(self):
        return self.state

    def log(self, event, **data):
        self.logs.append((event, data))


class SnapshotTests(unittest.TestCase):
    def test_two_lanes_are_copied_without_accessing_core_or_mutating_runtime(self):
        runtime = Runtime()
        before = deepcopy(runtime.state)
        publisher = ProcessingStatePublisher(runtime, NoCoreAccess(), "s", "jev", None)
        snapshot = publisher.snapshot()
        self.assertEqual(set(snapshot), {"type", "mode", "turn", "reply", "recent"})
        self.assertEqual(snapshot["mode"], "jev")
        snapshot["turn"]["label"] = "changed"
        snapshot["recent"][0]["label"] = "changed"
        self.assertEqual(runtime.state, before)
        self.assertEqual(runtime.logs, [])


class PublisherTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_publish_is_retried_without_stopping_dialogue(self):
        runtime = Runtime()
        calls = 0
        received = asyncio.Event()

        async def send(event):
            nonlocal calls
            calls += 1
            if calls <= 2:
                raise RuntimeError("private transport detail")
            self.assertEqual(event["type"], "processing_state")
            received.set()

        publisher = ProcessingStatePublisher(runtime, NoCoreAccess(), "s", "live", send, interval=.001)
        task = asyncio.create_task(publisher.run())
        try:
            await asyncio.wait_for(received.wait(), 2)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(len(runtime.logs), 1)
        self.assertNotIn("private", str(runtime.logs))
        self.assertIsNone(runtime.failure_message)

    async def test_cancellation_during_publish_propagates_without_error_log(self):
        runtime = Runtime()
        entered = asyncio.Event()

        async def send(_event):
            entered.set()
            await asyncio.Event().wait()

        publisher = ProcessingStatePublisher(runtime, NoCoreAccess(), "s", "jev", send)
        task = asyncio.create_task(publisher.run())
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(runtime.logs, [])


if __name__ == "__main__":
    unittest.main()

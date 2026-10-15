"""Deterministic voice/core ordering tests. No sockets or model APIs are used."""
import asyncio
from copy import deepcopy
from pathlib import Path
import tempfile
import threading
import time
import unittest

from voice_runtime import VoiceRuntime


HERE = Path(__file__).resolve().parents[1]


def output(mid, text="前の話の返事です。", kind="reply"):
    return {"id": mid, "role": "assistant", "content": text, "kind": kind,
            "delivery": {"status": "pending"}}


class Core:
    def __init__(self):
        self.messages = []
        self.inputs = []
        self.deliveries = []
        self.events = []
        self.before_submit = None
        self.error = None

    def pending(self, sid):
        return [{"id": m["id"], "text": m["content"], "kind": m["kind"]}
                for m in deepcopy(self.messages) if m["role"] == "assistant"
                and m.get("delivery", {}).get("status") == "pending" and m["kind"] != "error"]

    def view(self, sid):
        return {"id": sid, "messages": deepcopy(self.messages), "error": self.error}

    def delivery(self, sid, mid, status, heard_text=""):
        self.events.append(("delivery", mid, status))
        self.deliveries.append((mid, status, heard_text))
        for message in self.messages:
            if message["id"] == mid:
                message["delivery"]["status"] = status

    def submit(self, sid, text, rid):
        if self.before_submit:
            self.before_submit()
        self.events.append(("submit", text))
        self.inputs.append((text, rid))
        self.messages.append({"id": rid, "role": "user", "content": text, "kind": "user"})
        self.messages.append(output("new-reply", "新しい話への返事です。"))
        return self.view(sid)


class Client:
    def __init__(self):
        self.approvals = []
        self.instructions = []

    async def approve(self, text):
        self.approvals.append(text)

    async def instruct(self, text):
        self.instructions.append(text)


class VoiceRaceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.test_root = (HERE / ".runtime" / "test_races").resolve()
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.test_root)
        self.events = []
        self.core, self.client = Core(), Client()
        self.tasks = []
        self.releases = []

        async def send_json(event):
            self.events.append(event)

        async def send_bytes(_pcm):
            raise AssertionError("These tests should never release audio")

        self.r = VoiceRuntime(self.core, "a" * 32, send_json, send_bytes,
                              Path(self.temp.name), client=self.client)

    async def asyncTearDown(self):
        for release in self.releases:
            release.set()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await asyncio.gather(*list(self.r.delivery_tasks), *list(self.r.control_tasks), return_exceptions=True)
        self.r.logfile.close()
        self.assertTrue(Path(self.temp.name).resolve().is_relative_to(self.test_root))
        self.temp.cleanup()

    def block_call(self, name, *, snapshot_first=True):
        entered = asyncio.Event()
        release = threading.Event()
        self.releases.append(release)
        loop = asyncio.get_running_loop()
        original = getattr(self.core, name)

        def blocked(*args):
            value = original(*args) if snapshot_first else None
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(5):
                raise TimeoutError("test barrier was not released")
            return value if snapshot_first else original(*args)

        setattr(self.core, name, blocked)
        return entered, release

    def run_task(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)
        return task

    async def test_new_input_during_pending_lookup_cannot_reapprove_old_output(self):
        self.core.messages.append(output("old"))
        entered, release = self.block_call("pending")
        offer = self.run_task(self.r._offer_pending(0, 0))
        await asyncio.wait_for(entered.wait(), 2)
        await self.r.enqueue("新しい話です。")
        self.r.busy = True
        self.r.jobs.get_nowait()
        self.r.jobs.task_done()
        self.core.delivery(self.r.sid, "old", "cancelled")
        release.set()
        await asyncio.wait_for(offer, 2)
        self.assertEqual(self.client.approvals, [])
        self.assertIsNone(self.r.approved)

    async def test_finished_new_turn_during_view_lookup_still_invalidates_snapshot(self):
        self.core.messages.append(output("old"))
        entered, release = self.block_call("view")
        refresh = self.run_task(self.r._refresh_state())
        await asyncio.wait_for(entered.wait(), 2)
        # busy may have become true and false again while to_thread was pending.
        self.r.input_revision += 1
        release.set()
        await asyncio.wait_for(refresh, 2)
        self.assertEqual(self.client.approvals, [])
        self.assertEqual(self.events, [])

    async def test_input_during_session_notification_is_rechecked_before_offer(self):
        self.core.messages.append(output("old"))
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed_send(event):
            if event["type"] == "session":
                entered.set()
                await release.wait()
            self.events.append(event)

        self.r.send_json = delayed_send
        refresh = self.run_task(self.r._refresh_state())
        await asyncio.wait_for(entered.wait(), 2)
        await self.r.enqueue("別のことを話します。")
        release.set()
        await asyncio.wait_for(refresh, 2)
        self.assertEqual(self.client.approvals, [])

    async def test_output_generation_change_invalidates_pending_snapshot(self):
        self.core.messages.append(output("old"))
        entered, release = self.block_call("pending")
        offer = self.run_task(self.r._offer_pending(0, 0))
        await asyncio.wait_for(entered.wait(), 2)
        self.r.output_generation += 1
        release.set()
        await asyncio.wait_for(offer, 2)
        self.assertEqual(self.client.approvals, [])

    async def test_new_microphone_activity_invalidates_pending_snapshot(self):
        self.core.messages.append(output("old"))
        entered, release = self.block_call("pending")
        offer = self.run_task(self.r._offer_pending(0, 0))
        await asyncio.wait_for(entered.wait(), 2)
        self.r.input_last_voice = time.monotonic()
        release.set()
        await asyncio.wait_for(offer, 2)
        self.assertEqual(self.client.approvals, [])

    async def test_input_during_approval_does_not_restore_cancelled_message(self):
        self.core.messages.append(output("old"))
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed_approval(text):
            self.client.approvals.append(text)
            entered.set()
            await release.wait()

        self.client.approve = delayed_approval
        offer = self.run_task(self.r._offer_pending(0, 0))
        await asyncio.wait_for(entered.wait(), 2)
        await self.r.enqueue("少し待って。")
        release.set()
        await asyncio.wait_for(offer, 2)
        await asyncio.gather(*list(self.r.delivery_tasks), *list(self.r.control_tasks))
        self.assertIsNone(self.r.approved)
        self.assertEqual(self.core.deliveries, [("old", "cancelled", "")])
        self.assertFalse(any(e.get("text") == "返事を音声にしています。" for e in self.events))

    async def test_background_output_between_snapshot_and_submit_is_cancelled(self):
        self.core.before_submit = lambda: self.core.messages.append(output("racing-old-intro"))
        await self.r.enqueue("今の話を訂正したいです。")
        self.run_task(self.r.process_inputs())
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.assertEqual(self.core.deliveries, [("racing-old-intro", "cancelled", "")])
        self.assertEqual([m["id"] for m in self.core.pending(self.r.sid)], ["new-reply"])
        self.assertFalse(self.r.finished.is_set())

    async def test_failed_core_view_reports_failure_without_closing_conversation(self):
        self.core.error = "応答を最後まで確認できませんでした。"
        await self.r.enqueue("今日はうれしいことがありました。")
        worker = self.run_task(self.r.process_inputs())
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.assertFalse(worker.done())
        self.assertFalse(self.r.finished.is_set())
        self.assertIsNone(self.r.failure_message)
        self.assertTrue(any("このまま続けて" in e.get("text", "") for e in self.events))
        self.assertFalse(any(e["type"] == "error" for e in self.events))
        self.assertEqual(self.core.pending(self.r.sid), [])
        self.assertEqual(self.r.jobs.qsize(), 0)
        self.assertFalse(self.r.busy)

    async def test_completion_ack_is_committed_before_next_core_input(self):
        self.core.messages.append(output("heard"))
        self.r.playing = {"id": "heard", "kind": "reply", "text": "聞き終えた返事です。"}
        self.r.playing_started = time.monotonic() - 3
        self.r.playing_seconds = 1
        entered, release = self.block_call("delivery", snapshot_first=False)
        await self.r.played("heard")
        await asyncio.wait_for(entered.wait(), 2)
        await self.r.enqueue("そうなんだ。")
        self.run_task(self.r.process_inputs())
        await asyncio.sleep(.03)
        self.assertEqual(self.core.inputs, [])
        release.set()
        await asyncio.wait_for(self.r.jobs.join(), 2)
        self.assertEqual(self.core.events[0], ("delivery", "heard", "completed"))
        self.assertEqual(self.core.events[1][0], "submit")


if __name__ == "__main__":
    unittest.main()

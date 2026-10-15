"""Offline turn-boundary regressions for captured but unsent microphone audio."""
import asyncio
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from jev_runtime import JevSpeechRuntime
from test_jev_runtime import Judge
from test_speech_runtime import Live, Speech
from test_voice_races import Core
from voice_runtime import VoiceRuntime, INPUT_TRANSPORT_BARRIER_BYTES


class BarrierTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        parent = Path(__file__).resolve().parents[1] / ".runtime" / "input_transport_tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.runtimes, self.tasks, self.releases, self.events = [], [], [], []

    async def asyncTearDown(self):
        for release in self.releases:
            release.set()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        for runtime in self.runtimes:
            await runtime.close()
        self.temp.cleanup()

    def runtime(self, jev=False, send=None):
        async def send_json(event):
            self.events.append(event)
            if send:
                await send(event)

        async def send_bytes(_):
            pass

        client = Live()
        client.pending_audio = False
        core = Core()
        sid = f"{len(self.runtimes):032x}"
        if jev:
            runtime = JevSpeechRuntime(core, sid, send_json, send_bytes,
                Path(self.temp.name), client=client, speech=Speech(), jev=Judge())
        else:
            runtime = VoiceRuntime(core, sid, send_json, send_bytes,
                Path(self.temp.name), client=client)
        runtime.ready = True
        self.runtimes.append(runtime)
        return runtime

    def task(self, awaitable):
        task = asyncio.create_task(awaitable)
        self.tasks.append(task)
        return task

    def gate(self):
        event = asyncio.Event()
        self.releases.append(event)
        return event

    async def feed(self, runtime, text, item="one", sequence=0):
        await runtime._accept_transcript({
            "type": "session.input_transcript.completed", "item_id": item,
            "sequence": sequence, "segment_text": text, "text": text,
        })

    async def test_live_prefix_waits_and_full_continuation_commits_once(self):
        for source in ("fallback", "delegation", "response"):
            with self.subTest(source=source):
                runtime = self.runtime()
                now = time.monotonic()
                runtime.input_text = "前半ですが、"
                runtime.input_last_voice = runtime.input_last_change = now - 5
                if source == "delegation":
                    runtime.delegation = {"offset_ms": 1000}
                elif source == "response":
                    runtime.native_response_boundary = now - 1
                runtime.microphone_backlog_bytes = 48000 * 2
                self.assertFalse(await runtime._commit_live_input(now))
                self.assertTrue(runtime.jobs.empty())
                self.assertEqual(runtime.input_text, "前半ですが、")

                runtime.input_text += "まだ続きがあります。"
                runtime.input_revision += 1
                runtime.input_last_voice = runtime.input_last_change = now
                runtime.microphone_backlog_bytes = INPUT_TRANSPORT_BARRIER_BYTES
                self.assertTrue(await runtime._commit_live_input(now + 5))
                self.assertFalse(await runtime._commit_live_input(now + 6))
                self.assertEqual(runtime.jobs.qsize(), 1)
                self.assertEqual(runtime.jobs.get_nowait()[0], "前半ですが、まだ続きがあります。")
                self.assertEqual(runtime.input_text, "")

    async def test_live_rechecks_after_interrupt_await_without_losing_text(self):
        entered, release = asyncio.Event(), self.gate()

        async def send(event):
            if event["type"] == "interrupt":
                entered.set()
                await release.wait()

        runtime = self.runtime(send=send)
        runtime.playing = {"id": "bc", "kind": "backchannel", "text": "はい"}
        runtime.input_text = "前半"
        runtime.input_last_voice = runtime.input_last_change = time.monotonic() - 5
        commit = self.task(runtime._commit_live_input(time.monotonic()))
        await asyncio.wait_for(entered.wait(), 1)
        runtime.microphone_backlog_bytes = 48000
        runtime.input_text += "後半"
        runtime.input_revision += 1
        release.set()
        self.assertFalse(await commit)
        self.assertTrue(runtime.jobs.empty())
        self.assertEqual(runtime.input_text, "前半後半")
        self.assertIsNone(runtime.playing)

    async def test_worker_rechecks_backlog_after_status_before_core_submission(self):
        entered, release = asyncio.Event(), self.gate()

        async def send(event):
            if event["type"] == "status":
                entered.set()
                await release.wait()

        runtime = self.runtime(send=send)
        self.assertTrue(await runtime.enqueue("確定候補です。", "silence_recovery"))
        self.task(runtime.process_inputs())
        await asyncio.wait_for(entered.wait(), 1)
        runtime.microphone_backlog_bytes = 48000
        release.set()
        await asyncio.sleep(.06)
        self.assertEqual(runtime.core.inputs, [])
        runtime.microphone_backlog_bytes = 0
        await asyncio.wait_for(runtime.jobs.join(), 1)
        self.assertEqual([row[0] for row in runtime.core.inputs], ["確定候補です。"])

    async def test_executor_rechecks_audio_arriving_before_thread_starts(self):
        runtime = self.runtime()
        attempted = asyncio.Event()
        original_to_thread = asyncio.to_thread

        async def delayed_thread(function, *args, **kwargs):
            if function.__name__ == "submit_if_ready" and not attempted.is_set():
                runtime.microphone_backlog_bytes = 48000
                result = function(*args, **kwargs)
                attempted.set()
                return result
            return await original_to_thread(function, *args, **kwargs)

        await runtime.enqueue("送信直前の競合です。", "live_delegation")
        with patch("voice_runtime.asyncio.to_thread", delayed_thread):
            self.task(runtime.process_inputs())
            await asyncio.wait_for(attempted.wait(), 1)
            self.assertEqual(runtime.core.inputs, [])
            runtime.microphone_backlog_bytes = 0
            await asyncio.wait_for(runtime.jobs.join(), 1)
        self.assertEqual(len(runtime.core.inputs), 1)

    async def test_explicit_typed_input_and_interrupt_are_not_held(self):
        runtime = self.runtime()
        runtime.microphone_backlog_bytes = 48000 * 2
        runtime.playing = {"id": "bc", "kind": "backchannel", "text": "はい"}
        stopped_revision = runtime.input_revision
        self.assertTrue(await runtime.enqueue("止めてください", "typed"))
        self.assertIsNone(runtime.playing)
        self.assertIn({"type": "interrupt", "revision": stopped_revision}, self.events)
        self.task(runtime.process_inputs())
        await asyncio.wait_for(runtime.jobs.join(), 1)
        self.assertEqual(runtime.core.inputs[0][0], "止めてください")

    async def test_session_stop_before_executor_starts_prevents_core_submission(self):
        runtime = self.runtime()
        original_to_thread = asyncio.to_thread

        async def stop_before_thread(function, *args, **kwargs):
            if function.__name__ == "submit_if_ready":
                runtime.finished.set()
            return await original_to_thread(function, *args, **kwargs)

        await runtime.enqueue("終了直前の入力", "live_delegation")
        with patch("voice_runtime.asyncio.to_thread", stop_before_thread):
            await asyncio.wait_for(runtime.process_inputs(), 1)
        self.assertEqual(runtime.core.inputs, [])

    async def test_overlong_live_input_is_rejected_once_not_each_tick(self):
        runtime = self.runtime()
        runtime.input_text = "あ" * 2001
        runtime.input_last_voice = runtime.input_last_change = time.monotonic() - 5
        await runtime._commit_live_input(time.monotonic())
        await runtime._commit_live_input(time.monotonic())
        self.assertTrue(runtime.jobs.empty())
        self.assertEqual(len([e for e in self.events if e["type"] == "error"]), 1)

    async def test_session_stop_ends_deferred_worker_without_core_call(self):
        runtime = self.runtime()
        await runtime.enqueue("保留中の入力", "silence_recovery")
        runtime.microphone_backlog_bytes = 48000
        worker = self.task(runtime.process_inputs())
        await asyncio.sleep(.03)
        runtime.finished.set()
        await asyncio.wait_for(worker, 1)
        self.assertEqual(runtime.core.inputs, [])
        self.assertFalse(runtime.busy)
        self.assertEqual(runtime._input_job_sources, {})

    async def test_jev_inflight_judge_and_grace_cannot_commit_prefix(self):
        runtime = self.runtime(jev=True)
        runtime.jev.gate = self.gate()
        await self.feed(runtime, "前半です。")
        snapshot = runtime._snapshot("normal")
        runtime._start_judge(snapshot)
        await asyncio.wait_for(runtime.jev.started.wait(), 1)
        runtime.microphone_backlog_bytes = 48000
        runtime.jev.gate.set()
        await runtime.judge_task
        await runtime._advance(time.monotonic() + 5)
        await runtime._commit(snapshot, "jev_complete")
        self.assertTrue(runtime.jobs.empty())
        self.assertEqual(runtime._text("normal"), "前半です。")

        await self.feed(runtime, "前半です。後半もあります。")
        runtime.microphone_backlog_bytes = 0
        await runtime._advance(time.monotonic() + 1)
        if runtime.judge_task:
            await runtime.judge_task
        await runtime._advance(time.monotonic() + 2)
        await runtime._advance(time.monotonic() + 3)
        self.assertEqual(runtime.jobs.qsize(), 1)
        self.assertEqual(runtime.jobs.get_nowait()[0], "前半です。後半もあります。")

    async def test_jev_candidate_interrupt_survives_barrier_but_turn_waits(self):
        runtime = self.runtime(jev=True)
        runtime.playing = {"id": "bc", "kind": "backchannel", "text": "はい"}
        stopped_revision = runtime.input_revision
        await self.feed(runtime, "その説明は違います")
        runtime.microphone_backlog_bytes = 48000
        await runtime._advance(time.monotonic() + 1)
        await runtime.judge_task
        self.assertIsNone(runtime.playing)
        self.assertIn({"type": "interrupt", "revision": stopped_revision}, self.events)
        await runtime._advance(time.monotonic() + 2)
        self.assertTrue(runtime.jobs.empty())
        runtime.microphone_backlog_bytes = 0
        await runtime._advance(time.monotonic() + 3)
        self.assertEqual(runtime.jobs.qsize(), 1)

    async def test_jev_local_stop_interrupts_immediately_with_backlog(self):
        runtime = self.runtime(jev=True)
        runtime.playing = {"id": "bc", "kind": "backchannel", "text": "はい"}
        runtime.microphone_backlog_bytes = 48000
        stopped_revision = runtime.input_revision
        await self.feed(runtime, "止めて")
        self.assertIsNone(runtime.playing)
        self.assertIn({"type": "interrupt", "revision": stopped_revision}, self.events)
        self.assertEqual(runtime.jev.requests, [])
        await runtime._advance(time.monotonic() + 3)
        self.assertTrue(runtime.jobs.empty())


if __name__ == "__main__":
    unittest.main()

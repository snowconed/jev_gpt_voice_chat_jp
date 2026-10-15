from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import unittest

from chat_config import ChatSettings
from chat_core import ChatCore, SAFE_MODEL_ERROR, SAFE_RECOVERY_ERROR
from chat_model import ChatGenerationCancelled


class FakeResponder:
    def __init__(self, answer="今日はどんなことがありましたか？"):
        self.answer = answer
        self.calls = []
        self.closed = 0

    def respond(self, history, settings):
        self.calls.append(deepcopy(history))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer

    def close(self):
        self.closed += 1


class ChatCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.responder = FakeResponder()
        self.settings = ChatSettings(greeting="こんにちは。何を話しましょうか？")
        self.core = ChatCore(self.temp.name, settings=self.settings, responder=self.responder)
        self.addCleanup(self.core.close)
        self.state = self.core.create()
        self.sid = self.state["id"]

    def stored(self):
        return json.loads((Path(self.temp.name) / "sessions" / self.sid / "session.json").read_text(encoding="utf-8"))

    def test_local_greeting_and_public_view_are_independent(self):
        self.assertEqual(self.responder.calls, [])
        pending = self.core.pending(self.sid)
        self.assertEqual(pending, [{"id": self.state["messages"][0]["id"],
            "text": self.settings.greeting, "kind": "greeting"}])
        self.assertEqual(self.state["phase"], "chat")
        self.assertNotIn("requests", self.state)
        self.state["messages"].clear()
        self.assertEqual(len(self.core.view(self.sid)["messages"]), 1)
        self.assertEqual(self.stored()["messages"][0]["delivery"]["status"], "pending")

    def test_history_uses_only_heard_assistant_text(self):
        greeting = self.core.pending(self.sid)[0]
        self.core.delivery(self.sid, greeting["id"], "completed")
        first = self.core.submit(self.sid, "今日は買い物へ行きました。", "request_01")
        self.assertEqual(self.responder.calls[0], [
            {"role": "assistant", "content": self.settings.greeting},
            {"role": "user", "content": "今日は買い物へ行きました。"}])
        reply = first["messages"][-1]
        self.core.delivery(self.sid, reply["id"], "cancelled")
        second = self.core.submit(self.sid, "昨日のことでした。", "request_02")
        self.assertNotIn({"role": "assistant", "content": reply["content"]}, self.responder.calls[-1])
        self.core.delivery(self.sid, second["messages"][-1]["id"], "interrupted", "今日は")
        self.core.submit(self.sid, "少し話を変えます。", "request_03")
        self.assertIn({"role": "assistant", "content": "今日は"}, self.responder.calls[-1])
        self.assertNotIn({"role": "assistant", "content": reply["content"]}, self.responder.calls[-1])

    def test_pending_greeting_is_not_assumed_heard(self):
        self.core.submit(self.sid, "先に話します。", "request_01")
        self.assertEqual(self.responder.calls[0], [{"role": "user", "content": "先に話します。"}])

    def test_delivery_validation_and_terminal_idempotency(self):
        message = self.core.pending(self.sid)[0]
        for status, heard in (("interrupted", "何を"), ("completed", "こんにちは"),
                              ("cancelled", "こんにちは"), ("pending", "")):
            with self.subTest(status=status, heard=heard):
                with self.assertRaises(ValueError):
                    self.core.delivery(self.sid, message["id"], status, heard)
        one = self.core.delivery(self.sid, message["id"], "interrupted", "こんにちは。")
        two = self.core.delivery(self.sid, message["id"], "interrupted", "こんにちは。")
        self.assertEqual(one, two)
        with self.assertRaises(ValueError):
            self.core.delivery(self.sid, message["id"], "completed")
        with self.assertRaises(ValueError):
            self.core.delivery(self.sid, message["id"], "interrupted", "こんにちは")
        self.assertEqual(self.core.pending(self.sid), [])

    def test_completed_ack_with_and_without_text_is_identical(self):
        message = self.core.pending(self.sid)[0]
        one = self.core.delivery(self.sid, message["id"], "completed")
        two = self.core.delivery(self.sid, message["id"], "completed", message["text"])
        self.assertEqual(one, two)

    def test_request_idempotency_and_input_bounds(self):
        one = self.core.submit(self.sid, "こんにちは", "request_01")
        two = self.core.submit(self.sid, "こんにちは", "request_01")
        self.assertEqual(one, two)
        self.assertEqual(len(self.responder.calls), 1)
        for text, rid in (("違う内容", "request_01"), ("", "request_02"),
                          ("あ" * 2001, "request_02"), ("話", "../bad"), ("話", "short")):
            with self.subTest(rid=rid, length=len(text)):
                with self.assertRaises(ValueError):
                    self.core.submit(self.sid, text, rid)
        with self.assertRaises(ValueError):
            self.core.view("../outside")
        self.assertEqual(len(self.responder.calls), 1)

    def test_error_view_does_not_expose_provider_details_or_enqueue_error(self):
        greeting = self.core.pending(self.sid)[0]
        self.core.delivery(self.sid, greeting["id"], "completed")
        self.responder.answer = RuntimeError("SECRET_KEY provider-body")
        result = self.core.submit(self.sid, "話", "request_01")
        self.assertEqual(result["error"], SAFE_MODEL_ERROR)
        self.assertFalse(result["busy"])
        self.assertEqual(self.core.pending(self.sid), [])
        self.assertNotIn("SECRET", json.dumps(self.stored()))
        self.core.submit(self.sid, "話", "request_01")
        self.assertEqual(len(self.responder.calls), 1)

    def test_restart_cancels_unheard_output_and_preserves_confirmed_history(self):
        greeting = self.core.pending(self.sid)[0]
        self.core.delivery(self.sid, greeting["id"], "completed")
        self.core.submit(self.sid, "前の入力", "request_01")
        self.core.close()
        responder = FakeResponder()
        restarted = ChatCore(self.temp.name, settings=self.settings, responder=responder)
        self.addCleanup(restarted.close)
        self.assertEqual(responder.calls, [])
        self.assertEqual(restarted.pending(self.sid), [])
        self.assertEqual(restarted.view(self.sid)["messages"][-1]["delivery"]["status"], "cancelled")
        restarted.submit(self.sid, "次の入力", "request_02")
        self.assertEqual(responder.calls[-1], [{"role": "assistant", "content": self.settings.greeting},
            {"role": "user", "content": "前の入力"}, {"role": "user", "content": "次の入力"}])

    def test_safe_call_diagnostics_are_persisted_but_not_exposed_in_public_view(self):
        detail = {"status": "completed", "request_ms": 12.5,
                  "attempts": [{"status": "completed", "http_status": 200}]}
        self.responder.get_last_call = lambda: detail
        result = self.core.submit(self.sid, "話", "request_01")
        self.assertNotIn("diagnostics", result)
        self.assertEqual(self.stored()["requests"]["request_01"]["diagnostics"], detail)
        detail["attempts"][0]["status"] = "mutated"
        self.assertEqual(self.core._sessions[self.sid]["requests"]["request_01"]["diagnostics"]
                         ["attempts"][0]["status"], "completed")

    def test_failed_call_diagnostics_are_preserved(self):
        self.responder.answer = TimeoutError("private provider error")
        detail = {"status": "failed", "error_type": "APITimeoutError", "request_ms": 1000.0}
        self.responder.get_last_call = lambda: detail
        self.core.submit(self.sid, "話", "request_01")
        request = self.stored()["requests"]["request_01"]
        self.assertEqual(request["status"], "failed")
        self.assertEqual(request["diagnostics"], detail)
        self.assertNotIn("private", json.dumps(self.stored()))

    def test_invalid_optional_diagnostics_do_not_fail_successful_generation(self):
        for index, value in enumerate((None, {"invalid": object()}, {"invalid": float("nan")})):
            self.responder.get_last_call = lambda value=value: value
            result = self.core.submit(self.sid, "話", f"request_{index:02}")
            self.assertIsNone(result["error"])
            self.assertNotIn("diagnostics", self.stored()["requests"][f"request_{index:02}"])
        def failing_hook():
            raise RuntimeError("optional diagnostic hook failed")
        self.responder.get_last_call = failing_hook
        self.assertIsNone(self.core.submit(self.sid, "話", "request_03")["error"])

    def test_pre_cancelled_request_does_not_reuse_previous_call_diagnostics(self):
        self.responder.get_last_call = lambda: self.fail("No responder call was made")
        cancelled = threading.Event()
        cancelled.set()
        self.core.submit_cancellable(self.sid, "話", "request_01", cancelled)
        self.assertNotIn("diagnostics", self.stored()["requests"]["request_01"])

    def test_restart_marks_inflight_request_without_resending(self):
        self.core.close()
        state = self.stored()
        state["busy"] = True
        state["requests"]["request_01"] = {"text": "中断した入力", "status": "running"}
        file = Path(self.temp.name) / "sessions" / self.sid / "session.json"
        file.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        responder = FakeResponder()
        restarted = ChatCore(self.temp.name, settings=self.settings, responder=responder)
        self.addCleanup(restarted.close)
        result = restarted.submit(self.sid, "中断した入力", "request_01")
        self.assertEqual(result["error"], SAFE_RECOVERY_ERROR)
        self.assertFalse(result["busy"])
        self.assertEqual(responder.calls, [])
        self.assertEqual(self.stored()["requests"]["request_01"]["status"], "interrupted")

    def test_same_session_serializes_and_duplicate_inflight_is_not_resent(self):
        entered, release = threading.Event(), threading.Event()
        original = self.responder.respond
        def slow(history, settings):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test release missing")
            return original(history, settings)
        self.responder.respond = slow
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.core.submit, self.sid, "話", "request_01")
            self.assertTrue(entered.wait(2))
            second = pool.submit(self.core.submit, self.sid, "話", "request_01")
            self.assertFalse(second.done())
            release.set()
            self.assertEqual(first.result(3), second.result(3))
        self.assertEqual(len(self.responder.calls), 1)

    def test_other_session_is_not_blocked_by_model_call(self):
        entered, release = threading.Event(), threading.Event()
        original = self.responder.respond
        def slow(history, settings):
            if history[-1]["content"] == "遅い話":
                entered.set()
                if not release.wait(3):
                    raise TimeoutError("test release missing")
            return original(history, settings)
        self.responder.respond = slow
        second_sid = self.core.create()["id"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.core.submit, self.sid, "遅い話", "request_01")
            self.assertTrue(entered.wait(2))
            second = pool.submit(self.core.submit, second_sid, "早い話", "request_02")
            try:
                self.assertIsNone(second.result(2)["error"])
            finally:
                release.set()
            first.result(3)

    def test_close_during_generation_does_not_publish_late_reply(self):
        entered, release = threading.Event(), threading.Event()
        original = self.responder.respond
        def slow(history, settings):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test release missing")
            return original(history, settings)
        self.responder.respond = slow
        self.core.delivery(self.sid, self.core.pending(self.sid)[0]["id"], "completed")
        with ThreadPoolExecutor(max_workers=1) as pool:
            task = pool.submit(self.core.submit, self.sid, "話", "request_01")
            self.assertTrue(entered.wait(2))
            self.core.close()
            release.set()
            result = task.result(3)
        self.assertEqual(result["error"], SAFE_RECOVERY_ERROR)
        self.assertEqual(self.core.pending(self.sid), [])
        self.core.close()
        self.assertEqual(self.responder.closed, 1)
        with self.assertRaises(RuntimeError):
            self.core.create()
        with self.assertRaises(RuntimeError):
            self.core.submit(self.sid, "新しい話", "request_02")

    def test_cancellation_preserves_input_and_does_not_fail_the_session(self):
        entered = threading.Event()
        def cancellable(history, settings, cancel_event):
            self.responder.calls.append(deepcopy(history))
            entered.set()
            if not cancel_event.wait(2):
                raise TimeoutError("Cancellation was not delivered")
            raise ChatGenerationCancelled("cancelled")
        self.responder.respond_cancellable = cancellable
        with ThreadPoolExecutor(max_workers=1) as pool:
            task = pool.submit(self.core.submit, self.sid, "元の入力", "request_01")
            self.assertTrue(entered.wait(2))
            self.assertTrue(self.core.cancel(self.sid))
            result = task.result(2)
        self.assertEqual(result["request_status"], "interrupted")
        self.assertIsNone(result["error"])
        self.assertFalse(result["busy"])
        self.assertEqual(result["messages"][-1]["content"], "元の入力")
        self.assertEqual(result["messages"][-1]["role"], "user")
        self.assertEqual(self.stored()["requests"]["request_01"]["status"], "interrupted")
        self.assertEqual(self.core.submit(self.sid, "元の入力", "request_01"), result)
        self.assertEqual(len(self.responder.calls), 1)
        self.assertFalse(self.core.cancel(self.sid))
        del self.responder.respond_cancellable
        following = self.core.submit(self.sid, "訂正した入力", "request_02")
        self.assertEqual(following["request_status"], "completed")
        self.assertEqual([row["content"] for row in self.responder.calls[-1]], ["元の入力", "訂正した入力"])

    def test_pre_cancelled_submit_persists_input_without_model_call(self):
        cancel_event = threading.Event()
        cancel_event.set()
        result = self.core.submit_cancellable(self.sid, "追加前の入力", "request_01", cancel_event)
        self.assertEqual(result["request_status"], "interrupted")
        self.assertIsNone(result["error"])
        self.assertEqual(result["messages"][-1]["content"], "追加前の入力")
        self.assertEqual(self.responder.calls, [])
        self.assertEqual(self.stored()["requests"]["request_01"]["status"], "interrupted")

    def test_legacy_responder_late_reply_is_discarded_after_cancel(self):
        entered, release = threading.Event(), threading.Event()
        original = self.responder.respond
        def slow(history, settings):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("Test release missing")
            return original(history, settings)
        self.responder.respond = slow
        with ThreadPoolExecutor(max_workers=1) as pool:
            task = pool.submit(self.core.submit, self.sid, "元の入力", "request_01")
            self.assertTrue(entered.wait(2))
            self.assertTrue(self.core.cancel(self.sid))
            self.assertFalse(task.done())  # A legacy synchronous provider cannot be force-stopped.
            release.set()
            result = task.result(2)
        self.assertEqual(result["request_status"], "interrupted")
        self.assertIsNone(result["error"])
        self.assertEqual(result["messages"][-1]["role"], "user")

    def test_cancel_is_session_scoped_and_does_not_close_responder(self):
        entered = threading.Event()
        original = self.responder.respond
        def cancellable(history, settings, cancel_event):
            if history[-1]["content"] == "遅い話":
                entered.set()
                if not cancel_event.wait(2):
                    raise TimeoutError("Cancellation missing")
                raise ChatGenerationCancelled("cancelled")
            return original(history, settings)
        self.responder.respond_cancellable = cancellable
        second_sid = self.core.create()["id"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.core.submit, self.sid, "遅い話", "request_01")
            self.assertTrue(entered.wait(2))
            second = pool.submit(self.core.submit, second_sid, "早い話", "request_02")
            self.assertTrue(self.core.cancel(self.sid))
            self.assertEqual(first.result(2)["request_status"], "interrupted")
            self.assertEqual(second.result(2)["request_status"], "completed")
        self.assertEqual(self.responder.closed, 0)

    def test_corrupt_session_is_not_overwritten_or_loaded(self):
        self.core.close()
        file = Path(self.temp.name) / "sessions" / self.sid / "session.json"
        file.write_text('{"secret":"bad-record"}', encoding="utf-8")
        restarted = ChatCore(self.temp.name, settings=self.settings, responder=FakeResponder())
        self.addCleanup(restarted.close)
        self.assertEqual(restarted.recovery_errors, 1)
        with self.assertRaises(KeyError):
            restarted.view(self.sid)
        self.assertEqual(file.read_text(encoding="utf-8"), '{"secret":"bad-record"}')


if __name__ == "__main__":
    unittest.main()

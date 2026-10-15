"""Offline observations: snapshots must never advance or change a turn."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from speech_runtime import SpeechRuntime
from jev_runtime import JevSpeechRuntime
from turn_progress import ProcessingProgress


HERE = Path(__file__).resolve().parents[1]


class NoCoreCalls:
    def __getattr__(self, name):
        raise AssertionError("Snapshot must not call core: " + name)


class ProgressTests(unittest.TestCase):
    def setUp(self):
        root = HERE / ".runtime" / "progress_tests"
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.runtimes = []
        self.now = 100.0

    def tearDown(self):
        for r in self.runtimes: r.logfile.close()
        self.temp.cleanup()

    def runtime(self, jev=False):
        client = SimpleNamespace(pending_audio=False)
        async def forbidden(*args):
            raise AssertionError("A snapshot must not send messages or perform I/O")
        kwargs = {"client": client, "speech": object()}
        if jev: kwargs["jev"] = object()
        cls = JevSpeechRuntime if jev else SpeechRuntime
        r = cls(NoCoreCalls(), str(len(self.runtimes)), forbidden, forbidden, Path(self.temp.name), **kwargs)
        self.runtimes.append(r)
        r.ready = True
        r.started = 90
        r._processing.started = 90
        r._processing.resolved_input_at = 90
        return r

    def snapshot(self, r):
        return r._processing.snapshot(r, self.now)

    def transcript(self, r, *, candidate=False, final=True):
        r.segments["one"] = {"id": "one", "text": "今日は楽しかった", "sequence": 0, "arrival": 0,
                              "final": final, "lane": "candidate" if candidate else "normal", "context": {}}
        if not candidate: r.input_text = "今日は楽しかった"
        r.input_last_voice = r.input_last_change = 98

    def verdict(self, r, choice, *, due=100.2, candidate=False):
        r.verdict = {"snapshot": r._snapshot("candidate" if candidate else "normal"), "choice": choice, "due": due}

    def preparation(self, r, *, text="はい、楽しかったです", final=True, choice=None, due=100.25):
        r.segments["extra"] = {"id": "extra", "text": text, "sequence": 1, "arrival": 1,
                               "final": final, "lane": "preparation", "context": {}}
        r.input_last_voice = r.input_last_change = 98
        r._processing.configure(preparation_maximum=3.0, preparation_failure=2.0)
        if choice:
            r.verdict = {"snapshot": r._snapshot("preparation"), "choice": choice, "due": due}

    def test_connecting_error_and_close_override_stale_work(self):
        r = self.runtime(); r.ready = False
        self.assertEqual(self.snapshot(r)["turn"]["code"], "connecting")
        r.busy = True; r.approved = {"id": "old"}; r.closed = True
        self.assertEqual(self.snapshot(r)["reply"]["code"], "closed")
        r.failure_message = "接続に失敗しました。"
        self.assertEqual(self.snapshot(r)["reply"]["code"], "error")

    def test_backlog_wait_is_audio_quantity_not_completion_eta(self):
        r = self.runtime(); r.microphone_backlog_bytes = 24000
        turn = self.snapshot(r)["turn"]
        self.assertEqual(turn["code"], "audio_backlog")
        self.assertNotIn("remaining_ms", turn)
        self.assertIn("0.5秒分", turn["detail"])

    def test_live_quiet_and_text_stability_use_current_input(self):
        r = self.runtime(); r.input_text = "話"; r.input_last_voice = 99.8; r.input_last_change = 99.7
        a = self.snapshot(r)["turn"]
        self.assertEqual(a["code"], "live_stability"); self.assertEqual(a["remaining_ms"], 350)
        self.now += .2
        b = self.snapshot(r)["turn"]
        self.assertEqual(b["remaining_ms"], 150)
        self.assertEqual(self.snapshot(r)["recent"], [])

    def test_live_boundary_and_fallback_condition_are_not_api_eta(self):
        r = self.runtime(); r.input_text = "話"; r.input_last_voice = r.input_last_change = 98
        turn = self.snapshot(r)["turn"]
        self.assertEqual(turn["code"], "live_boundary_wait"); self.assertEqual(turn["remaining_ms"], 1500)
        r.delegation = {"offset_ms": 1000}
        self.assertEqual(self.snapshot(r)["turn"]["code"], "live_boundary_received")

    def test_explicit_interruption_wait_has_no_automatic_reply_countdown(self):
        for jev in (False, True):
            r = self.runtime(jev=jev)
            r.input_text = "待って。"
            r.waiting_for_continuation = "pause"
            r.input_last_voice = r.input_last_change = 80
            result = self.snapshot(r)
            self.assertEqual(result["turn"]["code"], "interruption_wait")
            self.assertEqual(result["reply"]["code"], "interruption_wait")
            self.assertNotIn("remaining_ms", result["turn"])
            self.assertNotIn("remaining_ms", result["reply"])

    def test_microphone_without_text_is_recognition_wait(self):
        r = self.runtime(); r.input_last_voice = 99
        self.assertEqual(self.snapshot(r)["turn"]["code"], "asr_wait")
        r._processing.event("input_committed", {"source": "live_delegation"}, 100)
        self.assertEqual(self.snapshot(r)["turn"]["code"], "listening")

    def test_preparation_ack_and_incomplete_tag_are_visible(self):
        r = self.runtime(); r.reply_input_text = "そう"; r.reply_input_started = 99
        self.assertEqual(self.snapshot(r)["turn"]["code"], "preparation_ack")
        self.assertEqual(self.snapshot(r)["turn"]["remaining_ms"], 1000)
        r.reply_input_text = ""; r.input_filter.pending = "[s"
        self.assertEqual(self.snapshot(r)["turn"]["code"], "transcript_fragment")

    def test_jev_api_active_and_stale_result_wait(self):
        r = self.runtime(jev=True); self.transcript(r)
        r.judge_task = SimpleNamespace(done=lambda: False)
        r._processing.judge_started(r._snapshot("normal"), 99.8)
        a = self.snapshot(r)["turn"]
        self.assertEqual(a["code"], "jev_judging"); self.assertNotIn("remaining_ms", a)
        r.input_revision += 1
        self.assertEqual(self.snapshot(r)["turn"]["code"], "jev_updated_input")

    def test_jev_final_asr_blocks_grace_display_until_actual_final(self):
        r = self.runtime(jev=True); self.transcript(r, final=False); self.verdict(r, "complete")
        self.assertEqual(self.snapshot(r)["turn"]["code"], "asr_final_wait")
        r.segments["one"]["final"] = True
        turn = self.snapshot(r)["turn"]
        self.assertEqual(turn["code"], "jev_complete_grace"); self.assertEqual(turn["remaining_ms"], 200)

    def test_jev_pending_audio_and_empty_pending_row_have_no_false_idle(self):
        r = self.runtime(jev=True); r.client.pending_audio = True
        self.assertEqual(self.snapshot(r)["turn"]["code"], "asr_final_wait")
        r.client.pending_audio = False; self.transcript(r, final=False); r.segments["one"]["text"] = ""
        self.assertEqual(self.snapshot(r)["turn"]["code"], "asr_final_wait")

    def test_short_long_and_uncertain_show_cached_wait_and_bounded_recovery(self):
        for choice in ("short", "long", "uncertain"):
            r = self.runtime(jev=True); self.transcript(r); self.verdict(r, choice, due=100.5)
            turn = self.snapshot(r)["turn"]
            self.assertEqual(turn["code"], "jev_"+choice+"_wait")
            self.assertEqual(turn["remaining_ms"], 6000)
            self.assertIn("判定を保持", turn["detail"])
            self.assertIn("6.0秒", turn["detail"])

    def test_failed_judge_recovery_and_candidate_ack_are_separate(self):
        r = self.runtime(jev=True); self.transcript(r); self.verdict(r, "error")
        self.assertEqual(self.snapshot(r)["turn"]["code"], "jev_failure_recovery")
        r = self.runtime(jev=True); self.transcript(r, candidate=True); r.input_last_voice = 99.6
        self.verdict(r, "backchannel", candidate=True)
        self.assertEqual(self.snapshot(r)["turn"]["code"], "jev_ack_wait")
        self.assertEqual(self.snapshot(r)["turn"]["remaining_ms"], 400)

    def test_stale_verdict_is_not_presented_as_current_decision(self):
        r = self.runtime(jev=True); self.transcript(r); self.verdict(r, "complete")
        r.buffer_version += 1
        self.assertEqual(self.snapshot(r)["turn"]["code"], "jev_judge_wait")
        self.assertIsNotNone(r.verdict)  # Observation did not clear it.

    def test_reply_phases_require_real_browser_start_and_matching_message(self):
        r = self.runtime(); r.approved = {"id": "one"}; r.approved_at = 98
        p = r._processing
        p.event("approved_text_sent", {"id": "one"}, 98)
        self.assertEqual(self.snapshot(r)["reply"]["code"], "tts_preparing")
        self.assertNotIn("remaining_ms", self.snapshot(r)["reply"])
        p.event("speech_ready", {"id": "one"}, 99)
        self.assertEqual(self.snapshot(r)["reply"]["code"], "tts_ready")
        r.playing, r.approved = r.approved, None
        self.assertEqual(self.snapshot(r)["reply"]["code"], "playback_buffering")
        p.event("browser_audio", {"stage": "playing", "id": "old"}, 99)
        self.assertEqual(self.snapshot(r)["reply"]["code"], "playback_buffering")
        p.event("browser_audio", {"stage": "playing", "id": "one"}, 99.5)
        self.assertEqual(self.snapshot(r)["reply"]["code"], "playing")
        self.assertEqual(self.snapshot(r)["reply"]["elapsed_ms"], 500)

    def test_new_approval_cannot_reuse_old_audio_ready_or_playing(self):
        r = self.runtime(); r.approved = {"id": "new"}; r.approved_at = 100
        p = r._processing
        p.event("approved_text_sent", {"id": "old"}, 98)
        p.event("speech_ready", {"id": "old"}, 99)
        p.event("approved_text_sent", {"id": "new"}, 100)
        p.event("speech_ready", {"id": "old"}, 100)
        self.assertEqual(self.snapshot(r)["reply"]["code"], "tts_preparing")

    def test_new_input_during_old_core_has_independent_turn_and_reply(self):
        r = self.runtime(); r.busy = True; r.input_text = "さらに"; r.input_last_change = 99.9
        r._processing.event("core_submit_started", {}, 97)
        snap = self.snapshot(r)
        self.assertEqual(snap["reply"]["code"], "core_processing")
        self.assertEqual(snap["reply"]["elapsed_ms"], 3000)
        self.assertEqual(snap["turn"]["code"], "live_stability")

    def test_preparation_empty_or_partial_input_waits_for_asr_with_reply_held(self):
        for text, final in (("", False), ("はい、でも", False)):
            r = self.runtime(jev=True); self.preparation(r, text=text, final=final); r.busy = True
            self.fill_jobs(r)
            snap = self.snapshot(r)
            self.assertEqual(snap["turn"]["code"], "jev_preparation_asr_wait" if text else "jev_audio_without_words")
            self.assertEqual(snap["reply"]["code"], "core_processing")
            self.assertIn("保持", snap["reply"]["detail"])
            self.assertNotIn("remaining_ms", snap["turn"])

    def test_empty_candidate_and_preparation_are_audio_recognition_not_jev_judging(self):
        for lane in ("candidate", "preparation"):
            r = self.runtime(jev=True); self.preparation(r, text="", final=False)
            r.segments["extra"]["lane"] = lane
            r.judge_task = SimpleNamespace(done=lambda: False)
            r._processing.judge_started(r._snapshot(lane), 99)
            turn = self.snapshot(r)["turn"]
            self.assertEqual(turn["code"], "jev_audio_without_words")
            self.assertEqual(turn["label"], "音を検出・言葉を確認中")
            self.assertIn("言葉はまだ認識されていません", turn["detail"])
            self.assertNotIn("remaining_ms", turn)

    def test_audio_only_confirmation_timer_does_not_promise_playback_at_deadline(self):
        r = self.runtime(jev=True); self.preparation(r, text="", final=False)
        r.noise_wait_started = 99
        r._noise_wait_can_release = lambda _now: False
        turn = self.snapshot(r)["turn"]
        self.assertEqual(turn["remaining_ms"], 1000)
        self.assertIn("他の条件", turn["wait_label"])
        self.now = 102
        turn = self.snapshot(r)["turn"]
        self.assertEqual(turn["code"], "jev_audio_without_words")
        self.assertNotIn("remaining_ms", turn)
        self.assertIn("空の確定結果", turn["detail"])
        r._noise_wait_can_release = lambda _now: True
        self.assertEqual(self.snapshot(r)["turn"]["code"], "jev_audio_release_ready")

    def test_audio_only_transport_barrier_never_claims_noise_release(self):
        r = self.runtime(jev=True); self.preparation(r, text="", final=False)
        r.noise_wait_started = 97
        r.microphone_backlog_bytes = 24000
        r._noise_wait_can_release = lambda _now: True
        turn = self.snapshot(r)["turn"]
        self.assertEqual(turn["code"], "jev_audio_without_words")
        self.assertIn("0.5秒分", turn["detail"])
        self.assertNotIn("remaining_ms", turn)

    def test_empty_noise_rows_do_not_mask_recognized_normal_input(self):
        r = self.runtime(jev=True); self.preparation(r, text="", final=False)
        r.segments["extra"]["lane"] = "candidate"
        self.transcript(r)
        self.verdict(r, "complete")
        self.assertEqual(self.snapshot(r)["turn"]["code"], "jev_complete_grace")

    def test_audio_only_ready_reply_distinguishes_asr_wait_from_semantic_judgment(self):
        r = self.runtime(jev=True); self.preparation(r, text="", final=False)
        r.approved = {"id": "first"}; r.approved_at = 98
        r._processing.event("approved_text_sent", {"id": "first"}, 98)
        reply = self.snapshot(r)["reply"]
        self.assertEqual(reply["code"], "tts_preparing")
        self.assertIn("音声認識を確認", reply["detail"])
        r._processing.event("speech_ready", {"id": "first"}, 99)
        reply = self.snapshot(r)["reply"]
        self.assertEqual(reply["code"], "tts_ready")
        self.assertIn("言葉はまだ認識されていません", reply["detail"])
        self.assertNotIn("追加入力の確認", reply["detail"])

    def test_preparation_transport_wait_does_not_claim_asr_or_api_completion_time(self):
        r = self.runtime(jev=True); self.preparation(r, choice="retain")
        r.microphone_backlog_bytes = 24000
        turn = self.snapshot(r)["turn"]
        self.assertEqual(turn["code"], "jev_preparation_transport_wait")
        self.assertIn("0.5秒分", turn["detail"])
        self.assertIn("保持", turn["detail"])
        self.assertNotIn("remaining_ms", turn)

    def test_preparation_quiet_and_stability_are_separate_from_judge_latency(self):
        r = self.runtime(jev=True); self.preparation(r); r.input_last_voice = 99.6
        turn = self.snapshot(r)["turn"]
        self.assertEqual(turn["code"], "jev_preparation_judge_wait")
        # Final words can be sent to Jev despite sound; changing text still waits.
        r.input_last_change = 99.9
        turn = self.snapshot(r)["turn"]
        self.assertEqual(turn["code"], "jev_preparation_stability")
        self.assertEqual(turn["remaining_ms"], 100)
        r.judge_task = SimpleNamespace(done=lambda: False)
        r._processing.judge_started(r._snapshot("preparation"), 99.9)
        turn = self.snapshot(r)["turn"]
        self.assertEqual(turn["code"], "jev_preparation_judging")
        self.assertNotIn("remaining_ms", turn)
        r.buffer_version += 1
        self.assertEqual(self.snapshot(r)["turn"]["code"], "jev_preparation_updated_input")

    def test_sound_without_text_changes_keeps_active_jev_request_current(self):
        for lane in ("normal", "preparation"):
            r = self.runtime(jev=True)
            self.preparation(r) if lane == "preparation" else self.transcript(r)
            r.judge_task = SimpleNamespace(done=lambda: False)
            r._processing.judge_started(r._snapshot(lane), 99)
            r.voice_epoch += 1
            r.input_last_voice = 99.99
            expected = "jev_preparation_judging" if lane == "preparation" else "jev_judging"
            self.assertEqual(self.snapshot(r)["turn"]["code"], expected)

    def test_cached_jev_result_waits_for_audio_without_claiming_api_reclassification(self):
        for lane in ("normal", "preparation"):
            r = self.runtime(jev=True)
            if lane == "preparation": self.preparation(r, choice="retain")
            else:
                self.transcript(r)
                self.verdict(r, "complete")
            r.voice_epoch += 1
            r.input_last_voice = 99.99
            r.client.pending_audio = True
            r.segments["noise"] = {"id": "noise", "text": "", "sequence": 4, "arrival": 4,
                                   "final": False, "lane": "candidate", "context": {}}
            r.noise_turn_key, r.noise_turn_started = ("sentinel",), 98
            def forbidden(*args):
                raise AssertionError("UI must not advance the acoustic recovery timer")
            r._turn_noise_recovery = forbidden
            expected = "jev_preparation_decision_audio_wait" if lane == "preparation" else "jev_decision_audio_wait"
            turn = self.snapshot(r)["turn"]
            self.assertEqual(turn["code"], expected)
            self.assertIn("判定結果を保持", turn["detail"])
            self.assertNotIn("再判定", turn["detail"])
            self.assertNotIn("remaining_ms", turn)
            self.assertEqual(r.noise_turn_key, ("sentinel",))
            self.assertEqual(r.noise_turn_started, 98)

    def test_final_normal_words_prepare_judgment_while_raw_sound_continues(self):
        r = self.runtime(jev=True); self.transcript(r)
        r.input_last_voice = 99.99
        r.client.pending_audio = True
        self.assertEqual(self.snapshot(r)["turn"]["code"], "jev_judge_wait")

    def test_preparation_retain_requires_final_asr_and_current_snapshot(self):
        r = self.runtime(jev=True); self.preparation(r, choice="retain", final=False)
        self.assertEqual(self.snapshot(r)["turn"]["code"], "jev_preparation_asr_wait")
        r.segments["extra"]["final"] = True
        turn = self.snapshot(r)["turn"]
        self.assertEqual(turn["code"], "jev_preparation_retain_grace")
        self.assertEqual(turn["remaining_ms"], 250)
        r.buffer_version += 1
        self.assertEqual(self.snapshot(r)["turn"]["code"], "jev_preparation_judge_wait")
        self.assertIsNotNone(r.verdict)

    def test_preparation_uncertain_and_failure_show_own_bounded_recovery(self):
        r = self.runtime(jev=True); self.preparation(r, choice="uncertain", due=100.5)
        turn = self.snapshot(r)["turn"]
        self.assertEqual(turn["code"], "jev_preparation_uncertain_wait")
        self.assertEqual(turn["remaining_ms"], 1000)
        self.assertIn("1.0秒", turn["detail"])
        r.verdict["due"] = 102
        turn = self.snapshot(r)["turn"]
        self.assertEqual(turn["remaining_ms"], 1000)
        self.assertEqual(turn["wait_label"], "追加入力を返答へ反映する条件まで")
        r.verdict["choice"] = "error"
        self.assertEqual(self.snapshot(r)["turn"]["code"], "jev_preparation_error_wait")
        self.assertEqual(self.snapshot(r)["turn"]["remaining_ms"], 0)
        r.verdict["choice"] = "revise"
        self.assertEqual(self.snapshot(r)["turn"]["code"], "jev_preparation_revise")

    def test_preparation_review_does_not_claim_timeout_will_replace_reply(self):
        r = self.runtime(jev=True); self.preparation(r, choice="uncertain")
        r.verdict["needs_review"] = True
        turn = self.snapshot(r)["turn"]
        self.assertEqual(turn["code"], "jev_preparation_review_wait")
        self.assertIn("一度だけ", turn["detail"])
        self.assertNotIn("remaining_ms", turn)

    def test_backchannel_dominant_wait_does_not_claim_timed_interruption(self):
        r = self.runtime(jev=True)
        self.transcript(r, candidate=True)
        self.verdict(r, "uncertain", candidate=True)
        r.verdict["backchannel_dominant"] = True
        turn = self.snapshot(r)["turn"]
        self.assertEqual(turn["code"], "jev_likely_ack_wait")
        self.assertIn("割込みに変わりません", turn["detail"])
        self.assertNotIn("remaining_ms", turn)

    def test_preparation_holds_reply_in_all_pre_playback_phases(self):
        r = self.runtime(jev=True); self.preparation(r); r.busy = True
        self.assertIn("保持", self.snapshot(r)["reply"]["detail"])
        r.busy = False; r.approved = {"id": "first"}; r.approved_at = 98
        r._processing.event("approved_text_sent", {"id": "first"}, 98)
        snap = self.snapshot(r)
        self.assertEqual(snap["reply"]["code"], "tts_preparing")
        self.assertIn("保持して準備を続けています", snap["reply"]["detail"])
        r._processing.event("speech_ready", {"id": "first"}, 99)
        snap = self.snapshot(r)
        self.assertEqual(snap["reply"]["code"], "tts_ready")
        self.assertIn("保持して再生を待っています", snap["reply"]["detail"])



    def test_playback_candidate_remains_visible_if_preparation_rows_exist(self):
        r = self.runtime(jev=True); self.preparation(r); self.transcript(r, candidate=True)
        self.verdict(r, "backchannel", candidate=True)
        self.assertEqual(self.snapshot(r)["turn"]["code"], "jev_ack_wait")

    def test_preparation_events_show_decisions_without_user_text(self):
        r = self.runtime(jev=True); p = r._processing
        p.event("jev_preparation_classification", {"preparation": {"choice": "retain"}, "text": "private"}, 98)
        self.assertIn("同じ内容", self.snapshot(r)["recent"][0]["detail"])
        p.event("preparation_retained", {"text": "private", "reason": "jev_redundant"}, 99)
        self.assertIn("保持しました", self.snapshot(r)["recent"][0]["label"])
        p.event("preparation_promoted", {"text": "private", "reason": "jev_revise"}, 100)
        self.assertIn("見直します", self.snapshot(r)["recent"][0]["label"])
        self.assertNotIn("private", json.dumps(self.snapshot(r)))

    def test_noise_release_event_is_visible_without_transcript_or_api_claim(self):
        r = self.runtime(jev=True)
        r._processing.event("noise_only_wait_released", {"waited_ms": 2300, "empty_asr_age_ms": 100}, 100)
        recent = self.snapshot(r)["recent"][0]
        self.assertEqual(recent["label"], "言葉のない音声区間を確認しました")
        self.assertNotIn("Jev", recent["detail"])
        self.assertEqual(r._processing.resolved_input_at, 100)

    def test_preparation_observation_never_promotes_or_consumes_input(self):
        r = self.runtime(jev=True); self.preparation(r, choice="retain", due=99)
        r.busy = True
        before = (deepcopy(r.segments), deepcopy(r.verdict), r.input_text, r.input_revision,
                  r.output_generation, r.jobs.qsize(), r.next_judge_at)
        for _ in range(20): self.snapshot(r)
        after = (r.segments, r.verdict, r.input_text, r.input_revision,
                 r.output_generation, r.jobs.qsize(), r.next_judge_at)
        self.assertEqual(before, after)

    def test_recent_is_event_only_bounded_and_omits_user_text(self):
        r = self.runtime(); p = r._processing
        p.event("session.input_transcript.delta", {"delta": "private-text"}, 98)
        p.event("input_committed", {"text": "private-text", "source": "typed"}, 99)
        recent = self.snapshot(r)["recent"]
        for i in range(30):
            self.now += .1; self.snapshot(r)
        self.assertEqual(self.snapshot(r)["recent"], recent)
        self.assertNotIn("private-text", json.dumps(recent))
        for i in range(8):
            p.event("input_committed" if i % 2 else "interrupted", {"source": "typed"}, self.now)
        self.assertEqual(len(self.snapshot(r)["recent"]), 6)
        value = self.snapshot(r); value["recent"][0]["label"] = "changed"
        self.assertNotEqual(self.snapshot(r)["recent"][0]["label"], "changed")

    def test_polling_does_not_touch_conversation_or_api(self):
        r = self.runtime(jev=True); self.transcript(r); self.verdict(r, "long")
        before = (deepcopy(r.segments), deepcopy(r.verdict), r.input_text, r.input_revision,
                  r.output_generation, r.jobs.qsize(), r.next_judge_at)
        for i in range(100): self.snapshot(r)
        after = (r.segments, r.verdict, r.input_text, r.input_revision, r.output_generation, r.jobs.qsize(), r.next_judge_at)
        self.assertEqual(before, after)

    def fill_jobs(self, r):
        for i in range(r.jobs.maxsize):
            r.jobs.put_nowait(("先の入力", str(i), i))

    def test_candidate_judging_is_not_hidden_by_backlog_or_full_queue(self):
        r = self.runtime(jev=True); self.transcript(r, candidate=True)
        r.microphone_backlog_bytes = 24000
        self.fill_jobs(r)
        r.judge_task = SimpleNamespace(done=lambda: False)
        r._processing.judge_started(r._snapshot("candidate"), 99.8)
        turn = self.snapshot(r)["turn"]
        self.assertEqual(turn["code"], "jev_judging")
        self.assertIn("終話・相槌の確定", turn["detail"])
        self.assertIn("音声の送信", turn["detail"])
        self.assertFalse(r.jobs.empty())

    def test_candidate_verdict_waits_for_transport_not_falsely_for_asr(self):
        r = self.runtime(jev=True); self.transcript(r, candidate=True)
        self.verdict(r, "backchannel", candidate=True)
        r.microphone_backlog_bytes = 24000
        self.assertEqual(self.snapshot(r)["turn"]["code"], "jev_candidate_transport_wait")

    def test_live_full_input_queue_explains_commit_barrier(self):
        r = self.runtime(); r.input_text = "次の話"; self.fill_jobs(r)
        turn = self.snapshot(r)["turn"]
        self.assertEqual(turn["code"], "input_queue_full")
        self.assertNotIn("remaining_ms", turn)
        r.input_text = ""
        self.assertNotEqual(self.snapshot(r)["turn"]["code"], "input_queue_full")

    def test_jev_full_queue_only_masks_a_commit_that_is_otherwise_ready(self):
        r = self.runtime(jev=True); self.transcript(r); self.fill_jobs(r)
        self.verdict(r, "complete", due=100.2)
        self.assertEqual(self.snapshot(r)["turn"]["code"], "jev_complete_grace")
        self.now = 100.3
        self.assertEqual(self.snapshot(r)["turn"]["code"], "input_queue_full")
        r.segments["one"]["final"] = False
        self.assertEqual(self.snapshot(r)["turn"]["code"], "asr_final_wait")

    def test_jev_full_queue_explains_error_and_bounded_recovery_barriers(self):
        for choice, waited in (("error", 3), ("long", 9), ("short", 9), ("uncertain", 9)):
            r = self.runtime(jev=True); self.transcript(r); self.fill_jobs(r)
            r.input_last_voice = r.input_last_change = self.now-waited
            self.verdict(r, choice, due=99)
            self.assertEqual(self.snapshot(r)["turn"]["code"], "input_queue_full")

    def test_full_queue_does_not_hide_candidate_ack_classification(self):
        r = self.runtime(jev=True); self.transcript(r, candidate=True); self.fill_jobs(r)
        self.verdict(r, "backchannel", candidate=True)
        self.assertEqual(self.snapshot(r)["turn"]["code"], "jev_ack_wait")

    def test_public_method_and_existing_log_hook(self):
        r = self.runtime()
        with patch("voice_runtime.time.monotonic", return_value=100):
            r.log("input_committed", source="live_delegation", text="private")
            snapshot = r.processing_snapshot()
        self.assertEqual(snapshot["recent"][0]["label"], "お話を確定しました")
        self.assertNotIn("private", json.dumps(snapshot))
        self.assertFalse(r.logfile.closed)


if __name__ == "__main__": unittest.main()

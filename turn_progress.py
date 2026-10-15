"""Read-only presentation of runtime state; never decides or schedules a turn.

Snapshots perform no I/O, send no messages, and never mutate conversation state.
Only actual runtime events enter the bounded recent-event list. Remaining times
describe existing wait conditions, never an estimate of API completion.
"""
from collections import deque
import math


def milliseconds(seconds):
    return max(0, round(seconds * 1000)) if math.isfinite(seconds) else 0


class ProcessingProgress:
    def __init__(self, started):
        self.started = started
        self.settings = {"mode": "live", "quiet": .55, "stable": .45, "recovery": 3.5,
                         "ack_stable": .75, "ack_quiet": .55, "ack_hold": 2.0}
        self.recent = deque(maxlen=6)
        self.event_times = {}
        self.resolved_input_at = started
        self.core_started = None
        self.core_active = False
        self.judge_started_at = None
        self.judge_snapshot = None
        self.speech_id = None
        self.speech_first = False
        self.speech_ready = False
        self.speech_ready_at = None
        self.browser_playing_id = None
        self.browser_playing_at = None
        self.phase_times = {}

    def configure(self, **settings):
        self.settings.update(settings)

    def judge_started(self, snapshot, now):
        self.judge_snapshot = snapshot
        self.judge_started_at = now

    def _remember(self, label, detail):
        value = {"label": label, "detail": detail}
        if not self.recent or self.recent[-1] != value:
            self.recent.append(value)

    def event(self, event, data, now):
        # PCM, waveform diagnostics and transcript deltas are deliberately not
        # retained here. The full existing log remains the diagnostic record.
        if event == "core_submit_started":
            self.core_started, self.core_active = now, True
        elif event == "core_submit_finished":
            self.core_active = False
        elif event == "approved_text_sent":
            self.speech_id = data.get("id")
            self.speech_first = self.speech_ready = False
            self.speech_ready_at = None
            self.browser_playing_id = None
            self.browser_playing_at = None
        elif event == "speech_first_audio" and data.get("id") == self.speech_id:
            self.speech_first = True
        elif event == "speech_ready" and data.get("id") == self.speech_id:
            self.speech_ready = True
            self.speech_ready_at = now
        elif event == "browser_audio" and data.get("stage") == "playing":
            self.browser_playing_id, self.browser_playing_at = data.get("id"), now
        elif event == "input_committed":
            self.resolved_input_at = now
            sources = {"typed": "文字入力を受け取りました。", "jev_complete": "Jevの区切り判定と待機条件を満たしました。",
                       "jev_failure_silence_fallback": "判定を取得できなかったため、静穏条件で確定しました。",
                       "jev_bounded_silence_fallback": "静穏・文字安定が回復待ちの上限に達しました。",
                       "live_delegation": "Liveの区切り通知を受けて確定しました。",
                       "live_response_boundary": "Liveの応答開始を区切りとして確定しました。",
                       "silence_recovery": "区切り通知がないため、静穏条件で確定しました。"}
            self._remember("お話を確定しました", sources.get(data.get("source"), "入力を返答処理へ渡しました。"))
        elif event in {"backchannel_ignored", "reply_preparation_ack"}:
            self.resolved_input_at = now
            self._remember("相槌として受け取りました", "準備中・再生中の返答を続けます。")
        elif event == "preparation_retained":
            self.resolved_input_at = now
            self._remember("同じ内容・相槌として返答を保持しました", "最初の返答の準備を続け、確認が済み次第再生します。")
        elif event == "preparation_promoted":
            self._remember("追加入力を反映して返答を見直します", "訂正や新しい内容を会話へ渡し、返答を確認します。")
        elif event == "noise_only_wait_released":
            self.resolved_input_at = now
            self._remember("言葉のない音声区間を確認しました", "認識された言葉がないことと待機条件を確認し、返答の再生へ進みます。")
        elif event == "jev_preparation_classification":
            choice = (data.get("preparation") or {}).get("choice")
            details = {"retain": "同じ内容・相槌の候補です。認識の確定と静穏を確認します。",
                       "revise": "返答へ反映すべき内容を検出しました。",
                       "uncertain": "返答を見直す必要があるか、認識文字の続きを確認します。"}
            self._remember("準備中の追加入力を判定しました", details.get(choice, "追加入力の扱いを確認しています。"))
        elif event == "session.input_transcript.completed" and data.get("segment_text") == "":
            self.resolved_input_at = now
            self._remember("認識結果が更新されました", "空の確定結果に合わせて入力を整理しました。")
        elif event == "jev_classification":
            choice = (data.get("turn") or {}).get("choice")
            labels = {"complete": "お話の区切りと判定しました", "short": "続きがあると判定しました",
                      "long": "考えている途中と判定しました"}
            interruption = (data.get("interruption") or {}).get("choice")
            if data.get("lane") == "candidate":
                detail = {"interrupt": "割込みの意図を検出しました。", "backchannel": "相槌の候補です。確定結果と静穏を確認します。",
                          "uncertain": "相槌か割込みかを引き続き確認します。"}.get(interruption, "入力の意図を確認しています。")
                self._remember("再生中のお話を判定しました", detail)
            elif choice in labels:
                self._remember(labels[choice], "認識結果と現在の待機条件を確認してから処理を進めます。")
        elif event == "jev_result_discarded":
            self._remember("最新のお話で判定を見直します", "入力や再生状態が変わったため、前の判定を使いません。")
        elif event == "jev_failed":
            self._remember("区切り判定を取得できませんでした", "認識の確定と静穏条件を確認して回復を待ちます。")
        elif event in {"local_explicit_stop", "interruption_promoted"}:
            self._remember("割込みを受け取りました", "読み上げの停止と、お話の区切りを別々に確認します。")
        elif event == "interrupted":
            self._remember("読み上げを停止しました", "次のお話を受け付けます。")
        elif event == "stale_reply_cancelled":
            self._remember("返答を切り替えます", "新しいお話が届いたため、前の返答を取り消しました。")
        elif event == "playback_completed":
            self._remember("再生完了を受け取りました", "会話への記録を確認しています。")
        elif event in {"core_error", "core_response_failed", "speech_failed", "speech_timeout", "task_error",
                       "live_error", "transcription_error", "connection_error"}:
            self._remember("処理中に問題が発生しました", "画面の案内を確認してください。")
        if event in {"input_committed", "interrupted", "playback_completed", "core_submit_started", "core_submit_finished"}:
            self.event_times[event] = now

    def _state(self, lane, now, code, label, detail, *, since=None, remaining=None, wait_label=None):
        old = self.phase_times.get(lane)
        if old is None or old[0] != code:
            self.phase_times[lane] = (code, now)
        result = {"code": code, "label": label, "detail": detail}
        # A phase observed for the first time starts at this snapshot unless an
        # actual runtime start time is available. No timer change creates events.
        result["elapsed_ms"] = milliseconds(now - (since if since is not None else self.phase_times[lane][1]))
        if remaining is not None:
            result["remaining_ms"] = milliseconds(remaining)
            result["wait_label"] = wait_label or "待機条件まで"
        return result

    def _turn(self, r, now):
        state = lambda *a, **kw: self._state("turn", now, *a, **kw)
        cfg = self.settings
        candidate_input = cfg["mode"] == "jev" and any(row["text"] for row in r._rows("candidate"))
        preparation_input = cfg["mode"] == "jev" and bool(r._rows("preparation"))
        if r.input_transport_pending and not candidate_input and not preparation_input:
            return state("audio_backlog", "音声の送信を待っています",
                         f"送信途中の音声が約{r.microphone_backlog_bytes / 48000:.1f}秒分あります。届くまでお話を確定しません。")
        if getattr(r, "waiting_for_continuation", None):
            return state("interruption_wait", "続きをお待ちしています",
                         "待って・否定の割込みを受け取りました。次の発話や文字入力があるまで返答を始めません。")
        if getattr(r, "reply_input_text", ""):
            return state("preparation_ack", "相槌か、お話の続きかを確認しています",
                         ("読み上げを続けながら相槌かどうかを確認します。訂正や続きがあれば停止します。" if r.playing else
                          "返答の音声は準備を続け、入力の確認が済むまで再生を待ちます。"),
                         since=r.reply_input_started,
                         remaining=cfg["ack_hold"] - (now-r.reply_input_started),
                         wait_label="入力保留の上限まで（返答の完成時間ではありません）")
        if getattr(getattr(r, "input_filter", None), "has_pending", False):
            return state("transcript_fragment", "認識文字の続きを確認しています",
                         "途中の注記や本文の続きがあるため、入力の確定を待っています。")
        if cfg["mode"] == "jev":
            result = self._jev_turn(r, now)
            if candidate_input and r.input_transport_pending:
                result["detail"] += " 終話・相槌の確定は、音声の送信が追いつくまで待ちます。"
            return result
        if r.input_text:
            if r.jobs.full():
                return self._queue_wait(now)
            quiet = now-r.input_last_voice
            stable = now-r.input_last_change
            remaining = max(cfg["quiet"]-quiet, cfg["stable"]-stable)
            if remaining > 0:
                return state("live_stability", "お話の区切りを確認しています",
                             "声と認識文字が落ち着くのを待っています。話し続けると待ち時間が更新されます。",
                             remaining=remaining, wait_label="声・文字が変わらない場合の確認待ち")
            native = r.delegation and r.delegation.get("offset_ms", 0) >= r.input_end_ms-350
            boundary = r.native_response_boundary >= r.input_last_voice and r.native_response_boundary > 0
            if native or boundary:
                return state("live_boundary_received", "区切りの合図を受け取りました", "入力を確定する条件を確認しています。")
            return state("live_boundary_wait", "Liveの区切りの合図を待っています",
                         "合図が先に届けば確定します。届かない場合は、声と文字が安定した状態が3.5秒続くと回復を試みます。",
                         remaining=cfg["recovery"]-min(quiet, stable), wait_label="区切り通知がない場合の回復条件まで")
        return self._idle_turn(r, now)

    def _queue_wait(self, now):
        return self._state("turn", now, "input_queue_full", "先の入力処理を待っています",
                           "入力の待ち行列が埋まっているため、処理が進むまで次のお話を確定できません。")

    def _idle_turn(self, r, now):
        state = lambda *a, **kw: self._state("turn", now, *a, **kw)
        if getattr(r.client, "pending_audio", False) or r.input_last_voice > self.resolved_input_at:
            return state("asr_wait", "認識結果を待っています", "音声を受け取りました。認識文字が届くのを待っています。")
        if r.busy or not r.jobs.empty():
            return state("input_committed", "お話は確定済みです", "返答を処理しています。続けて話すこともできます。",
                         since=self.event_times.get("input_committed"))
        return state("listening", "お話を待っています", "話しかけると音声と認識文字の状態を確認します。")

    def _jev_turn(self, r, now):
        state = lambda *a, **kw: self._state("turn", now, *a, **kw)
        cfg = self.settings
        if self._jev_audio_without_words(r):
            return self._jev_audio_wait(r, now)
        candidates = r._rows("candidate")
        lane = ("candidate" if any(row["text"] for row in candidates) else
                "preparation" if r._text("preparation") else "normal")
        if lane == "preparation":
            return self._jev_preparation(r, now)
        rows = candidates if lane == "candidate" else r._rows("normal")
        text = any(row["text"] for row in rows)
        if not text:
            if rows or candidates or getattr(r.client, "pending_audio", False):
                return state("asr_final_wait", "認識結果の確定を待っています", "音声区間の確定結果が届くまで、お話を確定しません。")
            return self._idle_turn(r, now)
        quiet = now-r.input_last_voice
        stable = now-r.input_last_change
        task = r.judge_task
        if task is not None and not task.done():
            if self.judge_snapshot is not None and not r._current(self.judge_snapshot):
                return state("jev_updated_input", "最新のお話で判定し直すのを待っています",
                             "先に送った判定が処理中です。更新前の結果は入力の確定に使いません。",
                             since=self.judge_started_at)
            return state("jev_judging", "Jevが相槌・割込みを判定しています" if lane == "candidate" else "Jevがお話の区切りを判定しています",
                         "判定の応答を待っています。APIの完了時刻は未定です。", since=self.judge_started_at)
        verdict = r.verdict
        if verdict and (verdict["snapshot"]["lane"] != lane or not r._current(verdict["snapshot"])):
            verdict = None
        words_final = r._words_final(lane)
        if (verdict and words_final and (not r._all_final(lane)
                or quiet < cfg["quiet"] and verdict["choice"] != "backchannel")):
            if lane == "candidate" and r.input_transport_pending:
                return state("jev_candidate_transport_wait", "再生中のお話の判定を確認しています",
                             "判定結果は届いています。送信途中の音声があるため、入力の確定を待っています。")
            return self._jev_cached_audio_wait(r, now, lane)
        remaining = max(cfg["stable"]-stable, cfg["quiet"]-quiet if lane == "normal" and not words_final else 0)
        if remaining > 0:
            return state("jev_stability", "認識文字の安定を待っています" if lane == "candidate" else "お話の区切りを確認しています",
                         "文字の更新" + ("" if lane == "candidate" else "と声") + "が落ち着くのを待っています。",
                         remaining=remaining, wait_label="声・文字が変わらない場合の確認待ち")
        if verdict:
            choice = verdict["choice"]
            if not r._all_final(lane):
                if lane == "candidate" and r.input_transport_pending:
                    return state("jev_candidate_transport_wait", "再生中のお話の判定を確認しています",
                                 "判定結果は届いています。送信途中の音声があるため、入力の確定を待っています。")
                return state("asr_final_wait", "認識結果の確定を待っています",
                             "判定は返っていますが、音声認識の確定結果が揃うまで入力を確定しません。")
            commit_ready = (choice == "complete" and now >= verdict["due"]
                            or choice == "error" and min(quiet, stable) >= cfg["failure"]
                            or choice in {"short", "long", "uncertain"} and min(quiet, stable) >= cfg["maximum"])
            if lane == "normal" and commit_ready and r.jobs.full():
                return self._queue_wait(now)
            if choice == "complete":
                return state("jev_complete_grace", "話し終わりの判定を確認しています",
                             "お話の再開や文字の更新がないか、短い確認時間を置いています。",
                             remaining=verdict["due"]-now, wait_label="再開がない場合の確認待ち")
            if choice == "backchannel":
                return state("jev_ack_wait", "相槌の判定を確認しています", "認識の確定と静穏を確認して、返答の再生を続けます。",
                             remaining=cfg["quiet"]-quiet, wait_label="声が再開しない場合の確認待ち")
            if lane == "candidate" and verdict.get("backchannel_dominant"):
                return state("jev_likely_ack_wait", "相槌の可能性が高い入力を確認しています",
                             "返答の再生を続けています。この判定は、待ち時間が経過しただけでは割込みに変わりません。追加の言葉があれば判定し直します。")
            if choice == "error":
                return state("jev_failure_recovery", "区切り判定の回復を待っています",
                             "判定を取得できませんでした。確定した認識と静穏条件で回復を試みます。",
                             remaining=cfg["failure"]-min(quiet, stable), wait_label="静穏・文字安定による回復条件まで")
            if choice in {"short", "long", "uncertain"}:
                recovery = max(0, cfg["maximum"]-min(quiet, stable))
                labels = {"short": "お話の続きを待っています", "long": "考えている途中として待っています", "uncertain": "相槌か割込みかを確認しています"}
                return state("jev_"+choice+"_wait", labels[choice],
                             f"同じ内容の判定を保持し、入力や文脈が変わった場合に判定し直します。静穏・文字安定による回復条件まで約{recovery:.1f}秒です。",
                             remaining=recovery,
                             wait_label="静穏・文字安定による回復条件まで")
        return state("jev_judge_wait", "区切りの判定を準備しています", "現在の入力で次の判定を開始するのを待っています。",
                     remaining=r.next_judge_at-now, wait_label="次の判定を開始できるまで")

    def _jev_cached_audio_wait(self, r, now, lane):
        prefix = "jev_preparation" if lane == "preparation" else "jev"
        detail = "言葉の認識とJevの判定は届いています。判定結果を保持し、追加の音に言葉が含まれるか、音声の送信・認識状態を確認しています。"
        if lane == "preparation": detail += " 最初の返答の準備は続けています。"
        # The runtime owns the acoustic recovery clock and may reset its key.
        # Never call its mutating recovery helper from a UI snapshot.
        return self._state("turn", now, prefix+"_decision_audio_wait", "Jevの判定を保持して音声を確認中", detail)

    def _jev_audio_without_words(self, r):
        return (self.settings["mode"] == "jev"
                and (r._rows("preparation") or r._rows("candidate")
                     or getattr(r, "noise_wait_started", 0) > 0)
                and not any(r._text(lane) for lane in ("normal", "candidate", "preparation")))

    def _jev_audio_wait(self, r, now):
        state = lambda *a, **kw: self._state("turn", now, *a, **kw)
        detail = "音は届いていますが、言葉はまだ認識されていません。Jevの意味判定ではなく、音声認識を確認しています。"
        if r.input_transport_pending:
            return state("jev_audio_without_words", "音を検出・言葉を確認中",
                         detail + f" 送信途中の音声が約{r.microphone_backlog_bytes / 48000:.1f}秒分あります。")
        can_release = getattr(r, "_noise_wait_can_release", None)
        if can_release is not None and can_release(now):
            return state("jev_audio_release_ready", "言葉のない音声区間を確認しました",
                         "認識された言葉がないことと待機条件を確認しました。準備済みの返答を再生する処理へ進みます。")
        started = getattr(r, "noise_wait_started", 0)
        remaining = self.settings.get("noise_wait", 2.0)-(now-started) if started > 0 else None
        if remaining is not None and remaining > 0:
            return state("jev_audio_without_words", "音を検出・言葉を確認中",
                         detail + " 空の確定結果なども確認するため、表示時間だけで再生が始まるとは限りません。",
                         since=started, remaining=remaining,
                         wait_label="言葉のない区間を再確認できるまで（他の条件も確認します）")
        return state("jev_audio_without_words", "音を検出・言葉を確認中",
                     detail + (" 再生の開始には、新しい空の確定結果と送信・認識状態の確認も必要です。" if started > 0 else ""),
                     since=started if started > 0 else None)

    def _jev_preparation(self, r, now):
        """Observe the separate input buffer without revoking the held reply."""
        state = lambda *a, **kw: self._state("turn", now, *a, **kw)
        cfg = self.settings
        held = "返答は保持して準備を続けています。"
        if r.input_transport_pending:
            return state("jev_preparation_transport_wait", "追加入力の音声が届くのを待っています",
                         held + f" 送信途中の音声が約{r.microphone_backlog_bytes / 48000:.1f}秒分あります。")
        if not r._text("preparation"):
            return state("jev_preparation_asr_wait", "追加入力の認識結果を待っています",
                         held + " 認識結果が届くまで、再生の開始を待ちます。")
        task = r.judge_task
        if task is not None and not task.done():
            if self.judge_snapshot is not None and not r._current(self.judge_snapshot):
                return state("jev_preparation_updated_input", "最新の追加入力で判定し直すのを待っています",
                             held + " 更新前の判定結果は追加入力の確定に使いません。", since=self.judge_started_at)
            return state("jev_preparation_judging", "追加入力を確認しています",
                         held + " Jevが相槌・繰り返しか、反映すべき新しい内容かを判定しています。",
                         since=self.judge_started_at)
        if not r._words_final("preparation"):
            return state("jev_preparation_asr_wait", "追加入力の認識結果の確定を待っています",
                         held + " 認識結果が揃うまで再生の開始を待ちます。")
        quiet, stable = now-r.input_last_voice, now-r.input_last_change
        verdict = r.verdict
        if verdict and (verdict["snapshot"]["lane"] != "preparation" or not r._current(verdict["snapshot"])):
            verdict = None
        if verdict and (not r._all_final("preparation") or quiet < cfg["quiet"]):
            return self._jev_cached_audio_wait(r, now, "preparation")
        remaining = cfg["stable"]-stable
        if remaining > 0:
            return state("jev_preparation_stability", "追加入力の認識文字の安定を確認しています",
                         held + " 認識文字が落ち着いてから、追加発話の意味を判定します。",
                         remaining=remaining, wait_label="文字が変わらない場合の確認待ち")
        if verdict:
            choice = verdict["choice"]
            if verdict.get("needs_review"):
                return state("jev_preparation_review_wait", "追加された意味を再確認します",
                             held + " 文末の補完か、新しい内容かを一度だけ確認してから、返答を作り直すか決めます。")
            if choice == "retain":
                return state("jev_preparation_retain_grace", "最初の返答を保持する判定を確認しています",
                             "同じ内容・相槌と判定しました。話の再開や文字の更新がないか、短い確認時間を置いています。",
                             remaining=verdict["due"]-now, wait_label="再開がない場合の確認待ち")
            if choice == "revise":
                return state("jev_preparation_revise", "追加入力を返答へ反映する準備をしています",
                             "訂正や新しい内容があるため、追加入力を使って返答を見直します。")
            if choice == "error":
                return state("jev_preparation_error_wait", "追加入力の判定を取得できませんでした",
                             held + " 静穏・文字安定の条件を満たすと、追加入力を会話へ渡して返答を見直します。",
                             remaining=cfg.get("preparation_failure", 2.0)-min(quiet, stable),
                             wait_label="追加入力を返答へ反映する条件まで")
            if choice == "uncertain":
                recovery = max(0, cfg.get("preparation_maximum", 3.0)-min(quiet, stable))
                return state("jev_preparation_uncertain_wait", "追加入力の扱いを引き続き確認しています",
                             held + f" 同じ内容の判定を保持し、静穏・文字安定があと約{recovery:.1f}秒続くと追加入力を返答へ反映します。",
                             remaining=recovery,
                             wait_label="追加入力を返答へ反映する条件まで")
        return state("jev_preparation_judge_wait", "追加入力の判定を準備しています",
                     held + " 現在の入力で次の判定を開始するのを待っています。",
                     remaining=r.next_judge_at-now, wait_label="次の判定を開始できるまで")

    def _reply(self, r, now):
        state = lambda *a, **kw: self._state("reply", now, *a, **kw)
        preparation = self.settings["mode"] == "jev" and bool(r._rows("preparation"))
        audio_only = self._jev_audio_without_words(r)
        if getattr(r, "interrupting", False) or r.draining:
            return state("stopping", "読み上げを止めています", "音声を停止し、配信結果を整理しています。")
        if getattr(r, "waiting_for_continuation", None):
            return state("interruption_wait", "返答を止めて待っています",
                         "続きのお話を聞いてから返答を作ります。")
        if r.playing:
            if self.browser_playing_id == r.playing["id"]:
                return state("playing", "お話ししています", "ブラウザで音声を再生しています。", since=self.browser_playing_at)
            return state("playback_buffering", "音声を再生へ渡しています", "音声の送信とブラウザの再生開始を待っています。")
        if r.approved:
            if self.speech_ready and self.speech_id == r.approved["id"]:
                detail = ("音は届いていますが、言葉はまだ認識されていません。音声認識の確認が済むまで、準備済みの音声を保持しています。" if audio_only else
                          "追加入力の確認が済むまで、準備済みの音声を保持して再生を待っています。" if preparation else
                          "入力の確認が済むまで再生を待っています。")
                return state("tts_ready", "音声の準備ができました", detail, since=self.speech_ready_at)
            detail = "音声データを受信しています。全文が揃ってから再生します。" if self.speech_first else "承認済みの返答を音声にしています。完了時刻は未定です。"
            if audio_only: detail += " 言葉はまだ認識されておらず、音声認識を確認しながら返答の準備を続けています。"
            elif preparation: detail += " 追加入力を確認中も、返答は保持して準備を続けています。"
            return state("tts_preparing", "返答を音声にしています",
                         detail,
                         since=r.approved_at)
        if r.busy or not r.jobs.empty():
            detail = "受け取ったお話への返答を作成・確認しています。"
            if r.input_transport_pending: detail = "入力音声の送信が追いつくのを待っています。"
            elif r.delivery_tasks: detail = "先の音声の配信結果を記録してから返答処理を進めます。"
            elif not self.core_active: detail = "入力を返答処理へ渡す準備や、処理結果の確認をしています。"
            if audio_only: detail += " 音声認識を確認しながら、最初の返答は保持しています。"
            elif preparation: detail += " 追加入力を確認しながら、最初の返答は保持しています。"
            return state("core_processing", "返答を考えています", detail,
                         since=self.core_started if self.core_active else None)
        if r.delivery_tasks:
            return state("confirming_delivery", "再生結果を記録しています", "再生完了を会話の記録へ反映しています。")
        return state("idle", "返答を待機しています", "入力が確定すると返答の処理を開始します。")

    def snapshot(self, r, now):
        if r.failure_message:
            turn = self._state("turn", now, "error", "処理が停止しました", r.failure_message)
            reply = self._state("reply", now, "error", "処理が停止しました", r.failure_message)
        elif r.closed or r.finished.is_set() or getattr(r, "speech_closing", False):
            turn = self._state("turn", now, "closed", "会話を終了しました", "音声の接続を終了しています。")
            reply = self._state("reply", now, "closed", "会話を終了しました", "新しい会話から再開できます。")
        elif not r.ready:
            turn = self._state("turn", now, "connecting", "音声へ接続しています", "接続の準備を待っています。", since=r.started)
            reply = self._state("reply", now, "idle", "返答を待機しています", "音声の接続が済むのを待っています。")
        else:
            turn, reply = self._turn(r, now), self._reply(r, now)
        return {"turn": turn, "reply": reply, "recent": [dict(item) for item in reversed(self.recent)]}

"""ASR segments must not become dialogue turns or lose PCM/corrections."""
import asyncio
import base64
import unittest

from transcription_client import TranscriptionClient


class Socket:
    closed = False

    def __init__(self):
        self.sent = []

    async def send_json(self, event):
        self.sent.append(event)

    async def close(self):
        self.closed = True


class TranscriptionTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def delta(client, item_id="quiet", text="よし"):
        return client.normalize_event({"type": "conversation.item.input_audio_transcription.delta",
                                       "item_id": item_id, "delta": text})

    @staticmethod
    def final(client, item_id="quiet", text="よし"):
        return client.normalize_event({"type": "conversation.item.input_audio_transcription.completed",
                                       "item_id": item_id, "transcript": text})

    @staticmethod
    def committed(client, item_id="quiet"):
        return client.normalize_event({"type": "input_audio_buffer.committed", "item_id": item_id})

    @staticmethod
    def commits(client):
        return sum(event["type"] == "input_audio_buffer.commit" for event in client.ws.sent)

    async def test_commit_on_silence_preserves_audio_before_and_after_boundary(self):
        client = TranscriptionClient()
        client.ws = Socket()
        # 100ms sound, 780ms silence, then one batch spanning the boundary.
        packets = [b"\x10\x27"*2400] + [bytes(4800)]*7 + [bytes(3840), bytes(960)+b"\x10\x27"*1920]
        for pcm in packets:
            await client.audio(pcm)
        events = client.ws.sent
        audio = b"".join(base64.b64decode(e["audio"]) for e in events if e["type"].endswith(".append"))
        self.assertEqual(audio, b"".join(packets))
        commits = [i for i,e in enumerate(events) if e["type"].endswith(".commit")]
        self.assertEqual(len(commits), 1)
        self.assertTrue(events[commits[0]+1]["type"].endswith(".append"))
        self.assertEqual(client.transport_metrics["audio_bytes_sent"], len(audio))

    async def test_silence_never_generates_a_turn_or_empty_commit(self):
        client = TranscriptionClient()
        client.ws = Socket()
        for _ in range(151):
            await client.audio(bytes(4800))
        types = [event["type"] for event in client.ws.sent]
        self.assertNotIn("input_audio_buffer.commit", types)
        self.assertIn("input_audio_buffer.clear", types)
        self.assertFalse(client.pending_audio)

    async def test_below_threshold_interim_is_committed_and_final_releases_barrier(self):
        client = TranscriptionClient()
        client.ws = Socket()
        await client.audio(b"\x64\x00" * 2400)  # RMS 100, below local 560.
        self.assertFalse(client._has_voice)
        self.delta(client)
        self.assertTrue(client.pending_audio)
        for _ in range(8):
            await client.audio(bytes(4800))
        self.assertEqual(self.commits(client), 1)
        self.assertTrue(client.pending_audio)
        self.committed(client)
        self.final(client, text="よし。")
        self.assertEqual(client._items["quiet"]["text"], "よし。")
        self.assertFalse(client.pending_audio)

    async def test_committed_late_delta_does_not_commit_silence_again(self):
        client = TranscriptionClient()
        client.ws = Socket()
        await client.audio(b"\x10\x27" * 2400)
        for _ in range(8):
            await client.audio(bytes(4800))
        self.committed(client)
        self.delta(client)
        for _ in range(151):
            await client.audio(bytes(4800))
        self.assertEqual(self.commits(client), 1)
        self.assertTrue(client.pending_audio, "clearing new silence cannot clear an older final wait")
        self.final(client)
        self.assertFalse(client.pending_audio)

    async def test_interim_during_final_append_prevents_silence_clear(self):
        client = TranscriptionClient()
        client.ws = Socket()
        for _ in range(149):
            await client.audio(bytes(4800))
        original_send = client.ws.send_json
        recognized = False

        async def receive_during_append(event):
            nonlocal recognized
            await original_send(event)
            if event["type"].endswith(".append") and not recognized:
                recognized = True
                self.delta(client)

        client.ws.send_json = receive_during_append
        for _ in range(10):
            await client.audio(bytes(4800))
        types = [event["type"] for event in client.ws.sent]
        self.assertNotIn("input_audio_buffer.clear", types)
        self.assertEqual(self.commits(client), 1)
        self.committed(client)
        self.final(client)
        self.assertFalse(client.pending_audio)

    async def test_delta_and_final_during_commit_send_consume_exactly_one_reservation(self):
        for interim_before_commit in (False, True):
            with self.subTest(interim_before_commit=interim_before_commit):
                client = TranscriptionClient()
                client.ws = Socket()
                original_send = client.ws.send_json

                async def receive_during_commit(event):
                    await original_send(event)
                    if event["type"].endswith(".commit"):
                        self.delta(client, text="よし")
                        self.final(client, text="よし。")
                        self.committed(client)  # Delayed ACK must not reopen it.

                client.ws.send_json = receive_during_commit
                await client.audio(b"\x10\x27" * 2400)
                if interim_before_commit:
                    self.delta(client, text="")
                    self.delta(client, text="よ")
                for _ in range(10):
                    await client.audio(bytes(4800))
                self.assertEqual(self.commits(client), 1)
                self.assertEqual(client._commits_inflight, 0)
                self.assertFalse(client.pending_audio)
                self.assertEqual(client._items["quiet"]["text"], "よし。")

    async def test_out_of_order_and_duplicate_finals_do_not_release_another_segment(self):
        client = TranscriptionClient()
        client.ws = Socket()
        for item_id in ("first", "second"):
            await client.audio(b"\x10\x27" * 2400)
            self.delta(client, item_id)
            for _ in range(8):
                await client.audio(bytes(4800))
            self.committed(client, item_id)
        self.assertEqual(self.commits(client), 2)
        self.final(client, "second")
        self.assertTrue(client.pending_audio)
        self.assertIsNone(self.final(client, "second"))
        self.assertIsNone(self.delta(client, "second", "遅れた重複"))
        self.assertTrue(client.pending_audio)
        self.final(client, "first", "")
        self.assertFalse(client.pending_audio)

    async def test_whitespace_interim_does_not_commit_silence(self):
        client = TranscriptionClient()
        client.ws = Socket()
        self.assertIsNone(self.delta(client, text=" \n"))
        for _ in range(151):
            await client.audio(bytes(4800))
        self.assertEqual(self.commits(client), 0)
        self.assertFalse(client.pending_audio)

    async def test_late_old_ack_does_not_consume_new_unbound_commit(self):
        client = TranscriptionClient()
        client.ws = Socket()
        for index in range(2):
            await client.audio(b"\x10\x27" * 2400)
            for _ in range(8):
                await client.audio(bytes(4800))
            if index == 0:
                self.final(client, "first")  # Final can precede commit ACK.
                self.assertFalse(client.pending_audio)
        self.assertEqual(client._commits_inflight, 1)
        self.assertIsNone(self.committed(client, "first"))
        self.assertEqual(client._commits_inflight, 1)
        self.committed(client, "second")
        self.final(client, "second", "")
        self.assertFalse(client.pending_audio)

    async def test_final_text_replaces_interim_and_keeps_committed_order(self):
        client = TranscriptionClient()
        for item in ("first", "second"):
            client.normalize_event({"type": "input_audio_buffer.committed", "item_id": item})
        delta = client.normalize_event({"type": "conversation.item.input_audio_transcription.delta", "item_id":"first", "delta":"げんいん"})
        second = client.normalize_event({"type":"conversation.item.input_audio_transcription.completed", "item_id":"second", "transcript":"うれしい"})
        first = client.normalize_event({"type":"conversation.item.input_audio_transcription.completed", "item_id":"first", "transcript":"原因"})
        self.assertEqual(delta["segment_text"], "げんいん")
        self.assertEqual(first["segment_text"], "原因")
        self.assertLess(first["sequence"], second["sequence"])
        self.assertEqual(first["item_id"], delta["item_id"])

    async def test_shutdown_does_not_claim_nonexistent_usage_finalization(self):
        client = TranscriptionClient()
        client.ws = Socket()
        await client.request_close()
        await client.audio(bytes(4800))
        self.assertEqual(client.ws.sent, [])
        self.assertTrue(client.ws.closed)
        self.assertFalse(client.finalized.is_set())

    async def test_provider_failure_body_is_not_forwarded(self):
        client = TranscriptionClient()
        error = client.normalize_event({"type":"conversation.item.input_audio_transcription.failed", "error":{"message":"private payload"}})
        self.assertEqual(error["type"], "session.error")
        self.assertNotIn("private payload", str(error))

    async def test_invalid_pcm_rejected(self):
        client = TranscriptionClient()
        client.ws = Socket()
        for value in (bytes(48002), b"x", "text"):
            with self.assertRaises(ValueError):
                await client.audio(value)
        self.assertEqual(client.ws.sent, [])

    async def test_new_audio_blocks_commit_until_corresponding_asr_final(self):
        client = TranscriptionClient()
        client.ws = Socket()
        await client.audio(b"\x10\x27"*2400)
        self.assertTrue(client.pending_audio)
        for _ in range(8):
            await client.audio(bytes(4800))
        self.assertTrue(client.pending_audio, "commit sent but no transcript received yet")
        pending = client.normalize_event({"type":"input_audio_buffer.committed", "item_id":"a"})
        self.assertEqual(pending["type"], "session.input_transcript.pending")
        self.assertTrue(client.pending_audio)
        client.normalize_event({"type":"conversation.item.input_audio_transcription.completed", "item_id":"a", "transcript":"完了"})
        self.assertFalse(client.pending_audio)
        await client.audio(b"\x10\x27"*2400)
        self.assertTrue(client.pending_audio, "speech resumed before any next delta")


if __name__ == "__main__":
    unittest.main()

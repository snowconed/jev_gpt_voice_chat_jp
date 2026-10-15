import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest

from voice_runtime import VoiceRuntime, normalized, rms

HERE = Path(__file__).resolve().parents[1]


class Core:
    def __init__(self):
        self.deliveries = []
        self.pending_messages = []
        self.inputs = []
    def delivery(self, sid, mid, status, heard_text=""):
        self.deliveries.append((mid,status,heard_text))
    def pending(self,sid): return self.pending_messages
    def view(self,sid): return {"id":sid,"messages":[]}
    def submit(self,sid,text,rid): self.inputs.append((text,rid))


class Client:
    def __init__(self): self.instructions=[]; self.approvals=[]; self.pcm=[]
    async def instruct(self,text,*args): self.instructions.append(text)
    async def approve(self,text,*args): self.approvals.append(text)
    async def audio(self,pcm): self.pcm.append(pcm)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temp_parent=HERE/".runtime"/"test_tmp"
        temp_parent.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=temp_parent)
        self.events=[];self.audio=[]
        self.core=Core();self.client=Client()
        async def js(e): self.events.append(e)
        async def pcm(p): self.audio.append(p)
        self.r=VoiceRuntime(self.core,"a"*32,js,pcm,Path(self.tmp.name),client=self.client)
        self.r.ready=True
    async def asyncTearDown(self):
        if self.r.delivery_tasks:
            await asyncio.gather(*list(self.r.delivery_tasks),return_exceptions=True)
        if self.r.control_tasks:
            await asyncio.gather(*list(self.r.control_tasks),return_exceptions=True)
        self.r.logfile.close()
        self.tmp.cleanup()
    def ready(self,text="どこがうれしかった？",actual=None):
        self.r.approved={"id":"question1","kind":"listening","text":text}
        self.r.approved_at=time.monotonic()
        self.r.approval_revision=self.r.input_revision
        self.r.output_text=text if actual is None else actual
        self.r.output_audio.extend(bytes(4800))
        self.r.output_has_voice=True
        self.r.output_last_voice=time.monotonic()-1
        self.r.output_last_text=time.monotonic()-1

    async def test_exact_text_released_but_not_committed_before_playback(self):
        self.ready(actual="どこがうれしかった?")
        await self.r.check_output(time.monotonic())
        self.assertTrue(self.audio)
        self.assertEqual(self.core.deliveries,[])
        await self.r.played("question1")
        if self.r.delivery_tasks: await asyncio.gather(*list(self.r.delivery_tasks))
        self.assertEqual(self.core.deliveries,[("question1","completed","")])

    async def test_added_question_never_reaches_playback(self):
        self.ready(actual="どこがうれしかった？誰と一緒だった？")
        await self.r.check_output(time.monotonic())
        self.assertFalse(self.audio)
        self.assertEqual(self.client.approvals,[])
        self.assertTrue(self.r.retry_pending)
        self.assertFalse(self.core.deliveries)

    async def test_changed_approved_reply_never_reaches_playback(self):
        self.ready("机の上で写真を見つけた。","公園で写真を見つけた。")
        await self.r.check_output(time.monotonic())
        self.assertFalse(self.audio)

    async def test_partial_transcript_waits(self):
        self.ready(actual="どこが")
        await self.r.check_output(time.monotonic())
        self.assertFalse(self.audio)
        self.assertFalse(self.client.approvals)

    async def test_unknown_output_blocked(self):
        self.ready(actual="私も旅行に行ったことがあります。")
        self.r.approved=None
        await self.r.check_output(time.monotonic())
        self.assertFalse(self.audio)

    async def test_allowed_backchannel_uses_same_audio_pipeline(self):
        self.ready(actual="うん。")
        self.r.approved=None
        await self.r.check_output(time.monotonic())
        self.assertTrue(self.audio)
        self.assertEqual(self.r.playing["kind"],"backchannel")
        self.assertFalse(self.core.deliveries)

    async def test_revision_change_cancels_old_audio(self):
        self.ready()
        self.r.input_revision += 1
        await self.r.check_output(time.monotonic())
        self.assertFalse(self.audio)
        self.assertIn({"type":"interrupt", "revision":self.r.input_revision},self.events)

    async def test_interruption_does_not_claim_unheard_words(self):
        self.ready()
        await self.r.check_output(time.monotonic())
        await self.r.interrupt()
        await asyncio.gather(*list(self.r.delivery_tasks))
        self.assertEqual(self.core.deliveries,[("question1","interrupted","")])
        await self.r.played("question1")
        self.assertEqual(len(self.core.deliveries),1)

    async def test_continuous_microphone_includes_silence(self):
        await self.r.microphone(bytes(4800))
        self.assertEqual(len(self.client.pcm),1)
        self.assertEqual(self.r.input_samples,2400)

    async def test_malformed_pcm_rejected(self):
        with self.assertRaises(ValueError): await self.r.microphone(b"x")

    async def test_typed_input_stays_verbatim(self):
        text="うれしかった。でも少し不安でした。"
        await self.r.enqueue(text)
        entry=await self.r.jobs.get()
        self.assertEqual(entry[0],text)

    async def test_early_playback_ack_does_not_commit(self):
        self.ready()
        self.r.output_audio=bytearray(48000*5)
        await self.r.check_output(time.monotonic())
        await self.r.played("question1")
        self.assertFalse(self.core.deliveries)
        self.assertIsNotNone(self.r.playing)

    def test_normalizer_does_not_allow_word_rewrites(self):
        self.assertEqual(normalized("うれしかった？"),normalized("うれしかった?"))
        self.assertNotEqual(normalized("うれしかった"),normalized("悲しかった"))
        self.assertEqual(rms(bytes(100)),0)


if __name__=="__main__": unittest.main()

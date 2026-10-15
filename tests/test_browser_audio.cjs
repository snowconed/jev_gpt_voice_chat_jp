// Offline audio-path checks. No browser, microphone, network, or OpenAI calls.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");

const webRoot = path.resolve(__dirname, "../web");
const appSource = fs.readFileSync(path.join(webRoot, "app.js"), "utf8")
  .replace(/initialize\(\);\s*$/, "");
const workletSource = fs.readFileSync(path.join(webRoot, "pcm-worklet.js"), "utf8");

function pipeline(rate = 48000, interruptionPolicy) {
  const elements = new Map();
  const element = () => ({
    style: { setProperty() {} }, classList: { add() {}, remove() {}, toggle() {} },
    addEventListener() {}, append() {}, replaceChildren() {}, focus() {},
    childNodes: [], scrollTop: 0, scrollHeight: 100, clientHeight: 100,
  });
  const timers = new Map();
  const replies = [];
  const captures = [];
  const workletMessages = [];
  let now = 0;
  let frameOffset = 0;
  const app = {
    document: {
      getElementById(id) {
        if (!elements.has(id)) elements.set(id, element());
        return elements.get(id);
      }, createElement: element, createDocumentFragment: element,
    },
    window: { addEventListener() {} },
    WebSocket: { OPEN: 1, CONNECTING: 0 },
    ArrayBuffer, Int16Array,
    performance: { now() { return now; } },
    navigator: { get mediaDevices() { throw new Error("The local audio test must not access a microphone"); } },
    fetch() { throw new Error("The local audio test must not call an API"); },
    setTimeout(fn, delay) { const token = {}; timers.set(token, { fn, due: now + delay }); return token; },
    clearTimeout(token) { timers.delete(token); },
  };
  vm.createContext(app);
  vm.runInContext(appSource + `
    globalThis.audioTest = {
      handleServerMessage, clearPlayback, acknowledgeAfterPlayback,
      handleWorkletMessage, connectWorkletHandlers, createAttempt, testAudio, testTonePCM,
      getActive() { return active; },
      setConfig(value) { config = value; },
      setActive(value) { active = value; }
    };`, app);

  let processor;
  let attempt;
  let node;
  const worklet = {
    sampleRate: rate, currentFrame: 0,
    AudioWorkletProcessor: class {
      constructor() {
        this.port = {
          postMessage(message, transfer = []) {
            const event = structuredClone(message, { transfer });
            workletMessages.push(event);
            node?.port.onmessage?.({ data: event });
          },
        };
      }
    },
    registerProcessor(_name, Constructor) { processor = new Constructor(); },
  };
  vm.createContext(worklet);
  vm.runInContext(workletSource, worklet);
  const contexts = [];
  class FakeAudioContext {
    constructor() {
      this.sampleRate = rate;
      this.state = "suspended";
      this.currentTime = 0;
      this.destination = {};
      this.audioWorklet = { async addModule(name) { assert.equal(name, "/pcm-worklet.js"); } };
      contexts.push(this);
    }
    async resume() { this.state = "running"; this.onstatechange?.(); }
    async close() { this.state = "closed"; this.onstatechange?.(); }
    getOutputTimestamp() { return { contextTime: this.currentTime }; }
  }
  class FakeAudioWorkletNode {
    constructor(context, name, options) {
      assert.equal(name, "voice-pcm");
      this.context = context;
      this.options = options;
      this.port = {
        onmessage: null,
        postMessage(data, transfer = []) { processor.receive(structuredClone(data, { transfer })); },
      };
      node = this;
    }
    connect(destination) { assert.equal(destination, this.context.destination); }
    disconnect() {}
  }
  app.window.AudioContext = FakeAudioContext;
  app.AudioWorkletNode = FakeAudioWorkletNode;
  if (interruptionPolicy !== undefined) app.audioTest.setConfig({ interruption_policy: interruptionPolicy });
  attempt = app.audioTest.createAttempt();
  attempt.context = new FakeAudioContext();
  attempt.context.state = "running";
  attempt.node = new FakeAudioWorkletNode(attempt.context, "voice-pcm", { numberOfInputs: 1 });
  attempt.socket = {
    readyState: 1, bufferedAmount: 0,
    send(message) {
      if (typeof message === "string") replies.push(JSON.parse(message));
      else captures.push(message);
    },
    close() { this.readyState = 3; },
  };
  app.audioTest.connectWorkletHandlers(attempt);
  app.audioTest.setActive(attempt);

  return {
    replies, captures, attempt, processor, workletMessages, elements, contexts, timers,
    played() { return replies.filter((message) => message.type === "played"); },
    diagnostics() { return replies.filter((message) => message.type === "audio_diagnostic"); },
    event(message) { app.audioTest.handleServerMessage(attempt, { data: JSON.stringify(message) }); },
    audio(bytes) { app.audioTest.handleServerMessage(attempt, { data: bytes }); },
    filler(clipId = "voice-hash-a", samples = 12000) {
      const buffer = tone(samples);
      const clip = {id:clipId,text:"ええと",audio_bytes:buffer.byteLength,audio_seconds:buffer.byteLength/48000,buffer};
      attempt.fillerEnabled = true;
      attempt.fillerClips.set(clip.id,clip);
      return {type:"filler",id:"filler-turn-1",clip_id:clip.id,text:clip.text,audio_bytes:clip.audio_bytes,audio_seconds:clip.audio_seconds,revision:1};
    },
    capture(level) { app.audioTest.handleWorkletMessage(attempt, { type: "capture", level, buffer: new ArrayBuffer(960) }); },
    async localTest() {
      app.audioTest.setActive(null);
      await app.audioTest.testAudio();
      return app.audioTest.getActive();
    },
    active() { return app.audioTest.getActive(); },
    tonePCM() { return app.audioTest.testTonePCM(); },
    render(seconds, inputSignal = null) {
      let energy = 0;
      let peak = 0;
      let processingMs = 0;
      const blockDurations = [];
      const total = Math.ceil(seconds * rate / 128) * 128;
      const rendered = new Float32Array(total);
      for (let frame = 0; frame < total; frame += 128) {
        worklet.currentFrame = frameOffset;
        for (const context of contexts) context.currentTime = frameOffset / rate;
        const output = new Float32Array(128);
        const input = inputSignal ? Float32Array.from({ length: 128 }, (_unused, i) => inputSignal((frameOffset + i) / rate)) : null;
        const processStarted = process.hrtime.bigint();
        processor.process(input ? [[input]] : [[]], [[output]]);
        const blockMs = Number(process.hrtime.bigint() - processStarted) / 1e6;
        processingMs += blockMs;
        blockDurations.push(blockMs);
        rendered.set(output, frame);
        for (const value of output) { energy += value * value; peak = Math.max(peak, Math.abs(value)); }
        frameOffset += 128;
        now += 128 / rate * 1000;
        for (const [token, entry] of [...timers]) {
          if (entry.due <= now) { timers.delete(token); entry.fn(); }
        }
      }
      now += 100;
      for (const context of contexts) context.currentTime += .1;
      for (const [token, entry] of [...timers]) {
        if (entry.due <= now) { timers.delete(token); entry.fn(); }
      }
      blockDurations.sort((a, b) => a - b);
      return { peak, energy, samples: rendered, processingMs,
        processP95Ms: blockDurations[Math.floor(blockDurations.length * .95)] };
    },
  };
}

function tone(samples = 2400) {
  const pcm = new Int16Array(samples);
  for (let i = 0; i < samples; i += 1) pcm[i] = Math.round(Math.sin(2 * Math.PI * 440 * i / 24000) * 8192);
  return pcm.buffer;
}

test("cached filler plays once with device ACK while keeping reply captions and processing state", () => {
  const p = pipeline(48000, "semantic"), message = p.filler();
  p.event({type:"output_transcript",text:"前の返答です。",replace:true});
  p.elements.get("status").textContent="返答を考えています";
  p.event(message);p.event(message);
  assert.equal(p.played().length,0);
  assert.equal(p.processor.segments.length,1);
  assert.equal(p.replies.filter(row=>row.type==="filler_skipped").length,0);
  assert(p.render(.15).energy>0);
  assert.equal(p.elements.get("output-caption").textContent,"前の返答です。");
  assert.equal(p.elements.get("status").textContent,"返答を考えています");
  assert.match(p.elements.get("filler-status").textContent,/ええと/);
  assert.equal(p.played().length,0,"partially played filler must not ACK");
  assert(p.render(.5).energy>0);
  assert.deepEqual(p.played(),[{type:"played",id:message.id}]);
  assert.equal(p.elements.get("filler-status").hidden,true);
  assert.equal(p.attempt.fillerClips.get(message.clip_id).buffer.byteLength,message.audio_bytes);
  p.event({...message,id:"filler-turn-2",revision:2});
  assert(p.render(.7).energy>0,"cached PCM remains reusable after transferable playback");
  assert.equal(p.played().length,2);
});

test("Jev filler preserves microphone capture and ignores energy-only noise/backchannel activity", () => {
  const p = pipeline(48000,"semantic"), message=p.filler();p.event(message);
  for(let i=0;i<20;i+=1)p.capture(.2);
  assert.equal(p.attempt.generation,0);
  assert(p.attempt.pendingPlayback.has(message.id));
  assert.equal(p.captures.length,20);
  assert.equal(p.replies.filter(row=>row.type==="user_activity").length,0);
  assert(p.render(.7).energy>0);assert.equal(p.played().length,1);
});

test("server interruption during filler drops remaining sound and stale ACK; next reply still plays", () => {
  const p=pipeline(), message=p.filler();p.event(message);p.render(.1);
  p.event({type:"interrupt",revision:1});
  p.event(message);
  assert.equal(p.render(.7).energy,0);
  assert.equal(p.played().length,0);
  assert.equal(p.elements.get("filler-status").hidden,true);
  p.event({type:"playback",id:"new-reply"});p.audio(tone());p.event({type:"playback_end",id:"new-reply"});
  assert(p.render(.2).energy>0);assert.deepEqual(p.played(),[{type:"played",id:"new-reply"}]);
});

test("filler_stop cancels only its ID and does not clear queued reply PCM or generation", () => {
  const p=pipeline(), message=p.filler();p.event(message);
  p.event({type:"playback",id:"reply"});p.audio(tone());p.event({type:"playback_end",id:"reply"});
  p.event({type:"filler_stop",id:message.id});
  assert.equal(p.attempt.generation,0);
  assert.deepEqual(Array.from(p.processor.segments,row=>row.id),["reply"]);
  assert(p.render(.2).energy>0);assert.deepEqual(p.played(),[{type:"played",id:"reply"}]);
  p.event({...message,revision:3});assert.equal(p.render(.6).energy,0);
});

test("unknown, disabled, wrong-voice, malformed, stale and late fillers fail open without touching reply", () => {
  const p=pipeline(), message=p.filler();
  p.attempt.fillerEnabled=false;p.event(message);p.attempt.fillerEnabled=true;
  p.event({...message,id:"unknown",clip_id:"another-voice"});
  p.event({...message,id:"bad-size",audio_bytes:1});
  p.event({...message,id:"bad-duration",audio_seconds:null});
  p.event({...message,id:"bad-text",text:"そうですね"});
  p.event({type:"interrupt",revision:3});p.event({...message,id:"stale",revision:2});
  p.event({type:"filler_stop",id:"early-stop"});p.event({...message,id:"early-stop",revision:4});
  p.event({type:"playback",id:"reply"});p.audio(tone());p.event({type:"playback_end",id:"reply"});
  p.event({...message,id:"late",revision:4});
  assert.equal(p.replies.filter(row=>row.type==="filler_skipped").length,8);
  assert(p.render(.2).energy>0);assert.deepEqual(p.played(),[{type:"played",id:"reply"}]);
  assert.equal(p.active(),p.attempt);
});

test("device-latency filler acknowledgement is cancelled by an ID-only stop", () => {
  const p=pipeline(),message=p.filler("voice",2400);p.event(message);
  p.attempt.context.getOutputTimestamp=()=>({contextTime:.001});
  p.render(.2);assert.equal(p.played().length,0);
  p.event({type:"filler_stop",id:message.id});
  p.attempt.context.getOutputTimestamp=()=>({contextTime:100});
  p.render(.2);assert.equal(p.played().length,0);
});

test("old filler cannot overtake an unacknowledged typed correction", () => {
  const p=pipeline(),message=p.filler();
  p.attempt.pendingTextInputs.add("text-1");p.attempt.pendingTextInputs.add("text-2");
  p.event(message);assert.equal(p.render(.2).energy,0);
  p.event({type:"input_accepted",client_input_id:"text-1"});
  p.event({...message,id:"still-old",revision:2});assert.equal(p.render(.2).energy,0);
  p.event({type:"input_accepted",client_input_id:"unknown"});assert.equal(p.attempt.pendingTextInputs.size,1);
  p.event({type:"input_accepted",client_input_id:"text-2"});
  p.event({...message,id:"new-turn",revision:3});assert(p.render(.7).energy>0);
  assert.deepEqual(p.played(),[{type:"played",id:"new-turn"}]);
});

for (const rate of [24000, 44100, 48000]) {
  test(`JSON start + binary PCM + end renders nonzero audio at ${rate}Hz`, () => {
    const p = pipeline(rate);
    p.event({ type: "playback", id: "approved" });
    p.audio(tone());
    p.event({ type: "playback_end", id: "approved" });
    assert.equal(p.played().length, 0, "receiving end must not claim audible completion");
    const output = p.render(.2);
    assert(output.peak > .24, "tone amplitude should survive the whole browser path");
    assert(output.energy > 10, "the rendered output must not be silence");
    assert.deepEqual(p.played(), [{ type: "played", id: "approved" }]);
    assert.deepEqual(p.diagnostics().map((message) => message.stage),
      ["playback_received", "playback_buffered", "playing", "rendered", "played"]);
    for (const diagnostic of p.diagnostics()) {
      assert.deepEqual(Object.keys(diagnostic).sort(),
        ["type", "stage", "id", "context_state", "sample_rate", "received_bytes"].sort());
      assert.equal(diagnostic.sample_rate, rate);
      assert.equal(diagnostic.context_state, "running");
      assert.equal(diagnostic.received_bytes, diagnostic.stage === "playback_received" ? 0 : 4800);
    }
    const rendered = p.workletMessages.find((message) => message.type === "rendered");
    assert(rendered.peak > .24);
    assert(Math.abs(rendered.renderedSamples - rate * .1) <= 1);
  });
}

test("interrupt drops old queued/in-flight PCM but permits the next generation", () => {
  const p = pipeline();
  p.event({ type: "playback", id: "old" });
  p.audio(tone());
  p.event({ type: "interrupt" });
  p.audio(tone());
  p.event({ type: "playback_end", id: "old" });
  assert.equal(p.render(.2).energy, 0);
  assert.equal(p.played().length, 0);
  p.event({ type: "playback", id: "new" });
  p.audio(tone());
  p.event({ type: "playback_end", id: "new" });
  assert(p.render(.2).energy > 10);
  assert.deepEqual(p.played(), [{ type: "played", id: "new" }]);
});

for (const rate of [24000, 44100, 48000]) {
  test(`queued reply segments join without inserted silence and ACK separately at ${rate}Hz`, () => {
    const p = pipeline(rate);
    const values = [4096, 8192, 12288];
    for (let index = 0; index < values.length; index += 1) {
      const id = `reply:s${index}`;
      p.event({ type: "playback", id, message_id: "reply" });
      p.audio(new Int16Array(2400).fill(values[index]).buffer);
      p.event({ type: "playback_end", id });
    }
    assert.equal(p.played().length, 0, "receiving all segments is not audible completion");
    const output = p.render(.4).samples;
    const starts = p.workletMessages.filter((message) => message.type === "playing");
    const ends = p.workletMessages.filter((message) => message.type === "rendered");
    assert.equal(starts.length, 3);
    assert.equal(ends.length, 3);
    for (let index = 0; index < values.length; index += 1) {
      assert.equal(starts[index].id, `reply:s${index}`);
      if (index) assert.equal(starts[index].frame, ends[index - 1].frame);
      assert(output.slice(starts[index].frame, ends[index].frame)
        .every((sample) => sample === values[index] / 32768));
    }
    assert(output.slice(ends.at(-1).frame).every((sample) => sample === 0));
    assert.deepEqual(p.played(), values.map((_value, index) => ({ type: "played", id: `reply:s${index}` })));
    p.render(.1);
    assert.equal(p.played().length, 3, "each segment is acknowledged only once");
  });
}

test("first segment plays while the next arrives, keeping speaking status and exact caption order", () => {
  const p = pipeline();
  p.event({ type: "playback", id: "reply:s0", message_id: "reply" });
  p.event({ type: "output_transcript", message_id: "reply", text: "最初の文です。" });
  p.audio(new Int16Array(24000).fill(4096).buffer);
  p.event({ type: "playback_end", id: "reply:s0" });
  assert(p.render(.2).energy > 0, "the first segment must not wait for later synthesis");
  assert.equal(p.elements.get("status").textContent, "お話ししています");
  p.event({ type: "playback", id: "reply:s1", message_id: "reply" });
  assert.equal(p.elements.get("status").textContent, "お話ししています");
  p.event({ type: "output_transcript", message_id: "reply", text: " 続きの文です。" });
  p.audio(new Int16Array(4800).fill(8192).buffer);
  p.event({ type: "playback_end", id: "reply:s1" });
  assert.equal(p.elements.get("output-caption").textContent, "最初の文です。 続きの文です。");
  const output = p.render(1.1).samples;
  const starts = p.workletMessages.filter((message) => message.type === "playing");
  const ends = p.workletMessages.filter((message) => message.type === "rendered");
  assert.equal(starts[1].frame, ends[0].frame, "already buffered next segment has no queue gap");
  assert.deepEqual(p.played(), [{ type: "played", id: "reply:s0" }, { type: "played", id: "reply:s1" }]);
  assert(output.some((sample) => sample !== 0));
});

test("interruption retains only completed segment ACKs and discards queued later segments", () => {
  const p = pipeline();
  for (const [id, samples] of [["reply:s0", 2400], ["reply:s1", 24000], ["reply:s2", 2400]]) {
    p.event({ type: "playback", id, message_id: "reply" });
    p.audio(tone(samples));
    p.event({ type: "playback_end", id });
  }
  p.render(.2);
  assert.deepEqual(p.played(), [{ type: "played", id: "reply:s0" }]);
  p.event({ type: "interrupt" });
  assert.equal(p.render(1.5).energy, 0);
  assert.deepEqual(p.played(), [{ type: "played", id: "reply:s0" }]);
});

test("Live noise keeps both delayed segment ACKs valid until a server interruption", () => {
  const p = pipeline(48000, "transcript");
  p.attempt.context.getOutputTimestamp = () => ({ contextTime: 0 });
  p.attempt.context.baseLatency = 10;
  for (const id of ["reply:s0", "reply:s1"]) {
    p.event({ type: "playback", id, message_id: "reply" });
    p.audio(tone());
    p.event({ type: "playback_end", id });
  }
  p.render(.3, (seconds) => .8 * Math.sin(2 * Math.PI * 1000 * seconds));
  assert.equal(p.attempt.generation, 0, "noise must not invalidate either segment");
  assert(p.captures.length >= 15, "loud audio must still reach recognition");
  assert.equal(p.replies.filter((message) => message.type === "user_activity").length, 0);
  assert.equal(p.played().length, 0);
  const lateChecks = [...p.timers.values()].map((entry) => entry.fn);
  const lateEvents = p.workletMessages.filter((message) => message.type === "rendered");
  assert.equal(lateEvents.length, 2);
  assert.equal(lateChecks.length, 2);
  p.event({ type: "interrupt" });
  p.attempt.context.currentTime = 100;
  for (const check of lateChecks) check();
  for (const data of lateEvents) p.attempt.node.port.onmessage({ data });
  assert.equal(p.played().length, 0);
  assert.equal(p.timers.size, 0);
});

test("explicit legacy energy interruption rejects a late next segment and caption from the same reply", () => {
  const p = pipeline(48000, "energy");
  p.event({ type: "playback", id: "old:s0", message_id: "old" });
  p.event({ type: "output_transcript", message_id: "old", text: "途中までの文。" });
  p.audio(tone(24000));
  p.event({ type: "playback_end", id: "old:s0" });
  for (let index = 0; index < 8; index += 1) p.capture(.025);
  assert.equal(p.attempt.generation, 1);
  // The server has not processed user_activity yet and has sent its next segment.
  p.event({ type: "playback", id: "old:s1", message_id: "old" });
  p.event({ type: "output_transcript", message_id: "old", text: "遅れて来た古い文。" });
  p.audio(tone());
  p.event({ type: "playback_end", id: "old:s1" });
  assert.equal(p.render(.3).energy, 0);
  assert.equal(p.played().length, 0);
  assert.equal(p.elements.get("output-caption").textContent, "途中までの文。");
  assert.equal(p.attempt.pendingPlayback.size, 0);
  p.event({ type: "playback", id: "new:s0", message_id: "new" });
  p.event({ type: "output_transcript", message_id: "new", text: "新しい返答。", replace: true });
  p.audio(tone());
  p.event({ type: "playback_end", id: "new:s0" });
  assert(p.render(.2).energy > 0);
  assert.deepEqual(p.played(), [{ type: "played", id: "new:s0" }]);
  assert.equal(p.elements.get("output-caption").textContent, "新しい返答。");
});

test("a text-only approved message cannot produce sound without playback PCM", () => {
  const p = pipeline();
  p.event({ type: "output_transcript", text: "テスト用の文字列" });
  assert.equal(p.render(.2).energy, 0);
  assert.equal(p.replies.length, 0);
});

test("microphone energy while a reply is being prepared cannot cancel that reply", () => {
  const p = pipeline();
  for (let i = 0; i < 20; i += 1) p.capture(.025);
  assert.equal(p.replies.length, 0);
  assert.equal(p.attempt.generation, 0);
  assert.equal(p.captures.length, 20, "continuous input must still reach the server");
});

test("explicit legacy energy policy stops queued playback on sustained microphone activity", () => {
  const p = pipeline(48000, "energy");
  p.event({ type: "playback", id: "speaking" });
  p.audio(tone());
  for (let i = 0; i < 7; i += 1) p.capture(.025);
  assert.equal(p.attempt.generation, 0, "less than 160ms must not interrupt");
  p.capture(.025);
  assert.equal(p.attempt.generation, 1);
  assert.equal(p.attempt.pendingPlayback.size, 0);
  assert.deepEqual(p.replies.filter((message) => message.type === "user_activity"), [{ type: "user_activity" }]);
  assert.equal(p.render(.2).energy, 0);
  assert.equal(p.played().length, 0);
  for (let i = 0; i < 20; i += 1) p.capture(.025);
  assert.equal(p.replies.filter((message) => message.type === "user_activity").length, 1);
});

for (const policy of ["transcript", undefined, "unknown"]) {
  test(`Live or unspecified policy (${policy}) forwards loud noise without stopping queued speech`, () => {
    const p = pipeline(48000, policy);
    p.event({ type: "playback", id: "speaking" });
    p.audio(tone(24000));
    p.event({ type: "playback_end", id: "speaking" });
    p.capture(1); // One maximum-level click.
    for (let i = 0; i < 100; i += 1) p.capture(.8); // Two seconds of continuous noise.
    assert.equal(p.attempt.generation, 0);
    assert(p.attempt.pendingPlayback.has("speaking"));
    assert.equal(p.captures.length, 101);
    assert.equal(p.replies.filter((message) => message.type === "user_activity").length, 0);
    assert.equal(p.diagnostics().filter((message) => message.stage === "local_interrupt").length, 0);
    assert(p.render(.2).energy > 0);
    p.event({ type: "interrupt" });
    assert.equal(p.render(.2).energy, 0, "recognized interruption from the server must still stop speech");
  });
}

test("Live keeps playing across two segments during continuous loud microphone noise", () => {
  const p = pipeline(48000, "transcript");
  for (const id of ["reply:s0", "reply:s1"]) {
    p.event({ type: "playback", id, message_id: "reply" });
    p.audio(new Int16Array(24000).fill(4096).buffer);
    p.event({ type: "playback_end", id });
  }
  const output = p.render(1.5, (seconds) => .8 * Math.sin(2 * Math.PI * 1000 * seconds));
  assert(output.samples.every((sample) => sample === .125), "noise must not create a gap at the segment boundary");
  assert(p.captures.length >= 75);
  assert.equal(p.attempt.generation, 0);
  assert.deepEqual(p.played(), [{ type: "played", id: "reply:s0" }]);
  assert(p.attempt.pendingPlayback.has("reply:s1"));
  assert.equal(p.replies.filter((message) => message.type === "user_activity").length, 0);
  p.event({ type: "interrupt" });
  assert.equal(p.render(.7).energy, 0);
  assert.deepEqual(p.played(), [{ type: "played", id: "reply:s0" }], "interrupted second segment must not be acknowledged");
});

test("user-click audio test renders two tones without microphone or API access", async () => {
  const p = pipeline();
  const local = await p.localTest();
  assert.equal(local.isTest, true);
  assert.equal(local.node.options.numberOfInputs, 0);
  assert.equal(local.socket, undefined);
  assert.equal(p.elements.get("test-audio").disabled, true);
  assert(p.render(.85).peak > .13);
  assert.equal(local.context.state, "closed");
  assert.equal(p.active(), null);
  assert.equal(p.timers.size, 0, "the timeout must be removed after success");
  assert.equal(p.elements.get("test-audio").disabled, false);
  assert.match(p.elements.get("audio-test-result").textContent, /2つの音が聞こえれば/);
  assert.match(p.elements.get("audio-test-result").textContent, /ミュート・Windowsの音量と出力先/);
  assert.equal(p.replies.length, 0);
  const pcm = new Int16Array(p.tonePCM());
  const maximum = (start, end) => pcm.slice(start * 24000, end * 24000).reduce((peak, value) => Math.max(peak, Math.abs(value)), 0);
  assert.equal(maximum(0, .08), 0);
  assert(maximum(.08, .30) > 4000);
  assert.equal(maximum(.30, .44), 0);
  assert(maximum(.44, .66) > 4000);
  assert.equal(maximum(.66, .82), 0);
});

test("zero rendered audio cannot report a successful local audio test", async () => {
  const p = pipeline();
  const local = await p.localTest();
  for (const segment of p.processor.segments) for (const chunk of segment.chunks) chunk.fill(0);
  assert.equal(p.render(.85).energy, 0);
  assert.equal(local.context.state, "closed");
  assert.match(p.elements.get("audio-test-result").textContent, /出力できませんでした/);
  assert.match(p.elements.get("error").textContent, /出力を確認できませんでした/);
});

test("0.4x real-time packet arrival waits for EOF then plays without silent holes", () => {
  const p = pipeline();
  p.event({ type: "playback", id: "slow_tts" });
  for (let i = 0; i < 12; i += 1) {
    const chunk = new Int16Array(2400).fill(4096); // 100ms of known nonzero PCM.
    p.audio(chunk.buffer);
    assert.equal(p.render(.25).energy, 0, "do not start 100ms audio every 250ms");
  }
  assert.equal(p.workletMessages.filter((event) => event.type === "playing").length, 0);
  assert.equal(p.workletMessages.filter((event) => event.type === "buffering").length, 1);
  p.event({ type: "playback_end", id: "slow_tts" });
  const output = p.render(1.3).samples;
  assert(output.slice(0, 57600).every((sample) => sample === .125),
    "the complete 1.2s utterance must have no inserted silence or skipped samples");
  assert(output.slice(57600).every((sample) => sample === 0));
  const started = p.workletMessages.find((event) => event.type === "playing");
  const ended = p.workletMessages.find((event) => event.type === "rendered");
  assert.equal(ended.frame - started.frame, 57600);
  assert.deepEqual(p.played(), [{ type: "played", id: "slow_tts" }]);
});

for (const rate of [24000, 44100, 48000]) {
  test(`audible microphone at ${rate}Hz keeps PCM volume and real-time 24kHz duration`, () => {
    const p = pipeline(rate);
    p.render(2.04, (seconds) => .1 * Math.sin(2 * Math.PI * 1000 * seconds));
    assert(p.captures.length >= 101 && p.captures.length <= 102,
      "2.04s microphone should send approximately 102 consecutive 20ms packets");
    let energy = 0;
    let count = 0;
    let peak = 0;
    for (const buffer of p.captures) {
      assert.equal(buffer.byteLength, 960);
      for (const sample of new Int16Array(buffer)) {
        const value = sample / 32768;
        energy += value * value;
        peak = Math.max(peak, Math.abs(value));
        count += 1;
      }
    }
    assert(Math.abs(count / 24000 - 2.04) <= .02);
    assert(Math.abs(Math.sqrt(energy / count) - .1 / Math.sqrt(2)) < .0002);
    assert(peak > .099 && peak < .101);
    const stats = p.diagnostics().filter((event) => event.stage === "capture_stats");
    assert.equal(stats.length, 1);
    assert(Math.abs(stats[0].capture_ms - stats[0].elapsed_ms) <= 4,
      "audio duration must follow the simulated real-time device clock");
    assert(Math.abs(stats[0].capture_rms - .1 / Math.sqrt(2)) < .0002);
    assert(stats[0].capture_peak > .099);
    assert.equal(stats[0].buffered_bytes, 0);
    assert.match(p.elements.get("microphone-status").textContent, /マイクの音を受信/);
  });
}

test("microphone continues to reach the server while TTS waits for EOF", () => {
  const p = pipeline();
  p.event({ type: "playback", id: "loading" });
  p.audio(tone());
  p.render(.5, (seconds) => .005 * Math.sin(2 * Math.PI * 440 * seconds));
  assert(p.captures.length >= 24);
  assert.equal(p.workletMessages.filter((event) => event.type === "playing").length, 0);
  assert.equal(p.attempt.pendingPlayback.has("loading"), true);
});

test("Jev mode forwards loud overlapping speech without a local energy interrupt", () => {
  const p = pipeline();
  p.attempt.interruptionPolicy = "semantic";
  p.event({ type: "playback", id: "semantic" });
  p.audio(tone(48000));
  p.event({ type: "playback_end", id: "semantic" });
  const output = p.render(.5, (seconds) => .1 * Math.sin(2 * Math.PI * 440 * seconds));
  assert(output.energy > 0);
  assert(p.captures.length >= 24);
  assert(p.attempt.pendingPlayback.has("semantic"));
  assert.equal(p.diagnostics().filter(e => e.stage === "local_interrupt").length, 0);
  p.event({ type: "interrupt" });
  assert.equal(p.render(.2).energy, 0, "server interruption must still stop playback");
});

test("socket backpressure is diagnosed without discarding normal microphone audio", () => {
  const p = pipeline();
  p.attempt.socket.bufferedAmount = 30000;
  p.capture(.01);
  assert.equal(p.captures.length, 1);
  const diagnostic = p.diagnostics().find((event) => event.stage === "send_backpressure");
  assert.equal(diagnostic.buffered_bytes, 30000);
});

test("arbitrary PCM chunk boundaries preserve every sample through continuous playback", () => {
  const p = pipeline(48000);
  const source = new Int16Array(12001);
  for (let i = 0; i < source.length; i += 1) source[i] = 1000 + i % 97;
  p.event({ type: "playback", id: "fragmented" });
  let position = 0;
  for (const length of [1, 2, 17, 479, 1, 1024, 256, 9, 1, 8192, 2019]) {
    if (position >= source.length) break;
    const end = Math.min(position + length, source.length);
    p.audio(source.slice(position, end).buffer);
    position = end;
  }
  if (position < source.length) p.audio(source.slice(position).buffer);
  p.event({ type: "playback_end", id: "fragmented" });
  const rendered = p.render(.55).samples;
  for (let i = 0; i < source.length; i += 1) {
    assert.equal(rendered[i * 2], source[i] / 32768);
    assert.equal(rendered[i * 2 + 1], (source[i] + source[Math.min(source.length - 1, i + 1)]) / 65536);
  }
  assert(rendered.slice(source.length * 2).every((sample) => sample === 0));
});

test("simultaneous audible capture and playback fit comfortably within the device clock", (t) => {
  const p = pipeline();
  p.attempt.activitySent = true; // Measure duplex DSP without intentionally triggering barge-in.
  p.event({ type: "playback", id: "duplex" });
  p.audio(tone(48000));
  p.event({ type: "playback_end", id: "duplex" });
  const output = p.render(2.04, (seconds) => .1 * Math.sin(2 * Math.PI * 1000 * seconds));
  assert.equal(p.played().length, 1);
  assert(p.captures.length >= 101);
  assert(output.processingMs < 2040, "processing must keep up with the real-time device");
  t.diagnostic(JSON.stringify({ device_rate: 48000, audio_ms: 2040,
    worklet_ms: Math.round(output.processingMs * 100) / 100,
    worklet_p95_ms: Math.round(output.processP95Ms * 1000) / 1000,
    device_block_budget_ms: 128 / 48 }));
});

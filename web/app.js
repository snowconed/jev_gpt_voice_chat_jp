const $ = (id) => document.getElementById(id);
const ui = {
  role: $("role"), start: $("start"), stop: $("stop"), status: $("status"),
  dot: $("status-dot"), hint: $("connection-hint"), error: $("error"),
  orb: $("voice-orb"), voice: $("voice-label"), input: $("input-caption"),
  output: $("output-caption"), messages: $("messages"),
  form: $("text-form"), text: $("text-input"), send: $("send-text"),
  testAudio: $("test-audio"), testResult: $("audio-test-result"),
  microphone: $("microphone-status"), micLevel: $("microphone-level"),
  fillerOptions: $("filler-options"), fillerEnabled: $("filler-enabled"),
  fillerHelp: $("filler-help"), fillerStatus: $("filler-status"),
};

let config = null;
let active = null;
let captions = { input: "", output: "" };
let lastMessages = "";
let configurationRevision = 0;
let fillerPreference = null;
const fillerCache = new Map();
const fillerHelp = "「ええと」などをすぐに再生し、その間に返答を準備します。対話を始める前に切り替えられます。";

const processingLanes = [
  ["turn", "ターン判断"], ["reply", "応答準備"],
];
let processingAnnouncement = "";
let processingRecent = "";

function replaceText(target, value) {
  // Timers can change every 0.5 seconds; unchanged content must not be
  // repeatedly announced or rebuild the recent list.
  if (target.textContent !== value) target.textContent = value;
}

function processingText(value, fallback = "") {
  return typeof value === "string" ? value : fallback;
}

function processingSeconds(value) {
  return `${(Math.max(0, value) / 1000).toFixed(1)}秒`;
}

function renderProcessing(state, stopped = false) {
  if (!state || typeof state !== "object") return;
  const terminal = [state.turn?.code, state.reply?.code].some((code) => code === "closed" || code === "error");
  if (terminal) {
    stopped = true;
    state = { ...state, recent: [] };
  }
  const mode = { live: "Live方式", jev: "Jev方式" }[state.mode];
  replaceText($("processing-mode"), mode || "接続後に方式を表示します");
  const announcement = [];
  for (const [key, title] of processingLanes) {
    const lane = state[key] || {};
    const label = processingText(lane.label, "状態を確認中");
    replaceText($(`processing-${key}-label`), label);
    replaceText($(`processing-${key}-detail`), processingText(lane.detail));
    $(`processing-${key}`).className = `processing-card${stopped ? " stopped" : ""}`;
    const timing = [];
    if (!stopped && Number.isFinite(lane.elapsed_ms) && lane.elapsed_ms >= 0) {
      timing.push(`経過 ${processingSeconds(lane.elapsed_ms)}`);
    }
    if (!stopped && Number.isFinite(lane.remaining_ms)) {
      const waitLabel = processingText(lane.wait_label, "次の再判定まで") || "次の再判定まで";
      timing.push(`${waitLabel} 約${processingSeconds(lane.remaining_ms)}`);
    }
    const timer = $(`processing-${key}-timing`);
    replaceText(timer, timing.join("\n"));
    timer.hidden = !timing.length;
    announcement.push(`${title}：${label}`);
  }
  const recent = Array.isArray(state.recent)
    ? state.recent.filter((item) => item && typeof item.label === "string").slice(0, 6) : [];
  const recentSignature = JSON.stringify(recent.map(({ label, detail }) => [label, detail]));
  if (recentSignature !== processingRecent) {
    processingRecent = recentSignature;
    $("processing-recent-list").replaceChildren(...recent.map((item) => {
      const row = document.createElement("li");
      const label = document.createElement("span");
      label.className = "processing-recent-label";
      label.textContent = item.label;
      const detail = document.createElement("span");
      detail.className = "processing-recent-detail";
      detail.textContent = processingText(item.detail);
      row.append(label, detail);
      return row;
    }));
  }
  $("processing-recent-list").hidden = !recent.length;
  $("processing-recent-empty").hidden = Boolean(recent.length);
  // Only stage labels enter the live region. Detail and elapsed/remaining
  // updates remain readable without repeatedly interrupting screen readers.
  const summary = announcement.join("。 ");
  if (summary !== processingAnnouncement) {
    processingAnnouncement = summary;
    replaceText($("processing-announcement"), summary);
  }
}

function resetProcessing(phase = "idle") {
  const detail = {
    idle: "対話を始めると、進み具合を表示します。",
    connecting: "接続の準備ができるまでお待ちください。",
    connected: "サーバーから現在の進み具合を確認しています。",
    stopped: "対話を終了しました。もう一度始めると、新しい対話の進み具合を表示します。",
    error: "対話は停止しています。エラーメッセージを確認してください。",
  }[phase];
  const label = { idle: "開始前", connecting: "接続準備中", connected: "状態を確認中", stopped: "終了", error: "停止（エラー）" }[phase];
  const ended = phase === "stopped" || phase === "error";
  renderProcessing({ mode: config?.turn_mode,
    turn: { label, detail }, reply: { label, detail }, recent: [],
  }, ended);
}

function setStatus(text, kind = "") {
  ui.status.textContent = text;
  ui.dot.className = `status-dot ${kind}`;
}

function showError(error) {
  ui.error.textContent = error;
  ui.error.hidden = !error;
}

function setControls() {
  const connected = Boolean(active?.socket?.readyState === WebSocket.OPEN);
  ui.start.disabled = Boolean(active);
  ui.start.textContent = config ? "音声対話を始める" : "接続を確認して始める";
  ui.role.disabled = !config || Boolean(active);
  ui.stop.disabled = !active;
  ui.testAudio.disabled = Boolean(active);
  ui.text.disabled = !connected;
  ui.send.disabled = !connected;
  ui.fillerEnabled.disabled = Boolean(active) || config?.filler_supported !== true;
  if (!active) ui.fillerHelp.textContent = fillerHelp;
}

function createAttempt(isTest = false) {
  return {
    isTest, cancelled: false, generation: 0, pendingPlayback: new Set(),
    playbackMessages: new Set(), discardedPlaybackMessages: new Set(),
    interruptionPolicy: config?.interruption_policy || "transcript",
    playbackTimers: new Set(), currentPlayback: null,
    speechMs: 0, quietMs: 0, activitySent: false, receivedBytes: new Map(),
    captureWindowStart: null, captureSamples: 0, captureEnergy: 0,
    capturePeak: 0, lastMeterAt: -Infinity, lastBackpressureAt: -Infinity,
    fillerEnabled: false, fillerClips: new Map(), fillerPlayback: new Map(),
    seenFillerIds: new Set(), lastFillerRevision: -1,
    pendingTextInputs: new Set(), textInputSequence: 0,
  };
}

function audioDiagnostic(attempt, stage, id = null, metrics = {}) {
  send(attempt, {
    type: "audio_diagnostic", stage, id,
    context_state: attempt.context?.state || "unavailable",
    sample_rate: attempt.context?.sampleRate || 0,
    received_bytes: id ? (attempt.receivedBytes?.get(id) || 0) : 0,
    ...metrics,
  });
}

function caption(kind, text, replace = false) {
  if (typeof text !== "string") return;
  captions[kind] = replace ? text : captions[kind] + text;
  // Keep the live caption bounded; the canonical dialogue remains in the log.
  if (captions[kind].length > 1800) captions[kind] = captions[kind].slice(-1800);
  const target = kind === "input" ? ui.input : ui.output;
  target.textContent = captions[kind];
  target.classList.toggle("placeholder", !captions[kind]);
  target.scrollTop = target.scrollHeight;
}

function updateMessages(session) {
  const messages = session?.messages;
  if (!Array.isArray(messages)) return;
  const serialized = JSON.stringify(messages);
  if (serialized === lastMessages) return;
  lastMessages = serialized;
  const nearBottom = ui.messages.scrollHeight - ui.messages.scrollTop - ui.messages.clientHeight < 70;
  const fragment = document.createDocumentFragment();
  for (const message of messages) {
    if (!message || typeof message.content !== "string") continue;
    const article = document.createElement("article");
    const role = message.role === "user" ? "user" : "assistant";
    article.className = `message ${role}`;
    const label = document.createElement("p");
    label.className = "message-label";
    label.textContent = role === "user" ? "あなた" : "AI";
    if (role === "assistant" && message.delivery) {
      const states = { pending: "音声準備中", completed: "再生済み", interrupted: "途中で停止", cancelled: "未再生" };
      label.textContent += ` · ${states[message.delivery.status] || "承認文"}`;
    }
    const body = document.createElement("p");
    body.className = "message-content";
    body.textContent = message.content;
    article.append(label, body);
    fragment.append(article);
  }
  if (!fragment.childNodes.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "対話を始めると、ここに記録が表示されます。";
    fragment.append(empty);
  }
  ui.messages.replaceChildren(fragment);
  if (nearBottom) ui.messages.scrollTop = ui.messages.scrollHeight;
}

function clearPlayback(attempt) {
  // A local interruption can precede the server receiving user_activity.
  // Reject later segments of every reply already received on this generation.
  for (const id of attempt.playbackMessages) attempt.discardedPlaybackMessages.add(id);
  attempt.playbackMessages.clear();
  attempt.generation += 1;
  attempt.currentPlayback = null;
  attempt.pendingPlayback.clear();
  attempt.receivedBytes?.clear();
  attempt.fillerPlayback.clear();
  ui.fillerStatus.hidden = true;
  ui.fillerStatus.textContent = "";
  for (const timer of attempt.playbackTimers) clearTimeout(timer);
  attempt.playbackTimers.clear();
  attempt.node?.port.postMessage({ type: "clear", generation: attempt.generation });
  ui.orb.classList.remove("speaking");
}

function send(attempt, message) {
  if (attempt.socket?.readyState === WebSocket.OPEN) attempt.socket.send(JSON.stringify(message));
}

// Worklet completion means samples have been rendered, not necessarily heard.
// Acknowledge only after the estimated device playhead has passed their end.
function acknowledgeAfterPlayback(attempt, event) {
  const endTime = event.frame / attempt.context.sampleRate;
  const isCurrent = () => active === attempt && !attempt.cancelled
    && event.generation === attempt.generation && attempt.pendingPlayback.has(event.id);
  const check = () => {
    if (!isCurrent()) return;
    const context = attempt.context;
    if (context.state !== "running") {
      fail(attempt, "音声の再生が一時停止しました。もう一度、対話を開始してください。");
      return;
    }
    const timestamp = context.getOutputTimestamp?.();
    const fallbackLatency = Math.max(.04, (context.baseLatency || 0) + (context.outputLatency || 0));
    const heardThrough = timestamp?.contextTime > 0
      ? timestamp.contextTime : context.currentTime - fallbackLatency;
    if (heardThrough >= endTime) {
      attempt.pendingPlayback.delete(event.id);
      if (attempt.fillerPlayback.delete(event.id)) {
        ui.fillerStatus.hidden = true;
        ui.fillerStatus.textContent = "";
      }
      audioDiagnostic(attempt, "played", event.id);
      send(attempt, { type: "played", id: event.id });
      attempt.receivedBytes?.delete(event.id);
      if (!attempt.pendingPlayback.size) ui.orb.classList.remove("speaking");
      if (attempt.isTest) {
        stopAttempt(attempt, "確認音を出力しました", false);
        ui.testResult.textContent = "2つの音が聞こえれば再生の準備は完了です。聞こえない場合は、ブラウザのミュート・Windowsの音量と出力先を確認してください。";
        ui.hint.textContent = "音が聞こえたら、音声対話を始めてください。";
      }
      return;
    }
    const timer = setTimeout(() => { attempt.playbackTimers.delete(timer); check(); }, 20);
    attempt.playbackTimers.add(timer);
  };
  check();
}

function disposeAudio(attempt) {
  if (attempt.testTimer) clearTimeout(attempt.testTimer);
  clearPlayback(attempt);
  attempt.stream?.getTracks().forEach((track) => track.stop());
  attempt.source?.disconnect();
  attempt.node?.disconnect();
  if (attempt.node) attempt.node.port.onmessage = null;
  if (attempt.context && attempt.context.state !== "closed") {
    attempt.context.onstatechange = null;
    attempt.context.close().catch(() => {});
  }
  ui.orb.style.setProperty("--level", "0");
  ui.micLevel.value = 0;
  ui.microphone.textContent = "マイクは停止しています。";
}

function stopAttempt(attempt, text = "対話を終了しました", tellServer = true) {
  if (attempt.cancelled) return;
  attempt.cancelled = true;
  attempt.setupAbort?.abort();
  if (tellServer) send(attempt, { type: "stop" });
  disposeAudio(attempt);
  if (attempt.socket?.readyState === WebSocket.OPEN) attempt.socket.close(1000, "conversation ended");
  else if (attempt.socket?.readyState === WebSocket.CONNECTING) {
    // Closing during the handshake still releases the server-side session.
    attempt.socket.close();
  }
  if (active === attempt) {
    active = null;
    if (!attempt.isTest) resetProcessing("stopped");
    setStatus(text);
    ui.hint.textContent = "もう一度始めると、新しい対話を開始します。";
    setControls();
  }
}

function fail(attempt, message) {
  if (active !== attempt || attempt.cancelled) return;
  audioDiagnostic(attempt, "client_error", attempt.currentPlayback);
  stopAttempt(attempt, attempt.isTest ? "音声テストが停止しました" : "対話が停止しました");
  if (!attempt.isTest) resetProcessing("error");
  if (attempt.isTest) ui.testResult.textContent = "確認音を出力できませんでした。下のメッセージをご確認ください。";
  showError(message);
}

function friendlyError(error) {
  if (error?.name === "NotAllowedError") return "マイクの使用が許可されていません。ブラウザのサイト設定でマイクを許可して、もう一度お試しください。";
  if (error?.name === "NotFoundError") return "マイクが見つかりません。接続と入力デバイスの設定を確認してください。";
  if (error?.name === "NotReadableError") return "マイクを利用できません。別のアプリが使用していないか確認してください。";
  return error?.message || "接続できませんでした。サーバーの起動とネットワークを確認してください。";
}

async function request(path, options = {}) {
  const response = await fetch(path, { credentials: "same-origin", ...options });
  let data;
  try { data = await response.json(); } catch { throw new Error("サーバーからの応答を読み取れませんでした。"); }
  if (!response.ok) throw new Error(data.error?.message || data.error || data.message || `接続エラー (${response.status})`);
  return data;
}

async function readConfiguration(signal) {
  const controller = new AbortController();
  const abort = () => controller.abort();
  if (signal?.aborted) abort();
  else signal?.addEventListener("abort", abort, { once: true });
  const timeout = setTimeout(abort, 10000);
  try {
    const data = await request("/api/config", { cache: "no-store", signal: controller.signal });
    if (typeof data.csrf_token !== "string" || !data.csrf_token || !Array.isArray(data.roles)
        || !data.roles.length || data.roles.some((role) => !role || typeof role.id !== "string" || !role.id)) {
      throw new Error("会話の設定を読み込めませんでした。サーバーの設定を確認してください。");
    }
    return data;
  } catch (error) {
    if (error?.name === "AbortError" && !signal?.aborted) {
      throw new Error("接続の確認が時間内に終わりませんでした。サーバーの起動を確認して、もう一度お試しください。");
    }
    if (error instanceof TypeError) {
      throw new Error("サーバーに接続できませんでした。サーバーを起動して、もう一度お試しください。");
    }
    throw error;
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener("abort", abort);
  }
}

function applyConfiguration(data, selectedRole = ui.role.value) {
  config = data;
  ui.role.replaceChildren(...data.roles.map((role) => {
    const option = document.createElement("option");
    option.value = role.id;
    option.textContent = role.label || role.name || role.id;
    return option;
  }));
  ui.role.value = data.roles.some((role) => role.id === selectedRole) ? selectedRole : data.roles[0].id;
  ui.voice.textContent = data.speech_voice_label || `同じ声で対話 · ${data.voice || "marin"}`;
  $("server-version").textContent = `音声版 ${data.version || "不明"}`;
  ui.fillerOptions.hidden = data.filler_supported !== true;
  if (data.filler_supported === true && fillerPreference === null) fillerPreference = true;
  ui.fillerEnabled.checked = fillerPreference !== false;
  ui.fillerHelp.textContent = fillerHelp;
}

function fillerCatalog(data) {
  if (data?.filler_supported !== true || !Array.isArray(data.filler_clips)
      || !data.filler_clips.length || data.filler_clips.length > 8) return [];
  const ids = new Set();
  for (const clip of data.filler_clips) {
    if (!clip || typeof clip.id !== "string" || !clip.id || ids.has(clip.id)
        || typeof clip.text !== "string" || !clip.text || clip.text.length > 40
        || typeof clip.url !== "string" || !clip.url.startsWith("/") || clip.url.startsWith("//")
        || !Number.isSafeInteger(clip.audio_bytes) || clip.audio_bytes <= 0
        || clip.audio_bytes % 2 || clip.audio_bytes > 48000 * 3
        || !Number.isFinite(clip.audio_seconds)
        || Math.abs(clip.audio_seconds - clip.audio_bytes / 48000) > .002) return [];
    // Only fetch the same server that supplied this voice configuration.
    try {
      const url = new URL(clip.url, location.href);
      if (url.origin !== new URL(location.href).origin) return [];
    } catch { return []; }
    ids.add(clip.id);
  }
  return data.filler_clips;
}

async function preloadFillers(data, signal) {
  const clips = fillerCatalog(data);
  if (!clips.length) return new Map();
  const controller = new AbortController();
  const abort = () => controller.abort();
  if (signal?.aborted) abort();
  else signal?.addEventListener("abort", abort, { once: true });
  const timeout = setTimeout(abort, 2500);
  try {
    const loaded = await Promise.all(clips.map(async (clip) => {
      const signature = JSON.stringify([clip.id, clip.url, clip.text, clip.audio_bytes, clip.audio_seconds]);
      let cached = fillerCache.get(clip.id);
      if (cached?.signature !== signature) {
        const response = await fetch(clip.url, { credentials: "same-origin", signal: controller.signal });
        if (!response.ok) throw new Error("filler unavailable");
        const buffer = await response.arrayBuffer();
        if (buffer.byteLength !== clip.audio_bytes) throw new Error("invalid filler audio");
        cached = { ...clip, signature, buffer };
        fillerCache.set(clip.id, cached);
        while (fillerCache.size > 16) fillerCache.delete(fillerCache.keys().next().value);
      }
      return [clip.id, cached];
    }));
    return signal?.aborted ? new Map() : new Map(loaded);
  } catch {
    // A missing optional clip must never prevent the actual conversation.
    controller.abort();
    return new Map();
  } finally {
    clearTimeout(timeout);
    signal?.removeEventListener("abort", abort);
  }
}

function stopFiller(attempt, id) {
  if (typeof id !== "string") return;
  attempt.seenFillerIds.add(id);
  if (!attempt.fillerPlayback.delete(id)) return;
  attempt.pendingPlayback.delete(id);
  attempt.receivedBytes.delete(id);
  attempt.node?.port.postMessage({ type: "cancel", id, generation: attempt.generation });
  ui.fillerStatus.hidden = true;
  ui.fillerStatus.textContent = "";
  if (!attempt.pendingPlayback.size) ui.orb.classList.remove("speaking");
}

function playFiller(attempt, message) {
  const clip = attempt.fillerClips.get(message.clip_id);
  const duplicate = attempt.seenFillerIds.has(message.id);
  if (typeof message.id === "string") attempt.seenFillerIds.add(message.id);
  if (!attempt.fillerEnabled || !clip || typeof message.id !== "string" || !message.id
      || duplicate || !Number.isSafeInteger(message.revision) || message.revision <= attempt.lastFillerRevision
      || message.audio_bytes !== clip.audio_bytes || Math.abs(message.audio_seconds - clip.audio_seconds) > .002
      || !Number.isFinite(message.audio_seconds) || message.text !== clip.text
      || attempt.pendingPlayback.size || attempt.pendingTextInputs.size || !attempt.node) {
    // A duplicate notification of the currently playing filler must not release
    // the server's wait for its original, still audible playback.
    if (typeof message.id === "string" && !attempt.fillerPlayback.has(message.id)) {
      send(attempt, { type: "filler_skipped", id: message.id });
    }
    return;
  }
  attempt.lastFillerRevision = message.revision;
  attempt.fillerPlayback.set(message.id, clip);
  attempt.pendingPlayback.add(message.id);
  attempt.receivedBytes.set(message.id, clip.audio_bytes);
  const buffer = clip.buffer.slice(0); // Transfer a copy; the voice cache survives every turn.
  const generation = attempt.generation;
  audioDiagnostic(attempt, "playback_received", message.id);
  attempt.node.port.postMessage({ type: "start", id: message.id, generation });
  attempt.node.port.postMessage({ type: "audio", id: message.id, buffer, generation }, [buffer]);
  attempt.node.port.postMessage({ type: "end", id: message.id, generation });
  audioDiagnostic(attempt, "playback_buffered", message.id);
}

function handleServerMessage(attempt, event) {
  if (active !== attempt || attempt.cancelled) return;
  if (event.data instanceof ArrayBuffer) {
    // An interruption can overtake audio already in transit. Those bytes must
    // be discarded, never replayed or treated as a new utterance.
    if (!attempt.currentPlayback || !attempt.pendingPlayback.has(attempt.currentPlayback)) return;
    if (event.data.byteLength % 2 !== 0) {
      fail(attempt, "音声の受信順序に問題がありました。もう一度、対話を開始してください。");
      return;
    }
    attempt.receivedBytes.set(attempt.currentPlayback,
      (attempt.receivedBytes.get(attempt.currentPlayback) || 0) + event.data.byteLength);
    attempt.node.port.postMessage({
      type: "audio", id: attempt.currentPlayback, buffer: event.data,
      generation: attempt.generation,
    }, [event.data]);
    return;
  }
  let message;
  try { message = JSON.parse(event.data); } catch { fail(attempt, "対話の情報を受信できませんでした。"); return; }
  switch (message.type) {
    case "input_accepted":
      if (typeof message.client_input_id === "string") attempt.pendingTextInputs.delete(message.client_input_id);
      break;
    case "filler":
      playFiller(attempt, message);
      break;
    case "filler_stop":
      stopFiller(attempt, message.id);
      break;
    case "status":
      setStatus(message.text || "接続中です", "active");
      break;
    case "session":
      updateMessages(message.session);
      break;
    case "processing_state":
      if (!attempt.isTest && !attempt.processingEnded) {
        attempt.processingEnded = [message.turn?.code, message.reply?.code]
          .some((code) => code === "closed" || code === "error");
        renderProcessing(message);
      }
      break;
    case "input_transcript":
      caption("input", message.text, message.replace === true);
      break;
    case "output_transcript":
      if (attempt.discardedPlaybackMessages.has(message.message_id)) break;
      caption("output", message.text, message.replace === true);
      break;
    case "playback": {
      const messageId = message.message_id ?? message.id;
      if (typeof message.id !== "string" || typeof messageId !== "string") {
        fail(attempt, "音声の再生情報を確認できませんでした。");
        return;
      }
      if (attempt.discardedPlaybackMessages.has(messageId)) {
        attempt.currentPlayback = null;
        return;
      }
      if (attempt.pendingPlayback.has(message.id)) {
        fail(attempt, "音声の再生情報を確認できませんでした。");
        return;
      }
      const alreadyQueued = attempt.pendingPlayback.size > 0;
      attempt.currentPlayback = message.id;
      attempt.playbackMessages.add(messageId);
      attempt.pendingPlayback.add(message.id);
      attempt.receivedBytes.set(message.id, 0);
      if (!attempt.isTest && !alreadyQueued) setStatus("声を準備しています", "busy");
      audioDiagnostic(attempt, "playback_received", message.id);
      attempt.node.port.postMessage({ type: "start", id: message.id, generation: attempt.generation });
      break;
    }
    case "playback_end":
      if (!attempt.pendingPlayback.has(message.id)) break;
      audioDiagnostic(attempt, "playback_buffered", message.id);
      attempt.node.port.postMessage({ type: "end", id: message.id, generation: attempt.generation });
      if (attempt.currentPlayback === message.id) attempt.currentPlayback = null;
      break;
    case "interrupt":
      if (Number.isSafeInteger(message.revision)) {
        attempt.lastFillerRevision = Math.max(attempt.lastFillerRevision, message.revision);
      }
      clearPlayback(attempt);
      break;
    case "error":
      fail(attempt, message.message || "対話中にエラーが発生しました。");
      break;
    case "closed":
      stopAttempt(attempt, "対話を終了しました", false);
      break;
  }
}

function handleWorkletMessage(attempt, data) {
  if (active !== attempt || attempt.cancelled) return;
  if (data.type === "capture") {
    const now = performance.now();
    if (now - attempt.lastMeterAt >= 100) {
      attempt.lastMeterAt = now;
      ui.orb.style.setProperty("--level", String(Math.min(1, data.level * 7)));
      ui.micLevel.value = Math.min(1, data.level * 7);
      ui.microphone.textContent = data.level >= .003
        ? "マイクの音を受信しています。文字起こしは上の欄に表示されます。"
        : "マイク入力は小さめです。話すとメーターが動きます。";
    }
    if (attempt.socket?.readyState === WebSocket.OPEN) {
      const samples = data.buffer.byteLength / 2;
      const elapsedAudioMs = samples / 24;
      if (attempt.captureWindowStart === null) attempt.captureWindowStart = now - elapsedAudioMs;
      attempt.captureSamples += samples;
      attempt.captureEnergy += data.level * data.level * samples;
      attempt.capturePeak = Math.max(attempt.capturePeak, data.peak ?? data.level);
      if (now - attempt.captureWindowStart >= 2000) {
        audioDiagnostic(attempt, "capture_stats", null, {
          capture_ms: Math.round(attempt.captureSamples / 24),
          elapsed_ms: Math.round(now - attempt.captureWindowStart),
          capture_rms: Number(Math.sqrt(attempt.captureEnergy / attempt.captureSamples).toFixed(6)),
          capture_peak: Number(attempt.capturePeak.toFixed(6)),
          buffered_bytes: attempt.socket.bufferedAmount,
        });
        attempt.captureWindowStart = now;
        attempt.captureSamples = 0;
        attempt.captureEnergy = 0;
        attempt.capturePeak = 0;
      }
      if (attempt.socket.bufferedAmount > 24000 && now - attempt.lastBackpressureAt >= 2000) {
        attempt.lastBackpressureAt = now;
        audioDiagnostic(attempt, "send_backpressure", null, { buffered_bytes: attempt.socket.bufferedAmount });
      }
      if (attempt.socket.bufferedAmount > 240000) {
        fail(attempt, "音声の送信が遅れています。ネットワークを確認して、もう一度お試しください。");
        return;
      }
      // Live and Jev wait for recognized speech at the server. Energy-only
      // interruption is a legacy opt-in, never the default for missing config.
      if (data.level >= .018) {
        attempt.speechMs += elapsedAudioMs;
        attempt.quietMs = 0;
        if (attempt.interruptionPolicy === "energy" && !attempt.activitySent && attempt.speechMs >= 160 && attempt.pendingPlayback.size > 0) {
          attempt.activitySent = true;
          audioDiagnostic(attempt, "local_interrupt", attempt.currentPlayback);
          clearPlayback(attempt);
          send(attempt, { type: "user_activity" });
        }
      } else {
        attempt.speechMs = 0;
        attempt.quietMs += elapsedAudioMs;
        if (attempt.quietMs >= 300) attempt.activitySent = false;
      }
      attempt.socket.send(data.buffer);
    }
  } else if (data.type === "buffering" && data.generation === attempt.generation) {
    audioDiagnostic(attempt, "buffering", data.id);
  } else if (data.type === "playing" && data.generation === attempt.generation) {
    if (!attempt.pendingPlayback.has(data.id)) return;
    audioDiagnostic(attempt, "playing", data.id);
    if (attempt.fillerPlayback.has(data.id)) {
      ui.fillerStatus.textContent = `フィラー「${attempt.fillerPlayback.get(data.id).text}」を再生中 · 返答の準備は続けています`;
      ui.fillerStatus.hidden = false;
    } else if (!attempt.isTest) setStatus("お話ししています", "active");
    ui.orb.classList.add("speaking");
  } else if (data.type === "rendered" && data.generation === attempt.generation) {
    audioDiagnostic(attempt, data.peak > 0 ? "rendered" : "rendered_silent", data.id);
    if (attempt.isTest && !(data.peak > 0 && data.renderedSamples > 0)) {
      fail(attempt, "確認音の出力を確認できませんでした。ブラウザを更新して、もう一度お試しください。");
      return;
    }
    acknowledgeAfterPlayback(attempt, data);
  }
}

function connectWorkletHandlers(attempt) {
  attempt.node.onprocessorerror = () => {
    audioDiagnostic(attempt, "processor_error", attempt.currentPlayback);
    fail(attempt, "音声の処理が停止しました。ブラウザを更新して、もう一度お試しください。");
  };
  attempt.node.port.onmessage = ({ data }) => handleWorkletMessage(attempt, data);
  attempt.context.onstatechange = () => audioDiagnostic(attempt, "context_state", attempt.currentPlayback);
}

function testTonePCM() {
  const rate = 24000;
  const samples = new Int16Array(Math.round(rate * .82));
  for (const [start, frequency] of [[.08, 523.25], [.44, 659.25]]) {
    const length = Math.round(rate * .22);
    const offset = Math.round(rate * start);
    for (let i = 0; i < length; i += 1) {
      const fade = Math.min(1, i / (rate * .012), (length - 1 - i) / (rate * .025));
      samples[offset + i] = Math.round(Math.sin(2 * Math.PI * frequency * i / rate) * 32767 * .14 * fade);
    }
  }
  return samples.buffer;
}

async function testAudio() {
  if (active) return;
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    showError("このブラウザは音声の再生に対応していません。最新版の Chrome または Edge をお使いください。");
    return;
  }
  const attempt = createAttempt(true);
  active = attempt;
  setControls();
  showError("");
  setStatus("確認音を再生しています", "busy");
  ui.testResult.textContent = "2つの短い音が聞こえるか、ご確認ください。";
  ui.hint.textContent = "このテストではマイクとAPIを使用しません。";
  try {
    attempt.context = new AudioContextClass({ latencyHint: "interactive" });
    await attempt.context.resume();
    if (attempt.cancelled) return;
    if (!attempt.context.audioWorklet) throw new Error("このブラウザは必要な音声機能に対応していません。最新版の Chrome または Edge をお使いください。");
    await attempt.context.audioWorklet.addModule("/pcm-worklet.js");
    if (attempt.cancelled) return;
    attempt.node = new AudioWorkletNode(attempt.context, "voice-pcm", {
      numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [1],
    });
    connectWorkletHandlers(attempt);
    attempt.node.connect(attempt.context.destination);
    const id = "local_audio_test";
    handleServerMessage(attempt, { data: JSON.stringify({ type: "playback", id }) });
    handleServerMessage(attempt, { data: testTonePCM() });
    handleServerMessage(attempt, { data: JSON.stringify({ type: "playback_end", id }) });
    attempt.testTimer = setTimeout(() => {
      fail(attempt, "確認音の再生が完了しませんでした。ブラウザの音声設定を確認して、もう一度お試しください。");
    }, 6000);
  } catch (error) {
    fail(attempt, friendlyError(error));
  }
}

async function startConversation() {
  if (active) return;
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!navigator.mediaDevices?.getUserMedia || !AudioContextClass) {
    showError("このブラウザではマイクを使用できません。最新版の Chrome または Edge で localhost のページを開いてください。");
    return;
  }
  const attempt = createAttempt();
  const revision = ++configurationRevision;
  const selectedRole = ui.role.value;
  const requestedFiller = ui.fillerEnabled.checked !== false;
  attempt.setupAbort = new AbortController();
  active = attempt;
  resetProcessing("connecting");
  showError("");
  setControls();
  setStatus("接続設定を確認しています", "busy");
  ui.hint.textContent = "サーバーへの接続を確認しています。";
  try {
    attempt.context = new AudioContextClass({ latencyHint: "interactive" });
    // Resume in the user's click handler, before awaiting network work. Each
    // attempt fetches the current server token; a prior process's token expires.
    const [freshConfig] = await Promise.all([
      readConfiguration(attempt.setupAbort.signal), attempt.context.resume(),
    ]);
    if (attempt.cancelled || active !== attempt || revision !== configurationRevision) return;
    applyConfiguration(freshConfig, selectedRole);
    attempt.csrfToken = freshConfig.csrf_token;
    attempt.roleId = ui.role.value;
    attempt.interruptionPolicy = freshConfig.interruption_policy || "transcript";
    const fillerLoading = requestedFiller ? preloadFillers(freshConfig, attempt.setupAbort.signal) : Promise.resolve(new Map());
    resetProcessing("connecting");
    setStatus("マイクと音声を準備しています", "busy");
    ui.hint.textContent = "マイクの使用を許可してください。";
    if (!attempt.context.audioWorklet) throw new Error("このブラウザは必要な音声機能に対応していません。最新版の Chrome または Edge をご利用ください。");
    await attempt.context.audioWorklet.addModule("/pcm-worklet.js");
    if (attempt.cancelled) return;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
      video: false,
    });
    if (attempt.cancelled) { stream.getTracks().forEach((track) => track.stop()); return; }
    attempt.stream = stream;
    ui.microphone.textContent = "マイクの入力を確認しています。";
    attempt.source = attempt.context.createMediaStreamSource(stream);
    attempt.node = new AudioWorkletNode(attempt.context, "voice-pcm", {
      numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
      channelCount: 1, channelCountMode: "explicit",
    });
    connectWorkletHandlers(attempt);
    attempt.source.connect(attempt.node);
    attempt.node.connect(attempt.context.destination);
    stream.getAudioTracks().forEach((track) => track.addEventListener("ended", () => {
      fail(attempt, "マイクの接続が切れました。マイクを確認して、もう一度お試しください。");
    }));
    setStatus("音声チャットにつないでいます", "busy");
    ui.hint.textContent = "準備ができるまで、少しお待ちください。";
    attempt.fillerClips = await fillerLoading;
    if (attempt.cancelled || active !== attempt) return;
    attempt.fillerEnabled = requestedFiller && attempt.fillerClips.size > 0;
    if (freshConfig.filler_supported === true) {
      ui.fillerHelp.textContent = attempt.fillerEnabled ? "この対話ではフィラーを使います。変更するときは対話を終了してください。"
        : requestedFiller ? "フィラー音声を読み込めなかったため、この対話はフィラーなしで続けます。"
          : "この対話ではフィラーを使いません。変更するときは対話を終了してください。";
    }
    const session = await request("/api/sessions", {
      method: "POST", headers: { "Content-Type": "application/json", "X-CSRF-Token": attempt.csrfToken },
      body: JSON.stringify({ role_id: attempt.roleId, filler_enabled: attempt.fillerEnabled }), signal: attempt.setupAbort.signal,
    });
    if (attempt.cancelled || active !== attempt) return;
    if (session.filler_enabled === false && attempt.fillerEnabled) {
      attempt.fillerEnabled = false;
      ui.fillerHelp.textContent = "サーバーのフィラー音声を利用できないため、この対話はフィラーなしで続けます。";
    }
    const socketUrl = new URL(`/api/voice/${encodeURIComponent(session.id)}`, location.href);
    socketUrl.protocol = location.protocol === "https:" ? "wss:" : "ws:";
    socketUrl.searchParams.set("token", attempt.csrfToken);
    const socket = new WebSocket(socketUrl);
    attempt.socket = socket;
    socket.binaryType = "arraybuffer";
    socket.addEventListener("open", () => {
      if (attempt.cancelled || active !== attempt) {
        send(attempt, { type: "stop" });
        socket.close(1000, "cancelled during setup");
        return;
      }
      captions = { input: "", output: "" };
      resetProcessing("connected");
      caption("input", "", true);
      caption("output", "", true);
      lastMessages = "";
      updateMessages(session);
      setStatus("音声チャットにつないでいます", "busy");
      audioDiagnostic(attempt, "connected");
      ui.hint.textContent = attempt.interruptionPolicy === "semantic"
        ? "普段の声でお話しください。AIが話している間の相槌や割込みも確認します。"
        : attempt.interruptionPolicy === "energy"
          ? "普段の声でお話しください。この設定では、AIの発話中に音が入ると読み上げを止める場合があります。"
          : "普段の声でお話しください。Live方式では、認識した発話に合わせて読み上げを止めます。雑音の音量だけでは止まりません。";
      setControls();
    });
    socket.addEventListener("message", (event) => handleServerMessage(attempt, event));
    socket.addEventListener("error", () => fail(attempt, "音声の接続に失敗しました。サーバーとネットワークを確認してください。"));
    socket.addEventListener("close", () => {
      if (!attempt.cancelled) fail(attempt, "音声の接続が終了しました。続ける場合は、もう一度対話を開始してください。");
    });
  } catch (error) {
    if (!attempt.cancelled) fail(attempt, friendlyError(error));
  }
}

ui.start.addEventListener("click", startConversation);
ui.testAudio.addEventListener("click", testAudio);
ui.fillerEnabled.addEventListener("change", () => {
  if (active) return;
  fillerPreference = ui.fillerEnabled.checked;
  if (fillerPreference) void preloadFillers(config);
});
ui.stop.addEventListener("click", () => { if (active) stopAttempt(active); });
ui.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = ui.text.value.trim();
  if (!text || !active || active.socket?.readyState !== WebSocket.OPEN) return;
  clearPlayback(active);
  const clientInputId = `text-${++active.textInputSequence}`;
  active.pendingTextInputs.add(clientInputId);
  send(active, { type: "text", text, client_input_id: clientInputId });
  ui.text.value = "";
  ui.text.focus();
});
window.addEventListener("pagehide", () => { if (active) stopAttempt(active); });

async function initialize() {
  const revision = ++configurationRevision;
  try {
    const data = await readConfiguration();
    if (revision !== configurationRevision || active) return;
    applyConfiguration(data);
    setStatus("声で話す準備ができました");
    if (fillerPreference !== false) void preloadFillers(data);
  } catch (error) {
    if (revision !== configurationRevision || active) return;
    config = null;
    resetProcessing("error");
    setStatus("設定を読み込めませんでした");
    ui.hint.textContent = "サーバーを起動して「接続を確認して始める」を押すと、再試行できます。";
    showError(friendlyError(error));
  }
  if (revision === configurationRevision) setControls();
}

initialize();

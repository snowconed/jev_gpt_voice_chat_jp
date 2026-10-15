/* Mono PCM transport. Capture and playback always use the actual device rate. */
class VoicePCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.captureEnabled = true;
    this.captureFrame = new Int16Array(480);
    this.captureLength = 0;
    this.inputIndex = 0;
    this.nextCaptureAt = 0;
    this.previousInput = 0;
    this.frameEnergy = 0;
    this.framePeak = 0;
    this.segments = [];
    this.byId = new Map();
    this.generation = 0;
    this.port.onmessage = ({ data }) => this.receive(data);
  }

  receive(data) {
    if (data.type === "clear") {
      this.segments = [];
      this.byId.clear();
      this.generation = data.generation;
      return;
    }
    if (data.type === "capture") {
      this.captureEnabled = Boolean(data.enabled);
      return;
    }
    if (data.generation !== this.generation) return;
    if (data.type === "cancel") {
      this.byId.delete(data.id);
      this.segments = this.segments.filter((segment) => segment.id !== data.id);
      return;
    }
    if (data.type === "start") {
      if (this.byId.has(data.id)) return;
      const segment = {
        id: data.id, chunks: [], head: 0, offset: 0, remaining: 0,
        fraction: 0, sealed: false, started: false, waiting: false,
        renderedSamples: 0, peak: 0,
      };
      this.byId.set(data.id, segment);
      this.segments.push(segment);
    } else {
      const segment = this.byId.get(data.id);
      if (!segment) return;
      if (data.type === "audio") {
        const samples = new Int16Array(data.buffer);
        if (samples.length) {
          segment.chunks.push(samples);
          segment.remaining += samples.length;
        }
      } else if (data.type === "end") {
        segment.sealed = true;
      }
    }
  }

  peek(segment, relativeIndex) {
    let head = segment.head;
    let offset = segment.offset + relativeIndex;
    while (head < segment.chunks.length) {
      const chunk = segment.chunks[head];
      if (offset < chunk.length) return chunk[offset] / 32768;
      offset -= chunk.length;
      head += 1;
    }
    return 0;
  }

  consume(segment, count) {
    count = Math.min(count, segment.remaining);
    segment.remaining -= count;
    while (count > 0) {
      const available = segment.chunks[segment.head].length - segment.offset;
      const take = Math.min(count, available);
      count -= take;
      segment.offset += take;
      if (segment.offset === segment.chunks[segment.head].length) {
        segment.head += 1;
        segment.offset = 0;
      }
    }
    if (segment.head > 32) {
      segment.chunks = segment.chunks.slice(segment.head);
      segment.head = 0;
    }
  }

  finish(segment, frame) {
    this.segments.shift();
    this.byId.delete(segment.id);
    this.port.postMessage({
      type: "rendered", id: segment.id, frame, generation: this.generation,
      renderedSamples: segment.renderedSamples, peak: segment.peak,
    });
  }

  capture(samples) {
    if (!samples) return;
    const step = sampleRate / 24000;
    for (let i = 0; i < samples.length; i += 1) {
      const current = samples[i];
      while (this.nextCaptureAt <= this.inputIndex) {
        const fraction = this.nextCaptureAt - (this.inputIndex - 1);
        const value = this.inputIndex === 0 ? current
          : this.previousInput + (current - this.previousInput) * fraction;
        const clipped = Math.max(-1, Math.min(1, value));
        this.captureFrame[this.captureLength++] = clipped < 0
          ? Math.round(clipped * 32768) : Math.round(clipped * 32767);
        this.frameEnergy += clipped * clipped;
        this.framePeak = Math.max(this.framePeak, Math.abs(clipped));
        this.nextCaptureAt += step;
        if (this.captureLength === this.captureFrame.length) {
          const buffer = this.captureFrame.buffer;
          this.port.postMessage({
            type: "capture", buffer,
            level: Math.sqrt(this.frameEnergy / this.captureLength),
            peak: this.framePeak,
          }, [buffer]);
          this.captureFrame = new Int16Array(480);
          this.captureLength = 0;
          this.frameEnergy = 0;
          this.framePeak = 0;
        }
      }
      this.previousInput = current;
      this.inputIndex += 1;
    }
  }

  process(inputs, outputs) {
    if (this.captureEnabled) this.capture(inputs[0]?.[0]);
    const output = outputs[0]?.[0];
    if (!output) return true;
    output.fill(0);
    // TTS may arrive slower than speech plays. Starting on the first network
    // packet would insert silence at each starvation. Wait for the whole
    // approved segment; following segments can queue while this one plays.
    // Capture continues independently while a segment is loading.
    if (this.segments.length && !this.segments[0].sealed) {
      const segment = this.segments[0];
      if (!segment.waiting) {
        segment.waiting = true;
        this.port.postMessage({ type: "buffering", id: segment.id, generation: this.generation });
      }
      return true;
    }
    const step = 24000 / sampleRate;
    for (let i = 0; i < output.length; i += 1) {
      while (this.segments.length) {
        const segment = this.segments[0];
        if (!segment.sealed) break;
        if (segment.remaining === 0 && segment.sealed) {
          this.finish(segment, currentFrame + i);
          continue;
        }
        if (!segment.started) {
          segment.started = true;
          this.port.postMessage({ type: "playing", id: segment.id, frame: currentFrame + i, generation: this.generation });
        }
        const first = this.peek(segment, 0);
        const second = segment.remaining > 1 ? this.peek(segment, 1) : first;
        output[i] = first + (second - first) * segment.fraction;
        segment.renderedSamples += 1;
        segment.peak = Math.max(segment.peak, Math.abs(output[i]));
        segment.fraction += step;
        const consumed = Math.floor(segment.fraction);
        segment.fraction -= consumed;
        this.consume(segment, consumed);
        if (segment.remaining === 0 && segment.sealed) this.finish(segment, currentFrame + i + 1);
        break;
      }
    }
    return true;
  }
}

registerProcessor("voice-pcm", VoicePCMProcessor);

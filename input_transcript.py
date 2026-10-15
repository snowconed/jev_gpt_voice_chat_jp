"""Conservative incremental cleanup of known Live non-speech annotations."""
from __future__ import annotations

TAG_HOLD_SECONDS = 1.0
NON_SPEECH_TAGS = frozenset({
    "gasp", "sigh", "breath", "breathing", "cough", "coughing",
    "laugh", "laughing", "laughter", "chuckle", "chuckling",
    "clears throat", "throat clearing",
})


class InputTranscriptFilter:
    """Remove only allowlisted tags, retaining unknown bracketed text verbatim.

    A bracket is held only while it can still become a known tag. A complete
    known name without its closing bracket is dropped after a bounded wait;
    ambiguous fragments such as '[s' are restored verbatim at that deadline.
    Raw deltas must be logged by the caller before using this projection.
    """

    def __init__(self):
        self.pending = ""
        self.started = 0.0
        self._discard_closer = False

    @property
    def has_pending(self):
        return bool(self.pending)

    def reset(self):
        self.pending = ""
        self.started = 0.0
        self._discard_closer = False

    def flush(self, now, *, force=False):
        if not self.pending or (not force and now - self.started < TAG_HOLD_SECONDS):
            return ""
        text, self.pending = self.pending, ""
        if text[1:].strip().casefold() in NON_SPEECH_TAGS:
            self._discard_closer = True
            return ""
        return text

    def feed(self, delta, now):
        output = [self.flush(now)]
        for char in delta:
            if self._discard_closer and not self.pending:
                if char == "]":
                    self._discard_closer = False
                    continue
                self._discard_closer = False
            if not self.pending:
                if char == "[":
                    self.pending, self.started = char, now
                else:
                    output.append(char)
                continue
            self.pending += char
            inner = self.pending[1:]
            if char == "]":
                if inner[:-1].strip().casefold() not in NON_SPEECH_TAGS:
                    output.append(self.pending)
                self.pending = ""
            elif not any(tag.startswith(inner.lstrip().casefold())
                         or inner.strip().casefold() == tag for tag in NON_SPEECH_TAGS):
                output.append(self.pending)
                self.pending = ""
            elif len(self.pending) > 64:
                output.append(self.pending)
                self.pending = ""
        return "".join(output)

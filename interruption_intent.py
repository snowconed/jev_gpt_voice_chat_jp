"""Explicit short-phrase rules for holding an already interrupted voice reply.

This is not model-based semantic judgment. Call only after recognizing speech
that interrupted assistant playback, preparation, or generation. A result asks
the caller to retain that input while waiting for the user's continuation;
it does not authorize dropping text or starting a replacement reply.
"""
from __future__ import annotations

import unicodedata
from typing import Literal


_IGNORABLE = frozenset("。 、，．,.!?！？「」『』\"“”‘’")
_PAUSES = frozenset({
    "待って", "待ってください", "待って下さい",
    "ちょっと待って", "ちょっと待ってください", "ちょっと待って下さい",
    "少し待って", "少し待ってください", "少し待って下さい",
    "少々待ってください", "少々待って下さい",
    "ストップ", "止めて", "止めてください", "止めて下さい",
    "やめて", "やめてください", "やめて下さい",
    "stop", "wait",
})
_CORRECTIONS = frozenset({
    "違う", "違うよ", "違うんです", "違います", "違いますよ",
    "いや", "いやいや", "いいえ", "いえ",
    "そうじゃない", "そうじゃないです", "そうではない", "そうではありません",
    "no",
})
_PHRASES = tuple((text, True) for text in _PAUSES) + tuple(
    (text, False) for text in _CORRECTIONS
)


def interruption_kind(text: str) -> Literal["pause", "correction"] | None:
    """Match the whole input using only explicit stop/denial phrases.

    Repetitions and combinations such as 'いや、違う、待って' are allowed.
    A pause phrase takes precedence over a denial. Any substantive remainder,
    including '待って、明日の話です', returns None so the caller can process
    the full correction normally. No partial-prefix or acknowledgment matching
    is performed. The original text is neither modified nor consumed.
    """
    if not isinstance(text, str):
        return None
    compact = "".join(character for character in unicodedata.normalize("NFKC", text).casefold()
                      if not character.isspace() and character not in _IGNORABLE)
    if not compact:
        return None

    # Each reachable position follows complete allowlisted phrases. Tracking
    # positions avoids recursive/exponential matching of repeated negatives.
    reachable = {0: False}
    for position in range(len(compact)):
        if position not in reachable:
            continue
        for phrase, is_pause in _PHRASES:
            if compact.startswith(phrase, position):
                end = position + len(phrase)
                reachable[end] = reachable.get(end, False) or reachable[position] or is_pause
    if len(compact) not in reachable:
        return None
    return "pause" if reachable[len(compact)] else "correction"

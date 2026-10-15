"""At most two independently buffered speech requests, preserving every character."""
import re


def split_speech(text):
    """Release an early sentence without fragmenting names, numbers or words.

    A short acknowledgement alone is not worth another request. Prefer a full
    Japanese sentence within 100 characters, then a long clause at a comma.
    No text is normalized, stripped or rewritten; joining the pieces is exact.
    """
    for pattern, minimum in ((r'[。！？!?]+[」』）”\"]*[ \t\u3000]*', 12),
                             (r'[、，]+[ \t\u3000]*', 30)):
        for match in re.finditer(pattern, text):
            end = match.end()
            if end > 100:
                break
            if end >= minimum and text[end:].strip():
                return [text[:end], text[end:]]
    return [text]

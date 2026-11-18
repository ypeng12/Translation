import re
from dataclasses import dataclass, field


@dataclass
class TextProcessor:
    """Conservative rules: keep negation, names, and meaningful repeated words."""

    glossary: dict[str, str] = field(default_factory=dict)

    def process(self, text: str) -> str:
        text = text.replace("’", "'")
        text = re.sub(r"\b(?:u+m+|u+h+|erm)\b[, ]*", "", text, flags=re.IGNORECASE)
        # Only remove discourse fillers at the start, not 'I know you know him'.
        text = re.sub(r"^\s*(?:you know|i mean)\s*[,;:]\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\b([A-Za-z]+)\s+\1\b", lambda m: m[1] if m[1].lower()
                      in {"the", "a", "an", "i", "and", "to"} else m[0], text, flags=re.IGNORECASE)
        for alias, canonical in sorted(self.glossary.items(), key=lambda p: -len(p[0])):
            text = re.sub(r"(?<!\w)" + re.escape(alias) + r"(?!\w)",
                          lambda _, canonical=canonical: canonical, text, flags=re.IGNORECASE)
        text = re.sub(r"\bi\b", "I", text)
        text = re.sub(r"\s+", " ", text).strip(" ,;:")
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        if not re.search(r"\w", text):
            return ""
        text = text[0].upper() + text[1:]
        if text[-1] not in ".!?":
            text += "."
        return text


@dataclass
class SentenceBuffer:
    """Optional punctuation-based buffering of RAW ASR, before adding punctuation."""

    max_chars: int = 400
    pending: str = ""

    def push(self, text: str, final: bool = False) -> list[str]:
        self.pending = " ".join(filter(None, [self.pending, text.strip()]))
        output = []
        while match := re.search(r"[.!?](?:\s|$)", self.pending):
            end = match.start() + 1
            output.append(self.pending[:end].strip())
            self.pending = self.pending[end:].strip()
        if self.pending and (final or len(self.pending) >= self.max_chars):
            output.append(self.pending)
            self.pending = ""
        return output

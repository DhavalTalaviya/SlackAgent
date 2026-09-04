import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Every ingested document's first line is always "[#channel] user (when):" --
# skip it when judging content shape, or every document (even a one-line
# chat message) would register as having a "speaker line".
_HEADER_LINE = 1

# A line like "Alice: let's ship it" or "Bob (PM): agreed" -- the shape a
# multi-speaker transcript or a Slack thread's aggregated replies take.
_SPEAKER_LINE_RE = re.compile(r"^[A-Za-z][\w .'\-]{0,40}:\s")

_CHAT_CHAR_THRESHOLD = 400
_DOC_CHAR_THRESHOLD = 1200
_TRANSCRIPT_MIN_LINES = 4
_TRANSCRIPT_SPEAKER_RATIO = 0.4

_SPLITTERS = {
    # Most chat messages are already under this size and won't be split at
    # all -- this config mainly matters for the occasional long thread that
    # doesn't read as a transcript (few distinct speakers, short turns).
    "chat": RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50),
    # Larger chunks, more overlap, and "\n" prioritized over mid-sentence
    # breaks -- a chunk boundary landing mid-speaker-turn loses the "who
    # said what" context a transcript question usually needs.
    "transcript": RecursiveCharacterTextSplitter(
        chunk_size=1800, chunk_overlap=300, separators=["\n\n", "\n", ". ", " ", ""]
    ),
    # Long single-author content (announcements, notes dumps) -- paragraph-
    # aware splitting via the default separator order, just with a bigger
    # budget than chat since the unit of meaning is a paragraph, not a line.
    "doc": RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200),
}


def classify_content(text: str) -> str:
    """Content-based, not source-based: a meeting transcript pasted as one
    Slack message and a busy multi-reply thread both get read from the text
    shape itself, not from how the message arrived."""
    if len(text) < _CHAT_CHAR_THRESHOLD:
        return "chat"

    body_lines = text.split("\n")[_HEADER_LINE:]
    lines = [line for line in body_lines if line.strip()]
    if lines:
        speaker_lines = sum(1 for line in lines if _SPEAKER_LINE_RE.match(line))
        if len(lines) >= _TRANSCRIPT_MIN_LINES and speaker_lines / len(lines) >= _TRANSCRIPT_SPEAKER_RATIO:
            return "transcript"

    if len(text) > _DOC_CHAR_THRESHOLD:
        return "doc"

    return "chat"


def chunk_document(doc: Document) -> list[Document]:
    """Classifies and splits one document with the splitter suited to its
    content shape. Tags content_type onto the metadata (inherited by every
    resulting chunk) so it's auditable and available for future filtering,
    not just an invisible internal decision."""
    content_type = classify_content(doc.page_content)
    doc.metadata["content_type"] = content_type
    return _SPLITTERS[content_type].split_documents([doc])


def chunk_documents(docs: list[Document]) -> list[Document]:
    chunks: list[Document] = []
    for doc in docs:
        chunks.extend(chunk_document(doc))
    return chunks

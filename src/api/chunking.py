import re

MAX_CHUNK_SIZE = 500


def chunk_text(text: str) -> list[str]:
    if not text:
        return []

    parts = re.split(r"(。|！|？|\n)", text)
    delimiters = {"。", "！", "？", "\n"}
    sentences = []
    current = ""
    for part in parts:
        current += part
        if part in delimiters:
            sentences.append(current)
            current = ""
    if current:
        sentences.append(current)

    chunks = []
    chunk = ""
    for sentence in sentences:
        if chunk and len(chunk) + len(sentence) > MAX_CHUNK_SIZE:
            chunks.append(chunk)
            chunk = sentence
        else:
            chunk += sentence
    if chunk:
        chunks.append(chunk)
    return chunks

import re
import unicodedata


_PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2014": "-",
        "\u2013": "-",
        "\u2212": "-",
    }
)


def sanitize_narration_text(text: str) -> str:
    """Normalize story text into a safer spoken form for TTS and subtitles."""

    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = normalized.translate(_PUNCT_TRANSLATION)
    normalized = re.sub(r"(?<=\w)[\-_\/](?=\w)", " ", normalized)
    normalized = normalized.replace("\\", " ")
    normalized = normalized.replace("|", " ")
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r'["“”]', "", normalized)
    normalized = re.sub(r"[()\[\]{}<>]", "", normalized)
    normalized = re.sub(r"[;:]+", ",", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s+([,.!?])", r"\1", normalized)
    return normalized.strip()


def normalize_alignment_token(text: str) -> str:
    """Normalize a single token for exact subtitle/timestamp comparisons."""

    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    normalized = normalized.translate(_PUNCT_TRANSLATION)
    normalized = re.sub(r"^[^\w']+|[^\w']+$", "", normalized)
    normalized = re.sub(r"[^a-z0-9']+", "", normalized)
    return normalized


def tokenize_alignment_text(text: str) -> list[str]:
    """Convert story text into the exact token stream used for timing checks."""

    sanitized = sanitize_narration_text(text)
    return [
        token
        for token in (
            normalize_alignment_token(raw_token)
            for raw_token in sanitized.split()
        )
        if token
    ]


def extract_alignment_tokens_from_words(words: list[dict]) -> list[str]:
    """Normalize Whisper-style word payloads into comparable alignment tokens."""

    return [
        token
        for token in (
            normalize_alignment_token(str(word.get("word", "") or ""))
            for word in words
        )
        if token
    ]


def summarize_token_alignment(
    expected_text: str,
    transcript_text: str,
    words: list[dict],
) -> dict:
    """Summarize whether a transcript is safe for exact word highlighting."""

    expected_tokens = tokenize_alignment_text(expected_text)
    transcript_tokens = tokenize_alignment_text(transcript_text)
    word_tokens = extract_alignment_tokens_from_words(words)

    if not expected_tokens:
        fallback_reason = "expected_text_empty"
    elif not word_tokens:
        fallback_reason = "missing_word_timestamps"
    elif len(word_tokens) != len(expected_tokens):
        fallback_reason = "word_count_mismatch"
    elif word_tokens != expected_tokens:
        fallback_reason = "token_mismatch"
    else:
        fallback_reason = None

    return {
        "expected_word_count": len(expected_tokens),
        "transcript_word_count": len(transcript_tokens),
        "timed_word_count": len(word_tokens),
        "transcript_text_matches": transcript_tokens == expected_tokens,
        "timed_words_match": word_tokens == expected_tokens,
        "usable_for_word_highlight": fallback_reason is None,
        "fallback_reason": fallback_reason,
    }

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .debug_utils import debug_print, summarize_wav_audio


DEFAULT_ALIGNMENT_BUNDLE = os.getenv(
    "STORYAI_ALIGNMENT_BUNDLE",
    "WAV2VEC2_ASR_BASE_960H",
)
_ALIGNMENT_MODEL_CACHE: dict[tuple[str, str], tuple[object, object, dict[str, int]]] = {}


@dataclass
class StoryToken:
    """One displayed story token and the normalized alignment words it maps to."""

    original_text: str
    normalized_words: list[str]


def get_alignment_runtime_config() -> dict[str, str]:
    """Return the currently configured local word-alignment backend."""

    return {
        "timing_engine": "local_forced_alignment",
        "alignment_bundle": DEFAULT_ALIGNMENT_BUNDLE,
    }


def align_story_text_to_audio(
    audio_path: Path,
    story_text: str,
    bundle_name: str = DEFAULT_ALIGNMENT_BUNDLE,
) -> dict:
    """Align a known story transcript to generated narration audio."""

    torch, torchaudio = _load_alignment_dependencies()
    bundle, model, token_dictionary = _load_alignment_bundle(
        torch=torch,
        torchaudio=torchaudio,
        bundle_name=bundle_name,
    )
    device = _resolve_alignment_device(torch)

    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.ndim != 2:
        raise RuntimeError(
            f"Unexpected audio shape for forced alignment: {tuple(waveform.shape)}"
        )
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            bundle.sample_rate,
        )
        sample_rate = bundle.sample_rate

    story_tokens = _build_story_token_map(story_text)
    normalized_words = [word for token in story_tokens for word in token.normalized_words]
    if not normalized_words:
        raise RuntimeError("Story text did not contain any alignable words.")

    unknown_characters = sorted(
        {
            character
            for word in normalized_words
            for character in word
            if character not in token_dictionary
        }
    )
    if unknown_characters:
        raise RuntimeError(
            "Forced alignment cannot tokenize these characters: "
            f"{''.join(unknown_characters)!r}"
        )

    flattened_targets = [
        token_dictionary[character]
        for word in normalized_words
        for character in word
    ]
    if not flattened_targets:
        raise RuntimeError("Forced alignment target sequence is empty.")

    request_payload = {
        "audio": summarize_wav_audio(audio_path),
        "story_text": story_text,
        "normalized_words": normalized_words,
        "bundle": bundle_name,
        "device": str(device),
    }
    debug_print("ALIGNMENT REQUEST", request_payload)

    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        targets = torch.tensor([flattened_targets], dtype=torch.int32, device=device)
        input_lengths = torch.tensor([emission.size(1)], dtype=torch.int32, device=device)
        target_lengths = torch.tensor([len(flattened_targets)], dtype=torch.int32, device=device)
        aligned_tokens, alignment_scores = torchaudio.functional.forced_align(
            emission,
            targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=token_dictionary["-"],
        )
        token_spans = torchaudio.functional.merge_tokens(
            aligned_tokens[0].cpu(),
            alignment_scores[0].cpu(),
            blank=token_dictionary["-"],
        )

    per_word_spans = _unflatten(
        token_spans,
        [len(word) for word in normalized_words],
    )
    if len(per_word_spans) != len(normalized_words):
        raise RuntimeError(
            "Forced alignment returned an unexpected number of word spans: "
            f"expected {len(normalized_words)}, got {len(per_word_spans)}."
        )

    duration_seconds = waveform.size(1) / sample_rate if sample_rate else 0.0
    frame_ratio = duration_seconds / emission.size(1) if emission.size(1) else 0.0

    words = []
    span_index = 0
    for token in story_tokens:
        piece_count = len(token.normalized_words)
        if piece_count == 0:
            continue

        piece_spans = per_word_spans[span_index : span_index + piece_count]
        span_index += piece_count
        if not piece_spans or not all(piece_spans):
            raise RuntimeError(
                f"Forced alignment failed to produce spans for token: {token.original_text!r}"
            )

        first_span = piece_spans[0][0]
        last_span = piece_spans[-1][-1]
        words.append(
            {
                "word": token.original_text,
                "normalized_word": " ".join(token.normalized_words),
                "start": round(first_span.start * frame_ratio, 3),
                "end": round(last_span.end * frame_ratio, 3),
                "score": round(_score_alignment(piece_spans), 4),
                "source": "local_forced_alignment",
            }
        )

    response = {
        "text": story_text,
        "words": words,
        "segments": [],
        "duration": round(duration_seconds, 3),
        "timing_engine": "local_forced_alignment",
        "alignment_bundle": bundle_name,
    }
    debug_print(
        "ALIGNMENT RESPONSE",
        {
            "word_count": len(words),
            "duration_seconds": response["duration"],
            "bundle": bundle_name,
            "device": str(device),
        },
    )
    debug_print("ALIGNMENT WORDS", words)
    return response


def _load_alignment_dependencies():
    """Import alignment dependencies only when the timing step runs."""

    try:
        import torch
        import torchaudio
    except ImportError as exc:
        raise RuntimeError(
            "Local forced alignment requires torch and torchaudio. "
            "Install the repo requirements in Colab before launching the app."
        ) from exc
    return torch, torchaudio


def _resolve_alignment_device(torch) -> object:
    """Choose the alignment device with an optional environment override."""

    configured = os.getenv("STORYAI_ALIGNMENT_DEVICE")
    if configured:
        return torch.device(configured)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_alignment_bundle(torch, torchaudio, bundle_name: str):
    """Load and cache the acoustic model and token dictionary."""

    device = str(_resolve_alignment_device(torch))
    cache_key = (bundle_name, device)
    if cache_key in _ALIGNMENT_MODEL_CACHE:
        return _ALIGNMENT_MODEL_CACHE[cache_key]

    try:
        bundle = getattr(torchaudio.pipelines, bundle_name)
    except AttributeError as exc:
        raise RuntimeError(f"Unknown torchaudio alignment bundle: {bundle_name}") from exc

    if not hasattr(bundle, "get_model") or not hasattr(bundle, "get_dict"):
        raise RuntimeError(
            f"Alignment bundle {bundle_name} does not expose model/dictionary helpers."
        )

    model = bundle.get_model().to(_resolve_alignment_device(torch)).eval()
    token_dictionary = bundle.get_dict()
    if "-" not in token_dictionary:
        raise RuntimeError(
            f"Alignment bundle {bundle_name} does not expose the blank token '-'."
        )

    resources = (bundle, model, token_dictionary)
    _ALIGNMENT_MODEL_CACHE[cache_key] = resources
    return resources


def _build_story_token_map(story_text: str) -> list[StoryToken]:
    """Map displayed story tokens to normalized alignment words."""

    tokens: list[StoryToken] = []
    for original_token in story_text.split():
        normalized_words = _normalize_alignment_token(original_token)
        if not normalized_words:
            continue
        tokens.append(
            StoryToken(
                original_text=original_token,
                normalized_words=normalized_words,
            )
        )
    return tokens


def _normalize_alignment_token(text: str) -> list[str]:
    """Normalize one display token into the upper-case words used by the ASR bundle."""

    normalized = unicodedata.normalize("NFKC", text).upper()
    normalized = normalized.translate(
        str.maketrans(
            {
                "-": " ",
                "/": " ",
            }
        )
    )
    normalized = re.sub(r"[^A-Z' ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.split() if normalized else []


def _score_alignment(piece_spans: list[list[object]]) -> float:
    """Compute a weighted average alignment confidence across merged token pieces."""

    weighted_score = 0.0
    total_length = 0
    for spans in piece_spans:
        for span in spans:
            span_length = len(span)
            weighted_score += span.score * span_length
            total_length += span_length
    if total_length == 0:
        raise RuntimeError("Forced alignment produced zero-length spans.")
    return weighted_score / total_length


def _unflatten(sequence: list[object], lengths: list[int]) -> list[list[object]]:
    """Group a flat token-span list back into the original word boundaries."""

    grouped: list[list[object]] = []
    start = 0
    for length in lengths:
        end = start + length
        grouped.append(sequence[start:end])
        start = end
    if start != len(sequence):
        raise RuntimeError(
            "Forced alignment returned token spans that did not match the expected "
            f"target length: expected {sum(lengths)}, got {len(sequence)}."
        )
    return grouped

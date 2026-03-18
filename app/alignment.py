import json
import os
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path

from .debug_utils import debug_print, summarize_path, summarize_wav_audio


DEFAULT_ALIGNMENT_ENGINE = "montreal_forced_aligner"
DEFAULT_MFA_ACOUSTIC_MODEL = os.getenv("STORYAI_MFA_ACOUSTIC_MODEL", "english_mfa")
DEFAULT_MFA_G2P_MODEL = os.getenv("STORYAI_MFA_G2P_MODEL", "english_us_mfa")
DEFAULT_MFA_BIN = os.getenv("STORYAI_MFA_BIN", "/content/mfa-env/bin/mfa")
ALIGNMENT_MODEL_LABEL = f"{DEFAULT_MFA_ACOUSTIC_MODEL} + {DEFAULT_MFA_G2P_MODEL}"
IGNORED_ALIGNMENT_LABELS = {"", "sp", "sil", "<eps>"}


def get_alignment_runtime_config() -> dict[str, str]:
    """Return the currently configured local word-alignment backend."""

    return {
        "timing_engine": DEFAULT_ALIGNMENT_ENGINE,
        "alignment_model": ALIGNMENT_MODEL_LABEL,
    }


def validate_alignment_runtime() -> dict[str, str]:
    """Fail early if the configured MFA alignment runtime is not available."""

    mfa_bin = _resolve_mfa_bin()
    _run_mfa_command([str(mfa_bin), "--help"], error_prefix="MFA is installed but not runnable")
    _run_mfa_command(
        [str(mfa_bin), "model", "inspect", "acoustic", DEFAULT_MFA_ACOUSTIC_MODEL],
        error_prefix=(
            "MFA acoustic model is missing. "
            "In Colab, run: /content/mfa-env/bin/mfa model download acoustic "
            f"{DEFAULT_MFA_ACOUSTIC_MODEL}"
        ),
    )
    _run_mfa_command(
        [str(mfa_bin), "model", "inspect", "g2p", DEFAULT_MFA_G2P_MODEL],
        error_prefix=(
            "MFA G2P model is missing. "
            "In Colab, run: /content/mfa-env/bin/mfa model download g2p "
            f"{DEFAULT_MFA_G2P_MODEL}"
        ),
    )

    return {
        "alignment_tool": str(mfa_bin.resolve()),
        "alignment_model": ALIGNMENT_MODEL_LABEL,
    }


def align_story_audio_batch(page_audio: list[dict], run_dir: Path) -> dict[str, dict]:
    """Align all page narration clips with Montreal Forced Aligner."""

    if not page_audio:
        return {}

    word_list_path = run_dir / "mfa_word_list.txt"
    dictionary_path = run_dir / "mfa_dictionary.txt"
    output_dir = run_dir / "mfa_alignment"
    temp_root = run_dir / "mfa_temp"
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    word_list = _collect_alignment_words(page_audio)
    if not word_list:
        raise RuntimeError("MFA alignment could not find any alignable words in the story text.")

    word_list_path.write_text("\n".join(word_list) + "\n", encoding="utf-8")
    debug_print(
        "ALIGNMENT WORD LIST",
        {
            "word_list_path": str(word_list_path.resolve()),
            "word_count": len(word_list),
            "preview": word_list[:50],
        },
    )

    _run_mfa_g2p(
        word_list_path=word_list_path,
        dictionary_path=dictionary_path,
        temporary_directory=temp_root / "g2p",
    )

    aligned_pages: dict[str, dict] = {}
    for item in page_audio:
        page_name = item["page"]
        audio_path = Path(item["audio_path"]).resolve()
        transcript_path = run_dir / f"{page_name}_alignment.txt"
        output_path = output_dir / f"{page_name}.json"
        normalized_transcript = _normalize_story_text_for_alignment(item["story_text"])
        transcript_path.write_text(normalized_transcript + "\n", encoding="utf-8")

        _run_mfa_align_one(
            audio_path=audio_path,
            transcript_path=transcript_path,
            dictionary_path=dictionary_path,
            output_path=output_path,
            temporary_directory=temp_root / page_name,
        )

        raw_alignment = _read_mfa_alignment_json(output_path)
        words = _extract_word_entries(raw_alignment, output_path)
        aligned_pages[str(audio_path)] = {
            "text": item["story_text"],
            "words": words,
            "segments": [],
            "duration": summarize_wav_audio(audio_path).get("duration_seconds"),
            "timing_engine": DEFAULT_ALIGNMENT_ENGINE,
            "alignment_model": ALIGNMENT_MODEL_LABEL,
            "mfa_output_path": str(output_path.resolve()),
            "alignment_transcript_path": str(transcript_path.resolve()),
            "alignment_dictionary_path": str(dictionary_path.resolve()),
            "alignment_word_list_path": str(word_list_path.resolve()),
        }

    debug_print(
        "ALIGNMENT BATCH RESPONSE",
        {
            "page_count": len(aligned_pages),
            "output_dir": str(output_dir.resolve()),
            "alignment_model": ALIGNMENT_MODEL_LABEL,
        },
    )
    return aligned_pages


def _resolve_mfa_bin() -> Path:
    """Resolve the MFA executable from the configured environment or PATH."""

    candidates = []
    if DEFAULT_MFA_BIN:
        candidates.append(Path(DEFAULT_MFA_BIN))

    which_path = shutil.which("mfa")
    if which_path:
        candidates.append(Path(which_path))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise RuntimeError(
        "Montreal Forced Aligner executable was not found. "
        "Set STORYAI_MFA_BIN to the mfa binary path, for example "
        "/content/mfa-env/bin/mfa in Colab."
    )


def _run_mfa_g2p(word_list_path: Path, dictionary_path: Path, temporary_directory: Path) -> None:
    """Generate a run-specific dictionary for the story words with MFA G2P."""

    mfa_bin = _resolve_mfa_bin()
    temporary_directory.mkdir(parents=True, exist_ok=True)
    command = [
        str(mfa_bin),
        "g2p",
        str(word_list_path.resolve()),
        DEFAULT_MFA_G2P_MODEL,
        str(dictionary_path.resolve()),
        "--temporary_directory",
        str(temporary_directory.resolve()),
        "--clean",
        "--no_use_mp",
        "--num_jobs",
        "1",
        "--quiet",
    ]
    debug_print(
        "ALIGNMENT G2P REQUEST",
        {
            "mfa_bin": str(mfa_bin.resolve()),
            "word_list_path": str(word_list_path.resolve()),
            "dictionary_path": str(dictionary_path.resolve()),
            "temporary_directory": str(temporary_directory.resolve()),
            "g2p_model": DEFAULT_MFA_G2P_MODEL,
        },
    )
    _run_mfa_command(command, error_prefix="MFA G2P failed")
    if not dictionary_path.exists():
        raise RuntimeError(
            "MFA G2P finished without producing the expected dictionary: "
            f"{dictionary_path}"
        )
    debug_print("ALIGNMENT DICTIONARY", summarize_path(dictionary_path))


def _run_mfa_align_one(
    audio_path: Path,
    transcript_path: Path,
    dictionary_path: Path,
    output_path: Path,
    temporary_directory: Path,
) -> None:
    """Align a single narration clip to a normalized transcript with MFA."""

    mfa_bin = _resolve_mfa_bin()
    temporary_directory.mkdir(parents=True, exist_ok=True)
    command = [
        str(mfa_bin),
        "align_one",
        str(audio_path.resolve()),
        str(transcript_path.resolve()),
        str(dictionary_path.resolve()),
        DEFAULT_MFA_ACOUSTIC_MODEL,
        str(output_path.resolve()),
        "--output_format",
        "json",
        "--temporary_directory",
        str(temporary_directory.resolve()),
        "--clean",
        "--single_speaker",
        "--no_use_mp",
        "--num_jobs",
        "1",
        "--quiet",
    ]
    debug_print(
        "ALIGNMENT REQUEST",
        {
            "mfa_bin": str(mfa_bin.resolve()),
            "audio_path": str(audio_path.resolve()),
            "transcript_path": str(transcript_path.resolve()),
            "dictionary_path": str(dictionary_path.resolve()),
            "output_path": str(output_path.resolve()),
            "temporary_directory": str(temporary_directory.resolve()),
            "acoustic_model": DEFAULT_MFA_ACOUSTIC_MODEL,
        },
    )
    _run_mfa_command(command, error_prefix="MFA alignment failed")
    if not output_path.exists():
        raise RuntimeError(
            "MFA alignment finished without producing the expected output file: "
            f"{output_path}"
        )


def _run_mfa_command(command: list[str], error_prefix: str) -> subprocess.CompletedProcess[str]:
    """Run an MFA command and surface stdout/stderr when it fails."""

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{error_prefix}. stdout:\n{exc.stdout}\n\nstderr:\n{exc.stderr}"
        ) from exc

    debug_print(
        "ALIGNMENT TOOL OUTPUT",
        {
            "command": command,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        },
    )
    return completed


def _collect_alignment_words(page_audio: list[dict]) -> list[str]:
    """Collect unique normalized words across all story pages for G2P."""

    unique_words: set[str] = set()
    for item in page_audio:
        unique_words.update(_tokenize_story_text(item["story_text"]))
    return sorted(unique_words)


def _normalize_story_text_for_alignment(story_text: str) -> str:
    """Create the transcript text passed into MFA alignment."""

    tokens = _tokenize_story_text(story_text)
    if not tokens:
        raise RuntimeError(f"Story text did not contain any alignable words: {story_text!r}")
    return " ".join(tokens)


def _tokenize_story_text(story_text: str) -> list[str]:
    """Normalize story text into MFA-friendly lowercase words."""

    normalized = unicodedata.normalize("NFKC", story_text).lower()
    normalized = normalized.translate(
        str.maketrans(
            {
                "’": "'",
                "‘": "'",
                "“": '"',
                "”": '"',
                "—": " ",
                "–": " ",
                "-": " ",
            }
        )
    )

    tokens: list[str] = []
    for raw_token in normalized.split():
        cleaned = re.sub(r"^[^\w']+|[^\w']+$", "", raw_token)
        cleaned = re.sub(r"[^a-z0-9']+", "", cleaned)
        if cleaned:
            tokens.append(cleaned)
    return tokens


def _read_mfa_alignment_json(output_path: Path) -> dict:
    """Read one MFA JSON alignment output file."""

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"MFA alignment output file was not found: {output_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"MFA alignment output was not valid JSON: {output_path}") from exc

    debug_print(
        "ALIGNMENT OUTPUT",
        {
            "output_path": str(output_path.resolve()),
            "tiers": sorted(payload.get("tiers", {}).keys()),
        },
    )
    return payload


def _extract_word_entries(payload: dict, output_path: Path) -> list[dict]:
    """Convert MFA JSON word tiers into the app word-timestamp shape."""

    tiers = payload.get("tiers", {})
    candidate_tier_names = ["words", "Words"]
    candidate_tier_names.extend(
        tier_name for tier_name in tiers.keys() if tier_name.lower().endswith("words")
    )

    entries = None
    for tier_name in candidate_tier_names:
        tier = tiers.get(tier_name)
        if tier and isinstance(tier.get("entries"), list):
            entries = tier["entries"]
            break

    if entries is None:
        raise RuntimeError(
            "MFA alignment output did not contain a usable words tier: "
            f"{output_path}"
        )

    words: list[dict] = []
    for entry in entries:
        if not isinstance(entry, list) or len(entry) != 3:
            raise RuntimeError(
                f"Unexpected MFA word entry format in {output_path}: {entry!r}"
            )
        start, end, label = entry
        label_text = str(label).strip()
        if label_text.lower() in IGNORED_ALIGNMENT_LABELS:
            continue
        start_value = float(start)
        end_value = float(end)
        if end_value < start_value:
            continue
        words.append(
            {
                "word": label_text,
                "start": round(start_value, 3),
                "end": round(end_value, 3),
                "duration": round(end_value - start_value, 3),
                "source": DEFAULT_ALIGNMENT_ENGINE,
            }
        )

    if not words:
        raise RuntimeError(
            "MFA alignment output did not contain any usable word intervals: "
            f"{output_path}"
        )

    debug_print(
        "ALIGNMENT WORDS",
        {
            "output_path": str(output_path.resolve()),
            "word_count": len(words),
            "preview": words[:10],
        },
    )
    return words

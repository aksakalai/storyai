import json
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from .debug_utils import debug_print, summarize_wav_audio


DEFAULT_ALIGNMENT_ENGINE = "nemo_forced_aligner"
DEFAULT_NEMO_MODEL = os.getenv(
    "STORYAI_NEMO_MODEL",
    "stt_en_citrinet_256_gamma_0_25",
)
DEFAULT_NEMO_REPO = Path(os.getenv("STORYAI_NEMO_REPO", "/content/NeMo"))
DEFAULT_NEMO_ALIGN_SCRIPT = os.getenv(
    "STORYAI_NEMO_ALIGN_SCRIPT",
    str(DEFAULT_NEMO_REPO / "tools" / "nemo_forced_aligner" / "align.py"),
)


def get_alignment_runtime_config() -> dict[str, str]:
    """Return the currently configured local word-alignment backend."""

    return {
        "timing_engine": DEFAULT_ALIGNMENT_ENGINE,
        "alignment_model": DEFAULT_NEMO_MODEL,
    }


def validate_alignment_runtime() -> dict[str, str]:
    """Fail early if the configured NeMo alignment runtime is not available."""

    align_script = _resolve_nemo_align_script()

    if importlib.util.find_spec("nemo") is None:
        raise RuntimeError(
            "NeMo Forced Aligner is configured, but nemo_toolkit is not installed. "
            "In Colab, install it with: pip install \"nemo_toolkit[asr]>=2.5.0\""
        )

    if importlib.util.find_spec("soundfile") is None:
        raise RuntimeError(
            "NeMo Forced Aligner is configured, but the Python soundfile package "
            "is not available. In Colab, install the README dependencies and "
            "apt-get install libsndfile1."
        )

    return {
        "alignment_script": str(align_script.resolve()),
        "alignment_model": DEFAULT_NEMO_MODEL,
    }


def align_story_audio_batch(page_audio: list[dict], run_dir: Path) -> dict[str, dict]:
    """Align all page narration clips in one NeMo batch run."""

    if not page_audio:
        return {}

    manifest_path = run_dir / "nemo_alignment_manifest.jsonl"
    output_dir = run_dir / "nemo_alignment"
    manifest_entries = []
    for item in page_audio:
        audio_path = Path(item["audio_path"]).resolve()
        manifest_entries.append(
            {
                "audio_filepath": str(audio_path),
                "text": item["story_text"],
            }
        )

    _write_manifest(manifest_path, manifest_entries)
    _run_nemo_forced_aligner(
        manifest_path=manifest_path,
        output_dir=output_dir,
        batch_size=len(page_audio),
    )

    raw_results = _read_nemo_output_manifest(manifest_path, output_dir)
    aligned_pages: dict[str, dict] = {}
    for item in page_audio:
        audio_key = str(Path(item["audio_path"]).resolve())
        if audio_key not in raw_results:
            raise RuntimeError(
                "NeMo Forced Aligner did not return output for audio file: "
                f"{audio_key}"
            )
        result = raw_results[audio_key]
        word_ctm_path = Path(result["word_level_ctm_filepath"])
        aligned_pages[audio_key] = {
            "text": item["story_text"],
            "words": _parse_word_ctm(word_ctm_path),
            "segments": [],
            "duration": summarize_wav_audio(audio_key).get("duration_seconds"),
            "timing_engine": DEFAULT_ALIGNMENT_ENGINE,
            "alignment_model": DEFAULT_NEMO_MODEL,
            "nemo_output_manifest_path": str(
                (output_dir / f"{manifest_path.stem}_with_output_file_paths.json").resolve()
            ),
            "word_level_ctm_filepath": str(word_ctm_path.resolve()),
            "segment_level_ctm_filepath": result.get("segment_level_ctm_filepath"),
            "token_level_ctm_filepath": result.get("token_level_ctm_filepath"),
        }

    debug_print(
        "ALIGNMENT BATCH RESPONSE",
        {
            "page_count": len(aligned_pages),
            "output_dir": str(output_dir.resolve()),
            "model": DEFAULT_NEMO_MODEL,
        },
    )
    return aligned_pages


def _write_manifest(manifest_path: Path, entries: list[dict]) -> None:
    """Write the NeMo alignment manifest as JSONL."""

    manifest_path.write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )
    debug_print(
        "ALIGNMENT MANIFEST",
        {
            "manifest_path": str(manifest_path.resolve()),
            "entries": entries,
        },
    )


def _run_nemo_forced_aligner(
    manifest_path: Path,
    output_dir: Path,
    batch_size: int,
) -> None:
    """Run the official NeMo Forced Aligner tool against a manifest."""

    align_script = _resolve_nemo_align_script()
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(align_script),
        f"pretrained_name={DEFAULT_NEMO_MODEL}",
        f"manifest_filepath={manifest_path.resolve()}",
        f"output_dir={output_dir.resolve()}",
        "align_using_pred_text=False",
        "transcribe_device=cpu",
        "viterbi_device=cpu",
        f"batch_size={max(batch_size, 1)}",
        "save_output_file_formats=[ctm]",
    ]
    debug_print(
        "ALIGNMENT REQUEST",
        {
            "script_path": str(align_script.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "output_dir": str(output_dir.resolve()),
            "model": DEFAULT_NEMO_MODEL,
            "batch_size": max(batch_size, 1),
        },
    )
    try:
        completed = subprocess.run(
            command,
            cwd=align_script.parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "NeMo Forced Aligner failed. "
            f"stdout:\n{exc.stdout}\n\nstderr:\n{exc.stderr}"
        ) from exc

    debug_print(
        "ALIGNMENT TOOL OUTPUT",
        {
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        },
    )


def _resolve_nemo_align_script() -> Path:
    """Resolve the NeMo align.py tool path from environment or default Colab paths."""

    candidates = []
    if DEFAULT_NEMO_ALIGN_SCRIPT:
        candidates.append(Path(DEFAULT_NEMO_ALIGN_SCRIPT))
    candidates.append(DEFAULT_NEMO_REPO / "tools" / "nemo_forced_aligner" / "align.py")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise RuntimeError(
        "NeMo Forced Aligner script was not found. "
        "Clone the NeMo tools folder in Colab and set STORYAI_NEMO_ALIGN_SCRIPT "
        "or STORYAI_NEMO_REPO before launching the app."
    )


def _read_nemo_output_manifest(manifest_path: Path, output_dir: Path) -> dict[str, dict]:
    """Read the NeMo output manifest and index it by absolute audio path."""

    output_manifest_path = output_dir / f"{manifest_path.stem}_with_output_file_paths.json"
    if not output_manifest_path.exists():
        raise RuntimeError(
            "NeMo Forced Aligner did not create the expected output manifest: "
            f"{output_manifest_path}"
        )

    indexed_results: dict[str, dict] = {}
    with output_manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            audio_key = str(Path(entry["audio_filepath"]).resolve())
            indexed_results[audio_key] = entry

    debug_print(
        "ALIGNMENT OUTPUT MANIFEST",
        {
            "output_manifest_path": str(output_manifest_path.resolve()),
            "entries": indexed_results,
        },
    )
    return indexed_results


def _parse_word_ctm(word_ctm_path: Path) -> list[dict]:
    """Parse one NeMo word-level CTM file into the app word-timestamp shape."""

    if not word_ctm_path.exists():
        raise RuntimeError(
            "Expected NeMo word-level CTM file was not found: "
            f"{word_ctm_path}"
        )

    words = []
    with word_ctm_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            fields = line.split()
            if len(fields) < 5:
                raise RuntimeError(
                    f"Unexpected CTM line format in {word_ctm_path}: {line!r}"
                )

            start = float(fields[2])
            duration = float(fields[3])
            token = fields[4].replace("<space>", " ")
            words.append(
                {
                    "word": token,
                    "start": round(start, 3),
                    "end": round(start + duration, 3),
                    "duration": round(duration, 3),
                    "source": DEFAULT_ALIGNMENT_ENGINE,
                }
            )

    if not words:
        raise RuntimeError(
            "NeMo Forced Aligner produced an empty word-level CTM file: "
            f"{word_ctm_path}"
        )

    debug_print(
        "ALIGNMENT WORDS",
        {
            "word_ctm_path": str(word_ctm_path.resolve()),
            "word_count": len(words),
            "preview": words[:10],
        },
    )
    return words

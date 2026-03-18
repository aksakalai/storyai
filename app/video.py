import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .debug_utils import debug_print, summarize_path
from .text_utils import normalize_alignment_token, sanitize_narration_text


VIDEO_WIDTH = 1024
VIDEO_HEIGHT = 1024
VIDEO_FPS = 30
CARD_X = 72
CARD_WIDTH = 880
CARD_BOTTOM_MARGIN = 64
CARD_MIN_HEIGHT = 220
CARD_MAX_HEIGHT = 420
CARD_PADDING_X = 88
CARD_PADDING_Y = 52
CARD_TEXT_WIDTH = CARD_WIDTH - (CARD_PADDING_X * 2)
FONT_NAME = os.getenv("STORYAI_SUBTITLE_FONT", "DejaVu Serif")
DEFAULT_FONT_SIZE = int(os.getenv("STORYAI_SUBTITLE_FONT_SIZE", "34"))
MIN_FONT_SIZE = int(os.getenv("STORYAI_MIN_SUBTITLE_FONT_SIZE", "22"))
MIN_WORD_DURATION = float(os.getenv("STORYAI_MIN_WORD_DURATION", "0.12"))
MAX_SUBTITLE_LINES = 6
LINE_HEIGHT_RATIO = 1.35
ASS_PAST_COLOR = "&HFFFFFF&"
ASS_CURRENT_COLOR = "&H7BD7F5&"
ASS_FUTURE_COLOR = "&HBFC9D1&"
DEFAULT_FONTS_DIR = Path("/usr/share/fonts/truetype/dejavu")


@dataclass
class AlignedToken:
    """One displayed token with a matched timing span."""

    text: str
    start: float
    end: float


@dataclass
class SubtitleEvent:
    """One subtitle state over a time interval."""

    start: float
    end: float
    current_index: int | None
    completed_index: int


@dataclass
class SubtitleLayout:
    """Resolved card and text placement for one subtitle script."""

    wrapped_lines: list[list[int]]
    font_size: int
    line_height: int
    line_count: int
    content_height: int
    card_height: int
    card_y: int
    text_center_x: int
    text_center_y: int


def render_page_video(
    image_path: Path,
    audio_path: Path,
    story_text: str,
    words: list[dict],
    duration_seconds: float,
    output_path: Path,
) -> dict:
    """Render one story page clip with a story card and persistent highlighting."""

    _ensure_ffmpeg_installed()
    duration_seconds = max(
        _coerce_float(duration_seconds, default=0.0),
        _coerce_float(words[-1].get("end"), default=0.0) if words else 0.0,
    )

    subtitles_path = output_path.with_name(f"{output_path.stem}_subtitles.ass")
    subtitle_summary = write_highlighted_subtitles(
        story_text=story_text,
        words=words,
        duration_seconds=duration_seconds,
        output_path=subtitles_path,
    )

    card_height = subtitle_summary["card_height"]
    card_y = subtitle_summary["card_y"]

    subtitles_filter = f"subtitles={subtitles_path.name}"
    if DEFAULT_FONTS_DIR.exists():
        subtitles_filter += f":fontsdir={DEFAULT_FONTS_DIR}"

    filter_chain = (
        f"[0:v]scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={VIDEO_WIDTH}:{VIDEO_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        f"drawbox=x={CARD_X}:y={card_y}:w={CARD_WIDTH}:h={card_height}:"
        f"color=black@0.45:t=fill,"
        f"{subtitles_filter}[v]"
    )

    request_payload = {
        "image_path": str(image_path.resolve()),
        "audio_path": str(audio_path.resolve()),
        "subtitles_path": str(subtitles_path.resolve()),
        "output_path": str(output_path.resolve()),
        "story_text": story_text,
        "word_count": len(words),
        "duration_seconds": round(duration_seconds, 3),
        "subtitle_line_count": subtitle_summary["line_count"],
        "subtitle_event_count": subtitle_summary["event_count"],
        "subtitle_font_size": subtitle_summary["font_size"],
        "story_card": {
            "x": CARD_X,
            "y": card_y,
            "width": CARD_WIDTH,
            "height": card_height,
        },
    }
    debug_print("VIDEO RENDER REQUEST", request_payload)

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-loop",
        "1",
        "-framerate",
        str(VIDEO_FPS),
        "-i",
        image_path.name,
        "-i",
        audio_path.name,
        "-filter_complex",
        filter_chain,
        "-map",
        "[v]",
        "-map",
        "1:a",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-shortest",
        output_path.name,
    ]
    try:
        subprocess.run(
            command,
            cwd=output_path.parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed while rendering {output_path.name}: {exc.stderr.strip()}"
        ) from exc

    response_summary = {
        "subtitle_script": summarize_path(subtitles_path),
        "output_video": summarize_path(output_path),
        "duration_seconds": round(duration_seconds, 3),
        "line_count": subtitle_summary["line_count"],
        "event_count": subtitle_summary["event_count"],
        "font_size": subtitle_summary["font_size"],
        "subtitle_mode": subtitle_summary["subtitle_mode"],
        "fallback_reason": subtitle_summary["fallback_reason"],
        "story_card": {
            "x": CARD_X,
            "y": card_y,
            "width": CARD_WIDTH,
            "height": card_height,
        },
    }
    debug_print("VIDEO RENDER RESPONSE", response_summary)

    return {
        "subtitle_script_path": str(subtitles_path.resolve()),
        "video_path": str(output_path.resolve()),
        "response": response_summary,
    }


def concatenate_page_videos(page_video_paths: list[Path], output_path: Path) -> dict:
    """Concatenate page clips into one final story video."""

    _ensure_ffmpeg_installed()

    concat_list_path = output_path.with_name("page_videos_concat.txt")
    concat_lines = [f"file '{video_path.name}'" for video_path in page_video_paths]
    concat_list_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

    request_payload = {
        "page_count": len(page_video_paths),
        "concat_list_path": str(concat_list_path.resolve()),
        "output_path": str(output_path.resolve()),
        "page_videos": [str(video_path.resolve()) for video_path in page_video_paths],
    }
    debug_print("FINAL VIDEO REQUEST", request_payload)

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_list_path.name,
        "-c",
        "copy",
        output_path.name,
    ]
    try:
        subprocess.run(
            command,
            cwd=output_path.parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed while concatenating the final video: {exc.stderr.strip()}"
        ) from exc

    response_summary = {
        "output_video": summarize_path(output_path),
        "concat_list": summarize_path(concat_list_path),
        "page_count": len(page_video_paths),
    }
    debug_print("FINAL VIDEO RESPONSE", response_summary)
    return response_summary


def write_highlighted_subtitles(
    story_text: str,
    words: list[dict],
    duration_seconds: float,
    output_path: Path,
) -> dict:
    """Write an ASS subtitle file with persistent storybook highlighting."""

    aligned_tokens, fallback_reason = _align_story_tokens(
        story_text,
        words,
        duration_seconds,
    )
    if aligned_tokens is None:
        subtitle_mode = "static_text_fallback"
        aligned_tokens = _build_static_tokens(story_text, duration_seconds)
        layout = _choose_subtitle_layout(aligned_tokens)
        events = _build_static_subtitle_events(aligned_tokens, duration_seconds)
    else:
        subtitle_mode = "word_highlight"
        layout = _choose_subtitle_layout(aligned_tokens)
        events = _build_subtitle_events(aligned_tokens, duration_seconds)

    script = _build_ass_script(
        aligned_tokens=aligned_tokens,
        layout=layout,
        events=events,
    )
    output_path.write_text(script, encoding="utf-8")

    response_summary = {
        "output_path": str(output_path.resolve()),
        "token_count": len(aligned_tokens),
        "line_count": layout.line_count,
        "event_count": len(events),
        "font_size": layout.font_size,
        "card_height": layout.card_height,
        "card_y": layout.card_y,
        "duration_seconds": round(duration_seconds, 3),
        "subtitle_mode": subtitle_mode,
        "fallback_reason": fallback_reason,
    }
    debug_print("SUBTITLE SCRIPT", response_summary)
    return response_summary


def _ensure_ffmpeg_installed() -> None:
    """Fail early with a Colab-friendly install hint when ffmpeg is missing."""

    if shutil.which("ffmpeg"):
        return
    raise RuntimeError(
        "ffmpeg is required for video rendering. Install it first, "
        "for example with `apt-get install -y ffmpeg` in Colab."
    )


def _align_story_tokens(
    story_text: str,
    words: list[dict],
    duration_seconds: float,
) -> tuple[list[AlignedToken] | None, str | None]:
    """Map sanitized story text onto Whisper word timings when exact proof exists."""

    display_tokens = _display_story_tokens(story_text)
    if not display_tokens:
        return None, "empty_story_text"

    normalized_story_tokens = [_normalize_for_timing(token) for token in display_tokens]
    normalized_story_tokens = [token for token in normalized_story_tokens if token]
    timed_words = []
    for word in words:
        normalized_word = _normalize_for_timing(str(word.get("word", "") or ""))
        if normalized_word:
            timed_words.append((normalized_word, word))

    normalized_words = [normalized_word for normalized_word, _ in timed_words]

    if len(normalized_story_tokens) != len(normalized_words):
        return None, "word_count_mismatch"

    previous_end = 0.0
    aligned_tokens: list[AlignedToken] = []

    for index, token in enumerate(display_tokens):
        normalized_token = _normalize_for_timing(token)
        if not normalized_token:
            return None, "token_normalization_failed"
        if normalized_token != normalized_words[index]:
            return None, "token_mismatch"

        matched_word = timed_words[index][1]

        start = _coerce_float(
            matched_word.get("start") if matched_word else previous_end,
            default=previous_end,
        )
        end = _coerce_float(
            matched_word.get("end") if matched_word else start,
            default=start,
        )
        if end < start:
            end = start
        previous_end = max(previous_end, end)

        aligned_tokens.append(AlignedToken(text=token, start=start, end=end))

    _fill_missing_timing_gaps(aligned_tokens, duration_seconds)
    return aligned_tokens, None


def _build_static_tokens(
    story_text: str,
    duration_seconds: float,
) -> list[AlignedToken]:
    """Create tokens for static subtitle fallback without word-level timing."""

    fallback_end = max(duration_seconds, MIN_WORD_DURATION)
    return [
        AlignedToken(text=token, start=0.0, end=fallback_end)
        for token in _display_story_tokens(story_text)
    ]


def _build_static_subtitle_events(
    tokens: list[AlignedToken],
    duration_seconds: float,
) -> list[SubtitleEvent]:
    """Render one full-page subtitle event when exact highlighting is not safe."""

    if not tokens:
        return []

    return [
        SubtitleEvent(
            start=0.0,
            end=max(duration_seconds, MIN_WORD_DURATION),
            current_index=None,
            completed_index=len(tokens) - 1,
        )
    ]


def _fill_missing_timing_gaps(
    aligned_tokens: list[AlignedToken],
    duration_seconds: float,
) -> None:
    """Give zero-length or missing spans a tiny minimum duration."""

    for index, token in enumerate(aligned_tokens):
        next_start = (
            aligned_tokens[index + 1].start
            if index + 1 < len(aligned_tokens)
            else duration_seconds
        )
        fallback_end = token.start + MIN_WORD_DURATION
        max_allowed_end = next_start if next_start > token.start else duration_seconds
        token.end = min(max(token.end, fallback_end), max_allowed_end)
        if token.end < token.start:
            token.end = token.start


def _choose_subtitle_layout(tokens: list[AlignedToken]) -> SubtitleLayout:
    """Choose a conservative wrapping and font size that stays inside the card."""

    if not tokens:
        card_height = CARD_MIN_HEIGHT
        card_y = VIDEO_HEIGHT - card_height - CARD_BOTTOM_MARGIN
        return SubtitleLayout(
            wrapped_lines=[],
            font_size=DEFAULT_FONT_SIZE,
            line_height=round(DEFAULT_FONT_SIZE * LINE_HEIGHT_RATIO),
            line_count=0,
            content_height=0,
            card_height=card_height,
            card_y=card_y,
            text_center_x=CARD_X + (CARD_WIDTH // 2),
            text_center_y=card_y + (card_height // 2),
        )

    candidate_sizes = []
    for font_size in range(DEFAULT_FONT_SIZE, MIN_FONT_SIZE - 1, -2):
        if font_size not in candidate_sizes:
            candidate_sizes.append(font_size)
    if MIN_FONT_SIZE not in candidate_sizes:
        candidate_sizes.append(MIN_FONT_SIZE)

    best_layout: SubtitleLayout | None = None
    for font_size in candidate_sizes:
        wrapped_lines = _wrap_tokens_for_font(tokens, font_size)
        line_count = len(wrapped_lines)
        line_height = round(font_size * LINE_HEIGHT_RATIO)
        content_height = line_count * line_height
        card_height = max(
            CARD_MIN_HEIGHT,
            min(CARD_MAX_HEIGHT, content_height + (CARD_PADDING_Y * 2)),
        )
        card_y = VIDEO_HEIGHT - card_height - CARD_BOTTOM_MARGIN
        layout = SubtitleLayout(
            wrapped_lines=wrapped_lines,
            font_size=font_size,
            line_height=line_height,
            line_count=line_count,
            content_height=content_height,
            card_height=card_height,
            card_y=card_y,
            text_center_x=CARD_X + (CARD_WIDTH // 2),
            text_center_y=card_y + (card_height // 2),
        )
        best_layout = layout

        fits_height = content_height + (CARD_PADDING_Y * 2) <= CARD_MAX_HEIGHT
        if fits_height and line_count <= MAX_SUBTITLE_LINES:
            return layout

    return best_layout


def _wrap_tokens_for_font(tokens: list[AlignedToken], font_size: int) -> list[list[int]]:
    """Greedily wrap tokens using a conservative estimated text width."""

    lines: list[list[int]] = []
    current_line: list[int] = []
    current_width_units = 0.0
    max_width_units = (CARD_TEXT_WIDTH / font_size) * 0.88
    space_width_units = _estimate_text_width_units(" ")

    for index, token in enumerate(tokens):
        token_width_units = _estimate_text_width_units(token.text)
        projected_width = token_width_units
        if current_line:
            projected_width = current_width_units + space_width_units + token_width_units

        if current_line and projected_width > max_width_units:
            lines.append(current_line)
            current_line = [index]
            current_width_units = token_width_units
        else:
            current_line.append(index)
            current_width_units = projected_width

    if current_line:
        lines.append(current_line)

    return lines


def _build_subtitle_events(
    tokens: list[AlignedToken],
    duration_seconds: float,
) -> list[SubtitleEvent]:
    """Create a timeline that highlights the current word and keeps past words lit."""

    if not tokens:
        return []

    events: list[SubtitleEvent] = []
    first_start = max(tokens[0].start, 0.0)
    if first_start > 0:
        events.append(
            SubtitleEvent(
                start=0.0,
                end=first_start,
                current_index=None,
                completed_index=-1,
            )
        )

    for index, token in enumerate(tokens):
        next_start = tokens[index + 1].start if index + 1 < len(tokens) else duration_seconds
        highlight_end = max(token.end, token.start + MIN_WORD_DURATION)
        if next_start > token.start:
            highlight_end = min(highlight_end, next_start)
        highlight_end = min(highlight_end, duration_seconds)
        if highlight_end > token.start:
            events.append(
                SubtitleEvent(
                    start=token.start,
                    end=highlight_end,
                    current_index=index,
                    completed_index=index - 1,
                )
            )

        if next_start > highlight_end:
            events.append(
                SubtitleEvent(
                    start=highlight_end,
                    end=next_start,
                    current_index=None,
                    completed_index=index,
                )
            )

    final_end = max(duration_seconds, tokens[-1].end)
    if final_end > events[-1].end:
        events.append(
            SubtitleEvent(
                start=events[-1].end,
                end=final_end,
                current_index=None,
                completed_index=len(tokens) - 1,
            )
        )

    return [event for event in events if event.end - event.start > 0.01]


def _build_ass_script(
    aligned_tokens: list[AlignedToken],
    layout: SubtitleLayout,
    events: list[SubtitleEvent],
) -> str:
    """Build the ASS subtitle text that ffmpeg will burn into the video."""

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {VIDEO_WIDTH}
PlayResY: {VIDEO_HEIGHT}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Story,{FONT_NAME},{layout.font_size},&H00FFFFFF,&H00FFFFFF,&H50000000,&H00000000,-1,0,0,0,100,100,0,0,1,1,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    event_lines = []
    for event in events:
        text = _render_event_text(
            aligned_tokens=aligned_tokens,
            wrapped_lines=layout.wrapped_lines,
            event=event,
        )
        event_lines.append(
            "Dialogue: 0,"
            f"{_format_ass_timestamp(event.start)},"
            f"{_format_ass_timestamp(event.end)},"
            "Story,,0,0,0,,"
            f"{{\\an5\\q2\\pos({layout.text_center_x},{layout.text_center_y})}}{text}"
        )

    return header + "\n".join(event_lines) + "\n"


def _render_event_text(
    aligned_tokens: list[AlignedToken],
    wrapped_lines: list[list[int]],
    event: SubtitleEvent,
) -> str:
    """Render the full multiline story text for one subtitle interval."""

    rendered_lines: list[str] = []
    for line in wrapped_lines:
        rendered_tokens = []
        for token_index in line:
            token = aligned_tokens[token_index]
            rendered_tokens.append(
                _decorate_ass_text(
                    token_text=token.text,
                    token_index=token_index,
                    current_index=event.current_index,
                    completed_index=event.completed_index,
                )
            )
        rendered_lines.append(" ".join(rendered_tokens))
    return r"\N".join(rendered_lines)


def _decorate_ass_text(
    token_text: str,
    token_index: int,
    current_index: int | None,
    completed_index: int,
) -> str:
    """Apply per-token color and weight for the current subtitle state."""

    if token_index == current_index:
        color = ASS_CURRENT_COLOR
        bold = 1
    elif token_index <= completed_index:
        color = ASS_PAST_COLOR
        bold = 0
    else:
        color = ASS_FUTURE_COLOR
        bold = 0

    escaped_text = _escape_ass_text(token_text)
    return f"{{\\1c{color}\\b{bold}}}{escaped_text}"


def _escape_ass_text(text: str) -> str:
    """Escape ASS control characters inside visible subtitle text."""

    escaped = text.replace("\\", r"\\")
    escaped = escaped.replace("{", "(").replace("}", ")")
    return escaped


def _estimate_text_width_units(text: str) -> float:
    """Estimate relative text width conservatively for proportional storybook text."""

    width = 0.0
    for character in text:
        if character == " ":
            width += 0.35
        elif character in "ilI'`.,:;!?":
            width += 0.34
        elif character in "mwMW@#%&":
            width += 0.95
        elif character.isupper():
            width += 0.74
        elif character.isdigit():
            width += 0.62
        else:
            width += 0.60
    return width


def _format_ass_timestamp(value: float) -> str:
    """Format a subtitle timestamp as H:MM:SS.cc."""

    safe_value = max(value, 0.0)
    hours = int(safe_value // 3600)
    minutes = int((safe_value % 3600) // 60)
    seconds = int(safe_value % 60)
    centiseconds = int(round((safe_value - int(safe_value)) * 100))
    if centiseconds == 100:
        seconds += 1
        centiseconds = 0
    if seconds == 60:
        minutes += 1
        seconds = 0
    if minutes == 60:
        hours += 1
        minutes = 0
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _display_story_tokens(story_text: str) -> list[str]:
    """Split sanitized subtitle-safe story text into visible display tokens."""

    return [token for token in sanitize_narration_text(story_text).split() if token]


def _normalize_for_timing(text: str) -> str:
    """Normalize token text so story words match transcription words exactly."""

    return normalize_alignment_token(text)

    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = normalized.translate(
        str.maketrans(
            {
                "’": "'",
                "‘": "'",
                "‗": "'",
                "“": '"',
                "”": '"',
                "—": "-",
                "–": "-",
            }
        )
    )
    normalized = re.sub(r"^[^\w']+|[^\w']+$", "", normalized)
    normalized = re.sub(r"[^a-z0-9']+", "", normalized)
    return normalized


def _coerce_float(value, default: float) -> float:
    """Safely coerce SDK values into floats."""

    try:
        if value is None:
            raise TypeError
        return float(value)
    except (TypeError, ValueError):
        return default

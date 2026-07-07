import html
import json
import logging
import os
import re
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

AI_BASE_URL = os.getenv("AI_BASE_URL", "https://api.cerebras.ai/v1")
AI_MODEL = os.getenv("AI_MODEL", "gpt-oss-120b")
MAX_TRANSCRIPT_CHARS = int(os.getenv("MAX_TRANSCRIPT_CHARS", "28000"))
TRANSCRIPT_SERVICE_URL = os.getenv(
    "TRANSCRIPT_SERVICE_URL",
    "https://transcript-service-zu4i.onrender.com"
).rstrip("/")

TRANSCRIPT_SERVICE_TIMEOUT = float(
    os.getenv("TRANSCRIPT_SERVICE_TIMEOUT", "30")
)
TRANSCRIPT_SERVICE_TIMEOUT = float(os.getenv("TRANSCRIPT_SERVICE_TIMEOUT", "30"))

logger = logging.getLogger(__name__)


USE_CASE_GUIDANCE = {
    "study_notes": "Create chapter-wise notes, exam points, definitions, important questions, and a quick revision sheet for a student.",
    "coding_tutorial": "Extract programming concepts, workflow, code ideas, debugging notes, practice tasks, and interview questions.",
    "podcast_summary": "Capture discussion themes, speaker opinions, memorable short quotes, decisions, and listener action items.",
    "business_insights": "Build SWOT, business model notes, market opportunities, risks, growth strategy, and decision points.",
    "startup_ideas": "Find startup concepts, customer pains, MVP ideas, revenue models, unfair advantages, and an execution plan.",
    "research_notes": "Summarize abstract, method, findings, evidence quality, limitations, citations mentioned, and research gaps.",
    "marketing_plan": "Extract audience, positioning, channels, campaign ideas, copy angles, metrics, and next actions.",
    "finance_brief": "Extract financial drivers, assumptions, risks, opportunities, metrics, and plain-language recommendations.",
    "career_coach": "Create skill gaps, learning path, portfolio tasks, interview prep, and weekly action steps.",
    "content_creator": "Extract hooks, content angles, reusable clips, title ideas, thumbnails, and audience takeaways.",
}


def _mode_guidance(mode: str = "normal", custom_prompt: Optional[str] = None) -> str:
    return {
        "normal": "Write a clear, balanced summary for a general audience.",
        "advanced": "Include technical detail, cause/effect, and important nuance.",
        "pro": "Write executive-grade notes with decisions, risks, and action items.",
        "custom": custom_prompt or "Follow the user's custom intent.",
        **USE_CASE_GUIDANCE,
    }.get(mode, "Write a clear, balanced summary for a general audience.")


class TranscriptUnavailableError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


def _configured_keys() -> dict[str, Optional[str]]:
    return {
        "standard": os.getenv("CEREBRAS_API_KEY"),
        "fast": os.getenv("CEREBRAS_API_KEY_FAST") or os.getenv("CEREBRAS_API_KEY"),
        "third": os.getenv("CEREBRAS_API_KEY_THIRD") or os.getenv("CEREBRAS_API_KEY_FAST") or os.getenv("CEREBRAS_API_KEY"),
        "comparison": os.getenv("CEREBRAS_API_KEY_COMPARISON") or os.getenv("CEREBRAS_API_KEY_THIRD") or os.getenv("CEREBRAS_API_KEY"),
        "usecase": os.getenv("CEREBRAS_API_KEY_USE_CASE") or os.getenv("CEREBRAS_API_KEY_COMPARISON") or os.getenv("CEREBRAS_API_KEY"),
        "presentation": os.getenv("CEREBRAS_API_KEY_PRESENTATION") or os.getenv("CEREBRAS_API_KEY_USE_CASE") or os.getenv("CEREBRAS_API_KEY"),
    }


def _client(slot: str = "standard") -> OpenAI:
    key = _configured_keys().get(slot)
    if not key:
        raise RuntimeError(
            f"No AI API key configured for {slot}. Add CEREBRAS_API_KEY, CEREBRAS_API_KEY_COMPARISON, CEREBRAS_API_KEY_USE_CASE, or CEREBRAS_API_KEY_PRESENTATION to backend/.env."
        )
    return OpenAI(api_key=key, base_url=AI_BASE_URL)


def _truncate_transcript(transcript: str) -> str:
    if len(transcript) <= MAX_TRANSCRIPT_CHARS:
        return transcript
    return transcript[:MAX_TRANSCRIPT_CHARS] + "\n\n[Transcript truncated for model context.]"


def _json_from_text(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _chat_json(system_prompt: str, user_prompt: str, slot: str = "standard") -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        response = _client(slot).chat.completions.create(
            model=AI_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
        )
    except Exception:
        response = _client(slot).chat.completions.create(
            model=AI_MODEL,
            messages=[
                *messages,
                {"role": "user", "content": "Return the answer as valid JSON only."},
            ],
            temperature=0.2,
        )
    return _json_from_text(response.choices[0].message.content or "{}")


def _chat_text(prompt: str, slot: str = "fast") -> str:
    response = _client(slot).chat.completions.create(
        model=AI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def _seconds_to_time(seconds: float) -> str:
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _clean_caption_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def build_caption_windows(
    segments: list[dict[str, Any]], duration: Optional[int] = None, window_seconds: int = 45
) -> list[dict[str, Any]]:
    if not segments:
        return []

    last_second = duration or int(max(segment.get("end_seconds", 0) for segment in segments)) + 1
    windows = []
    for start in range(0, max(last_second, window_seconds), window_seconds):
        end = start + window_seconds
        window_segments = [
            segment
            for segment in segments
            if start <= float(segment.get("start_seconds", 0)) < end
        ]
        text = " ".join(segment["text"] for segment in window_segments).strip()
        if not text:
            continue
        windows.append(
            {
                "start_seconds": start,
                "end_seconds": min(end, last_second),
                "start_time": _seconds_to_time(start),
                "end_time": _seconds_to_time(min(end, last_second)),
                "captions": window_segments,
                "text": text,
                "summary": "",
            }
        )
    return windows


YOUTUBE_VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _extract_youtube_video_id(youtube_url: str) -> str:
    value = (youtube_url or "").strip()
    if YOUTUBE_VIDEO_ID_PATTERN.fullmatch(value):
        return value

    parsed = urlparse(value)
    host = parsed.netloc.lower()
    path_parts = [part for part in parsed.path.split("/") if part]

    query_video_id = parse_qs(parsed.query).get("v", [""])[0]
    if YOUTUBE_VIDEO_ID_PATTERN.fullmatch(query_video_id):
        return query_video_id

    if host.endswith("youtu.be") and path_parts and YOUTUBE_VIDEO_ID_PATTERN.fullmatch(path_parts[0]):
        return path_parts[0]

    if host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        for marker in ("shorts", "embed", "live", "v"):
            if marker in path_parts:
                index = path_parts.index(marker)
                if len(path_parts) > index + 1 and YOUTUBE_VIDEO_ID_PATTERN.fullmatch(path_parts[index + 1]):
                    return path_parts[index + 1]

    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/|/v/)([A-Za-z0-9_-]{11})", value)
    if match:
        return match.group(1)

    raise TranscriptUnavailableError(
        "Invalid YouTube URL. Please provide a valid YouTube video link.",
        status_code=400,
    )


def _video_metadata(youtube_url: str, video_id: str) -> dict[str, Any]:
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"
    metadata = {
        "title": "YouTube Video",
        "channel": "",
        "duration": None,
        "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
    }

    try:
        response = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": youtube_url if urlparse(youtube_url).netloc else canonical_url, "format": "json"},
            timeout=10,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.RequestException:
        return metadata
    except ValueError:
        return metadata

    return {
        "title": payload.get("title") or metadata["title"],
        "channel": payload.get("author_name") or metadata["channel"],
        "duration": None,
        "thumbnail": payload.get("thumbnail_url") or metadata["thumbnail"],
    }


def _normalize_caption_segments(raw_segments: Any, transcript: str) -> list[dict[str, Any]]:
    segments = []
    if isinstance(raw_segments, list):
        for item in raw_segments:
            if not isinstance(item, dict):
                continue
            text = _clean_caption_text(str(item.get("text") or ""))
            if not text:
                continue
            start = float(item.get("start_seconds") or item.get("start") or 0)
            end = float(item.get("end_seconds") or item.get("end") or start)
            if end < start:
                end = start
            segments.append(
                {
                    "time": item.get("time") or _seconds_to_time(start),
                    "end_time": item.get("end_time") or _seconds_to_time(end),
                    "start_seconds": start,
                    "end_seconds": end,
                    "text": text,
                }
            )

    if not segments and transcript.strip():
        text = _clean_caption_text(transcript)
        segments.append(
            {
                "time": "00:00",
                "end_time": "00:00",
                "start_seconds": 0,
                "end_seconds": 0,
                "text": text,
            }
        )

    return segments


def _is_caption_unavailable_error(message: str) -> bool:
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in (
            "caption",
            "subtitle",
            "transcript",
            "no usable",
            "not available",
            "unavailable",
            "disabled",
            "private",
            "deleted",
        )
    )


def _request_transcript_from_service(youtube_url: str, video_id: str, language: Optional[str]) -> dict[str, Any]:
    service_url = f"{TRANSCRIPT_SERVICE_URL}/transcript"
    payload = {"youtube_url": youtube_url, "video_id": video_id}
    if language:
        payload["language"] = language

    try:
        response = requests.post(service_url, json=payload, timeout=TRANSCRIPT_SERVICE_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        logger.exception("Transcript service request failed: %s", exc.__class__.__name__)
        raise TranscriptUnavailableError(
            "Transcript service is temporarily unavailable. Please try again shortly.",
            status_code=503,
        ) from exc

    try:
        data = response.json()
    except ValueError as exc:
        logger.error("Transcript service returned a non-JSON response with status %s", response.status_code)
        raise TranscriptUnavailableError(
            "Transcript service returned an invalid response. Please try again shortly.",
            status_code=502,
        ) from exc

    if response.status_code >= 500:
        logger.error("Transcript service error %s: %s", response.status_code, data.get("error"))
        raise TranscriptUnavailableError(
            "Transcript service is temporarily unavailable. Please try again shortly.",
            status_code=502,
        )

    if not data.get("success"):
        message = str(data.get("error") or "No captions were found for this video.")
        status_code = 422 if response.status_code == 422 or _is_caption_unavailable_error(message) else 502
        raise TranscriptUnavailableError(message, status_code=status_code)

    return data


def get_video_transcript(youtube_url: str, language: Optional[str] = None) -> dict[str, Any]:
    video_id = _extract_youtube_video_id(youtube_url)
    metadata = _video_metadata(youtube_url, video_id)
    data = _request_transcript_from_service(youtube_url, video_id, language)

    transcript = _clean_caption_text(str(data.get("transcript") or ""))
    segments = _normalize_caption_segments(data.get("segments"), transcript)
    if not segments:
        raise TranscriptUnavailableError(
            "No usable captions were found for this video. Try a YouTube video with captions or automatic captions enabled.",
            status_code=422,
        )

    if not transcript:
        transcript = " ".join(segment["text"] for segment in segments)

    duration = int(max(segment.get("end_seconds", 0) for segment in segments)) if segments else None
    logger.info(
        "Transcript service returned %s grouped caption segments in %s for video %s",
        len(segments),
        data.get("language") or language or "default language",
        video_id,
    )
    return {
        "title": data.get("title") or metadata["title"],
        "channel": data.get("channel") or metadata["channel"],
        "duration": data.get("duration") or duration,
        "thumbnail": data.get("thumbnail") or metadata["thumbnail"],
        "transcript": transcript,
        "caption_segments": segments,
        "caption_windows": build_caption_windows(segments, duration),
    }

async def generate_synopsis(
    transcript: str, mode: str = "normal", custom_prompt: Optional[str] = None,
    output_language: str = "English",
) -> dict[str, Any]:
    transcript = _truncate_transcript(transcript)
    mode_guidance = _mode_guidance(mode, custom_prompt)
    generation_slot = "usecase" if mode in USE_CASE_GUIDANCE else "standard"

    system_prompt = """
You generate structured YouTube study, product, business, coding, podcast, startup, and research notes.
Return only valid JSON with these keys:
summary: markdown string with useful section headings that exactly follow the selected mode guidance
keywords: array of 6 to 12 short strings
chapters: array of objects with title and description
"""

    return _chat_json(
        system_prompt,
        f"Write all generated content in {output_language}.\nMode guidance: {mode_guidance}\n\nTranscript:\n{transcript}",
        slot=generation_slot,
    )


async def generate_feature(
    transcript: str,
    feature_type: str,
    output_language: str = "English",
    mode: str = "normal",
    custom_prompt: Optional[str] = None,
) -> Any:
    transcript = _truncate_transcript(transcript)
    guidance = _mode_guidance(mode, custom_prompt)
    slot = "usecase" if mode in USE_CASE_GUIDANCE else "fast"

    if feature_type == "key_points":
        data = _chat_json(
            "Return only valid JSON with key_points as an array of concise strings.",
            f"Write in {output_language}. Follow this purpose: {guidance}\nExtract the 8 most useful key points from this transcript:\n{transcript}",
            slot=slot,
        )
        return data.get("key_points", [])

    if feature_type == "questions":
        data = _chat_json(
            "Return only valid JSON with questions as an array of objects containing type, question, answer, options, and correct_answer. Mix multiple_choice, true_false, and short_answer. Multiple choice questions must have exactly 4 options. Answers must be supported by the transcript and useful for the selected purpose.",
            f"Write in {output_language}. Follow this purpose: {guidance}\nGenerate 8 varied questions from this transcript:\n{transcript}",
            slot="third",
        )
        return data.get("questions", [])

    if feature_type == "action_items":
        data = _chat_json(
            "Return only valid JSON with action_items as an array of concise strings.",
            f"Write in {output_language}. Follow this purpose: {guidance}\nExtract practical action items or next steps from this transcript:\n{transcript}",
            slot="third",
        )
        return data.get("action_items", [])

    if feature_type == "transcript":
        return _chat_text(
            "Clean this transcript for readability. Keep meaning unchanged and do not invent timestamps:\n"
            + transcript,
            slot="fast",
        )

    raise ValueError(f"Unknown feature type: {feature_type}")


def _fallback_window_summary(text: str) -> str:
    words = text.split()
    if len(words) <= 26:
        return text
    return " ".join(words[:26]) + "..."


async def summarize_caption_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not windows:
        return []

    compact_windows = [
        {
            "index": index,
            "time_range": f"{window['start_time']} - {window['end_time']}",
            "captions": window["text"],
        }
        for index, window in enumerate(windows)
    ]

    try:
        data = _chat_json(
            "Return only valid JSON with summaries as an array of objects with index and summary. Each summary must be one short sentence.",
            "Summarize each 30-second caption window separately. Do not merge windows.\n\n"
            + json.dumps({"windows": compact_windows}, ensure_ascii=False),
            slot="fast",
        )
        summaries = {
            int(item.get("index")): item.get("summary", "").strip()
            for item in data.get("summaries", [])
            if item.get("summary")
        }
    except Exception:
        summaries = {}

    return [
        {
            **window,
            "summary": summaries.get(index) or _fallback_window_summary(window["text"]),
        }
        for index, window in enumerate(windows)
    ]


async def answer_summary_question(
    question: str,
    summary: str = "",
    transcript: str = "",
    caption_summaries: Optional[list[dict[str, Any]]] = None,
    selected_window: Optional[dict[str, Any]] = None,
) -> str:
    caption_context = caption_summaries or []
    compact_caption_context = [
        {
            "time_range": f"{item.get('start_time')} - {item.get('end_time')}",
            "summary": item.get("summary", ""),
            "captions": item.get("text", ""),
        }
        for item in caption_context[:80]
    ]

    focused = ""
    if selected_window:
        focused = (
            f"\nSelected window: {selected_window.get('start_time')} - {selected_window.get('end_time')}\n"
            f"Selected captions: {selected_window.get('text', '')}\n"
            f"Selected summary: {selected_window.get('summary', '')}\n"
        )

    return _chat_text(
        "You are the Ask AI assistant for a YouTube video summary page. "
        "Answer only from the provided summary, transcript, and caption windows. "
        "If the answer is not supported by the provided data, say that clearly.\n\n"
        f"User question: {question}\n"
        f"{focused}\n"
        f"Overall summary:\n{summary}\n\n"
        f"30-second caption summaries:\n{json.dumps(compact_caption_context, ensure_ascii=False)}\n\n"
        f"Transcript:\n{_truncate_transcript(transcript)}",
        slot="third",
    )


async def generate_all_features(
    transcript: str, mode: str = "normal", custom_prompt: Optional[str] = None,
    output_language: str = "English",
) -> dict[str, Any]:
    synopsis = await generate_synopsis(transcript, mode, custom_prompt, output_language)
    key_points = await generate_feature(transcript, "key_points", output_language, mode, custom_prompt)
    questions = await generate_feature(transcript, "questions", output_language, mode, custom_prompt)
    action_items = await generate_feature(transcript, "action_items", output_language, mode, custom_prompt)

    return {
        "summary": synopsis.get("summary", ""),
        "keywords": synopsis.get("keywords", []),
        "chapters": synopsis.get("chapters", []),
        "key_points": key_points,
        "questions": questions,
        "action_items": action_items,
    }


async def translate_summary_content(payload: dict[str, Any], output_language: str) -> dict[str, Any]:
    return _chat_json(
        "Translate the supplied summary content. Return valid JSON with summary, transcript, key_points, questions, action_items, chapters, and keywords. Preserve structure and meaning.",
        f"Translate all text to {output_language}. Keep URLs and timestamps unchanged.\n\n{json.dumps(payload, ensure_ascii=False)}",
        slot="standard",
    )

async def generate_video_comparison(
    video_1: dict[str, Any],
    video_2: dict[str, Any],
    comparison_goal: Optional[str] = None,
    output_language: str = "English",
) -> dict[str, Any]:
    payload = {
        "goal": comparison_goal or "Compare the two videos for practical learning value.",
        "video_1": {
            "title": video_1.get("title"),
            "channel": video_1.get("channel"),
            "duration": video_1.get("duration"),
            "transcript": _truncate_transcript(video_1.get("transcript", "")),
        },
        "video_2": {
            "title": video_2.get("title"),
            "channel": video_2.get("channel"),
            "duration": video_2.get("duration"),
            "transcript": _truncate_transcript(video_2.get("transcript", "")),
        },
    }

    return _chat_json(
        """
You compare two YouTube videos for a product-quality comparison page.
Return only valid JSON with these keys:
combined_summary: concise paragraph covering both videos
common_points: array of 4 to 10 short common themes
differences: array of objects with topic, video1, and video2
best_takeaways: object with video1_best, video2_best, combined_recommendation, gold, silver, bronze
verdict: object with students, entrepreneurs, developers, content_creators, professionals; each value has winner and reasoning
best_overall_video: object with winner and reasoning
""",
        f"Write all generated content in {output_language}. Compare these videos against the goal.\n\n{json.dumps(payload, ensure_ascii=False)}",
        slot="comparison",
    )


async def improve_slide_content(slide: dict[str, Any], context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    data = _chat_json(
        "Return only valid JSON with slide as an improved slide object. Keep the same schema. Improve readability, reduce text clutter, and improve layout.",
        json.dumps({"slide": slide, "context": context or {}}, ensure_ascii=False),
        slot="presentation",
    )
    return data.get("slide", slide)




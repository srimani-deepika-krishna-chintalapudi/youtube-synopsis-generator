from typing import Any, Optional

import datetime
import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session
from fastapi.responses import RedirectResponse

from database import SessionLocal, get_db
from engine import (
    answer_summary_question,
    generate_all_features,
    generate_feature,
    generate_video_comparison,
    get_video_transcript,
    improve_slide_content,
    summarize_caption_windows,
    TranscriptUnavailableError,
    translate_summary_content,
)
from models import Comparison, Presentation, Slide, Summary, User
from security import (
    create_access_token,
    get_current_user,
    get_password_hash,
    require_admin,
    verify_password,
)

app = FastAPI(
    title="AI Video Synopsis Generator API",
    docs_url="/docs",
    redoc_url="/redoc"
)

default_allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://yousynopsis.vercel.app",
]

env_allowed_origins = [
    origin.strip().rstrip("/")
    for origin in os.environ.get("CORS_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[*default_allowed_origins, *env_allowed_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def ensure_admin_user() -> None:
    load_dotenv(override=True)
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@synopsis.local").strip().lower()
    admin_password = os.environ.get("ADMIN_PASSWORD", "Admin@12345")
    admin_name = os.environ.get("ADMIN_NAME", "Synopsis Admin")

    db = SessionLocal()
    try:
        admin = db.query(User).filter(func.lower(User.email) == admin_email).first()
        if admin:
            changed = False
            if admin.role != "Admin":
                admin.role = "Admin"
                changed = True
            if admin.email != admin_email:
                admin.email = admin_email
                changed = True
            if not admin.name:
                admin.name = admin_name
                changed = True
            try:
                password_matches = verify_password(admin_password, admin.password_hash)
            except Exception:
                password_matches = False
            if not password_matches:
                admin.password_hash = get_password_hash(admin_password)
                changed = True
            if changed:
                db.commit()
            return

        db.add(
            User(
                email=admin_email,
                password_hash=get_password_hash(admin_password),
                name=admin_name,
                role="Admin",
                location="Admin console",
                bio="Backend administrator account.",
            )
        )
        db.commit()
    finally:
        db.close()


@app.on_event("startup")
def startup_tasks():
    ensure_admin_user()


class UserCreate(BaseModel):
    email: str
    password: str
    name: str
    role: Optional[str] = "User"
    location: Optional[str] = ""
    bio: Optional[str] = ""


class VideoRequest(BaseModel):
    youtube_url: str
    mode: str = "normal"
    custom_prompt: Optional[str] = None
    transcript: Optional[str] = None
    output_language: Optional[str] = "English"


class CompareVideosRequest(BaseModel):
    youtube_url_1: str
    youtube_url_2: str
    comparison_goal: Optional[str] = None
    output_language: Optional[str] = "English"


class PresentationSaveRequest(BaseModel):
    title: str
    slides: list[dict[str, Any]]
    source_type: Optional[str] = None
    source_id: Optional[int] = None


class SlideImproveRequest(BaseModel):
    slide: dict[str, Any]
    context: Optional[dict[str, Any]] = None


class UserProfileUpdate(BaseModel):
    name: str
    role: Optional[str] = "User"
    location: Optional[str] = ""
    bio: Optional[str] = ""


class SummaryChatRequest(BaseModel):
    question: str
    summary: Optional[str] = ""
    transcript: Optional[str] = ""
    caption_summaries: Optional[list[dict[str, Any]]] = None
    selected_window: Optional[dict[str, Any]] = None


class SummaryTranslateRequest(BaseModel):
    language: str
    data: dict[str, Any]


def serialize_user(user: User) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "role": user.role,
        "location": user.location,
        "bio": user.bio,
    }


def serialize_admin_user(user: User) -> dict[str, Any]:
    summaries = sorted(user.summaries or [], key=lambda item: item.created_at or datetime.datetime.min, reverse=True)
    comparisons = sorted(user.comparisons or [], key=lambda item: item.created_at or datetime.datetime.min, reverse=True)
    presentations = sorted(user.presentations or [], key=lambda item: item.created_at or datetime.datetime.min, reverse=True)
    activity_dates = [
        item.created_at
        for item in [*summaries, *comparisons, *presentations]
        if item.created_at is not None
    ]

    data = serialize_user(user)
    summary_work = [
        {
            "id": summary.id,
            "type": "summary",
            "title": summary.title or "YouTube Video",
            "url": summary.youtube_url,
            "language": summary.language or "English",
            "created_at": summary.created_at.isoformat() if summary.created_at else None,
            "keywords": summary.keywords or [],
            "key_points": summary.key_points or [],
        }
        for summary in summaries
    ]
    comparison_work = [
        {
            "id": comparison.id,
            "type": "comparison",
            "title": comparison.goal or "Video comparison",
            "url_1": comparison.youtube_url_1,
            "url_2": comparison.youtube_url_2,
            "language": comparison.language or "English",
            "created_at": comparison.created_at.isoformat() if comparison.created_at else None,
            "best_overall_video": comparison.best_overall_video or {},
        }
        for comparison in comparisons
    ]
    presentation_work = [
        {
            "id": presentation.id,
            "type": "presentation",
            "title": presentation.title,
            "source_type": presentation.source_type,
            "source_id": presentation.source_id,
            "slide_count": len(presentation.slides_json or []),
            "slides": presentation.slides_json or [],
            "created_at": presentation.created_at.isoformat() if presentation.created_at else None,
            "updated_at": presentation.updated_at.isoformat() if presentation.updated_at else None,
        }
        for presentation in presentations
    ]
    recent_activity = sorted(
        [*summary_work, *comparison_work, *presentation_work],
        key=lambda item: item.get("created_at") or "",
        reverse=True,
    )
    data["usage"] = {
        "summaries": len(summaries),
        "comparisons": len(comparisons),
        "presentations": len(presentations),
        "total_requests": len(summaries) + len(comparisons) + len(presentations),
        "last_activity": max(activity_dates).isoformat() if activity_dates else None,
        "web_usage": {
            "summary_urls": [
                {
                    "id": summary.id,
                    "url": summary.youtube_url,
                    "title": summary.title or "YouTube Video",
                    "language": summary.language or "English",
                    "created_at": summary.created_at.isoformat() if summary.created_at else None,
                }
                for summary in summaries[:5]
            ],
            "comparisons": [
                {
                    "id": comparison.id,
                    "url_1": comparison.youtube_url_1,
                    "url_2": comparison.youtube_url_2,
                    "goal": comparison.goal or "",
                    "language": comparison.language or "English",
                    "created_at": comparison.created_at.isoformat() if comparison.created_at else None,
                }
                for comparison in comparisons[:5]
            ],
        },
        "work": {
            "summaries": summary_work,
            "comparisons": comparison_work,
            "presentations": presentation_work,
            "recent_activity": recent_activity,
        },
    }
    return data


def serialize_summary(summary: Summary) -> dict[str, Any]:
    return {
        "id": summary.id,
        "youtube_url": summary.youtube_url,
        "title": summary.title or "YouTube Video",
        "channel": summary.channel or "",
        "duration": summary.duration,
        "thumbnail": summary.thumbnail or "",
        "transcript": summary.transcript or "",
        "caption_segments": summary.caption_segments or [],
        "caption_summaries": summary.caption_summaries or [],
        "summary": summary.summary_text or "",
        "keywords": summary.keywords or [],
        "chapters": summary.chapters or [],
        "key_points": summary.key_points or [],
        "questions": summary.questions or [],
        "action_items": summary.action_items or [],
        "language": summary.language or "English",
        "created_at": summary.created_at.isoformat() if summary.created_at else None,
    }


def _video_info(video: dict[str, Any], youtube_url: str) -> dict[str, Any]:
    return {
        "youtube_url": youtube_url,
        "title": video.get("title", "YouTube Video"),
        "channel": video.get("channel", ""),
        "duration": video.get("duration"),
        "thumbnail": video.get("thumbnail", ""),
    }


def serialize_comparison(comparison: Comparison) -> dict[str, Any]:
    return {
        "id": comparison.id,
        "goal": comparison.goal or "",
        "language": comparison.language or "English",
        "video_1": comparison.video_1 or {},
        "video_2": comparison.video_2 or {},
        "combined_summary": comparison.combined_summary or "",
        "common_points": comparison.common_points or [],
        "differences": comparison.differences or [],
        "best_takeaways": comparison.best_takeaways or {},
        "verdict": comparison.verdict or {},
        "best_overall_video": comparison.best_overall_video or {},
        "created_at": comparison.created_at.isoformat() if comparison.created_at else None,
    }


def serialize_presentation(presentation: Presentation) -> dict[str, Any]:
    return {
        "id": presentation.id,
        "title": presentation.title,
        "source_type": presentation.source_type,
        "source_id": presentation.source_id,
        "slides": presentation.slides_json or [],
        "created_at": presentation.created_at.isoformat() if presentation.created_at else None,
        "updated_at": presentation.updated_at.isoformat() if presentation.updated_at else None,
    }


def transcript_error_response(exc: TranscriptUnavailableError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))

@app.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")
@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/auth/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = User(
        email=user.email,
        password_hash=get_password_hash(user.password),
        name=user.name,
        role=user.role or "User",
        location=user.location or "",
        bio=user.bio or "",
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully", "user": serialize_user(new_user)}


@app.post("/api/auth/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    ensure_admin_user()
    email = form_data.username.strip().lower()
    user = db.query(User).filter(func.lower(User.email) == email).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    access_token = create_access_token(data={"sub": user.email, "role": user.role})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": serialize_user(user),
    }


@app.get("/api/users/me")
def get_my_profile(current_user: User = Depends(get_current_user)):
    return serialize_user(current_user)


@app.put("/api/users/profile")
def update_profile(
    profile_data: UserProfileUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    current_user.name = profile_data.name
    current_user.location = profile_data.location or ""
    current_user.bio = profile_data.bio or ""
    db.commit()
    db.refresh(current_user)
    return {"message": "Profile updated successfully", "user": serialize_user(current_user)}


@app.get("/api/admin/users")
def list_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    return [serialize_admin_user(user) for user in db.query(User).order_by(User.id.desc()).all()]


@app.get("/api/admin/usage")
def admin_usage_dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    users = db.query(User).order_by(User.id.desc()).all()
    admin_users = [serialize_admin_user(user) for user in users]
    totals = {
        "users": len(admin_users),
        "summaries": db.query(Summary).count(),
        "comparisons": db.query(Comparison).count(),
        "presentations": db.query(Presentation).count(),
    }
    totals["total_requests"] = totals["summaries"] + totals["comparisons"] + totals["presentations"]

    return {
        "totals": totals,
        "users": admin_users,
    }
    


@app.post("/api/summarize")
async def summarize_video(
    request: VideoRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        video = get_video_transcript(request.youtube_url)
        caption_summaries = await summarize_caption_windows(video.get("caption_windows", []))
        generated = await generate_all_features(
            video["transcript"], request.mode, request.custom_prompt, request.output_language or "English"
        )
    except TranscriptUnavailableError as exc:
        raise transcript_error_response(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    new_summary = Summary(
        user_id=current_user.id,
        youtube_url=request.youtube_url,
        title=video.get("title", "YouTube Video"),
        channel=video.get("channel", ""),
        duration=video.get("duration"),
        thumbnail=video.get("thumbnail", ""),
        transcript=video["transcript"],
        caption_segments=video.get("caption_segments", []),
        caption_summaries=caption_summaries,
        summary_text=generated.get("summary", ""),
        keywords=generated.get("keywords", []),
        chapters=generated.get("chapters", []),
        key_points=generated.get("key_points", []),
        questions=generated.get("questions", []),
        action_items=generated.get("action_items", []),
        language=request.output_language or "English",
    )
    db.add(new_summary)
    db.commit()
    db.refresh(new_summary)

    return {"message": "Video successfully summarized", "data": serialize_summary(new_summary)}


@app.post("/api/compare-videos")
async def compare_videos(
    request: CompareVideosRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    existing = (
        db.query(Comparison)
        .filter(
            Comparison.user_id == current_user.id,
            Comparison.youtube_url_1 == request.youtube_url_1,
            Comparison.youtube_url_2 == request.youtube_url_2,
            Comparison.goal == (request.comparison_goal or ""),
            Comparison.language == (request.output_language or "English"),
        )
        .order_by(Comparison.created_at.desc())
        .first()
    )
    if existing:
        return {"message": "Cached comparison loaded", "data": serialize_comparison(existing)}

    try:
        video_1 = get_video_transcript(request.youtube_url_1)
        video_2 = get_video_transcript(request.youtube_url_2)
        generated = await generate_video_comparison(
            video_1,
            video_2,
            request.comparison_goal,
            request.output_language or "English",
        )
    except TranscriptUnavailableError as exc:
        raise transcript_error_response(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    comparison = Comparison(
        user_id=current_user.id,
        youtube_url_1=request.youtube_url_1,
        youtube_url_2=request.youtube_url_2,
        goal=request.comparison_goal or "",
        language=request.output_language or "English",
        video_1=_video_info(video_1, request.youtube_url_1),
        video_2=_video_info(video_2, request.youtube_url_2),
        combined_summary=generated.get("combined_summary", ""),
        common_points=generated.get("common_points", []),
        differences=generated.get("differences", []),
        best_takeaways=generated.get("best_takeaways", {}),
        verdict=generated.get("verdict", {}),
        best_overall_video=generated.get("best_overall_video", {}),
    )
    db.add(comparison)
    db.commit()
    db.refresh(comparison)

    return {"message": "Videos successfully compared", "data": serialize_comparison(comparison)}


@app.post("/api/video/features")
async def get_features(
    request: VideoRequest,
    current_user: User = Depends(get_current_user),
):
    try:
        transcript = request.transcript
        caption_segments = []
        if not transcript:
            video = get_video_transcript(request.youtube_url)
            transcript = video["transcript"]
            caption_segments = video.get("caption_segments", [])
            caption_windows = video.get("caption_windows", [])
        else:
            caption_windows = []

        return {
            "transcript": transcript,
            "caption_segments": caption_segments,
            "caption_summaries": await summarize_caption_windows(caption_windows),
            "key_points": await generate_feature(transcript, "key_points"),
            "questions": await generate_feature(transcript, "questions"),
            "action_items": await generate_feature(transcript, "action_items"),
        }
    except TranscriptUnavailableError as exc:
        raise transcript_error_response(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/summaries/{summary_id}/hydrate")
async def hydrate_saved_summary(
    summary_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = db.query(Summary).filter(Summary.id == summary_id, Summary.user_id == current_user.id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Summary not found")

    try:
        needs_video = not row.transcript or not row.caption_segments or not row.caption_summaries
        video: dict[str, Any] = {}
        if needs_video:
            try:
                video = get_video_transcript(row.youtube_url)
                row.title = row.title or video.get("title", "YouTube Video")
                row.channel = row.channel or video.get("channel", "")
                row.duration = row.duration or video.get("duration")
                row.thumbnail = row.thumbnail or video.get("thumbnail", "")
                row.transcript = row.transcript or video["transcript"]
                row.caption_segments = row.caption_segments or video.get("caption_segments", [])
                row.caption_summaries = row.caption_summaries or await summarize_caption_windows(video.get("caption_windows", []))
            except TranscriptUnavailableError:
                if not row.transcript:
                    raise

        transcript = row.transcript or video.get("transcript", "")
        if transcript and (not row.key_points or not row.questions or not row.action_items):
            if not row.key_points:
                row.key_points = await generate_feature(transcript, "key_points", row.language or "English")
            if not row.questions:
                row.questions = await generate_feature(transcript, "questions", row.language or "English")
            if not row.action_items:
                row.action_items = await generate_feature(transcript, "action_items", row.language or "English")

        db.commit()
        db.refresh(row)
        return {"data": serialize_summary(row)}
    except TranscriptUnavailableError as exc:
        raise transcript_error_response(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/summary/chat")
async def ask_summary_ai(
    request: SummaryChatRequest,
    current_user: User = Depends(get_current_user),
):
    try:
        answer = await answer_summary_question(
            question=request.question,
            summary=request.summary or "",
            transcript=request.transcript or "",
            caption_summaries=request.caption_summaries or [],
            selected_window=request.selected_window,
        )
        return {"answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/summary/translate")
async def translate_summary(
    request: SummaryTranslateRequest,
    current_user: User = Depends(get_current_user),
):
    try:
        translated = await translate_summary_content(request.data, request.language)
        return {"data": {**request.data, **translated, "language": request.language}}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/summaries/recent")
def recent_summaries(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = db.query(Summary).filter(Summary.user_id == current_user.id).order_by(Summary.created_at.desc()).all()
    return [serialize_summary(row) for row in rows]


@app.get("/api/comparisons/recent")
def recent_comparisons(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = db.query(Comparison).filter(Comparison.user_id == current_user.id).order_by(Comparison.created_at.desc()).all()
    return [serialize_comparison(row) for row in rows]


@app.post("/api/presentations")
def save_presentation(
    request: PresentationSaveRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    presentation = Presentation(
        user_id=current_user.id,
        title=request.title,
        source_type=request.source_type,
        source_id=request.source_id,
        slides_json=request.slides,
    )
    db.add(presentation)
    db.commit()
    db.refresh(presentation)

    for index, slide_payload in enumerate(request.slides):
        db.add(Slide(
            presentation_id=presentation.id,
            slide_index=index,
            title=slide_payload.get("title", f"Slide {index + 1}"),
            content=slide_payload,
        ))
    db.commit()
    db.refresh(presentation)
    return {"message": "Presentation saved", "data": serialize_presentation(presentation)}


@app.post("/api/presentations/improve-slide")
async def improve_slide(
    request: SlideImproveRequest,
    current_user: User = Depends(get_current_user),
):
    try:
        slide = await improve_slide_content(request.slide, request.context)
        return {"slide": slide}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc



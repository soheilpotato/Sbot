import os
import hmac
import hashlib
import re
import json
import io
import time
import uuid
import subprocess
import threading
import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQueryResultArticle,
    InputTextMessageContent,
    Update,
    BotCommand,
    WebAppInfo,
    MenuButtonWebApp,
)
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackQueryHandler,
    InlineQueryHandler,
)
from telegram.request import HTTPXRequest
from groq import Groq, BadRequestError, APIStatusError
from google import genai
from google.genai import types as genai_types
from ddgs import DDGS  # free web search, no API key / no billing - pip install ddgs
import edge_tts  # free text-to-speech (Microsoft Edge), no API key - pip install edge-tts

# Language auto-detection for translation/TTS across many languages (not just
# Persian/English). Pure-Python, no API key - pip install langdetect. The bot
# still runs without it (falls back to script-based detection + English),
# but add it to requirements.txt for full multi-language accuracy.
try:
    from langdetect import detect as _langdetect_detect, DetectorFactory
    DetectorFactory.seed = 0  # deterministic results across runs
except ImportError:  # pragma: no cover
    _langdetect_detect = None

# PDF reading (the "send me a PDF and ask questions about it" feature).
# Pure-Python, no API key - pip install pypdf. The bot still boots without
# it; PDF uploads just get a "this feature isn't installed" reply instead of
# crashing, so a missing dependency can never take the whole bot down.
try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

# ----------------------------------------------------------------------------------------------------------------------
# Load .env locally. On Render this file won't exist, and that's fine -
# Render injects environment variables directly, so os.environ still has them.
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

TOKEN = os.environ.get("TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
BOT_USERNAME = "@MyBigPotatobot"
WEBAPP_URL = (os.environ.get("WEBAPP_URL") or os.environ.get("RENDER_EXTERNAL_URL") or "").rstrip("/")
WEBAPP_DIR = Path(__file__).parent

if not TOKEN or not GROQ_API_KEY or not GEMINI_API_KEY:
    raise RuntimeError("Missing TOKEN, GROQ_API_KEY, or GEMINI_API_KEY environment variables.")

# ---------------------- OWNER RECOGNITION ----------------------
# Only this exact Telegram user ID gets addressed as "پدر"/"Father" -
# everyone else gets normal polite treatment. Defaults to your ID below;
# still overridable by setting OWNER_TELEGRAM_ID in .env / Render env vars
# (e.g. if you ever want to change it without editing code).
_owner_id_raw = os.environ.get("OWNER_TELEGRAM_ID", "2023076168")
try:
    OWNER_TELEGRAM_ID = int(_owner_id_raw) if _owner_id_raw else None
except ValueError:
    OWNER_TELEGRAM_ID = None


def is_owner(update: Update) -> bool:
    user = update.effective_user
    return OWNER_TELEGRAM_ID is not None and user is not None and user.id == OWNER_TELEGRAM_ID


# ---------------------- ADMIN PANEL: STATS/USER STORE (Upstash Redis) ----------------------
# Render's free plan wipes local disk on every deploy, so bot_stats.json
# doesn't actually survive updates there. Upstash Redis is a free, external
# key-value store - we save/load one JSON blob to/from it over its REST API
# (just plain HTTP, using the `requests` library already imported above), so
# the data lives outside Render entirely and survives redeploys.
#
# Set these two in your .env locally / Render's Environment tab (from your
# Upstash database's "REST API" section):
#   UPSTASH_REDIS_REST_URL
#   UPSTASH_REDIS_REST_TOKEN
#
# If they're not set (e.g. you haven't set Upstash up yet), the bot falls
# back to the old local JSON file so nothing breaks - it just won't survive
# a Render redeploy until you add those two variables.
UPSTASH_REDIS_REST_URL = os.environ.get("UPSTASH_REDIS_REST_URL")
UPSTASH_REDIS_REST_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
STATS_REDIS_KEY = "bot_stats"
STATS_FILE = Path(__file__).parent / "bot_stats.json"  # fallback only, see note above
_stats_lock = threading.Lock()

if not (UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN):
    print(
        "[stats] UPSTASH_REDIS_REST_URL/UPSTASH_REDIS_REST_TOKEN not set - "
        "falling back to a local JSON file, which will NOT survive a Render "
        "redeploy. Add those two env vars to fix this."
    )


def _upstash_command(*command_parts) -> dict:
    """Run one Redis command against Upstash's REST API. Blocking (uses
    `requests`) - callers on the event loop should wrap this in
    asyncio.to_thread so it doesn't stall the bot while the network call is
    in flight."""
    resp = requests.post(
        UPSTASH_REDIS_REST_URL,
        headers={"Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}"},
        json=list(command_parts),
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _load_stats() -> dict:
    if UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN:
        try:
            result = _upstash_command("GET", STATS_REDIS_KEY).get("result")
            if result:
                return json.loads(result)
        except Exception as e:
            print(f"[stats] failed to load from Upstash, starting fresh: {e}")
        return {"users": {}, "total_messages": 0, "maintenance": False}

    # Fallback: local file (only survives plain restarts, not redeploys)
    if STATS_FILE.exists():
        try:
            return json.loads(STATS_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[stats] failed to load {STATS_FILE}, starting fresh: {e}")
    return {"users": {}, "total_messages": 0, "maintenance": False}


def _save_stats() -> None:
    with _stats_lock:
        if UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN:
            try:
                _upstash_command("SET", STATS_REDIS_KEY, json.dumps(BOT_STATS, ensure_ascii=False))
                return
            except Exception as e:
                print(f"[stats] failed to save to Upstash: {e}")
                return
        # Fallback: local file
        try:
            STATS_FILE.write_text(json.dumps(BOT_STATS, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[stats] failed to save: {e}")


BOT_STATS = _load_stats()
BOT_STATS.setdefault("users", {})
BOT_STATS.setdefault("total_messages", 0)
BOT_STATS.setdefault("maintenance", False)
BOT_BOOTED_AT = datetime.now(ZoneInfo("UTC"))


# ---------------------- LONG-TERM USER MEMORY (Upstash Redis) ----------------------
# This is what actually fixes "I told the bot my name and it forgot after
# redeploy": `context.user_data` (used for conversation history below) lives
# only in RAM and is wiped every time the process restarts - a Render
# redeploy, a crash, anything. It was never connected to Upstash at all, so
# ANY fact the AI said it would "remember" was only ever real for as long as
# the process stayed up.
#
# This store is separate from conversation history: it's a small, per-user
# list of short facts (name, preferences, ongoing situations) that the AI
# explicitly chooses to save via the remember_fact tool (see TOOLS below),
# persisted in the same Upstash Redis database already used for bot_stats -
# so it survives redeploys, crashes, and even long conversations where the
# regular history gets trimmed.
USER_MEMORY_KEY_PREFIX = "user_memory:"
MAX_MEMORY_FACTS = 60  # keep it bounded - oldest facts drop off first


def _load_user_memory(user_id: int) -> list:
    """Returns the list of remembered fact strings for this user, or []
    if Upstash isn't configured / nothing is saved yet / the call fails.
    Blocking - callers on the event loop should wrap this in
    asyncio.to_thread."""
    if not (UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN):
        return []
    try:
        result = _upstash_command("GET", f"{USER_MEMORY_KEY_PREFIX}{user_id}").get("result")
        if result:
            return json.loads(result)
    except Exception as e:
        print(f"[memory] failed to load memory for user {user_id}: {e}")
    return []


def _save_user_memory(user_id: int, facts: list) -> None:
    """Blocking - callers on the event loop should wrap this in
    asyncio.to_thread. Silently no-ops if Upstash isn't configured, same
    fallback behaviour as bot_stats above."""
    if not (UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN):
        return
    try:
        _upstash_command(
            "SET", f"{USER_MEMORY_KEY_PREFIX}{user_id}", json.dumps(facts, ensure_ascii=False)
        )
    except Exception as e:
        print(f"[memory] failed to save memory for user {user_id}: {e}")


async def remember_fact_tool(user_id: int, fact: str) -> str:
    """Called when the AI decides something the user said is worth
    remembering long-term (name, preference, ongoing situation, etc.)."""
    fact = (fact or "").strip()
    if not fact:
        return "هیچ متنی برای ذخیره‌سازی داده نشد."

    def _do() -> int:
        facts = _load_user_memory(user_id)
        if fact not in facts:
            facts.append(fact)
            if len(facts) > MAX_MEMORY_FACTS:
                facts[:] = facts[-MAX_MEMORY_FACTS:]
            _save_user_memory(user_id, facts)
        return len(facts)

    count = await asyncio.to_thread(_do)
    return f"ذخیره شد - همیشه یادم می‌مونه. الان {count} مورد درباره این کاربر ذخیره شده."


# context.user_data already caches conversation history per user for free
# (it's just RAM); piggyback on it to also cache the long-term-memory GET,
# so a chatty session doesn't re-fetch the same facts from Upstash on every
# message. Short TTL rather than "forever this process" so a fact saved via
# remember_fact still shows up in Gemini mode (which rebuilds its system
# prompt every message) reasonably soon, without needing to plumb a
# cache-invalidation callback through the tool-call machinery.
MEMORY_CACHE_TTL_SECONDS = 300


async def _get_memory_facts_cached(context: ContextTypes.DEFAULT_TYPE, user_id: int) -> list:
    cache = context.user_data.get("_memory_cache")
    now = time.monotonic()
    if cache and now - cache["loaded_at"] < MEMORY_CACHE_TTL_SECONDS:
        return cache["facts"]
    facts = await asyncio.to_thread(_load_user_memory, user_id)
    context.user_data["_memory_cache"] = {"facts": facts, "loaded_at": now}
    return facts


async def forget_fact_tool(user_id: int, fact: str) -> str:
    """Called when a previously remembered fact is no longer true or the
    user explicitly asks to have it forgotten."""
    fact = (fact or "").strip()

    def _do() -> bool:
        facts = _load_user_memory(user_id)
        new_facts = [f for f in facts if f != fact]
        changed = len(new_facts) != len(facts)
        if changed:
            _save_user_memory(user_id, new_facts)
        return changed

    changed = await asyncio.to_thread(_do)
    return "پاک شد و دیگه بهش استناد نمی‌کنم." if changed else "همچین موردی بین چیزهای ذخیره‌شده پیدا نکردم."


# ---------------------- PER-USER PERSONALITY (Upstash Redis) ----------------------
# The bot used to have exactly one personality baked into BASE_SYSTEM_PROMPT
# for everybody. Now each user picks their own (or writes a custom one), it's
# saved next to their long-term memory in Upstash, and it survives restarts.
# The owner's "father" rule is separate and always applies on top, whatever
# personality is chosen.
PERSONALITY_KEY_PREFIX = "personality:"
DEFAULT_PERSONALITY = "cute"
MAX_CUSTOM_PERSONALITY_CHARS = 400

# Each entry: short labels for the buttons (fa/en, everything else falls back
# to en) plus the actual instruction block injected into the system prompt.
PERSONALITY_PRESETS = {
    "cute": {
        "labels": {"fa": "🌸 دخترونه و بامزه", "en": "🌸 Cute & girly"},
        "prompt": (
            "- Your personality has a small, natural touch of girly/cute charm - think a\n"
            "  friendly, upbeat young woman texting a friend, not an over-the-top anime\n"
            "  character. Keep it light and occasional, never constant.\n"
            "- This can show up as: a warm, cheerful tone, an occasional soft emoji\n"
            "  (like 🌸✨💕😊 - pick one at most per message, and often use none at all),\n"
            "  or a playful/affectionate word choice here and there.\n"
            "- Do NOT use baby talk, \"uwu\"/\"owo\"-style speech, excessive giggling,\n"
            "  stretched-out words (\"heyyy\", \"yesss\"), or stacks of emoji/kaomoji. That\n"
            "  reads as cringy, not cute - avoid it entirely."
        ),
    },
    "friendly": {
        "labels": {"fa": "😊 دوستانه و ساده", "en": "😊 Friendly & casual"},
        "prompt": (
            "- You talk like a relaxed, friendly person - casual, warm, everyday language.\n"
            "- No corporate stiffness, no cutesy act, no heavy emoji use (one now and then\n"
            "  at most). Just a normal friend who happens to know a lot."
        ),
    },
    "pro": {
        "labels": {"fa": "🎩 رسمی و حرفه‌ای", "en": "🎩 Formal & professional"},
        "prompt": (
            "- You are precise, professional, and polite, the way a good colleague writes\n"
            "  at work. Full sentences, correct terminology, no slang.\n"
            "- Do not use emoji. Do not use pet names or affectionate wording.\n"
            "- Structure longer answers with short numbered points when that helps."
        ),
    },
    "funny": {
        "labels": {"fa": "😎 شوخ و بامزه", "en": "😎 Witty & funny"},
        "prompt": (
            "- You are quick-witted and a bit cheeky - light jokes, playful asides, the\n"
            "  occasional bit of harmless sarcasm.\n"
            "- The joke never replaces the answer: be funny on the way to being useful,\n"
            "  and drop the humour entirely if the user seems upset or the topic is serious.\n"
            "- Never mock the user, their questions, or anyone's identity."
        ),
    },
    "tutor": {
        "labels": {"fa": "📚 معلم و توضیح‌دهنده", "en": "📚 Patient tutor"},
        "prompt": (
            "- You explain things like a patient teacher: start from what the user already\n"
            "  seems to know, build up step by step, and use concrete examples or analogies.\n"
            "- Check understanding with a short question at the end when it makes sense.\n"
            "- Never make the user feel slow for asking something basic."
        ),
    },
    "calm": {
        "labels": {"fa": "🕊️ آرام و همدل", "en": "🕊️ Calm & supportive"},
        "prompt": (
            "- Your tone is calm, gentle, and grounded. You listen first and don't rush\n"
            "  the user toward a solution.\n"
            "- You acknowledge how something feels before giving practical advice, and you\n"
            "  keep advice small and doable.\n"
            "- You are not a therapist and never pretend to be one; if someone is in real\n"
            "  distress, gently suggest talking to a person they trust or a professional."
        ),
    },
    "short": {
        "labels": {"fa": "⚡ کوتاه و سرراست", "en": "⚡ Short & direct"},
        "prompt": (
            "- Answer in as few words as the question honestly allows. No preamble, no\n"
            "  recap of the question, no filler, no emoji.\n"
            "- Lead with the answer itself; add detail only if it's genuinely needed."
        ),
    },
}


def personality_label(code: str, ui_lang: str) -> str:
    preset = PERSONALITY_PRESETS.get(code)
    if not preset:
        return code
    labels = preset["labels"]
    return labels.get(ui_lang) or labels.get("en") or code


def _load_personality(user_id: int) -> dict:
    """Blocking - wrap in asyncio.to_thread. Returns {} when nothing is
    saved (which means 'use the default preset')."""
    if not (UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN):
        return {}
    try:
        result = _upstash_command("GET", f"{PERSONALITY_KEY_PREFIX}{user_id}").get("result")
        if result:
            data = json.loads(result)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[personality] failed to load for user {user_id}: {e}")
    return {}


def _save_personality(user_id: int, data: dict) -> None:
    """Blocking - wrap in asyncio.to_thread."""
    if not (UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN):
        return
    try:
        _upstash_command(
            "SET", f"{PERSONALITY_KEY_PREFIX}{user_id}", json.dumps(data, ensure_ascii=False)
        )
    except Exception as e:
        print(f"[personality] failed to save for user {user_id}: {e}")


async def get_personality(context: ContextTypes.DEFAULT_TYPE, user_id: int) -> dict:
    """Cached per session in user_data, same idea as the memory cache - the
    personality changes rarely, so re-fetching it every message is waste."""
    cached = context.user_data.get("_personality")
    if cached is not None:
        return cached
    data = await asyncio.to_thread(_load_personality, user_id)
    if not data:
        data = {"preset": DEFAULT_PERSONALITY}
    context.user_data["_personality"] = data
    return data


async def set_personality(
    context: ContextTypes.DEFAULT_TYPE, user_id: int, preset: str, custom_text: str = ""
) -> None:
    data = {"preset": preset}
    if custom_text:
        data["custom"] = custom_text[:MAX_CUSTOM_PERSONALITY_CHARS]
    context.user_data["_personality"] = data
    await asyncio.to_thread(_save_personality, user_id, data)
    # The system prompt (personality included) is baked in when a session's
    # history starts, so old histories have to go or the change wouldn't show
    # up until the user happened to start a fresh conversation.
    for key in ("history_groq", "history_groq_search", "history_gemini", "history_groq_images"):
        context.user_data.pop(key, None)


def personality_block(data: dict) -> str:
    """Turn a stored personality record into the PERSONALITY section of the
    system prompt."""
    preset = (data or {}).get("preset", DEFAULT_PERSONALITY)
    if preset == "custom" and (data or {}).get("custom"):
        return (
            "- The user has described, in their own words, how they want you to behave:\n"
            f"  \"{data['custom']}\"\n"
            "- Follow that description as your personality and tone, as long as it doesn't\n"
            "  ask you to be harmful, deceptive, or to abandon accuracy. Never invent facts\n"
            "  or agree with something false just to stay in character."
        )
    return PERSONALITY_PRESETS.get(preset, PERSONALITY_PRESETS[DEFAULT_PERSONALITY])["prompt"]


# ---------------------- REMINDERS (Upstash Redis + PTB JobQueue) ----------------------
# Reminders are the one feature that makes the bot message the user instead of
# only answering them. Two halves:
#   1. Storage: every reminder is kept in one JSON list in the same Upstash DB
#      as everything else, so a Render redeploy doesn't silently delete
#      everybody's alarms.
#   2. Scheduling: python-telegram-bot's JobQueue fires them. On boot,
#      reschedule_all_reminders() re-registers every stored reminder with the
#      new process (jobs live in RAM and die with it - the stored list is the
#      source of truth).
# JobQueue needs the extra: pip install "python-telegram-bot[job-queue]".
REMINDERS_REDIS_KEY = "bot_reminders"
REMINDERS_FILE = Path(__file__).parent / "bot_reminders.json"
MAX_REMINDERS_PER_USER = 20
_reminders_lock = threading.Lock()

REPEAT_INTERVALS = {
    "daily": 24 * 3600,
    "weekly": 7 * 24 * 3600,
    "hourly": 3600,
}

# Set in __main__ so background helpers (tool calls, job callbacks) can reach
# the running Application without it being threaded through every function.
BOT_APP = None


def _load_reminders() -> list:
    """Blocking - wrap in asyncio.to_thread."""
    if UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN:
        try:
            result = _upstash_command("GET", REMINDERS_REDIS_KEY).get("result")
            if result:
                data = json.loads(result)
                return data if isinstance(data, list) else []
        except Exception as e:
            print(f"[reminders] failed to load from Upstash: {e}")
        return []
    if REMINDERS_FILE.exists():
        try:
            return json.loads(REMINDERS_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[reminders] failed to load {REMINDERS_FILE}: {e}")
    return []


def _save_reminders(items: list) -> None:
    """Blocking - wrap in asyncio.to_thread."""
    with _reminders_lock:
        if UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN:
            try:
                _upstash_command(
                    "SET", REMINDERS_REDIS_KEY, json.dumps(items, ensure_ascii=False)
                )
            except Exception as e:
                print(f"[reminders] failed to save to Upstash: {e}")
            return
        try:
            REMINDERS_FILE.write_text(
                json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            print(f"[reminders] failed to save: {e}")


def _store_reminder(reminder: dict) -> int:
    """Add/replace one reminder in the stored list. Returns how many this
    user now has. Blocking - wrap in asyncio.to_thread."""
    items = _load_reminders()
    items = [r for r in items if r.get("id") != reminder["id"]]
    items.append(reminder)
    _save_reminders(items)
    return sum(1 for r in items if r.get("user_id") == reminder["user_id"])


def _delete_reminder(reminder_id: str) -> bool:
    """Blocking - wrap in asyncio.to_thread."""
    items = _load_reminders()
    remaining = [r for r in items if r.get("id") != reminder_id]
    changed = len(remaining) != len(items)
    if changed:
        _save_reminders(remaining)
    return changed


def _user_reminders(user_id: int) -> list:
    """Blocking - wrap in asyncio.to_thread."""
    return sorted(
        (r for r in _load_reminders() if r.get("user_id") == user_id),
        key=lambda r: r.get("due_ts", 0),
    )


def format_reminder_time(reminder: dict) -> str:
    tz_name = reminder.get("tz", "Asia/Tehran")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("UTC")
    when = datetime.fromtimestamp(reminder.get("due_ts", 0), tz)
    label = when.strftime("%Y-%m-%d %H:%M")
    repeat = reminder.get("repeat", "none")
    if repeat in REPEAT_INTERVALS:
        label += f" ({repeat})"
    return label


async def _reminder_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fires when a reminder comes due. Sends the message, then either
    reschedules the next occurrence (repeating) or removes it (one-off)."""
    reminder = context.job.data
    try:
        await context.bot.send_message(
            chat_id=reminder["chat_id"],
            text=f"⏰ {reminder.get('text', '')}",
        )
    except Exception as e:
        print(f"[reminders] failed to deliver {reminder.get('id')}: {e}")

    repeat = reminder.get("repeat", "none")
    interval = REPEAT_INTERVALS.get(repeat)
    if interval:
        now = time.time()
        due = reminder.get("due_ts", now) + interval
        while due <= now:  # bot was down for a while - roll forward, don't spam
            due += interval
        reminder["due_ts"] = due
        await asyncio.to_thread(_store_reminder, reminder)
        schedule_reminder_job(reminder)
    else:
        await asyncio.to_thread(_delete_reminder, reminder["id"])


def schedule_reminder_job(reminder: dict) -> bool:
    """Register one reminder with the JobQueue. Returns False if the
    JobQueue isn't available (the [job-queue] extra isn't installed)."""
    if BOT_APP is None or BOT_APP.job_queue is None:
        return False
    delay = max(1.0, reminder.get("due_ts", 0) - time.time())
    BOT_APP.job_queue.run_once(
        _reminder_job, when=delay, data=reminder, name=f"reminder:{reminder['id']}"
    )
    return True


def cancel_reminder_job(reminder_id: str) -> None:
    if BOT_APP is None or BOT_APP.job_queue is None:
        return
    for job in BOT_APP.job_queue.get_jobs_by_name(f"reminder:{reminder_id}"):
        job.schedule_removal()


async def add_reminder(
    user_id: int, chat_id: int, text: str, due_ts: float, repeat: str = "none", tz: str = "Asia/Tehran"
) -> dict | None:
    """Save + schedule one reminder. Returns None if the user is already at
    MAX_REMINDERS_PER_USER."""
    existing = await asyncio.to_thread(_user_reminders, user_id)
    if len(existing) >= MAX_REMINDERS_PER_USER:
        return None
    reminder = {
        "id": uuid.uuid4().hex[:10],
        "user_id": user_id,
        "chat_id": chat_id,
        "text": (text or "").strip()[:500],
        "due_ts": float(due_ts),
        "repeat": repeat if repeat in REPEAT_INTERVALS else "none",
        "tz": tz,
    }
    await asyncio.to_thread(_store_reminder, reminder)
    schedule_reminder_job(reminder)
    return reminder


async def reschedule_all_reminders() -> int:
    """Called once at startup: re-register every stored reminder with this
    fresh process. Overdue one-offs (bot was down when they were due) fire a
    few seconds after boot rather than being silently dropped."""
    items = await asyncio.to_thread(_load_reminders)
    now = time.time()
    count = 0
    for reminder in items:
        interval = REPEAT_INTERVALS.get(reminder.get("repeat", "none"))
        if interval:
            due = reminder.get("due_ts", now)
            while due <= now:
                due += interval
            reminder["due_ts"] = due
        elif reminder.get("due_ts", 0) <= now:
            reminder["due_ts"] = now + 5  # deliver it late rather than never
        if schedule_reminder_job(reminder):
            count += 1
    if items:
        await asyncio.to_thread(_save_reminders, items)
    return count


# Parsing helpers shared by the /remind command and the AI's set_reminder tool.
_DURATION_RE = re.compile(
    r"^(\d+)\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days|"
    r"ثانیه|دقیقه|دقیقه‌|ساعت|روز)$",
    re.IGNORECASE,
)
_CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


def parse_duration_to_seconds(token: str) -> int | None:
    """'20m' / '2h' / '30s' / '3d' / '۲ساعت'-style tokens -> seconds."""
    match = _DURATION_RE.match((token or "").strip())
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2).lower()
    if unit in ("s", "sec", "secs", "second", "seconds", "ثانیه"):
        return amount
    if unit in ("m", "min", "mins", "minute", "minutes", "دقیقه", "دقیقه‌"):
        return amount * 60
    if unit in ("h", "hr", "hrs", "hour", "hours", "ساعت"):
        return amount * 3600
    return amount * 86400


def next_clock_time(clock: str, tz_name: str = "Asia/Tehran") -> float | None:
    """'08:30' -> the timestamp of the next time it's 08:30 in that timezone."""
    match = _CLOCK_RE.match((clock or "").strip())
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo("Asia/Tehran")
    now = datetime.now(tz)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target.timestamp()


def record_user_activity(update: Update) -> None:
    """Called on every incoming update (group=-1, before any other handler)
    so the admin panel has real numbers to show. Only updates BOT_STATS in
    memory - saving it (a network call now, with Upstash) is the caller's
    job via asyncio.to_thread, so this stays fast and never blocks the
    event loop. Best-effort - never allowed to break the actual message
    handling."""
    try:
        user = update.effective_user
        if user is None:
            return
        uid = str(user.id)
        now = datetime.now(ZoneInfo("UTC")).isoformat()
        entry = BOT_STATS["users"].setdefault(uid, {"first_seen": now, "message_count": 0})
        entry["username"] = user.username
        entry["first_name"] = user.first_name
        entry["last_seen"] = now
        entry["message_count"] = entry.get("message_count", 0) + 1
        BOT_STATS["total_messages"] = BOT_STATS.get("total_messages", 0) + 1
    except Exception as e:
        print(f"[stats] record_user_activity failed: {e}")


# Full BOT_STATS blob only gets actually PUSHED to Upstash at most this
# often, instead of on every single message. record_user_activity() still
# updates the in-memory numbers instantly on every message (the admin panel
# always reads current numbers), but re-uploading the whole growing JSON
# blob on every message is the single biggest driver of data-transfer usage
# for a bot with any real traffic - a batch of messages in the same
# window now costs one write instead of one each. Worst case if the
# process dies mid-window: the last few minutes of message counts aren't
# reflected in Upstash yet - nothing this bot treats as critical.
STATS_SAVE_INTERVAL_SECONDS = 30
_stats_last_flush = 0.0
_stats_flush_lock = threading.Lock()


async def track_activity(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global _stats_last_flush
    record_user_activity(update)

    now = time.monotonic()
    with _stats_flush_lock:
        should_flush = now - _stats_last_flush >= STATS_SAVE_INTERVAL_SECONDS
        if should_flush:
            _stats_last_flush = now
    if should_flush:
        # Network call (Upstash) - pushed to a thread so it never blocks
        # the bot while it waits on the request.
        await asyncio.to_thread(_save_stats)


def _format_duration(delta) -> str:
    total_seconds = int(delta.total_seconds())
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days} روز")
    if hours:
        parts.append(f"{hours} ساعت")
    if minutes or not parts:
        parts.append(f"{minutes} دقیقه")
    return " و ".join(parts)


groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "openai/gpt-oss-120b"  # check console.groq.com/docs/models - Groq's free lineup changes often

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"

# How many past messages (user+model combined) to keep per user/provider, to stop memory growing forever
MAX_HISTORY = 20

BASE_SYSTEM_PROMPT = """
You are a helpful, warm, and unfailingly polite AI assistant inside a Telegram bot.

PERSONALITY:
{personality_block}
- Never let this override clarity or usefulness. For serious, technical, or
  sensitive topics, drop the stylistic touches and just be direct and helpful.

LANGUAGE:
- You are fully multilingual. You understand and can fluently write in any
  language the user uses - including but not limited to Persian, English,
  Japanese, Arabic, Chinese, Korean, French, German, Spanish, Russian, Turkish,
  Italian, Portuguese, Hindi, Urdu, and Dutch.
- ALWAYS reply in the SAME language the user just wrote their message in,
  unless they explicitly ask you to answer in a different language. Never
  default to Persian or English just because it's your "usual" language.
- When writing Japanese, Chinese, Korean, Arabic, or any other non-Latin
  script, write it correctly and natively in that script - never
  transliterate into Latin letters unless the user specifically asks for
  romanization.

You have real tools available - web_search, fetch_page, and get_current_datetime.
Use them whenever they would make your answer more accurate, instead of guessing:
- For ANY question about the current date, time, or "what time is it [somewhere]",
  you MUST call get_current_datetime with the correct IANA timezone, and ONLY
  that tool. Never use web_search or fetch_page for this - a live clock is not
  something a search result or a web page can give you reliably, so trying to
  browse for the time is always wrong. Never guess or calculate the time
  yourself either.
- For questions about recent events, current facts, prices, news, or anything
  that might have changed since your training, call web_search first.
- If the search results aren't detailed enough to answer properly, call
  fetch_page on the most relevant result's URL to read the full page, the way
  a person would click into a search result.
- Don't mention these tools by name to the user or explain that you're using
  them - just use them and answer naturally.

IMPORTANT OUTPUT RULES:
- Always reply using plain text only.
- Never output HTML or HTML tags such as <p>, <br>, <b>, <i>, <div>, <a>, etc.
- Never output XML or other markup/code intended to be interpreted as a web page.
- Do not use Markdown formatting such as **bold**, __bold__, [links](...), or ```code blocks```.
- Use normal text that can be sent directly as a Telegram message.
- You may use normal line breaks, simple numbered lists, and simple bullet points.
- If you need to show code, write it as plain text without Markdown code fences.
"""

# Appended only for the one specific Telegram user configured as
# OWNER_TELEGRAM_ID above - never applied to any other user.
OWNER_ADDRESS_INSTRUCTION = """
SPECIAL RULE ABOUT WHO YOU ARE TALKING TO:
The person you are currently talking to is your creator/owner. Always be
especially warm, polite and respectful toward them - like a close friend, not
a formal assistant. You may address them with a warm word equivalent to
"father"/"پدر" in meaning, but use it sparingly and only where it feels
natural and human, the way a real person would drop in a name now and then -
not as a fixed tag on every message. Good spots: when greeting them, when
asking them a question, when answering a direct question of theirs, or when
saying something warm/reassuring. Do NOT attach it to routine replies,
confirmations, error messages, or every single sentence - that reads as
robotic. If in doubt, leave it out.
IMPORTANT: always translate this address into whatever language you are
replying in at that moment - never insert the English word "Father" or the
Persian word "پدر" into a reply written in a different language. Use the
natural, warm equivalent a native speaker of that language would actually
use for a father figure (for example, in Japanese something like お父さん,
not a raw English/Persian word dropped into the sentence). This address is
reserved ONLY for this person; never call anyone else this way.
"""


# Only appended when remember_fact/forget_fact are actually wired up as
# real, callable tools for this provider (currently: groq, groq_search).
# Gemini mode below does NOT get this, on purpose - it has no live tool
# use here, so promising a "remember_fact tool" it can't actually call
# would recreate the exact "bot said it'll remember but doesn't" bug this
# was meant to fix. Gemini still gets to *see* previously saved facts
# (via the plain facts block), just not save new ones itself.
MEMORY_TOOL_INSTRUCTIONS = """
LONG-TERM MEMORY:
- You have a remember_fact tool. Unlike web_search/fetch_page/
  get_current_datetime (which only help with THIS one reply), remember_fact
  permanently saves a short fact about the user - their name, a preference,
  an ongoing situation, anything they'd expect you to still know the next
  time they talk to you, even after the bot restarts.
- The moment the user tells you something like this, actually CALL
  remember_fact right away. Don't just say "I'll remember that" in your
  reply text without calling the tool - the tool call is what makes it real.
- If a remembered fact turns out to be outdated or the user asks you to
  forget something, call forget_fact.
- If you genuinely cannot save something (the tool call fails), be honest
  about that instead of claiming you saved it.
- Don't narrate the mechanism of saving - a natural "حتما یادم می‌مونه" is
  fine, explaining tools/Redis/etc. is not.
- If a "LONG-TERM MEMORY ABOUT THIS USER" section appears below, those facts
  are already saved from earlier sessions - use them naturally, don't ask
  the user to repeat them, and don't re-save the same fact again.
"""


EXTRA_TOOL_INSTRUCTIONS = """
IMAGES:
- You have a search_images tool that finds real photos on the web and sends
  them to the user as actual Telegram photos.
- Use it whenever the user asks to SEE something - "a picture of X", "show me
  X", "عکس X رو بفرست", "what does X look like" - and also when a picture
  would obviously help (a species, a place, a landmark, a dish, a diagram of
  a concept, a product).
- Think like a good researcher, not a single-keyword search box: pick a
  specific, descriptive English query that will actually return the right
  images (for example, for "عکس باکتری" use something like
  "bacteria microscope photograph", not "bacteria"), and ask for several
  images (3-6) so the user gets a proper set, not one random photo.
- If the subject has clearly different aspects worth showing (different
  species, angles, stages, before/after), call search_images more than once
  with different queries instead of repeating the same one.
- The images are sent automatically alongside your reply. So write a real
  answer explaining what's in them - never say "here is the image" and
  nothing else, and never paste image URLs into your text.

REMINDERS:
- You have a set_reminder tool that makes the bot message the user later.
- Use it whenever the user asks to be reminded, woken up, nudged, or told
  something at a certain time ("یادم بنداز...", "remind me in 20 minutes",
  "every morning at 8 tell me to...").
- For a relative time use delay_minutes. For a clock time use at_time with
  "HH:MM" in 24-hour format, and set repeat to "daily"/"weekly"/"hourly" if
  they want it to keep happening.
- Put ONLY the thing to be reminded of in `text`, written the way the user
  would want to read it later, in their language.
- After the tool succeeds, confirm naturally what you'll remind them of and
  when. If it fails, say so honestly instead of pretending it worked.
"""


def build_system_prompt(
    owner: bool,
    memory_facts: list | None = None,
    tools_available: bool = True,
    personality: dict | None = None,
) -> str:
    prompt = BASE_SYSTEM_PROMPT.replace("{personality_block}", personality_block(personality))
    if tools_available:
        prompt += "\n" + MEMORY_TOOL_INSTRUCTIONS + "\n" + EXTRA_TOOL_INSTRUCTIONS
    if memory_facts:
        facts_block = "\n".join(f"- {f}" for f in memory_facts)
        prompt += (
            "\n\nLONG-TERM MEMORY ABOUT THIS USER (persisted across sessions, "
            "already true, don't ask again):\n" + facts_block
        )
    if owner:
        prompt += "\n" + OWNER_ADDRESS_INSTRUCTION
    return prompt
# ----------------------------------------------------------------------------------------------------------------------


# ---------------------- WEB APP + HEALTH SERVER ----------------------
# The Flask service also serves the Telegram Mini App and its AI API.
flask_app = Flask(__name__)

_WEB_HISTORY: dict[int, list] = {}
_WEB_HISTORY_LOCK = threading.Lock()
_WEB_CONTEXT_CACHE = {}


def _telegram_webapp_user(init_data: str) -> dict | None:
    """Validate Telegram Mini App initData and return the Telegram user."""
    if not init_data or not TOKEN:
        return None
    try:
        from urllib.parse import parse_qsl
        pairs = dict(parse_qsl(init_data, keep_blank_values=True))
        received_hash = pairs.pop("hash", None)
        if not received_hash:
            return None
        data_check_string = "\n".join(
            f"{k}={v}" for k, v in sorted(pairs.items())
        )
        secret_key = hmac.new(
            b"WebAppData", TOKEN.encode(), hashlib.sha256
        ).digest()
        calculated = hmac.new(
            secret_key, data_check_string.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(calculated, received_hash):
            return None
        auth_date = int(pairs.get("auth_date", "0"))
        if auth_date and time.time() - auth_date > 86400:
            return None
        user = json.loads(pairs.get("user", "{}"))
        return user if isinstance(user, dict) and user.get("id") else None
    except Exception as e:
        print(f"[webapp] initData validation failed: {e}")
        return None


class _WebContext:
    def __init__(self):
        self.user_data = {}


def _web_context(user_id: int):
    with _WEB_HISTORY_LOCK:
        return _WEB_CONTEXT_CACHE.setdefault(user_id, _WebContext())


@flask_app.route("/")
def health_check():
    return "Potato AI is alive", 200


@flask_app.route("/app")
def web_app():
    return send_from_directory(WEBAPP_DIR, "index.html")


@flask_app.route("/api/chat", methods=["POST"])
def web_chat():
    data = request.get_json(silent=True) or {}
    user = _telegram_webapp_user(data.get("initData", ""))
    if not user:
        return jsonify({
            "ok": False,
            "error": "Telegram session is missing or expired. Re-open the app from Telegram."
        }), 401

    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"ok": False, "error": "Message is empty."}), 400
    if len(message) > 12000:
        return jsonify({"ok": False, "error": "Message is too long."}), 400

    user_id = int(user["id"])
    provider = data.get("provider", "groq")
    if provider not in ("groq", "gemini", "groq_search", "groq_images"):
        provider = "groq"

    async def do_chat():
        ctx = _web_context(user_id)

        # Snapshot the history without holding the lock across network calls.
        with _WEB_HISTORY_LOCK:
            history = _WEB_HISTORY.setdefault(user_id, [])

        # Load the same persistent memory/personality used by Telegram.
        memory_facts = await _get_memory_facts_cached(ctx, user_id)
        personality = await get_personality(ctx, user_id)

        if provider != "gemini":
            with _WEB_HISTORY_LOCK:
                history = _WEB_HISTORY.setdefault(user_id, [])
                if history and history[0].get("role") == "system":
                    history[0]["content"] = build_system_prompt(
                        user_id == OWNER_TELEGRAM_ID,
                        memory_facts,
                        personality=personality,
                    )
                history.append({"role": "user", "content": message})
                if len(history) > MAX_HISTORY:
                    history[:] = [history[0]] + history[-(MAX_HISTORY - 1):]

            image_batches = []
            tool_ctx = {"chat_id": user_id, "images": image_batches}
            reply = await run_groq_agent(
                history,
                user_id,
                model=GROQ_MODEL,
                force_search=(provider == "groq_search"),
                force_images=(provider == "groq_images"),
                tool_ctx=tool_ctx,
            )
            with _WEB_HISTORY_LOCK:
                history.append({"role": "assistant", "content": reply})
            return reply

        # Gemini uses its own system instruction and model-role history.
        with _WEB_HISTORY_LOCK:
            history = _WEB_HISTORY.setdefault(user_id, [])
            history.append({"role": "user", "content": message})
            compact = [
                {
                    "role": ("model" if x["role"] == "assistant" else x["role"]),
                    "content": x["content"],
                }
                for x in history
                if x["role"] in ("user", "assistant", "model")
            ]
            if len(compact) > MAX_HISTORY:
                compact = compact[-MAX_HISTORY:]

        reply = await call_gemini(
            compact,
            build_system_prompt(
                user_id == OWNER_TELEGRAM_ID,
                memory_facts,
                tools_available=False,
                personality=personality,
            ),
        )
        with _WEB_HISTORY_LOCK:
            history.append({"role": "model", "content": reply})
        return reply

    try:
        reply = asyncio.run(do_chat())
        return jsonify({"ok": True, "reply": reply})
    except Exception as e:
        print(f"[webapp] chat failed for user {user_id}: {e}")
        return jsonify({
            "ok": False,
            "error": "Sorry, the AI request failed. Please try again."
        }), 500


def run_flask():
    port = int(os.environ.get("PORT", 10000))
    flask_app.run(host="0.0.0.0", port=port, threaded=True)


# ---------------------- START MENU ----------------------
def build_start_keyboard(context: ContextTypes.DEFAULT_TYPE) -> InlineKeyboardMarkup:
    """Every feature the bot has, on one screen. Anything not listed here may
    as well not exist - people don't read /help."""
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(ui(context, "btn_ai_groq"), callback_data="ai_groq"),
                InlineKeyboardButton(ui(context, "btn_ai_gemini"), callback_data="ai_gemini"),
            ],
            [
                InlineKeyboardButton(ui(context, "btn_web_search"), callback_data="ai_groq_search"),
                InlineKeyboardButton(ui(context, "btn_images"), callback_data="ai_images"),
            ],
            [
                InlineKeyboardButton(ui(context, "btn_translate_menu"), callback_data="ai_translate_menu"),
                InlineKeyboardButton(ui(context, "btn_pdf"), callback_data="pdf_help"),
            ],
            [
                InlineKeyboardButton(ui(context, "btn_personality"), callback_data="personality_menu"),
                InlineKeyboardButton(ui(context, "btn_reminders"), callback_data="reminders_menu"),
            ],
            [
                # switch_inline_query opens Telegram's chat picker and drops
                # "@thisbot " into whichever chat they choose - the cheapest
                # word-of-mouth loop there is, since their friends see the bot
                # working inside a chat they're already in.
                InlineKeyboardButton(ui(context, "btn_share"), switch_inline_query=""),
            ],
        ]
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        ui(context, "start_greeting") + "\n\n" + ui(context, "start_features"),
        reply_markup=build_start_keyboard(context),
    )


# ---------------------- HELP ----------------------
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = ui(context, "help_text") + "\n" + ui(context, "help_extra")
    if is_owner(update):
        text += "\n\n🔐 /admin — پنل مدیریت ربات (فقط قابل مشاهده برای تو)"
    await update.message.reply_text(text)


# ---------------------- SHARED MODE-ACTIVATION HELPERS ----------------------
# Used by both the inline buttons (button_click) and the equivalent slash
# commands below, so the two entry points never drift out of sync.
def activate_groq_mode(context: ContextTypes.DEFAULT_TYPE) -> str:
    context.user_data["ai_mode"] = True
    context.user_data["ai_off"] = False  # an explicit choice always overrides a previous /stop
    context.user_data["ai_provider"] = "groq"
    context.user_data.setdefault("history_groq", [])
    return ui(context, "mode_groq_on")


def activate_gemini_mode(context: ContextTypes.DEFAULT_TYPE) -> str:
    context.user_data["ai_mode"] = True
    context.user_data["ai_off"] = False  # an explicit choice always overrides a previous /stop
    context.user_data["ai_provider"] = "gemini"
    context.user_data.setdefault("history_gemini", [])
    return ui(context, "mode_gemini_on")


def activate_search_mode(context: ContextTypes.DEFAULT_TYPE) -> str:
    context.user_data["ai_mode"] = True
    context.user_data["ai_off"] = False  # an explicit choice always overrides a previous /stop
    context.user_data["ai_provider"] = "groq_search"
    context.user_data.setdefault("history_groq_search", [])
    return ui(context, "mode_search_on")


def activate_images_mode(context: ContextTypes.DEFAULT_TYPE) -> str:
    context.user_data["ai_mode"] = True
    context.user_data["ai_off"] = False  # an explicit choice always overrides a previous /stop
    context.user_data["ai_provider"] = "groq_images"
    context.user_data.setdefault("history_groq_images", [])
    return ui(context, "mode_images_on")


def activate_translate_mode(context: ContextTypes.DEFAULT_TYPE, target: str) -> str:
    """target can be 'auto', a known LANGUAGES code, or an arbitrary
    free-text language name (e.g. from the /translate command) that isn't
    in the LANGUAGES dict at all - the underlying AI translator understands
    language names directly, so this isn't limited to the dict's contents."""
    context.user_data["ai_mode"] = True
    context.user_data["ai_off"] = False  # an explicit choice always overrides a previous /stop
    context.user_data["ai_provider"] = "translate"
    context.user_data["translate_target"] = target
    if target == "auto":
        return ui(context, "translate_auto_on")
    target_label = LANGUAGES[target]["name"] if target in LANGUAGES else target
    return ui(context, "translate_target_on", target=target_label)


# ---------------------- BUTTON HANDLER ----------------------
async def button_click(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()  # tells telegram the button press was received

    if query.data == "noop":
        return  # the page-indicator button in paginated language keyboards - does nothing

    if query.data == "ai_groq":
        await query.edit_message_text(activate_groq_mode(context))

    elif query.data == "ai_gemini":
        await query.edit_message_text(activate_gemini_mode(context))

    elif query.data == "ai_groq_search":
        await query.edit_message_text(activate_search_mode(context))

    elif query.data == "ai_images":
        await query.edit_message_text(activate_images_mode(context))

    elif query.data == "pdf_help":
        await query.edit_message_text(ui(context, "pdf_help"))

    elif query.data == "personality_menu":
        data = await get_personality(context, update.effective_user.id)
        await query.edit_message_text(
            ui(context, "personality_prompt"),
            reply_markup=build_personality_keyboard(context, data.get("preset", DEFAULT_PERSONALITY)),
        )

    elif query.data.startswith("pers:"):
        await handle_personality_button(update, context)

    elif query.data == "reminders_menu":
        await _render_reminders(update, context, edit=True)

    elif query.data.startswith("remdel:"):
        await handle_reminder_delete_button(update, context)

    elif query.data == "ai_translate_menu":
        await query.edit_message_text(
            ui(context, "translate_menu_prompt"),
            reply_markup=build_paginated_lang_keyboard("translate_lang", page=0, include_auto=True),
        )

    elif query.data.startswith("translate_lang:"):
        target_code = query.data.split(":", 1)[1]
        await query.edit_message_text(activate_translate_mode(context, target_code))

    elif query.data.startswith("langpage:"):
        # "langpage:<kind>:<page>" - flip a page of a paginated language keyboard.
        _, kind, page_str = query.data.split(":", 2)
        page = int(page_str)
        if kind == "translate_lang":
            await query.edit_message_reply_markup(
                reply_markup=build_paginated_lang_keyboard("translate_lang", page, include_auto=True)
            )

    elif query.data.startswith("ui_lang:"):
        lang_code = query.data.split(":", 1)[1]
        context.user_data["ui_lang"] = lang_code
        lang_label = LANGUAGES[lang_code]["label"]
        await query.edit_message_text(ui(context, "language_set", lang=lang_label))

    elif query.data.startswith("tts:"):
        await handle_tts_button(update, context)

    elif query.data.startswith("ttsrestore:"):
        await handle_tts_restore_button(update, context)

    elif query.data.startswith("trmenu:"):
        await handle_translate_menu_button(update, context)

    elif query.data.startswith("trcancel:"):
        await handle_translate_cancel_button(update, context)

    elif query.data.startswith("trlang:"):
        await handle_translate_lang_button(update, context)

    elif query.data.startswith("admin_"):
        await handle_admin_button(update, context)


# ---------------------- "JUST TALK TO ME" GATE ----------------------
# Originally every message was ignored unless the user had gone through
# /start -> button first, which meant new people messaged the bot, got
# silence, and left. Now a plain message in a private chat just works: AI
# mode switches itself on the first time and stays on.
#
# Two things keep that from becoming annoying:
#   - /stop sets "ai_off", which this respects until the user turns AI mode
#     back on themselves (otherwise /stop would do nothing at all).
#   - In groups the bot still only answers when it's actually being talked
#     to - mentioned by username, or replied to. Auto-answering every message
#     in a group is the fastest way to get a bot kicked out.
def _strip_bot_mention(text: str, username: str | None) -> str:
    if not username:
        return text
    return re.sub(rf"@{re.escape(username)}\b", "", text, flags=re.IGNORECASE).strip()


async def ensure_ai_ready(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Returns True if this message should be handled by the AI. Switches AI
    mode on by itself when needed, so users never have to 'activate' anything
    to get an answer."""
    chat = update.effective_chat
    message = update.message
    if message is None:
        return False

    if chat.type != "private":
        text = message.text or message.caption or ""
        username = context.bot.username
        mentioned = bool(username) and f"@{username}".lower() in text.lower()
        replied_to_bot = bool(
            message.reply_to_message
            and message.reply_to_message.from_user
            and message.reply_to_message.from_user.id == context.bot.id
        )
        if not (mentioned or replied_to_bot):
            return False

    if context.user_data.get("ai_off"):
        return False

    if not context.user_data.get("ai_mode"):
        activate_groq_mode(context)
    return True


# ---------------------- STOP AI MODE (optional command) ----------------------
async def stop_ai(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["ai_mode"] = False
    # Without this flag, the "just talk to me" gate above would helpfully
    # switch AI mode straight back on with the user's very next message, and
    # /stop would look broken.
    context.user_data["ai_off"] = True
    await update.message.reply_text(ui(context, "stop_ai_msg"))


# ---------------------- DIRECT MODE COMMANDS (shortcuts for the /start buttons) ----------------------
async def groq_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(activate_groq_mode(context))


async def gemini_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(activate_gemini_mode(context))


async def websearch_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(activate_search_mode(context))


async def images_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/images on its own switches into image mode; /images <query> does one
    search straight away, without changing the current mode."""
    query = " ".join(context.args).strip() if context.args else ""
    if not query:
        await update.message.reply_text(activate_images_mode(context))
        return

    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)
    results = await search_images(query, MAX_IMAGES_PER_BATCH)
    sent = await send_image_batch(
        context.bot,
        chat_id,
        {"query": query, "results": results, "count": MAX_IMAGES_PER_BATCH},
        reply_to_message_id=update.message.message_id,
    )
    if not sent:
        await update.message.reply_text(ui(context, "images_none", query=query))


# ---------------------- /personality COMMAND (per-user tone) ----------------------
def build_personality_keyboard(context: ContextTypes.DEFAULT_TYPE, current: str) -> InlineKeyboardMarkup:
    lang = get_ui_lang(context)
    rows, row = [], []
    for code in PERSONALITY_PRESETS:
        label = personality_label(code, lang)
        if code == current:
            label = "✅ " + label
        row.append(InlineKeyboardButton(label, callback_data=f"pers:{code}"))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    custom_label = ui(context, "btn_personality_custom")
    if current == "custom":
        custom_label = "✅ " + custom_label
    rows.append([InlineKeyboardButton(custom_label, callback_data="pers:custom")])
    return InlineKeyboardMarkup(rows)


async def personality_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    data = await get_personality(context, update.effective_user.id)
    current = data.get("preset", DEFAULT_PERSONALITY)
    await update.message.reply_text(
        ui(context, "personality_prompt"),
        reply_markup=build_personality_keyboard(context, current),
    )


async def handle_personality_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    code = query.data.split(":", 1)[1]
    user_id = update.effective_user.id

    if code == "custom":
        # The next plain text message this user sends becomes their custom
        # personality - handled at the top of text_router.
        context.user_data["awaiting_personality"] = True
        await query.edit_message_text(ui(context, "personality_custom_prompt"))
        return

    if code not in PERSONALITY_PRESETS:
        return
    await set_personality(context, user_id, code)
    await query.edit_message_text(
        ui(context, "personality_set", name=personality_label(code, get_ui_lang(context)))
    )


# ---------------------- /remind + /reminders COMMANDS ----------------------
async def remind_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args or []
    if len(args) < 2:
        await update.message.reply_text(ui(context, "remind_usage"))
        return

    if BOT_APP is None or BOT_APP.job_queue is None:
        await update.message.reply_text(ui(context, "remind_unavailable"))
        return

    when_token = args[0]
    rest = args[1:]

    repeat = "none"
    if rest and rest[0].lower() in REPEAT_INTERVALS:
        repeat = rest[0].lower()
        rest = rest[1:]
    text = " ".join(rest).strip()
    if not text:
        await update.message.reply_text(ui(context, "remind_usage"))
        return

    tz = context.user_data.get("tz", "Asia/Tehran")
    seconds = parse_duration_to_seconds(when_token)
    if seconds is not None:
        due_ts = time.time() + seconds
    else:
        due_ts = next_clock_time(when_token, tz)
    if due_ts is None:
        await update.message.reply_text(ui(context, "remind_usage"))
        return

    reminder = await add_reminder(
        update.effective_user.id, update.effective_chat.id, text, due_ts, repeat, tz
    )
    if reminder is None:
        await update.message.reply_text(ui(context, "remind_limit", limit=MAX_REMINDERS_PER_USER))
        return
    await update.message.reply_text(
        ui(context, "remind_set", when=format_reminder_time(reminder), text=reminder["text"])
    )


def build_reminders_keyboard(context: ContextTypes.DEFAULT_TYPE, items: list) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(
                f"🗑 {format_reminder_time(r)} — {r['text'][:25]}",
                callback_data=f"remdel:{r['id']}",
            )
        ]
        for r in items[:10]
    ]
    return InlineKeyboardMarkup(rows)


async def _render_reminders(update: Update, context: ContextTypes.DEFAULT_TYPE, edit: bool) -> None:
    items = await asyncio.to_thread(_user_reminders, update.effective_user.id)
    if not items:
        text, markup = ui(context, "reminders_empty"), None
    else:
        lines = [ui(context, "reminders_header")]
        lines += [f"• {format_reminder_time(r)} — {r['text']}" for r in items[:10]]
        text = "\n".join(lines) + "\n\n" + ui(context, "reminders_delete_hint")
        markup = build_reminders_keyboard(context, items)

    if edit:
        await update.callback_query.edit_message_text(text, reply_markup=markup)
    else:
        await update.message.reply_text(text, reply_markup=markup)


async def reminders_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _render_reminders(update, context, edit=False)


async def handle_reminder_delete_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    reminder_id = update.callback_query.data.split(":", 1)[1]
    items = await asyncio.to_thread(_user_reminders, update.effective_user.id)
    # Only ever delete a reminder that belongs to whoever pressed the button.
    if not any(r["id"] == reminder_id for r in items):
        return
    cancel_reminder_job(reminder_id)
    await asyncio.to_thread(_delete_reminder, reminder_id)
    await _render_reminders(update, context, edit=True)


# ---------------------- /language COMMAND (change the bot's OWN interface language) ----------------------
async def language_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        ui(context, "language_menu_prompt"),
        reply_markup=build_ui_language_keyboard(),
    )


# ---------------------- /translate COMMAND (quick one-off translation or enabling translate mode) ----------------------
def _resolve_translate_target(raw: str) -> str:
    """Turn a user-typed language token into either a known LANGUAGES code
    or, if it doesn't match anything in the dict, the raw text itself (the
    AI translator can work from a plain language name it's never seen a
    code for, e.g. 'swahili')."""
    lowered = raw.strip().lower()
    if lowered in LANGUAGES:
        return lowered
    for code, info in LANGUAGES.items():
        if info["name"].lower() == lowered:
            return code
    return raw.strip()


async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args or []
    if not args:
        await update.message.reply_text(
            ui(context, "translate_menu_prompt"),
            reply_markup=build_paginated_lang_keyboard("translate_lang", page=0, include_auto=True),
        )
        return

    target = _resolve_translate_target(args[0])
    remaining_text = " ".join(args[1:]).strip()

    if not remaining_text:
        # Just "/translate <lang>" - no text yet, so switch into continuous
        # translate mode targeting that language, same as picking it from the menu.
        await update.message.reply_text(activate_translate_mode(context, target))
        return

    # "/translate <lang> <text>" - a one-off translation, doesn't touch ai_mode/history at all.
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    try:
        translated = await translate_text(remaining_text, target)
    except Exception as e:
        print(f"[translate_command] translation failed: {e}")
        await update.message.reply_text(ui(context, "translate_fail"))
        return
    await send_ai_reply(update, context, translated)


# ---------------------- FREE WEB SEARCH (DuckDuckGo, no API key, no billing) ----------------------
def _ddg_search_sync(query: str, max_results: int = 5) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"(جستجوی اینترنتی با خطا مواجه شد: {e})"

    if not results:
        return "(نتیجه‌ای پیدا نشد.)"

    lines = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        lines.append(f"{i}. {title}\n{body}\nمنبع: {href}")
    return "\n\n".join(lines)


async def web_search(query: str, max_results: int = 5) -> str:
    # DDGS() does blocking network calls, so push it to a thread just like
    # the Groq/Gemini calls, to avoid blocking the bot's event loop.
    return await asyncio.to_thread(_ddg_search_sync, query, max_results)


# Phrases that mean "go search the web for this" when they show up in a
# message, regardless of which AI mode (Groq/Gemini) the user is currently
# in. Kept lowercase for the English ones since we lowercase user_text before
# checking; Persian doesn't have a case distinction.
SEARCH_TRIGGERS = [
    "سرچ کن",
    "جستجو کن",
    "جست و جو کن",
    "search it",
    "search this",
    "search for",
    "google it",
    "look it up",
]


def wants_web_search(text: str) -> bool:
    lowered = text.lower()
    return any(trigger in text or trigger in lowered for trigger in SEARCH_TRIGGERS)


# ---------------------- VISIT A SPECIFIC PAGE (like clicking a search result) ----------------------
def _fetch_page_sync(url: str, max_chars: int = 4000) -> str:
    try:
        resp = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (compatible; TelegramBot/1.0)"},
        )
        resp.raise_for_status()
    except Exception as e:
        return f"(نتونستم این صفحه رو باز کنم: {e})"

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = " ".join(soup.stripped_strings)
    if len(text) > max_chars:
        text = text[:max_chars] + " ...(متن بریده شد)"
    return text or "(این صفحه متنی برای خوندن نداشت.)"


async def fetch_page(url: str) -> str:
    return await asyncio.to_thread(_fetch_page_sync, url)


# ---------------------- IMAGE SEARCH (DuckDuckGo images, no API key) ----------------------
# "Ask about a bacterium, get a set of real photos back" - the model picks a
# proper English search query itself via the search_images tool, so this works
# like a real assistant researching images rather than a dumb keyword lookup.
MAX_IMAGES_PER_BATCH = 3  # was 6 - kept low so "show me a picture" doesn't flood the chat
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # skip anything huge - Telegram rejects it anyway
TOTAL_IMAGES_PER_REPLY = 4  # hard cap across every batch in a single AI reply


def _ddg_images_sync(query: str, max_results: int = 5) -> list:
    try:
        with DDGS() as ddgs:
            # type_image="photo" filters out clipart/icons/line-art, which is
            # where most of the "totally unrelated picture" results come
            # from; safesearch="moderate" also seems to correlate with more
            # on-topic results than the unfiltered default.
            try:
                results = list(
                    ddgs.images(
                        query,
                        max_results=max_results,
                        safesearch="moderate",
                        type_image="photo",
                    )
                )
            except TypeError:
                # Older/newer ddgs versions may not accept these kwargs.
                results = list(ddgs.images(query, max_results=max_results))
    except Exception as e:
        print(f"[images] search failed for '{query}': {e}")
        return []
    cleaned = []
    query_words = {w for w in re.findall(r"[a-zA-Z]{3,}", query.lower())}
    scored = []
    for r in results:
        url = r.get("image") or r.get("thumbnail")
        if not url:
            continue
        title = (r.get("title") or "").strip()[:150]
        # Cheap relevance check: how many of the query's words actually show
        # up in the result's own title. Results with zero overlap are pushed
        # to the back instead of dropped outright, since some genuinely
        # relevant images just have unhelpful titles.
        title_words = set(re.findall(r"[a-zA-Z]{3,}", title.lower()))
        overlap = len(query_words & title_words) if query_words else 0
        scored.append((overlap, {"url": url, "title": title, "source": r.get("url") or ""}))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    cleaned = [item for _, item in scored]
    return cleaned


async def search_images(query: str, count: int = 4) -> list:
    count = max(1, min(int(count or 4), MAX_IMAGES_PER_BATCH))
    # Ask for more than needed: with per-image fallback (see send_image_batch)
    # a handful of dead/blocked links no longer sinks the whole batch, but we
    # still want enough candidates left over to reach `count` real sends.
    return await asyncio.to_thread(_ddg_images_sync, query, count + 6)


def _download_image_sync(url: str) -> bytes | None:
    try:
        parsed_root = re.match(r"^(https?://[^/]+)", url)
        headers = {
            # Sites that hotlink-protect their images tend to check both of
            # these; DDG's own generic UA/no-referer request gets a 403 from
            # a lot of hosts (this was the actual cause of "found the search
            # results but couldn't send a picture").
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Referer": parsed_root.group(1) + "/" if parsed_root else "https://duckduckgo.com/",
        }
        resp = requests.get(url, timeout=12, stream=True, headers=headers)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type and not url.lower().split("?")[0].endswith(
            (".jpg", ".jpeg", ".png", ".webp", ".gif")
        ):
            return None
        data = b""
        for chunk in resp.iter_content(64 * 1024):
            data += chunk
            if len(data) > MAX_IMAGE_BYTES:
                return None
        return data or None
    except Exception:
        return None


async def send_image_batch(
    bot, chat_id: int, batch: dict, reply_to_message_id: int | None = None
) -> int:
    """Send one search_images result set as real Telegram photos.

    Two strategies, tried in order per image:
      1. Hand Telegram the URL directly and let ITS servers fetch it. This
         works for most hosts and is instant - no download through Render's
         IP at all, which a lot of image hosts throttle or block outright.
      2. If Telegram can't fetch it (private/broken/hotlink-protected host),
         download it ourselves with browser-like headers and send the raw
         bytes instead.
    Sent one at a time rather than as a single album: an album is all-or-
    nothing, so one bad link used to silently sink every other photo in the
    batch - which is exactly what was happening before.
    """
    query = batch.get("query", "")
    candidates = batch.get("results", [])[: MAX_IMAGES_PER_BATCH + 6]
    wanted = batch.get("count", MAX_IMAGES_PER_BATCH)

    sent = 0
    for i, result in enumerate(candidates):
        if sent >= wanted:
            break
        caption = f"🖼 {query}" if sent == 0 else None
        reply_id = reply_to_message_id if sent == 0 else None

        try:
            await bot.send_photo(
                chat_id=chat_id,
                photo=result["url"],
                caption=caption,
                reply_to_message_id=reply_id,
            )
            sent += 1
            continue
        except Exception as e:
            print(f"[images] direct-URL send failed for {result['url']}: {e}")

        data = await asyncio.to_thread(_download_image_sync, result["url"])
        if not data:
            continue
        try:
            await bot.send_photo(
                chat_id=chat_id,
                photo=io.BytesIO(data),
                caption=caption,
                reply_to_message_id=reply_id,
            )
            sent += 1
        except Exception as e:
            print(f"[images] fallback download-send also failed for {result['url']}: {e}")
            continue

    return sent


# ---------------------- REAL CURRENT TIME (deterministic, no guessing) ----------------------
def get_current_datetime(timezone: str) -> str:
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        return (
            f"منطقه‌ی زمانی '{timezone}' معتبر نیست. باید یه نام استاندارد IANA "
            "باشه، مثلاً Asia/Tehran برای ایران یا Europe/Amsterdam برای هلند."
        )
    now = datetime.now(tz)
    return now.strftime(f"%Y-%m-%d %H:%M:%S %A ({timezone})")


# Direct, deterministic handling of plain "what time/date is it" questions.
# This completely bypasses the LLM and its tool-calling loop for the single
# most common case that kept breaking - no model call, no tools, no way for
# a Groq-side tool-calling quirk to fail it. Anything more open-ended
# ("چند ساعت تا ظهر ایران مونده؟") still goes through the normal agent, which
# can call get_current_datetime itself as a tool.
TIME_QUESTION_TRIGGERS = [
    "ساعت", "چه ساعتی", "چند ساعته", "الان ساعت", "امروز چندمه", "تاریخ امروز", "چه تاریخی",
    "what time", "current time", "what's the time", "whats the time", "current date", "today's date", "what date",
]

LOCATION_TO_TIMEZONE = {
    "ایران": "Asia/Tehran", "تهران": "Asia/Tehran", "iran": "Asia/Tehran", "tehran": "Asia/Tehran",
    "دبی": "Asia/Dubai", "امارات": "Asia/Dubai", "dubai": "Asia/Dubai", "uae": "Asia/Dubai",
    "ترکیه": "Europe/Istanbul", "استانبول": "Europe/Istanbul", "turkey": "Europe/Istanbul", "istanbul": "Europe/Istanbul",
    "لندن": "Europe/London", "انگلیس": "Europe/London", "london": "Europe/London",
    "آلمان": "Europe/Berlin", "germany": "Europe/Berlin", "berlin": "Europe/Berlin",
    "آمریکا": "America/New_York", "usa": "America/New_York", "new york": "America/New_York",
}


def detect_plain_time_query(text: str) -> str | None:
    lowered = text.lower()
    if not any(trigger in text or trigger in lowered for trigger in TIME_QUESTION_TRIGGERS):
        return None
    for place, tz in LOCATION_TO_TIMEZONE.items():
        if place in text or place in lowered:
            return tz
    # A time question with no place named - default to Iran, since that's
    # this bot's primary audience/language.
    return "Asia/Tehran"


def format_time_reply(timezone: str) -> str:
    return f"الان زمان تقریبی: {get_current_datetime(timezone)}"


# Tool definitions the Groq model can call on its own, in OpenAI-compatible
# function-calling format.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Free web search (DuckDuckGo) for current information, news, facts, or anything that might not be in your training data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch and read the full text of a specific web page URL, like clicking into a search result for more detail than the snippet gives.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The full URL to open."}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the real, current date and time for a given IANA timezone. Always use this for any 'what time/date is it' question instead of guessing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone name, e.g. 'Asia/Tehran' for Iran, 'Europe/Amsterdam' for the Netherlands, 'America/New_York' for US Eastern time.",
                    }
                },
                "required": ["timezone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": (
                "Permanently save one short fact about the user (name, "
                "preference, ongoing situation, anything they'd expect you "
                "to still know next time) so it survives a bot restart. "
                "Call this immediately when the user shares something worth "
                "remembering long-term - don't just say you'll remember it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The short fact to remember, in the user's own language, e.g. 'اسم کاربر علی است' or 'User's name is Ali'.",
                    }
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_images",
            "description": (
                "Search the web for real photos and send them to the user as "
                "actual Telegram images. Use this for any 'show me / picture of "
                "/ what does X look like / عکس X' request. Write a specific, "
                "descriptive English query (e.g. 'bacteria under microscope "
                "photograph', not just 'bacteria') - a vague query is the main "
                "reason irrelevant pictures get sent. Call this AT MOST ONCE per "
                "reply, for one subject at a time; do not call it repeatedly to "
                "cover different aspects, since every call sends more real "
                "photos to the user's chat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A specific, descriptive image search query, preferably in English for better results.",
                    },
                    "count": {
                        "type": "integer",
                        "description": (
                            f"How many images to send (1-{MAX_IMAGES_PER_BATCH}). "
                            "If the user gave a number ('one picture', 'دو تا عکس'), use exactly that "
                            "number. Otherwise default to 1-2 - never send more than the user "
                            "actually needs to see."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": (
                "Schedule a reminder: the bot will message the user at the given "
                "time with the given text. Use delay_minutes for relative times "
                "('in 20 minutes'), or at_time ('HH:MM', 24-hour) for clock "
                "times, optionally with repeat for recurring reminders."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "What to remind the user about, in their own language, phrased as they'd want to read it later.",
                    },
                    "delay_minutes": {
                        "type": "number",
                        "description": "Fire this many minutes from now. Use this for 'in X minutes/hours' requests.",
                    },
                    "at_time": {
                        "type": "string",
                        "description": "Clock time in 'HH:MM' 24-hour format; fires at the next occurrence of that time. Use instead of delay_minutes.",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone for at_time, e.g. 'Asia/Tehran'. Defaults to Asia/Tehran if unknown.",
                    },
                    "repeat": {
                        "type": "string",
                        "description": "One of 'none', 'hourly', 'daily', 'weekly'. Use 'daily' for 'every morning/day' style requests.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_fact",
            "description": "Remove a previously remembered fact about the user that is no longer true or that the user asked you to forget.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The exact fact text (as it was remembered) to remove.",
                    }
                },
                "required": ["fact"],
            },
        },
    },
]


async def execute_tool(tool_call, user_id: int, tool_ctx: dict | None = None) -> str:
    """tool_ctx carries the things a tool needs that aren't in the model's
    arguments: which chat this is, and a list to drop image batches into so
    ai_handler can send them after the text reply."""
    name = tool_call.function.name
    tool_ctx = tool_ctx if tool_ctx is not None else {}
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}

    if name == "search_images":
        query = (args.get("query") or "").strip()
        if not query:
            return "(هیچ عبارتی برای جستجوی عکس داده نشد.)"
        count = args.get("count") or 2
        results = await search_images(query, count)
        if not results:
            return f"(برای «{query}» عکسی پیدا نشد.)"
        sink = tool_ctx.get("images")
        if sink is not None and len(sink) < 3:  # at most 3 albums per reply
            sink.append(
                {"query": query, "results": results, "count": max(1, min(int(count), MAX_IMAGES_PER_BATCH))}
            )
        titles = "; ".join(r["title"] for r in results[:6] if r["title"])
        return (
            f"{len(results)} عکس برای «{query}» پیدا شد و همین الان برای کاربر ارسال می‌شه. "
            f"عنوان‌ها: {titles or 'بدون عنوان'}. "
            "حالا درباره‌ی موضوع توضیح بده - نیازی نیست بگی «عکس رو فرستادم» یا لینک بذاری."
        )

    if name == "set_reminder":
        text = (args.get("text") or "").strip()
        chat_id = tool_ctx.get("chat_id")
        if not text or chat_id is None:
            return "(یادآوری ثبت نشد: متن یا چت مشخص نیست.)"
        tz = args.get("timezone") or "Asia/Tehran"
        repeat = (args.get("repeat") or "none").lower()
        due_ts = None
        if args.get("delay_minutes") is not None:
            try:
                due_ts = time.time() + max(0.2, float(args["delay_minutes"])) * 60
            except (TypeError, ValueError):
                due_ts = None
        if due_ts is None and args.get("at_time"):
            due_ts = next_clock_time(str(args["at_time"]), tz)
        if due_ts is None:
            return "(زمان یادآوری مشخص نیست - از کاربر بپرس دقیقاً کِی یادآوری بشه.)"
        if BOT_APP is None or BOT_APP.job_queue is None:
            return "(قابلیت یادآوری روی این سرور فعال نیست - صادقانه به کاربر بگو که نتونستی ثبتش کنی.)"
        reminder = await add_reminder(user_id, chat_id, text, due_ts, repeat, tz)
        if reminder is None:
            return f"(کاربر به سقف {MAX_REMINDERS_PER_USER} یادآوری رسیده - باید اول با /reminders یکی رو حذف کنه.)"
        return f"یادآوری ثبت شد برای {format_reminder_time(reminder)} با متن: {reminder['text']}"

    if name == "web_search":
        return await web_search(args.get("query", ""))
    if name == "fetch_page":
        return await fetch_page(args.get("url", ""))
    if name == "get_current_datetime":
        return get_current_datetime(args.get("timezone", "UTC"))
    if name == "remember_fact":
        return await remember_fact_tool(user_id, args.get("fact", ""))
    if name == "forget_fact":
        return await forget_fact_tool(user_id, args.get("fact", ""))
    return f"ابزار ناشناخته: {name}"


# ---------------------- GROQ CALL (with real tool use) ----------------------
MAX_TOOL_STEPS = 6  # how many search/fetch/time round-trips the model can make per message

# Groq's free tier caps openai/gpt-oss-120b at 8,000 tokens PER MINUTE (this
# is per Groq's own rate-limit table, not something we control). That cap
# covers the whole request - system prompt + tool schemas + every message
# still sitting in this user's history + the reply itself. MAX_HISTORY above
# only limits the *number* of turns kept, not their size, so a chat can
# quietly sit right at that ceiling; then even a couple of extra lines from
# the user tips a single request over 8,000 tokens and Groq answers with a
# 413 "request too large" error - which is the "works for 2 lines, breaks
# after that" symptom. GROQ_TOKEN_BUDGET keeps what we actually *send* well
# under the ceiling, leaving headroom for the tool schemas and the model's
# own reply (which count against the same per-minute budget).
GROQ_TOKEN_BUDGET = 4500


def _estimate_tokens(text: str) -> int:
    # No tokenizer dependency - just a deliberately generous estimate (~3
    # chars/token). Non-Latin scripts like Persian/Arabic tend to tokenize
    # heavier per character than English, so erring high here is safer than
    # under-counting and tripping the same 413 we're trying to avoid.
    return max(1, len(text) // 3)


def _trim_messages_to_budget(messages: list, budget: int = GROQ_TOKEN_BUDGET) -> list:
    """Keep any system message(s) plus as many of the most-recent messages
    as fit in `budget` estimated tokens, dropping the oldest ones first.
    Always keeps at least the single most recent message even if it alone
    is bigger than the budget - there's nothing smaller we can send."""
    if not messages:
        return messages
    system_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]

    used = sum(_estimate_tokens(m.get("content") or "") for m in system_msgs)
    kept = []
    for m in reversed(other_msgs):
        cost = _estimate_tokens(m.get("content") or "")
        if kept and used + cost > budget:
            break
        used += cost
        kept.append(m)
    kept.reverse()
    return system_msgs + kept

# Nudge used for the dedicated search button. This is a *soft* instruction,
# not a hard API-level constraint: gpt-oss on Groq doesn't reliably obey a
# forced tool_choice (it can refuse to call the forced tool, or keep trying
# to call one even when told "none"), and either mismatch causes a 400. A
# plain instruction message is more reliable in practice.
FORCE_SEARCH_HINT = {
    "role": "system",
    "content": (
        "کاربر از حالت «جستجوی اینترنتی» استفاده می‌کنه. مگر اینکه پیامش کاملاً "
        "بی‌ربط به اطلاعات بیرونی باشه (مثلاً فقط تشکر کردن)، حتماً قبل از "
        "پاسخ نهایی از ابزار web_search استفاده کن."
    ),
}


FORCE_IMAGES_HINT = {
    "role": "system",
    "content": (
        "کاربر از حالت «جستجوی عکس» استفاده می‌کنه. پیامش رو به عنوان موضوعی که "
        "می‌خواد ببینه در نظر بگیر: حتماً ابزار search_images رو با یک عبارت "
        "دقیق و توصیفی (ترجیحاً انگلیسی) و با count بین ۳ تا ۶ صدا بزن، و اگه "
        "موضوع چند جنبه‌ی متفاوت داره بیشتر از یک بار با عبارت‌های مختلف. بعد "
        "یک توضیح کوتاه و مفید درباره‌ی چیزی که تو عکس‌هاست بنویس."
    ),
}


async def run_groq_agent(
    history: list,
    user_id: int,
    model: str = GROQ_MODEL,
    force_search: bool = False,
    force_images: bool = False,
    tool_ctx: dict | None = None,
) -> str:
    # The user's actual question, so we can rebuild a clean prompt later.
    original_user_text = history[-1]["content"] if history and history[-1].get("role") == "user" else ""

    # Work on a local copy: the back-and-forth tool-call turns are only needed
    # to produce this one reply and are never saved into the persisted
    # per-user history (that would bloat it fast and isn't needed later).
    # Trimmed to Groq's per-minute token budget first - see GROQ_TOKEN_BUDGET
    # above for why this is necessary even though MAX_HISTORY already caps
    # the number of turns.
    working = _trim_messages_to_budget(list(history))
    if force_search:
        working = working + [FORCE_SEARCH_HINT]
    if force_images:
        working = working + [FORCE_IMAGES_HINT]

    gathered_info = []  # plain-text notes on what each tool call found

    for step in range(MAX_TOOL_STEPS):
        try:
            response = await asyncio.to_thread(
                groq_client.chat.completions.create,
                model=model,
                messages=working,
                tools=TOOLS,
                tool_choice="auto",
            )
        except APIStatusError as e:
            status = getattr(e, "status_code", None)
            if status in (413, 429):
                # 413 = this single request (system prompt + tool schemas +
                # history) is bigger than the 8,000 TPM ceiling by itself.
                # 429 = the account has burned through its per-minute budget
                # across recent calls. Either way, retrying with the exact
                # same payload will just fail again - so on the FIRST time
                # this happens in this loop, cut the context hard and retry
                # once; only give up if that still doesn't fit.
                print(f"[run_groq_agent] Groq {status} on step {step}, trimming and retrying: {e}")
                smaller = _trim_messages_to_budget(working, budget=GROQ_TOKEN_BUDGET // 2)
                if smaller == working:
                    # Nothing left to trim (e.g. the user's own message is
                    # simply too long on its own) - no point retrying.
                    return (
                        "پیامت برای این مدل خیلی طولانیه و سقف حجم مجاز رو رد می‌کنه. "
                        "میشه کوتاه‌ترش کنی یا در چند پیام جدا بفرستیش؟"
                    )
                working = smaller
                try:
                    response = await asyncio.to_thread(
                        groq_client.chat.completions.create,
                        model=model,
                        messages=working,
                        tools=TOOLS,
                        tool_choice="auto",
                    )
                except APIStatusError as e2:
                    print(f"[run_groq_agent] retry after trim also failed: {e2}")
                    if getattr(e2, "status_code", None) == 429:
                        return "الان درخواست‌ها به سقف مجاز هوش مصنوعی (Groq) رسیده. یکی-دو دقیقه دیگه دوباره امتحان کن."
                    return "ببخشید، الان نمی‌تونم جواب بدم. یکم بعد دوباره امتحان کن."
                msg = response.choices[0].message
                if not msg.tool_calls:
                    return msg.content or ""
                # fall through to normal tool-call handling below with the
                # smaller `working` list
            elif isinstance(e, BadRequestError):
                # Groq's gpt-oss endpoint can get into a bad state once a
                # transcript has several tool-call turns in it - it may keep
                # trying to emit more tool calls almost regardless of what we
                # ask for next, and any mismatch throws a 400. Once that
                # happens, stop asking it for more tools and go straight to
                # the clean, tool-free final answer below instead of
                # retrying the same broken pattern.
                print(f"[run_groq_agent] tool round failed, giving up on tools: {e}")
                break
            else:
                print(f"[run_groq_agent] unexpected Groq error on step {step}: {e}")
                raise

        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content or ""

        working.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        for tc in msg.tool_calls:
            result = await execute_tool(tc, user_id, tool_ctx)
            gathered_info.append(f"[{tc.function.name}] {result}")
            working.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": result,
                }
            )

    # Final answer: build a completely FRESH, tool-free message list instead
    # of re-sending `working` (which contains assistant/tool turns tied to
    # tool_calls). Re-sending that transcript without offering tools again
    # is exactly what was triggering the 400s, regardless of tool_choice
    # value - so we never do that. Instead, fold whatever the tools found
    # into one plain user message on top of the original clean history.
    if gathered_info:
        info_block = "\n\n".join(gathered_info[-6:])  # keep it bounded
        final_prompt = (
            f"{original_user_text}\n\n"
            "اطلاعاتی که از ابزارها (جستجو/خواندن صفحه/زمان واقعی) به دست اومده:\n"
            f"{info_block}\n\n"
            "با توجه به این اطلاعات (در صورت مرتبط بودن)، مستقیم و طبیعی جواب بده - "
            "به همون زبونی که کاربر پیامش رو نوشته (فارسی، انگلیسی، ژاپنی، یا هر زبان دیگه)."
        )
    else:
        final_prompt = original_user_text

    final_messages = _trim_messages_to_budget(
        history[:-1] + [{"role": "user", "content": final_prompt}]
    )

    try:
        response = await asyncio.to_thread(
            groq_client.chat.completions.create,
            model=model,
            messages=final_messages,
        )
        return response.choices[0].message.content or "ببخشید، نتونستم جواب بدم. دوباره امتحان کن."
    except APIStatusError as e:
        status = getattr(e, "status_code", None)
        if status in (413, 429):
            # Same size/rate ceiling as above, but here there's no cheaper
            # fallback left to try - trim once more and retry, otherwise
            # give a message that actually explains what happened instead
            # of a generic failure.
            smaller = _trim_messages_to_budget(final_messages, budget=GROQ_TOKEN_BUDGET // 2)
            if smaller != final_messages:
                try:
                    response = await asyncio.to_thread(
                        groq_client.chat.completions.create,
                        model=model,
                        messages=smaller,
                    )
                    return response.choices[0].message.content or "ببخشید، نتونستم جواب بدم. دوباره امتحان کن."
                except Exception as e2:
                    print(f"[run_groq_agent] final trimmed retry also failed: {e2}")
            if status == 429:
                return "الان درخواست‌ها به سقف مجاز هوش مصنوعی (Groq) رسیده. یکی-دو دقیقه دیگه دوباره امتحان کن."
            return (
                "پیامت (یا تاریخچه‌ی گفتگو) برای این مدل خیلی طولانیه. "
                "میشه کوتاه‌ترش کنی یا با /groq یه گفتگوی تازه شروع کنی؟"
            )
        print(f"[run_groq_agent] final plain-text attempt also failed: {e}")
        return "ببخشید، الان نمی‌تونم جواب بدم. یکم بعد دوباره امتحان کن."
    except Exception as e:
        print(f"[run_groq_agent] final plain-text attempt also failed: {e}")
        return "ببخشید، الان نمی‌تونم جواب بدم. یکم بعد دوباره امتحان کن."


# ---------------------- GEMINI CALL ----------------------
async def call_gemini(history: list, system_prompt: str) -> str:
    # Gemini uses role "model" instead of "assistant", and takes a separate
    # system_instruction field rather than a system role inside the messages.
    contents = [
        {"role": turn["role"], "parts": [{"text": turn["content"]}]}
        for turn in history
    ]
    response = await asyncio.to_thread(
        gemini_client.models.generate_content,
        model=GEMINI_MODEL,
        contents=contents,
        config=genai_types.GenerateContentConfig(system_instruction=system_prompt),
    )
    return response.text


# ---------------------- LANGUAGE SUPPORT (shared by translation menu + TTS voice picking) ----------------------
# code -> button label (with flag), display name (for prompts), and the
# edge-tts neural voice used when speaking that language out loud. edge-tts
# (Microsoft Edge's free TTS service, no API key) has solid native voices
# for all of these - including Japanese - unlike Groq's own TTS (Orpheus),
# which only covers English/Arabic, or gTTS, whose Persian voice is unreliable.
LANGUAGES = {
    "fa":    {"label": "🇮🇷 فارسی",     "name": "Persian",    "voice": "fa-IR-DilaraNeural"},
    "en":    {"label": "🇬🇧 English",    "name": "English",    "voice": "en-US-AriaNeural"},
    "ar":    {"label": "🇸🇦 العربية",    "name": "Arabic",     "voice": "ar-SA-ZariyahNeural"},
    "ja":    {"label": "🇯🇵 日本語",     "name": "Japanese",   "voice": "ja-JP-NanamiNeural"},
    "ko":    {"label": "🇰🇷 한국어",     "name": "Korean",     "voice": "ko-KR-SunHiNeural"},
    "zh-cn": {"label": "🇨🇳 中文",       "name": "Chinese",    "voice": "zh-CN-XiaoxiaoNeural"},
    "fr":    {"label": "🇫🇷 Français",   "name": "French",     "voice": "fr-FR-DeniseNeural"},
    "de":    {"label": "🇩🇪 Deutsch",    "name": "German",     "voice": "de-DE-KatjaNeural"},
    "es":    {"label": "🇪🇸 Español",    "name": "Spanish",    "voice": "es-ES-ElviraNeural"},
    "ru":    {"label": "🇷🇺 Русский",    "name": "Russian",    "voice": "ru-RU-SvetlanaNeural"},
    "tr":    {"label": "🇹🇷 Türkçe",     "name": "Turkish",    "voice": "tr-TR-EmelNeural"},
    "it":    {"label": "🇮🇹 Italiano",   "name": "Italian",    "voice": "it-IT-ElsaNeural"},
    "pt":    {"label": "🇵🇹 Português",  "name": "Portuguese", "voice": "pt-BR-FranciscaNeural"},
    "hi":    {"label": "🇮🇳 हिन्दी",      "name": "Hindi",      "voice": "hi-IN-SwaraNeural"},
    "ur":    {"label": "🇵🇰 اردو",       "name": "Urdu",       "voice": "ur-PK-UzmaNeural"},
    "nl":    {"label": "🇳🇱 Nederlands", "name": "Dutch",      "voice": "nl-NL-ColetteNeural"},
    # -------- extra languages (added on top of the original 16 above) --------
    "az":    {"label": "🇦🇿 Azərbaycan",      "name": "Azerbaijani", "voice": "az-AZ-BanuNeural"},
    "uk":    {"label": "🇺🇦 Українська",      "name": "Ukrainian",   "voice": "uk-UA-PolinaNeural"},
    "pl":    {"label": "🇵🇱 Polski",          "name": "Polish",      "voice": "pl-PL-ZofiaNeural"},
    "sv":    {"label": "🇸🇪 Svenska",         "name": "Swedish",     "voice": "sv-SE-SofieNeural"},
    "el":    {"label": "🇬🇷 Ελληνικά",        "name": "Greek",       "voice": "el-GR-AthinaNeural"},
    "he":    {"label": "🇮🇱 עברית",          "name": "Hebrew",      "voice": "he-IL-HilaNeural"},
    "vi":    {"label": "🇻🇳 Tiếng Việt",      "name": "Vietnamese",  "voice": "vi-VN-HoaiMyNeural"},
    "th":    {"label": "🇹🇭 ไทย",             "name": "Thai",        "voice": "th-TH-PremwadeeNeural"},
    "id":    {"label": "🇮🇩 Bahasa Indonesia", "name": "Indonesian", "voice": "id-ID-GadisNeural"},
    "bn":    {"label": "🇧🇩 বাংলা",           "name": "Bengali",     "voice": "bn-BD-NabanitaNeural"},
    "ka":    {"label": "🇬🇪 ქართული",        "name": "Georgian",    "voice": "ka-GE-EkaNeural"},
    "kk":    {"label": "🇰🇿 Қазақша",         "name": "Kazakh",      "voice": "kk-KZ-AigulNeural"},
    "hu":    {"label": "🇭🇺 Magyar",          "name": "Hungarian",   "voice": "hu-HU-NoemiNeural"},
    "cs":    {"label": "🇨🇿 Čeština",         "name": "Czech",       "voice": "cs-CZ-VlastaNeural"},
    "ro":    {"label": "🇷🇴 Română",          "name": "Romanian",    "voice": "ro-RO-AlinaNeural"},
    "da":    {"label": "🇩🇰 Dansk",           "name": "Danish",      "voice": "da-DK-ChristelNeural"},
    "fi":    {"label": "🇫🇮 Suomi",           "name": "Finnish",     "voice": "fi-FI-NooraNeural"},
    "no":    {"label": "🇳🇴 Norsk",           "name": "Norwegian",   "voice": "nb-NO-PernilleNeural"},
    "sw":    {"label": "🇰🇪 Kiswahili",       "name": "Swahili",     "voice": "sw-KE-ZuriNeural"},
    "ms":    {"label": "🇲🇾 Bahasa Melayu",   "name": "Malay",       "voice": "ms-MY-YasminNeural"},
    "uz":    {"label": "🇺🇿 Oʻzbekcha",       "name": "Uzbek",       "voice": "uz-UZ-MadinaNeural"},
    "ps":    {"label": "🇦🇫 پښتو",           "name": "Pashto",      "voice": "ps-AF-LatifaNeural"},
    "ta":    {"label": "🇮🇳 தமிழ்",           "name": "Tamil",       "voice": "ta-IN-PallaviNeural"},
    "sr":    {"label": "🇷🇸 Српски",          "name": "Serbian",     "voice": "sr-RS-SophieNeural"},
    "bg":    {"label": "🇧🇬 Български",       "name": "Bulgarian",   "voice": "bg-BG-KalinaNeural"},
    "hr":    {"label": "🇭🇷 Hrvatski",        "name": "Croatian",    "voice": "hr-HR-GabrijelaNeural"},
    "sk":    {"label": "🇸🇰 Slovenčina",      "name": "Slovak",      "voice": "sk-SK-ViktoriaNeural"},
    "fil":   {"label": "🇵🇭 Filipino",        "name": "Filipino",    "voice": "fil-PH-BlessicaNeural"},
}
DEFAULT_TTS_VOICE = "en-US-AriaNeural"

# The original, smaller set of languages - used for the compact "quick
# translate" keyboard attached under every AI reply (so that keyboard stays
# short and doesn't force the user to page through 40+ buttons just to
# translate a single message). The full LANGUAGES dict above (all 40+) is
# used everywhere else: the main /translate menu, language auto-detection,
# and TTS voice picking.
QUICK_LANGUAGES = ["fa", "en", "ar", "ja", "ko", "zh-cn", "fr", "de", "es", "ru", "tr", "it", "pt", "hi", "ur", "nl"]

# The (smaller, curated) set of languages the bot's OWN interface text -
# menus, confirmations, error messages - can be displayed in. Kept separate
# and small on purpose: these strings are hand-translated (see UI_STRINGS
# below), not machine-translated, so this list only grows when someone adds
# a full translation for that language to UI_STRINGS.
UI_LANGUAGES = ["fa", "en", "ar", "tr", "ru", "fr", "de", "es"]

# ---------------------- BOT UI LANGUAGE (interface text, separate from translation) ----------------------
# All of the bot's OWN messages (menus, confirmations, error strings) live
# here, one full translation per supported UI language. This is separate
# from the *translation* feature above: /translate can turn user text into
# 40+ languages, but the bot's own buttons/menus are only fully translated
# into the languages listed in UI_LANGUAGES, since these are hand-written,
# not machine-translated.
UI_STRINGS = {
    "fa": {
        "start_greeting": "خوش آمدید! من ربات شخصی شما هستم، هرطور که مایلید ازم استفاده کنید.",
        "btn_ai_groq": "🤖 AI Groq",
        "btn_ai_gemini": "🤖 AI Gemini",
        "btn_web_search": "🔎 جستجوی اینترنتی",
        "btn_translate_menu": "🌐 ترجمه چندزبانه",
        "help_text": (
            "این کارایی هستن که از دستم برمیاد:\n\n"
            "🤖 هوش مصنوعی (Groq یا Gemini) - چت آزاد و پاسخ به سوالات\n"
            "🔎 جستجوی اینترنتی - برای گرفتن اطلاعات به‌روز\n"
            "🌐 ترجمه چندزبانه - ترجمه هر متنی به بیش از ۴۰ زبون\n"
            "🎙️ تبدیل متن به صدا و صدا به متن\n\n"
            "دستورات:\n"
            "/start - نمایش منوی اصلی\n"
            "/stop - خاموش کردن حالت هوش مصنوعی\n"
            "/help - همین راهنما\n"
            "/groq - فعال کردن مستقیم حالت Groq\n"
            "/gemini - فعال کردن مستقیم حالت Gemini\n"
            "/websearch - فعال کردن مستقیم حالت جستجوی اینترنتی\n"
            "/translate [زبان] [متن] - ترجمهٔ سریع یا فعال کردن حالت ترجمه\n"
            "/language - تغییر زبان رابط ربات\n\n"
            "برای شروع فقط /start رو بزن و از دکمه‌ها انتخاب کن."
        ),
        "mode_groq_on": "حالت هوش مصنوعی (Groq) فعال شد🤖!",
        "mode_gemini_on": "حالت هوش مصنوعی (Gemini) فعال شد🤖!",
        "mode_search_on": "حالت جستجوی اینترنتی (Groq) فعال شد🔎!",
        "translate_menu_prompt": "زبان مقصد رو انتخاب کن؛ از این به بعد هر متنی بفرستی، به همون زبون ترجمه می‌کنم:",
        "translate_auto_on": "حالت ترجمه خودکار فعال شد🌐 (فارسی⇄انگلیسی). هر متنی بفرستی ترجمه می‌کنم.",
        "translate_target_on": (
            "حالت ترجمه فعال شد🌐! هر متنی به هر زبونی بفرستی، به {target} ترجمه می‌کنم.\n"
            "برای عوض کردن زبان مقصد دوباره /start رو بزن یا از دستور /translate استفاده کن."
        ),
        "stop_ai_msg": "حالت هوش مصنوعی خاموش شد. تا وقتی /start یا /groq رو نزنی به پیام‌هات جواب نمی‌دم.",
        "translate_fail": "ترجمه با خطا مواجه شد. لطفاً دوباره امتحان کن.",
        "ai_call_fail": "یه خطا توی گرفتن جواب از هوش مصنوعی پیش اومد (ممکنه موقتی باشه). لطفاً دوباره امتحان کن.",
        "global_error": "یه خطای غیرمنتظره پیش اومد. لطفاً دوباره امتحان کن.",
        "msg_translate_unavailable": "این پیام دیگه برای ترجمه در دسترس نیست.",
        "tts_unavailable": "این پیام دیگه برای تبدیل به صدا در دسترس نیست.",
        "tts_fail": "تبدیل به صدا با خطا مواجه شد. لطفاً دوباره امتحان کن.",
        "voice_transcribe_fail": "نتونستم صدا رو تبدیل به متن کنم. لطفاً دوباره امتحان کن یا پیامت رو تایپ کن.",
        "voice_empty": "متنی از پیام صوتی استخراج نشد. لطفاً دوباره امتحان کن.",
        "btn_cancel": "❌ لغو",
        "btn_tts": "🔊 تبدیل به صدا",
        "btn_text": "📝 بازگرداندن متن",
        "btn_translate": "🌐 ترجمه",
        "btn_auto_translate": "🔄 خودکار (فارسی⇄انگلیسی)",
        "language_menu_prompt": "زبان رابط ربات رو انتخاب کن:",
        "language_set": "زبان رابط ربات روی {lang} تنظیم شد✅.",
        "translate_cmd_usage": (
            "استفاده: /translate <زبان مقصد> <متن>\n"
            "مثال: /translate en سلام دنیا\n"
            "یا فقط /translate en برای فعال‌کردن حالت ترجمهٔ مداوم به انگلیسی.\n"
            "زبان مقصد می‌تونه یه کد باشه (en، fr، ja...) یا حتی اسم هر زبونی که توی لیستم نیست، "
            "مثلاً: /translate swahili سلام."
        ),
        # --- new features ---
        "start_features": (
            "چیزهایی که ازم برمیاد:\n"
            "🤖 چت هوش مصنوعی  •  🔎 جستجوی اینترنتی  •  🖼 جستجوی عکس\n"
            "🌐 ترجمه  •  📄 پرسش از روی PDF  •  ⏰ یادآوری\n"
            "🎭 شخصیت دلخواه  •  🎙️ صدا به متن و متن به صدا\n\n"
            "همینطوری پیام بده یا صدا بفرست — لازم نیست هر بار دکمه بزنی. "
            "دکمه‌ها فقط برای وقتیه که حالت خاصی می‌خوای:"
        ),
        "btn_images": "🖼 جستجوی عکس",
        "btn_pdf": "📄 سوال از PDF",
        "btn_personality": "🎭 شخصیت ربات",
        "btn_reminders": "⏰ یادآوری‌ها",
        "btn_share": "🔗 استفاده در چت‌های دیگه",
        "btn_personality_custom": "✍️ شخصیت دلخواه خودم",
        "mode_images_on": (
            "حالت جستجوی عکس فعال شد🖼!\n"
            "اسم هر چیزی رو بفرست (مثلاً «باکتری» یا «کهکشان راه شیری») تا چند تا عکس واقعی "
            "پیدا کنم و با یه توضیح کوتاه برات بفرستم."
        ),
        "images_none": "برای «{query}» عکس قابل ارسالی پیدا نکردم. یه عبارت دیگه امتحان کن.",
        "personality_prompt": (
            "می‌خوای چطوری باهات حرف بزنم؟ یکی رو انتخاب کن — هر وقت خواستی با /personality عوضش کن."
        ),
        "personality_set": "شخصیت من روی «{name}» تنظیم شد✅. از همین الان اعمال میشه.",
        "personality_custom_prompt": (
            "توی یه پیام بنویس دوست داری چطور باشم — لحن، اسمی که صدات کنم، رسمی یا خودمونی، "
            "هر چیزی.\n\nمثال: «خودمونی حرف بزن، صدام کن رفیق، جواب‌ها کوتاه و بدون تعارف باشه.»"
        ),
        "personality_custom_set": "گرفتم✅ از این به بعد همونطوری که گفتی باهات حرف می‌زنم.",
        "remind_usage": (
            "استفاده: /remind <زمان> [تکرار] <متن>\n\n"
            "مثال‌ها:\n"
            "/remind 20m قرص بخور\n"
            "/remind 2h به مامان زنگ بزن\n"
            "/remind 08:00 daily صبحونه بخور\n\n"
            "زمان می‌تونه فاصله باشه (30s, 20m, 2h, 3d) یا ساعت (08:00).\n"
            "تکرار اختیاریه: hourly، daily یا weekly.\n"
            "راستش رو بخوای لازم هم نیست این دستور رو یاد بگیری — توی حالت هوش مصنوعی "
            "فقط بگو «یه ربع دیگه یادم بنداز آب بخورم» و خودم ثبتش می‌کنم."
        ),
        "remind_set": "باشه⏰ ساعت {when} یادت می‌ندازم: {text}",
        "remind_limit": "به سقف {limit} یادآوری رسیدی. اول با /reminders یکی رو حذف کن.",
        "remind_unavailable": (
            "قابلیت یادآوری روی این سرور فعال نیست. برای فعال شدنش باید این نصب بشه:\n"
            "pip install \"python-telegram-bot[job-queue]\""
        ),
        "reminders_header": "⏰ یادآوری‌های فعال تو:",
        "reminders_empty": "الان هیچ یادآوری فعالی نداری. با /remind یکی بساز یا همینطوری بهم بگو «فردا ساعت ۸ یادم بنداز...».",
        "reminders_delete_hint": "برای حذف، روی هرکدوم بزن:",
        "pdf_help": (
            "📄 یه فایل PDF برام بفرست، بعدش هر سوالی ازش داشتی بپرس — خلاصه‌ش کن، "
            "یه نکته‌ی خاص رو پیدا کن، یا ازش سوال امتحانی دربیار.\n\n"
            "برای پاک کردن فایل از حافظه: /forgetpdf"
        ),
        "pdf_only": "فعلاً فقط فایل PDF رو می‌تونم بخونم. لطفاً فایل رو به صورت PDF بفرست.",
        "pdf_missing_lib": "خوندن PDF روی این سرور نصب نیست (pip install pypdf).",
        "pdf_too_big": "این فایل خیلی بزرگه (سقف ۲۰ مگابایته). یه نسخه‌ی کوچیک‌تر یا چند تیکه بفرست.",
        "pdf_reading": "📄 دارم فایل رو می‌خونم...",
        "pdf_fail": "نتونستم این فایل رو باز کنم. مطمئنی سالم و قفل‌نشده‌ست؟",
        "pdf_no_text": (
            "این PDF متن قابل خوندن نداره — احتمالاً اسکن یا عکسه. "
            "یه نسخه‌ی متنی‌ش رو بفرست تا بتونم بخونمش."
        ),
        "pdf_ready": "✅ «{name}» رو خوندم ({pages} صفحه). حالا هر سوالی ازش داری بپرس.",
        "pdf_cleared": "فایل از حافظه پاک شد.",
        "pdf_none": "فایلی توی حافظه نبود.",
        "inline_result_title": "📩 ارسال جواب هوش مصنوعی",
        "help_extra": (
            "\nقابلیت‌های جدید:\n"
            "🖼 /images [موضوع] - جستجوی عکس (یا فقط بگو «عکس فلان چیز رو بفرست»)\n"
            "📄 فرستادن فایل PDF - بعدش هر سوالی ازش بپرس (/forgetpdf برای پاک کردن)\n"
            "⏰ /remind 20m متن - یادآوری (لیست: /reminders)\n"
            "🎭 /personality - انتخاب لحن و شخصیت ربات برای خودت\n"
            f"🔗 توی هر چتی بنویس {BOT_USERNAME} و سوالت - بدون اینکه ربات عضو اون چت باشه"
        ),
    },
    "en": {
        "start_greeting": "Welcome! I'm your personal bot — use me however you like.",
        "btn_ai_groq": "🤖 AI Groq",
        "btn_ai_gemini": "🤖 AI Gemini",
        "btn_web_search": "🔎 Web Search",
        "btn_translate_menu": "🌐 Multilingual Translate",
        "help_text": (
            "Here's what I can do:\n\n"
            "🤖 AI (Groq or Gemini) - free chat and answering questions\n"
            "🔎 Web search - to get up-to-date information\n"
            "🌐 Multilingual translation - translate any text into 40+ languages\n"
            "🎙️ Text-to-speech and speech-to-text\n\n"
            "Commands:\n"
            "/start - show the main menu\n"
            "/stop - turn off AI mode\n"
            "/help - this help message\n"
            "/groq - switch straight into Groq mode\n"
            "/gemini - switch straight into Gemini mode\n"
            "/websearch - switch straight into web search mode\n"
            "/translate [language] [text] - quick translate or enable translate mode\n"
            "/language - change the bot's interface language\n\n"
            "To get started, just tap /start and pick from the buttons."
        ),
        "mode_groq_on": "AI mode (Groq) activated🤖!",
        "mode_gemini_on": "AI mode (Gemini) activated🤖!",
        "mode_search_on": "Web search mode (Groq) activated🔎!",
        "translate_menu_prompt": "Pick a target language; from now on I'll translate any text you send into it:",
        "translate_auto_on": "Auto-translate mode activated🌐 (Persian⇄English). Send any text and I'll translate it.",
        "translate_target_on": (
            "Translate mode activated🌐! Send text in any language and I'll translate it into {target}.\n"
            "To change the target language, tap /start again or use the /translate command."
        ),
        "stop_ai_msg": "AI mode turned off. I won't reply to your messages until you tap /start or /groq.",
        "translate_fail": "Translation failed. Please try again.",
        "ai_call_fail": "There was an error getting a response from the AI (it might be temporary). Please try again.",
        "global_error": "An unexpected error occurred. Please try again.",
        "msg_translate_unavailable": "This message is no longer available for translation.",
        "tts_unavailable": "This message is no longer available for text-to-speech.",
        "tts_fail": "Text-to-speech failed. Please try again.",
        "voice_transcribe_fail": "I couldn't transcribe that voice message. Please try again or type your message.",
        "voice_empty": "No text was extracted from the voice message. Please try again.",
        "btn_cancel": "❌ Cancel",
        "btn_tts": "🔊 Convert to voice",
        "btn_text": "📝 Return to text",
        "btn_translate": "🌐 Translate",
        "btn_auto_translate": "🔄 Auto (Persian⇄English)",
        "language_menu_prompt": "Pick the bot's interface language:",
        "language_set": "Bot interface language set to {lang}✅.",
        "translate_cmd_usage": (
            "Usage: /translate <target language> <text>\n"
            "Example: /translate en سلام دنیا\n"
            "Or just /translate en to turn on continuous translation into English.\n"
            "The target can be a code (en, fr, ja...) or even any language name not in my list, "
            "e.g. /translate swahili hello."
        ),
        # --- new features ---
        "start_features": (
            "Here's what I can do:\n"
            "🤖 AI chat  •  🔎 Web search  •  🖼 Image search\n"
            "🌐 Translation  •  📄 Ask about a PDF  •  ⏰ Reminders\n"
            "🎭 Custom personality  •  🎙️ Voice in and voice out\n\n"
            "Just message me (or send a voice note) — no command needed. "
            "The buttons are only for when you want a specific mode:"
        ),
        "btn_images": "🖼 Image search",
        "btn_pdf": "📄 Ask about a PDF",
        "btn_personality": "🎭 My personality",
        "btn_reminders": "⏰ Reminders",
        "btn_share": "🔗 Use me in any chat",
        "btn_personality_custom": "✍️ Write my own",
        "mode_images_on": (
            "Image search mode is on🖼!\n"
            "Send me anything (\"bacteria\", \"the Milky Way\") and I'll find real photos "
            "of it and send them with a short explanation."
        ),
        "images_none": "I couldn't get any usable images for \"{query}\". Try different wording.",
        "personality_prompt": "How do you want me to talk to you? Pick one — you can change it any time with /personality.",
        "personality_set": "Personality set to \"{name}\"✅. It applies from now on.",
        "personality_custom_prompt": (
            "Describe how you want me to be, in one message — tone, what to call you, "
            "formal or casual, anything.\n\nExample: \"Be casual, call me chief, keep answers "
            "short and skip the pleasantries.\""
        ),
        "personality_custom_set": "Got it✅ I'll talk to you exactly like that from now on.",
        "remind_usage": (
            "Usage: /remind <when> [repeat] <text>\n\n"
            "Examples:\n"
            "/remind 20m take the pills\n"
            "/remind 2h call mum\n"
            "/remind 08:00 daily have breakfast\n\n"
            "When can be a delay (30s, 20m, 2h, 3d) or a clock time (08:00).\n"
            "Repeat is optional: hourly, daily or weekly.\n"
            "You don't really need this command though — in AI mode just say "
            "\"remind me in 15 minutes to drink water\" and I'll set it up."
        ),
        "remind_set": "Done⏰ I'll remind you at {when}: {text}",
        "remind_limit": "You've hit the limit of {limit} reminders. Delete one with /reminders first.",
        "remind_unavailable": (
            "Reminders aren't enabled on this server. To turn them on, install:\n"
            "pip install \"python-telegram-bot[job-queue]\""
        ),
        "reminders_header": "⏰ Your active reminders:",
        "reminders_empty": "No active reminders. Set one with /remind, or just tell me \"remind me tomorrow at 8 to...\".",
        "reminders_delete_hint": "Tap one to delete it:",
        "pdf_help": (
            "📄 Send me a PDF, then ask anything about it — summarise it, find a specific "
            "detail, or turn it into practice questions.\n\n"
            "To clear the file from memory: /forgetpdf"
        ),
        "pdf_only": "I can only read PDF files for now. Please send it as a PDF.",
        "pdf_missing_lib": "PDF reading isn't installed on this server (pip install pypdf).",
        "pdf_too_big": "That file is too big (20 MB max). Try a smaller version or split it up.",
        "pdf_reading": "📄 Reading the file...",
        "pdf_fail": "I couldn't open that file. Is it intact and not password-protected?",
        "pdf_no_text": (
            "This PDF has no readable text layer — it's probably a scan or images. "
            "Send a text-based version and I'll read it."
        ),
        "pdf_ready": "✅ I've read \"{name}\" ({pages} pages). Ask me anything about it.",
        "pdf_cleared": "File cleared from memory.",
        "pdf_none": "There was no file in memory.",
        "inline_result_title": "📩 Send the AI's answer",
        "help_extra": (
            "\nNew features:\n"
            "🖼 /images [topic] - image search (or just say \"show me a picture of X\")\n"
            "📄 Send a PDF - then ask anything about it (/forgetpdf to clear it)\n"
            "⏰ /remind 20m text - set a reminder (list them with /reminders)\n"
            "🎭 /personality - pick how the bot talks to you\n"
            f"🔗 Type {BOT_USERNAME} plus your question in ANY chat - it doesn't need to be a member"
        ),
    },
    "ar": {
        # --- new features (rest of the new strings fall back to English) ---
        "start_features": (
            "ما أستطيع فعله:\n"
            "🤖 دردشة ذكاء اصطناعي  •  🔎 بحث في الإنترنت  •  🖼 بحث عن الصور\n"
            "🌐 ترجمة  •  📄 أسئلة عن ملف PDF  •  ⏰ تذكيرات\n"
            "🎭 شخصية مخصصة  •  🎙️ صوت إلى نص ونص إلى صوت\n\n"
            "اختر من الأزرار:"
        ),
        "btn_images": "🖼 بحث عن الصور",
        "btn_pdf": "📄 أسئلة عن PDF",
        "btn_personality": "🎭 شخصية البوت",
        "btn_reminders": "⏰ التذكيرات",
        "btn_share": "🔗 استخدمني في أي دردشة",
        "btn_personality_custom": "✍️ شخصية من كتابتي",
        "personality_prompt": "كيف تريدني أن أتحدث معك؟ اختر واحدة — يمكنك تغييرها في أي وقت بـ /personality.",
        "start_greeting": "مرحبًا بك! أنا بوتك الشخصي، استخدمني بأي طريقة تحب.",
        "btn_ai_groq": "🤖 AI Groq",
        "btn_ai_gemini": "🤖 AI Gemini",
        "btn_web_search": "🔎 بحث في الإنترنت",
        "btn_translate_menu": "🌐 ترجمة متعددة اللغات",
        "help_text": (
            "هذه هي الأشياء التي أستطيع فعلها:\n\n"
            "🤖 الذكاء الاصطناعي (Groq أو Gemini) - محادثة حرة والإجابة على الأسئلة\n"
            "🔎 بحث في الإنترنت - للحصول على معلومات محدثة\n"
            "🌐 ترجمة متعددة اللغات - ترجمة أي نص إلى أكثر من ٤٠ لغة\n"
            "🎙️ تحويل النص إلى صوت والصوت إلى نص\n\n"
            "الأوامر:\n"
            "/start - عرض القائمة الرئيسية\n"
            "/stop - إيقاف وضع الذكاء الاصطناعي\n"
            "/help - هذه المساعدة\n"
            "/groq - تفعيل وضع Groq مباشرة\n"
            "/gemini - تفعيل وضع Gemini مباشرة\n"
            "/websearch - تفعيل وضع البحث في الإنترنت مباشرة\n"
            "/translate [لغة] [نص] - ترجمة سريعة أو تفعيل وضع الترجمة\n"
            "/language - تغيير لغة واجهة البوت\n\n"
            "للبدء، اضغط /start فقط واختر من الأزرار."
        ),
        "mode_groq_on": "تم تفعيل وضع الذكاء الاصطناعي (Groq)🤖!",
        "mode_gemini_on": "تم تفعيل وضع الذكاء الاصطناعي (Gemini)🤖!",
        "mode_search_on": "تم تفعيل وضع البحث في الإنترنت (Groq)🔎!",
        "translate_menu_prompt": "اختر اللغة الهدف؛ من الآن سأترجم أي نص ترسله إليها:",
        "translate_auto_on": "تم تفعيل وضع الترجمة التلقائية🌐 (فارسي⇄إنجليزي). أرسل أي نص وسأترجمه.",
        "translate_target_on": (
            "تم تفعيل وضع الترجمة🌐! أرسل نصًا بأي لغة وسأترجمه إلى {target}.\n"
            "لتغيير اللغة الهدف، اضغط /start مرة أخرى أو استخدم أمر /translate."
        ),
        "stop_ai_msg": "تم إيقاف وضع الذكاء الاصطناعي.",
        "translate_fail": "فشلت الترجمة. حاول مرة أخرى من فضلك.",
        "ai_call_fail": "حدث خطأ أثناء الحصول على رد من الذكاء الاصطناعي (قد يكون مؤقتًا). حاول مرة أخرى من فضلك.",
        "global_error": "حدث خطأ غير متوقع. حاول مرة أخرى من فضلك.",
        "msg_translate_unavailable": "هذه الرسالة لم تعد متاحة للترجمة.",
        "tts_unavailable": "هذه الرسالة لم تعد متاحة للتحويل إلى صوت.",
        "tts_fail": "فشل التحويل إلى صوت. حاول مرة أخرى من فضلك.",
        "voice_transcribe_fail": "لم أتمكن من تحويل الرسالة الصوتية إلى نص. حاول مرة أخرى أو اكتب رسالتك.",
        "voice_empty": "لم يتم استخراج أي نص من الرسالة الصوتية. حاول مرة أخرى من فضلك.",
        "btn_cancel": "❌ إلغاء",
        "btn_tts": "🔊 تحويل إلى صوت",
        "btn_text": "📝 إعادة النص",
        "btn_translate": "🌐 ترجمة",
        "btn_auto_translate": "🔄 تلقائي (فارسي⇄إنجليزي)",
        "language_menu_prompt": "اختر لغة واجهة البوت:",
        "language_set": "تم تعيين لغة واجهة البوت إلى {lang}✅.",
        "translate_cmd_usage": (
            "الاستخدام: /translate <اللغة الهدف> <النص>\n"
            "مثال: /translate en سلام دنیا\n"
            "أو فقط /translate en لتفعيل الترجمة المستمرة إلى الإنجليزية.\n"
            "يمكن أن تكون اللغة الهدف رمزًا (en، fr، ja...) أو حتى اسم أي لغة غير موجودة في قائمتي، "
            "مثل /translate swahili hello."
        ),
    },
    "tr": {
        # --- new features (rest of the new strings fall back to English) ---
        "start_features": (
            "Yapabildiklerim:\n"
            "🤖 Yapay zekâ sohbeti  •  🔎 Web araması  •  🖼 Görsel arama\n"
            "🌐 Çeviri  •  📄 PDF hakkında soru  •  ⏰ Hatırlatıcılar\n"
            "🎭 Kişiselleştirilmiş karakter  •  🎙️ Sesten yazıya ve yazıdan sese\n\n"
            "Aşağıdan birini seç:"
        ),
        "btn_images": "🖼 Görsel arama",
        "btn_pdf": "📄 PDF'e soru sor",
        "btn_personality": "🎭 Karakterim",
        "btn_reminders": "⏰ Hatırlatıcılar",
        "btn_share": "🔗 Beni her sohbette kullan",
        "btn_personality_custom": "✍️ Kendim yazayım",
        "personality_prompt": "Seninle nasıl konuşmamı istersin? Birini seç — istediğin zaman /personality ile değiştirebilirsin.",
        "start_greeting": "Hoş geldin! Ben senin kişisel botunum, beni istediğin gibi kullanabilirsin.",
        "btn_ai_groq": "🤖 AI Groq",
        "btn_ai_gemini": "🤖 AI Gemini",
        "btn_web_search": "🔎 İnternette Arama",
        "btn_translate_menu": "🌐 Çok Dilli Çeviri",
        "help_text": (
            "İşte yapabildiklerim:\n\n"
            "🤖 Yapay zeka (Groq veya Gemini) - serbest sohbet ve soru cevaplama\n"
            "🔎 İnternette arama - güncel bilgi almak için\n"
            "🌐 Çok dilli çeviri - herhangi bir metni 40'tan fazla dile çevirme\n"
            "🎙️ Metni sese, sesi metne çevirme\n\n"
            "Komutlar:\n"
            "/start - ana menüyü göster\n"
            "/stop - yapay zeka modunu kapat\n"
            "/help - bu yardım metni\n"
            "/groq - doğrudan Groq modunu aç\n"
            "/gemini - doğrudan Gemini modunu aç\n"
            "/websearch - doğrudan internet arama modunu aç\n"
            "/translate [dil] [metin] - hızlı çeviri ya da çeviri modunu aç\n"
            "/language - botun arayüz dilini değiştir\n\n"
            "Başlamak için sadece /start'a bas ve butonlardan seç."
        ),
        "mode_groq_on": "Yapay zeka modu (Groq) etkinleştirildi🤖!",
        "mode_gemini_on": "Yapay zeka modu (Gemini) etkinleştirildi🤖!",
        "mode_search_on": "İnternette arama modu (Groq) etkinleştirildi🔎!",
        "translate_menu_prompt": "Hedef dili seç; bundan sonra gönderdiğin her metni o dile çeviririm:",
        "translate_auto_on": "Otomatik çeviri modu etkinleştirildi🌐 (Farsça⇄İngilizce). Ne gönderirsen çeviririm.",
        "translate_target_on": (
            "Çeviri modu etkinleştirildi🌐! Hangi dilde metin gönderirsen {target} diline çeviririm.\n"
            "Hedef dili değiştirmek için tekrar /start'a bas ya da /translate komutunu kullan."
        ),
        "stop_ai_msg": "Yapay zeka modu kapatıldı.",
        "translate_fail": "Çeviri başarısız oldu. Lütfen tekrar dene.",
        "ai_call_fail": "Yapay zekadan yanıt alırken bir hata oluştu (geçici olabilir). Lütfen tekrar dene.",
        "global_error": "Beklenmeyen bir hata oluştu. Lütfen tekrar dene.",
        "msg_translate_unavailable": "Bu mesaj artık çeviri için kullanılamıyor.",
        "tts_unavailable": "Bu mesaj artık sese çevirme için kullanılamıyor.",
        "tts_fail": "Sese çevirme başarısız oldu. Lütfen tekrar dene.",
        "voice_transcribe_fail": "Sesli mesajı metne çeviremedim. Lütfen tekrar dene ya da mesajını yaz.",
        "voice_empty": "Sesli mesajdan metin çıkarılamadı. Lütfen tekrar dene.",
        "btn_cancel": "❌ İptal",
        "btn_tts": "🔊 Sese çevir",
        "btn_text": "📝 Metne döndür",
        "btn_translate": "🌐 Çevir",
        "btn_auto_translate": "🔄 Otomatik (Farsça⇄İngilizce)",
        "language_menu_prompt": "Botun arayüz dilini seç:",
        "language_set": "Bot arayüz dili {lang} olarak ayarlandı✅.",
        "translate_cmd_usage": (
            "Kullanım: /translate <hedef dil> <metin>\n"
            "Örnek: /translate en hello world\n"
            "Ya da sadece /translate en yazarak sürekli İngilizce çeviri modunu aç.\n"
            "Hedef dil bir kod olabilir (en, fr, ja...) ya da listemde olmayan herhangi bir dilin adı, "
            "örn. /translate swahili hello."
        ),
    },
    "ru": {
        # --- new features (rest of the new strings fall back to English) ---
        "start_features": (
            "Вот что я умею:\n"
            "🤖 ИИ-чат  •  🔎 Поиск в интернете  •  🖼 Поиск картинок\n"
            "🌐 Перевод  •  📄 Вопросы по PDF  •  ⏰ Напоминания\n"
            "🎭 Своя личность бота  •  🎙️ Голос в текст и текст в голос\n\n"
            "Выбери кнопку ниже:"
        ),
        "btn_images": "🖼 Поиск картинок",
        "btn_pdf": "📄 Вопросы по PDF",
        "btn_personality": "🎭 Личность бота",
        "btn_reminders": "⏰ Напоминания",
        "btn_share": "🔗 Использовать в любом чате",
        "btn_personality_custom": "✍️ Напишу свою",
        "personality_prompt": "Как мне с тобой общаться? Выбери вариант — поменять можно в любой момент через /personality.",
        "start_greeting": "Добро пожаловать! Я твой личный бот, используй меня как захочешь.",
        "btn_ai_groq": "🤖 AI Groq",
        "btn_ai_gemini": "🤖 AI Gemini",
        "btn_web_search": "🔎 Поиск в интернете",
        "btn_translate_menu": "🌐 Многоязычный перевод",
        "help_text": (
            "Вот что я умею:\n\n"
            "🤖 ИИ (Groq или Gemini) - свободное общение и ответы на вопросы\n"
            "🔎 Поиск в интернете - чтобы получать актуальную информацию\n"
            "🌐 Многоязычный перевод - перевод любого текста на 40+ языков\n"
            "🎙️ Преобразование текста в речь и речи в текст\n\n"
            "Команды:\n"
            "/start - показать главное меню\n"
            "/stop - выключить режим ИИ\n"
            "/help - эта справка\n"
            "/groq - сразу включить режим Groq\n"
            "/gemini - сразу включить режим Gemini\n"
            "/websearch - сразу включить режим поиска в интернете\n"
            "/translate [язык] [текст] - быстрый перевод или включение режима перевода\n"
            "/language - сменить язык интерфейса бота\n\n"
            "Чтобы начать, просто нажми /start и выбери кнопку."
        ),
        "mode_groq_on": "Режим ИИ (Groq) активирован🤖!",
        "mode_gemini_on": "Режим ИИ (Gemini) активирован🤖!",
        "mode_search_on": "Режим поиска в интернете (Groq) активирован🔎!",
        "translate_menu_prompt": "Выбери целевой язык; теперь любой присланный текст я буду переводить на него:",
        "translate_auto_on": "Режим автоперевода активирован🌐 (персидский⇄английский). Присылай любой текст, и я переведу.",
        "translate_target_on": (
            "Режим перевода активирован🌐! Присылай текст на любом языке, и я переведу его на {target}.\n"
            "Чтобы сменить целевой язык, снова нажми /start или используй команду /translate."
        ),
        "stop_ai_msg": "Режим ИИ выключен.",
        "translate_fail": "Перевод не удался. Попробуй ещё раз.",
        "ai_call_fail": "Произошла ошибка при получении ответа от ИИ (возможно, временная). Попробуй ещё раз.",
        "global_error": "Произошла непредвиденная ошибка. Попробуй ещё раз.",
        "msg_translate_unavailable": "Это сообщение больше недоступно для перевода.",
        "tts_unavailable": "Это сообщение больше недоступно для преобразования в голос.",
        "tts_fail": "Не удалось преобразовать в голос. Попробуй ещё раз.",
        "voice_transcribe_fail": "Не удалось распознать голосовое сообщение. Попробуй ещё раз или напиши текстом.",
        "voice_empty": "Из голосового сообщения не удалось извлечь текст. Попробуй ещё раз.",
        "btn_cancel": "❌ Отмена",
        "btn_tts": "🔊 Преобразовать в голос",
        "btn_text": "📝 Вернуть текст",
        "btn_translate": "🌐 Перевести",
        "btn_auto_translate": "🔄 Авто (персидский⇄английский)",
        "language_menu_prompt": "Выбери язык интерфейса бота:",
        "language_set": "Язык интерфейса бота установлен на {lang}✅.",
        "translate_cmd_usage": (
            "Использование: /translate <целевой язык> <текст>\n"
            "Пример: /translate en hello world\n"
            "Или просто /translate en, чтобы включить постоянный перевод на английский.\n"
            "Целевой язык может быть кодом (en, fr, ja...) или даже названием любого языка, "
            "которого нет в моём списке, напр. /translate swahili hello."
        ),
    },
    "fr": {
        # --- new features (rest of the new strings fall back to English) ---
        "start_features": (
            "Ce que je sais faire :\n"
            "🤖 Chat IA  •  🔎 Recherche web  •  🖼 Recherche d'images\n"
            "🌐 Traduction  •  📄 Questions sur un PDF  •  ⏰ Rappels\n"
            "🎭 Personnalité au choix  •  🎙️ Voix vers texte et texte vers voix\n\n"
            "Choisis un bouton :"
        ),
        "btn_images": "🖼 Recherche d'images",
        "btn_pdf": "📄 Questions sur un PDF",
        "btn_personality": "🎭 Ma personnalité",
        "btn_reminders": "⏰ Rappels",
        "btn_share": "🔗 M'utiliser dans n'importe quel chat",
        "btn_personality_custom": "✍️ La écrire moi-même",
        "personality_prompt": "Comment veux-tu que je te parle ? Choisis — tu peux changer à tout moment avec /personality.",
        "start_greeting": "Bienvenue ! Je suis ton bot personnel, utilise-moi comme tu veux.",
        "btn_ai_groq": "🤖 IA Groq",
        "btn_ai_gemini": "🤖 IA Gemini",
        "btn_web_search": "🔎 Recherche web",
        "btn_translate_menu": "🌐 Traduction multilingue",
        "help_text": (
            "Voici ce que je peux faire :\n\n"
            "🤖 IA (Groq ou Gemini) - discussion libre et réponses aux questions\n"
            "🔎 Recherche web - pour obtenir des informations à jour\n"
            "🌐 Traduction multilingue - traduire n'importe quel texte dans plus de 40 langues\n"
            "🎙️ Conversion texte-parole et parole-texte\n\n"
            "Commandes :\n"
            "/start - afficher le menu principal\n"
            "/stop - désactiver le mode IA\n"
            "/help - cette aide\n"
            "/groq - activer directement le mode Groq\n"
            "/gemini - activer directement le mode Gemini\n"
            "/websearch - activer directement le mode recherche web\n"
            "/translate [langue] [texte] - traduction rapide ou activation du mode traduction\n"
            "/language - changer la langue de l'interface du bot\n\n"
            "Pour commencer, appuie sur /start et choisis un bouton."
        ),
        "mode_groq_on": "Mode IA (Groq) activé🤖 !",
        "mode_gemini_on": "Mode IA (Gemini) activé🤖 !",
        "mode_search_on": "Mode recherche web (Groq) activé🔎 !",
        "translate_menu_prompt": "Choisis la langue cible ; à partir de maintenant je traduirai tout texte envoyé dans celle-ci :",
        "translate_auto_on": "Mode traduction automatique activé🌐 (persan⇄anglais). Envoie n'importe quel texte et je le traduis.",
        "translate_target_on": (
            "Mode traduction activé🌐 ! Envoie un texte dans n'importe quelle langue et je le traduirai en {target}.\n"
            "Pour changer la langue cible, retape /start ou utilise la commande /translate."
        ),
        "stop_ai_msg": "Mode IA désactivé.",
        "translate_fail": "La traduction a échoué. Merci de réessayer.",
        "ai_call_fail": "Une erreur est survenue en obtenant une réponse de l'IA (peut-être temporaire). Merci de réessayer.",
        "global_error": "Une erreur inattendue est survenue. Merci de réessayer.",
        "msg_translate_unavailable": "Ce message n'est plus disponible pour la traduction.",
        "tts_unavailable": "Ce message n'est plus disponible pour la conversion en voix.",
        "tts_fail": "La conversion en voix a échoué. Merci de réessayer.",
        "voice_transcribe_fail": "Je n'ai pas pu transcrire ce message vocal. Réessaie ou tape ton message.",
        "voice_empty": "Aucun texte n'a pu être extrait du message vocal. Merci de réessayer.",
        "btn_cancel": "❌ Annuler",
        "btn_tts": "🔊 Convertir en voix",
        "btn_text": "📝 Restaurer le texte",
        "btn_translate": "🌐 Traduire",
        "btn_auto_translate": "🔄 Auto (persan⇄anglais)",
        "language_menu_prompt": "Choisis la langue de l'interface du bot :",
        "language_set": "Langue de l'interface réglée sur {lang}✅.",
        "translate_cmd_usage": (
            "Utilisation : /translate <langue cible> <texte>\n"
            "Exemple : /translate en bonjour le monde\n"
            "Ou juste /translate en pour activer la traduction continue vers l'anglais.\n"
            "La langue cible peut être un code (en, fr, ja...) ou même le nom de n'importe quelle langue "
            "absente de ma liste, par ex. /translate swahili hello."
        ),
    },
    "de": {
        # --- new features (rest of the new strings fall back to English) ---
        "start_features": (
            "Das kann ich:\n"
            "🤖 KI-Chat  •  🔎 Websuche  •  🖼 Bildersuche\n"
            "🌐 Übersetzung  •  📄 Fragen zu einer PDF  •  ⏰ Erinnerungen\n"
            "🎭 Eigene Persönlichkeit  •  🎙️ Sprache zu Text und Text zu Sprache\n\n"
            "Wähl unten etwas aus:"
        ),
        "btn_images": "🖼 Bildersuche",
        "btn_pdf": "📄 Fragen zur PDF",
        "btn_personality": "🎭 Meine Persönlichkeit",
        "btn_reminders": "⏰ Erinnerungen",
        "btn_share": "🔗 In jedem Chat nutzen",
        "btn_personality_custom": "✍️ Selbst schreiben",
        "personality_prompt": "Wie soll ich mit dir reden? Such dir etwas aus — änderbar jederzeit mit /personality.",
        "start_greeting": "Willkommen! Ich bin dein persönlicher Bot, nutze mich wie du magst.",
        "btn_ai_groq": "🤖 KI Groq",
        "btn_ai_gemini": "🤖 KI Gemini",
        "btn_web_search": "🔎 Websuche",
        "btn_translate_menu": "🌐 Mehrsprachige Übersetzung",
        "help_text": (
            "Das kann ich für dich tun:\n\n"
            "🤖 KI (Groq oder Gemini) - freies Chatten und Fragen beantworten\n"
            "🔎 Websuche - für aktuelle Informationen\n"
            "🌐 Mehrsprachige Übersetzung - jeden Text in über 40 Sprachen übersetzen\n"
            "🎙️ Text-zu-Sprache und Sprache-zu-Text\n\n"
            "Befehle:\n"
            "/start - Hauptmenü anzeigen\n"
            "/stop - KI-Modus ausschalten\n"
            "/help - diese Hilfe\n"
            "/groq - direkt in den Groq-Modus wechseln\n"
            "/gemini - direkt in den Gemini-Modus wechseln\n"
            "/websearch - direkt in den Websuche-Modus wechseln\n"
            "/translate [Sprache] [Text] - Schnellübersetzung oder Übersetzungsmodus aktivieren\n"
            "/language - Bot-Oberflächensprache ändern\n\n"
            "Um loszulegen, tippe einfach /start und wähle einen Button."
        ),
        "mode_groq_on": "KI-Modus (Groq) aktiviert🤖!",
        "mode_gemini_on": "KI-Modus (Gemini) aktiviert🤖!",
        "mode_search_on": "Websuche-Modus (Groq) aktiviert🔎!",
        "translate_menu_prompt": "Wähle die Zielsprache; ab jetzt übersetze ich jeden gesendeten Text in diese Sprache:",
        "translate_auto_on": "Auto-Übersetzungsmodus aktiviert🌐 (Persisch⇄Englisch). Sende einen Text und ich übersetze ihn.",
        "translate_target_on": (
            "Übersetzungsmodus aktiviert🌐! Sende Text in jeder Sprache und ich übersetze ihn ins {target}.\n"
            "Um die Zielsprache zu ändern, tippe erneut /start oder nutze den Befehl /translate."
        ),
        "stop_ai_msg": "KI-Modus ausgeschaltet.",
        "translate_fail": "Übersetzung fehlgeschlagen. Bitte versuche es erneut.",
        "ai_call_fail": "Beim Abrufen einer Antwort von der KI ist ein Fehler aufgetreten (evtl. vorübergehend). Bitte versuche es erneut.",
        "global_error": "Ein unerwarteter Fehler ist aufgetreten. Bitte versuche es erneut.",
        "msg_translate_unavailable": "Diese Nachricht steht nicht mehr zur Übersetzung zur Verfügung.",
        "tts_unavailable": "Diese Nachricht steht nicht mehr für die Sprachumwandlung zur Verfügung.",
        "tts_fail": "Die Umwandlung in Sprache ist fehlgeschlagen. Bitte versuche es erneut.",
        "voice_transcribe_fail": "Ich konnte die Sprachnachricht nicht transkribieren. Bitte versuche es erneut oder tippe deine Nachricht.",
        "voice_empty": "Aus der Sprachnachricht konnte kein Text extrahiert werden. Bitte versuche es erneut.",
        "btn_cancel": "❌ Abbrechen",
        "btn_tts": "🔊 In Sprache umwandeln",
        "btn_text": "📝 Text wiederherstellen",
        "btn_translate": "🌐 Übersetzen",
        "btn_auto_translate": "🔄 Automatisch (Persisch⇄Englisch)",
        "language_menu_prompt": "Wähle die Oberflächensprache des Bots:",
        "language_set": "Bot-Oberflächensprache auf {lang} eingestellt✅.",
        "translate_cmd_usage": (
            "Verwendung: /translate <Zielsprache> <Text>\n"
            "Beispiel: /translate en hello world\n"
            "Oder einfach /translate en, um die fortlaufende Übersetzung ins Englische zu aktivieren.\n"
            "Die Zielsprache kann ein Code sein (en, fr, ja...) oder sogar der Name jeder Sprache, "
            "die nicht in meiner Liste steht, z. B. /translate swahili hello."
        ),
    },
    "es": {
        # --- new features (rest of the new strings fall back to English) ---
        "start_features": (
            "Esto es lo que puedo hacer:\n"
            "🤖 Chat con IA  •  🔎 Búsqueda web  •  🖼 Búsqueda de imágenes\n"
            "🌐 Traducción  •  📄 Preguntas sobre un PDF  •  ⏰ Recordatorios\n"
            "🎭 Personalidad a tu gusto  •  🎙️ Voz a texto y texto a voz\n\n"
            "Elige un botón:"
        ),
        "btn_images": "🖼 Búsqueda de imágenes",
        "btn_pdf": "📄 Preguntas sobre un PDF",
        "btn_personality": "🎭 Mi personalidad",
        "btn_reminders": "⏰ Recordatorios",
        "btn_share": "🔗 Úsame en cualquier chat",
        "btn_personality_custom": "✍️ Escribirla yo mismo",
        "personality_prompt": "¿Cómo quieres que hable contigo? Elige una — puedes cambiarla cuando quieras con /personality.",
        "start_greeting": "¡Bienvenido! Soy tu bot personal, úsame como quieras.",
        "btn_ai_groq": "🤖 IA Groq",
        "btn_ai_gemini": "🤖 IA Gemini",
        "btn_web_search": "🔎 Búsqueda web",
        "btn_translate_menu": "🌐 Traducción multilingüe",
        "help_text": (
            "Esto es lo que puedo hacer:\n\n"
            "🤖 IA (Groq o Gemini) - chat libre y respuesta a preguntas\n"
            "🔎 Búsqueda web - para obtener información actualizada\n"
            "🌐 Traducción multilingüe - traducir cualquier texto a más de 40 idiomas\n"
            "🎙️ Conversión de texto a voz y de voz a texto\n\n"
            "Comandos:\n"
            "/start - mostrar el menú principal\n"
            "/stop - desactivar el modo IA\n"
            "/help - esta ayuda\n"
            "/groq - activar directamente el modo Groq\n"
            "/gemini - activar directamente el modo Gemini\n"
            "/websearch - activar directamente el modo de búsqueda web\n"
            "/translate [idioma] [texto] - traducción rápida o activar el modo traducción\n"
            "/language - cambiar el idioma de la interfaz del bot\n\n"
            "Para empezar, solo pulsa /start y elige un botón."
        ),
        "mode_groq_on": "Modo IA (Groq) activado🤖!",
        "mode_gemini_on": "Modo IA (Gemini) activado🤖!",
        "mode_search_on": "Modo de búsqueda web (Groq) activado🔎!",
        "translate_menu_prompt": "Elige el idioma de destino; a partir de ahora traduciré cualquier texto que envíes a ese idioma:",
        "translate_auto_on": "Modo de traducción automática activado🌐 (persa⇄inglés). Envía cualquier texto y lo traduzco.",
        "translate_target_on": (
            "¡Modo de traducción activado🌐! Envía texto en cualquier idioma y lo traduciré al {target}.\n"
            "Para cambiar el idioma de destino, pulsa /start de nuevo o usa el comando /translate."
        ),
        "stop_ai_msg": "Modo IA desactivado.",
        "translate_fail": "La traducción falló. Por favor, inténtalo de nuevo.",
        "ai_call_fail": "Ocurrió un error al obtener una respuesta de la IA (puede ser temporal). Por favor, inténtalo de nuevo.",
        "global_error": "Ocurrió un error inesperado. Por favor, inténtalo de nuevo.",
        "msg_translate_unavailable": "Este mensaje ya no está disponible para traducir.",
        "tts_unavailable": "Este mensaje ya no está disponible para convertir a voz.",
        "tts_fail": "La conversión a voz falló. Por favor, inténtalo de nuevo.",
        "voice_transcribe_fail": "No pude transcribir ese mensaje de voz. Inténtalo de nuevo o escribe tu mensaje.",
        "voice_empty": "No se pudo extraer texto del mensaje de voz. Por favor, inténtalo de nuevo.",
        "btn_cancel": "❌ Cancelar",
        "btn_tts": "🔊 Convertir a voz",
        "btn_text": "📝 Volver al texto",
        "btn_translate": "🌐 Traducir",
        "btn_auto_translate": "🔄 Automático (persa⇄inglés)",
        "language_menu_prompt": "Elige el idioma de la interfaz del bot:",
        "language_set": "Idioma de la interfaz del bot configurado en {lang}✅.",
        "translate_cmd_usage": (
            "Uso: /translate <idioma destino> <texto>\n"
            "Ejemplo: /translate en hello world\n"
            "O simplemente /translate en para activar la traducción continua al inglés.\n"
            "El idioma de destino puede ser un código (en, fr, ja...) o incluso el nombre de cualquier "
            "idioma que no esté en mi lista, p. ej. /translate swahili hello."
        ),
    },
}


def get_ui_lang(context: ContextTypes.DEFAULT_TYPE) -> str:
    return context.user_data.get("ui_lang", "fa")


def ui(context: ContextTypes.DEFAULT_TYPE, key: str, **kwargs) -> str:
    """Look up a bot interface string in the user's chosen UI language
    (falls back to Persian for any language/key that isn't translated)."""
    lang = get_ui_lang(context)
    table = UI_STRINGS.get(lang, UI_STRINGS["fa"])
    # Newer strings are only hand-written in Persian and English, so anything
    # missing falls back to English first (readable for most users) and only
    # then to Persian, rather than dropping an Arabic/Turkish/French user
    # straight into Persian.
    template = table.get(key) or UI_STRINGS["en"].get(key) or UI_STRINGS["fa"].get(key, key)
    return template.format(**kwargs) if kwargs else template


def build_ui_language_keyboard() -> InlineKeyboardMarkup:
    rows, row = [], []
    for code in UI_LANGUAGES:
        row.append(InlineKeyboardButton(LANGUAGES[code]["label"], callback_data=f"ui_lang:{code}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return InlineKeyboardMarkup(rows)


# Persian and Arabic share most of their script, which trips up both simple
# char-range checks and langdetect. These letters exist in Persian but not in
# standard Arabic, so their presence is a reliable Persian signal.
_PERSIAN_ONLY_CHARS = set("پچژگ")


def _script_hint(text: str) -> str | None:
    # Fast, dependency-free pre-check for scripts that are cheap to recognize
    # directly and that langdetect (a statistical, Latin-script-biased
    # detector) sometimes gets wrong on short strings.
    if any(ch in _PERSIAN_ONLY_CHARS for ch in text):
        return "fa"
    if any("\u3040" <= ch <= "\u30ff" for ch in text):  # hiragana/katakana - uniquely Japanese
        return "ja"
    if any("\uac00" <= ch <= "\ud7a3" for ch in text):  # hangul - uniquely Korean
        return "ko"
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):  # CJK ideographs w/o kana/hangul -> Chinese
        return "zh-cn"
    if any("\u0600" <= ch <= "\u06ff" for ch in text):  # Arabic-script, no Persian-only letters
        return "ar"
    return None


def detect_language(text: str) -> str:
    """Best-effort ISO-ish language code for the given text (e.g. 'fa', 'ja', 'en')."""
    hint = _script_hint(text)
    if hint:
        return hint
    if _langdetect_detect is not None:
        try:
            return _langdetect_detect(text)
        except Exception:
            pass
    return "en"


def _tts_voice_for_language(lang_code: str) -> str:
    if lang_code in LANGUAGES:
        return LANGUAGES[lang_code]["voice"]
    base = lang_code.split("-")[0]
    for code, info in LANGUAGES.items():
        if code.split("-")[0] == base:
            return info["voice"]
    return DEFAULT_TTS_VOICE


LANG_PAGE_SIZE = 9  # 3 columns x 3 rows of language buttons per page


def build_paginated_lang_keyboard(
    callback_prefix: str, page: int = 0, include_auto: bool = False
) -> InlineKeyboardMarkup:
    """A language-picker keyboard that pages through the full LANGUAGES dict
    (40+ entries) instead of dumping them all into one giant, unscrollable
    wall of buttons. `callback_prefix` is the prefix used for each language
    button's callback_data (e.g. "translate_lang" -> "translate_lang:en"),
    and also identifies this keyboard in the "langpage:<prefix>:<page>"
    pagination callback."""
    codes = list(LANGUAGES.keys())
    total_pages = max(1, -(-len(codes) // LANG_PAGE_SIZE))  # ceil division
    page = max(0, min(page, total_pages - 1))
    chunk = codes[page * LANG_PAGE_SIZE : (page + 1) * LANG_PAGE_SIZE]

    rows, row = [], []
    for code in chunk:
        row.append(InlineKeyboardButton(LANGUAGES[code]["label"], callback_data=f"{callback_prefix}:{code}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)

    if total_pages > 1:
        nav = []
        if page > 0:
            nav.append(InlineKeyboardButton("◀️", callback_data=f"langpage:{callback_prefix}:{page - 1}"))
        nav.append(InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="noop"))
        if page < total_pages - 1:
            nav.append(InlineKeyboardButton("▶️", callback_data=f"langpage:{callback_prefix}:{page + 1}"))
        rows.append(nav)

    if include_auto:
        rows.append([InlineKeyboardButton("🔄 فارسی⇄انگلیسی", callback_data=f"{callback_prefix}:auto")])
    return InlineKeyboardMarkup(rows)


# ---------------------- TEXT-TO-SPEECH (turn an AI reply into a voice note) ----------------------
def _detect_tts_voice(text: str) -> str:
    return _tts_voice_for_language(detect_language(text))


async def _edge_tts_mp3(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def _mp3_to_ogg_opus_sync(mp3_bytes: bytes) -> bytes:
    # Telegram's round, tap-to-play voice bubble (send_voice) requires OGG
    # container + Opus codec specifically - edge-tts only gives mp3, so this
    # shells out to ffmpeg to convert. ffmpeg must be installed and on PATH.
    process = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", "pipe:0",
            "-vn",
            "-c:a", "libopus",
            "-b:a", "64k",
            "-ac", "1",
            "-f", "ogg",
            "pipe:1",
        ],
        input=mp3_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {process.stderr.decode(errors='ignore')[-500:]}")
    return process.stdout


async def synthesize_speech(text: str) -> bytes:
    voice = _detect_tts_voice(text)
    mp3_bytes = await _edge_tts_mp3(text, voice)
    # ffmpeg is a blocking subprocess call, so push it to a thread like the
    # other blocking calls in this file, to avoid blocking the bot's event loop.
    return await asyncio.to_thread(_mp3_to_ogg_opus_sync, mp3_bytes)


# In-memory store mapping a short id -> (AI reply text, the user message it
# was replying to), so the "🔊 to voice" button (which can only carry ~64
# bytes of callback_data) can look the full text back up when pressed, and
# the resulting voice note can keep the same "reply to" thread. Capped so it
# can't grow forever.
MAX_STORED_TTS_TEXTS = 200


def _remember_tts_text(context: ContextTypes.DEFAULT_TYPE, text: str, reply_to_message_id: int) -> str:
    store = context.chat_data.setdefault("tts_texts", {})
    if len(store) > MAX_STORED_TTS_TEXTS:
        for old_key in list(store.keys())[: len(store) - MAX_STORED_TTS_TEXTS]:
            store.pop(old_key, None)
    tts_id = uuid.uuid4().hex[:12]
    store[tts_id] = {"text": text, "reply_to_message_id": reply_to_message_id}
    return tts_id


def build_message_translate_keyboard(context: ContextTypes.DEFAULT_TYPE, tts_id: str) -> InlineKeyboardMarkup:
    # Deliberately uses the smaller QUICK_LANGUAGES list (not the full 40+ in
    # LANGUAGES) so this keyboard, attached under every single AI reply,
    # stays short. For the full language list, use /translate or the
    # "🌐 Multilingual Translate" menu from /start.
    rows, row = [], []
    for code in QUICK_LANGUAGES:
        row.append(InlineKeyboardButton(LANGUAGES[code]["label"], callback_data=f"trlang:{tts_id}:{code}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton(ui(context, "btn_cancel"), callback_data=f"trcancel:{tts_id}")])
    return InlineKeyboardMarkup(rows)


def build_ai_reply_keyboard(context: ContextTypes.DEFAULT_TYPE, tts_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(ui(context, "btn_tts"), callback_data=f"tts:{tts_id}")],
            [InlineKeyboardButton(ui(context, "btn_translate"), callback_data=f"trmenu:{tts_id}")],
        ]
    )


def build_voice_reply_keyboard(context: ContextTypes.DEFAULT_TYPE, tts_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(ui(context, "btn_text"), callback_data=f"ttsrestore:{tts_id}")],
        ]
    )


async def send_ai_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, reply_text: str) -> None:
    # Every AI reply goes out through here so it always gets the "convert to
    # voice" and "translate" buttons underneath it. reply_to_message_id is
    # remembered so the voice version (sent later, from a button press) can
    # still quote the same original user message.
    tts_id = _remember_tts_text(context, reply_text, update.message.message_id)
    keyboard = build_ai_reply_keyboard(context, tts_id)
    await update.message.reply_text(reply_text, parse_mode=None, reply_markup=keyboard)


async def handle_translate_menu_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    tts_id = query.data.split(":", 1)[1]
    if tts_id not in context.chat_data.get("tts_texts", {}):
        await query.message.reply_text(ui(context, "msg_translate_unavailable"))
        return
    await query.edit_message_reply_markup(reply_markup=build_message_translate_keyboard(context, tts_id))


async def handle_translate_cancel_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    tts_id = query.data.split(":", 1)[1]
    await query.edit_message_reply_markup(reply_markup=build_ai_reply_keyboard(context, tts_id))


async def handle_translate_lang_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    _, tts_id, target_code = query.data.split(":", 2)

    store = context.chat_data.get("tts_texts", {})
    entry = store.get(tts_id)
    if not entry:
        await query.edit_message_reply_markup(reply_markup=build_ai_reply_keyboard(context, tts_id))
        await query.message.reply_text(ui(context, "msg_translate_unavailable"))
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    try:
        translated = await translate_text(entry["text"], target_code)
    except Exception as e:
        print(f"[handle_translate_lang_button] translation failed: {e}")
        await query.edit_message_reply_markup(reply_markup=build_ai_reply_keyboard(context, tts_id))
        await query.message.reply_text(ui(context, "translate_fail"))
        return

    # Replace the message in place with the translation, and remember the
    # translated text under the same id so the tts/translate buttons on this
    # same message now act on the translated version.
    entry["text"] = translated
    await query.edit_message_text(
        translated, parse_mode=None, reply_markup=build_ai_reply_keyboard(context, tts_id)
    )


async def handle_tts_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    tts_id = query.data.split(":", 1)[1]
    entry = context.chat_data.get("tts_texts", {}).get(tts_id)
    if not entry:
        await query.message.reply_text(ui(context, "tts_unavailable"))
        return

    text = entry["text"]
    reply_to_message_id = entry["reply_to_message_id"]
    chat_id = update.effective_chat.id

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.RECORD_VOICE)

    try:
        ogg_bytes = await synthesize_speech(text)
    except Exception as e:
        print(f"[handle_tts_button] TTS failed: {e}")
        await query.message.reply_text(ui(context, "tts_fail"))
        return

    # Remove the text message and replace it with the voice note.
    try:
        await query.message.delete()
    except Exception:
        pass  # message may already be gone / too old to delete - not critical

    voice_message = await context.bot.send_voice(
        chat_id=chat_id,
        voice=io.BytesIO(ogg_bytes),
        filename="voice.ogg",
        reply_to_message_id=reply_to_message_id,
        reply_markup=build_voice_reply_keyboard(context, tts_id),
    )

    # Remember the current voice message so the "return to text" button can
    # replace this voice note with the original/translated text.
    entry["voice_message_id"] = voice_message.message_id


async def handle_tts_restore_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    tts_id = query.data.split(":", 1)[1]
    entry = context.chat_data.get("tts_texts", {}).get(tts_id)
    if not entry:
        await query.message.reply_text(ui(context, "tts_unavailable"))
        return

    text = entry["text"]
    reply_to_message_id = entry["reply_to_message_id"]
    chat_id = update.effective_chat.id

    # Remove the voice message and restore the text in its place.
    try:
        await query.message.delete()
    except Exception:
        pass  # message may already be gone / too old to delete - not critical

    text_message = await context.bot.send_message(
        chat_id=chat_id,
        text=text,
        parse_mode=None,
        reply_to_message_id=reply_to_message_id,
        reply_markup=build_ai_reply_keyboard(context, tts_id),
    )

    # Keep track of whichever representation is currently active.
    entry["text_message_id"] = text_message.message_id
    entry.pop("voice_message_id", None)


WHISPER_MODEL = "whisper-large-v3-turbo"  # fast + free-tier friendly Groq transcription model


async def transcribe_voice(file_bytes: bytes) -> str:
    # Groq's transcription endpoint just wants (filename, raw_bytes) - it can
    # read Telegram's native .ogg/Opus voice notes directly, no conversion needed.
    transcription = await asyncio.to_thread(
        groq_client.audio.transcriptions.create,
        file=("voice.ogg", file_bytes),
        model=WHISPER_MODEL,
    )
    return (transcription.text or "").strip()


async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Same gate as text messages: a voice note in a private chat is answered
    # straight away, without the user activating anything first.
    if not await ensure_ai_ready(update, context):
        return

    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    voice = update.message.voice
    tg_file = await context.bot.get_file(voice.file_id)
    file_bytes = bytes(await tg_file.download_as_bytearray())

    try:
        user_text = await transcribe_voice(file_bytes)
    except Exception as e:
        print(f"[voice_handler] transcription failed: {e}")
        await update.message.reply_text(ui(context, "voice_transcribe_fail"))
        return

    if not user_text:
        await update.message.reply_text(ui(context, "voice_empty"))
        return

    await ai_handler(update, context, user_text)


# ---------------------- TRANSLATION (any language -> chosen target language) ----------------------
# Deliberately stateless (no persisted history): every message is translated
# fresh on its own, independent of anything said before, so context from a
# previous translation never leaks in and skews the next one.
_AUTO_TRANSLATE_PROMPT_TEXT = (
    "You are an expert professional Persian<->English translator. "
    "Automatically detect whether the given text is Persian or English, then "
    "translate it into the other language. Translate the meaning, tone, and "
    "nuance naturally and idiomatically, the way a skilled human translator "
    "would - never a stiff, literal word-for-word conversion. "
    "Output ONLY the translation itself: plain text, no quotation marks, no "
    "notes, no explanations, no labels like 'Translation:', nothing except "
    "the translated text."
)


def build_translate_system_prompt(target_code: str) -> dict:
    if not target_code or target_code == "auto":
        content = _AUTO_TRANSLATE_PROMPT_TEXT
    else:
        # target_code is either a known LANGUAGES key (e.g. "fr") or, from
        # the /translate command, an arbitrary free-text language name the
        # user typed that isn't in the dict at all (e.g. "Swahili") - the
        # model understands language names directly, so both work the same way.
        target_name = LANGUAGES[target_code]["name"] if target_code in LANGUAGES else target_code
        content = (
            "You are an expert professional translator, fluent and native-level in "
            "every major world language (Persian, English, Japanese, Arabic, Chinese, "
            "Korean, French, German, Spanish, Russian, Turkish, Italian, Portuguese, "
            "Hindi, Urdu, Dutch, and more). "
            f"Automatically detect the language of the given text. "
            f"If it is already written in {target_name}, translate it into English "
            f"instead. Otherwise, translate it into natural, idiomatic {target_name}, "
            "the way a skilled native-speaking human translator would - never a stiff, "
            "literal word-for-word conversion. Preserve the original meaning, tone, "
            "register (formal/informal), and any names or numbers exactly. "
            "Output ONLY the translation itself: plain text, no quotation marks, no "
            "notes, no explanations, no labels like 'Translation:', nothing except "
            "the translated text."
        )
    return {"role": "system", "content": content}


async def translate_text(text: str, target_code: str = "auto") -> str:
    system_prompt = build_translate_system_prompt(target_code)
    response = await asyncio.to_thread(
        groq_client.chat.completions.create,
        model=GROQ_MODEL,
        messages=[system_prompt, {"role": "user", "content": text}],
        temperature=0.3,  # lower temperature - translation wants accuracy, not creative variation
    )
    return (response.choices[0].message.content or "").strip()


# ---------------------- PDF / DOCUMENT Q&A ----------------------
# Upload a PDF, then ask questions about it. The whole document never goes to
# the model - Groq's free tier has a small per-request budget (see
# GROQ_TOKEN_BUDGET) and a 40-page PDF would blow straight through it. Instead
# the text is split into chunks once, and each question pulls in only the few
# chunks that actually look relevant. That's a tiny retrieval system, and it's
# what lets a long document work at all on a free-tier model.
MAX_PDF_BYTES = 20 * 1024 * 1024  # Telegram's own bot-API download limit
MAX_PDF_CHARS = 120_000
PDF_CHUNK_CHARS = 1200
PDF_CONTEXT_CHARS = 2600  # how much document text rides along with one question


def _extract_pdf_text_sync(data: bytes) -> tuple[str, int]:
    reader = PdfReader(io.BytesIO(data))
    parts, total = [], 0
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text:
            parts.append(text)
            total += len(text)
        if total > MAX_PDF_CHARS:
            break
    return "\n".join(parts), len(reader.pages)


def _chunk_text(text: str, size: int = PDF_CHUNK_CHARS) -> list:
    words, chunks, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > size:
            chunks.append(current.strip())
            current = ""
        current += word + " "
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _relevant_chunks(chunks: list, question: str, max_chars: int = PDF_CONTEXT_CHARS) -> str:
    """Score chunks by how many of the question's words they contain. Crude,
    but it's dependency-free and works well enough on a single document -
    and it beats sending page 1 every time, which is what naive versions do."""
    terms = [w for w in re.split(r"\W+", question.lower()) if len(w) > 2]
    if not terms:
        scored = list(enumerate(chunks))[:2]
    else:
        ranked = sorted(
            enumerate(chunks),
            key=lambda pair: sum(pair[1].lower().count(term) for term in terms),
            reverse=True,
        )
        scored = [pair for pair in ranked if any(term in pair[1].lower() for term in terms)][:3]
        if not scored:  # nothing matched - fall back to the start of the document
            scored = list(enumerate(chunks))[:2]

    # Put them back in document order so the excerpt reads naturally.
    scored.sort(key=lambda pair: pair[0])
    out, used = [], 0
    for _, chunk in scored:
        if used + len(chunk) > max_chars:
            chunk = chunk[: max(0, max_chars - used)]
        if not chunk:
            break
        out.append(chunk)
        used += len(chunk)
    return "\n...\n".join(out)


def build_pdf_context_message(doc: dict, question: str) -> dict:
    excerpt = _relevant_chunks(doc["chunks"], question)
    return {
        "role": "system",
        "content": (
            f"کاربر فایلی به اسم «{doc['name']}» ({doc['pages']} صفحه) آپلود کرده و "
            "سوالش درباره‌ی همین فایله. بخش‌های مرتبط از متن فایل:\n\n"
            f"{excerpt}\n\n"
            "فقط بر اساس همین متن جواب بده. اگه جواب توی این بخش‌ها نبود، صادقانه "
            "بگو که این قسمت رو توی فایل پیدا نکردی و از کاربر بخواه دقیق‌تر بپرسه - "
            "چیزی از خودت نساز."
        ),
    }


async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    doc = update.message.document
    if doc is None:
        return

    # In a group, only take a file that's actually aimed at the bot - it
    # shouldn't grab every PDF people share with each other. In a private
    # chat, uploading a file is itself the request, so it always counts.
    if update.effective_chat.type != "private" and not await ensure_ai_ready(update, context):
        return

    name = doc.file_name or "document.pdf"
    is_pdf = (doc.mime_type == "application/pdf") or name.lower().endswith(".pdf")
    if not is_pdf:
        await update.message.reply_text(ui(context, "pdf_only"))
        return
    if PdfReader is None:
        await update.message.reply_text(ui(context, "pdf_missing_lib"))
        return
    if doc.file_size and doc.file_size > MAX_PDF_BYTES:
        await update.message.reply_text(ui(context, "pdf_too_big"))
        return

    status = await update.message.reply_text(ui(context, "pdf_reading"))
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        tg_file = await context.bot.get_file(doc.file_id)
        data = bytes(await tg_file.download_as_bytearray())
        text, pages = await asyncio.to_thread(_extract_pdf_text_sync, data)
    except Exception as e:
        print(f"[pdf] failed to read {name}: {e}")
        await status.edit_text(ui(context, "pdf_fail"))
        return

    if not text.strip():
        # Almost always a scanned PDF: images of text, no text layer to pull.
        await status.edit_text(ui(context, "pdf_no_text"))
        return

    context.user_data["pdf_doc"] = {
        "name": name,
        "pages": pages,
        "chunks": _chunk_text(text),
    }
    # A PDF is useless if the user then has to remember to turn AI mode on, so
    # uploading one puts them straight into a mode that can answer about it.
    if context.user_data.get("ai_provider") not in ("groq", "groq_search"):
        activate_groq_mode(context)
    else:
        context.user_data["ai_mode"] = True
    context.user_data["ai_off"] = False  # an explicit choice always overrides a previous /stop

    await status.edit_text(ui(context, "pdf_ready", name=name, pages=pages))


async def forget_pdf_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    had = context.user_data.pop("pdf_doc", None)
    await update.message.reply_text(
        ui(context, "pdf_cleared") if had else ui(context, "pdf_none")
    )


# ---------------------- SHARED AI HANDLER ----------------------
async def ai_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, user_text: str) -> None:
    # If this user hasn't activated AI mode, ignore their message here
    if not context.user_data.get("ai_mode"):
        return

    provider = context.user_data.get("ai_provider", "groq")
    chat_id = update.effective_chat.id
    owner = is_owner(update)

    # Maintenance mode (toggled from the admin panel) pauses the bot for
    # everyone except the owner, e.g. while you're debugging or redeploying.
    if BOT_STATS.get("maintenance") and not owner:
        await update.message.reply_text(
            "ربات موقتاً در حالت تعمیره و به‌زودی برمی‌گرده. لطفاً یکم بعد دوباره امتحان کن. 🌸"
        )
        return

    if provider == "translate":
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        target_code = context.user_data.get("translate_target", "auto")
        try:
            translated = await translate_text(user_text, target_code)
        except Exception as e:
            print(f"[ai_handler] translation failed: {e}")
            await update.message.reply_text(ui(context, "translate_fail"))
            return
        await send_ai_reply(update, context, translated)
        return

    user_id = update.effective_user.id

    history_key = f"history_{provider}"
    history = context.user_data.setdefault(history_key, [])

    # Groq (OpenAI-style) wants the system prompt as the first message in the
    # list. Gemini takes its system prompt separately via config (handled in
    # its own branch below), so it's never added to its own history list.
    # groq_search uses the same OpenAI-style chat format as groq, just with
    # a different model. The prompt is built per-user so only the configured
    # owner gets addressed as "پدر"/"Father" - everyone else gets the normal
    # polite prompt, and includes whatever long-term facts are on file for
    # this user. Long-term facts are only fetched here - once, when a fresh
    # session starts - not on every message; they're baked into the system
    # message and ride along with `history` for the rest of the session.
    if not history and provider in ("groq", "groq_search", "groq_images"):
        memory_facts = await _get_memory_facts_cached(context, user_id)
        personality = await get_personality(context, user_id)
        history.append(
            {
                "role": "system",
                "content": build_system_prompt(owner, memory_facts, personality=personality),
            }
        )

    # Add the user's new message to this provider's own memory
    history.append({"role": "user", "content": user_text})

    # Trim history so it doesn't grow forever. Keep the system message (index
    # 0, if present) no matter what - otherwise a long conversation quietly
    # loses its instructions (and the long-term-memory facts baked into it
    # above) once it scrolls past MAX_HISTORY turns, which looked just like
    # "the bot forgot" even without a redeploy.
    if len(history) > MAX_HISTORY:
        if history and history[0].get("role") == "system":
            history[:] = [history[0]] + history[-(MAX_HISTORY - 1):]
        else:
            history[:] = history[-MAX_HISTORY:]

    # Show "typing..." while we wait on the AI
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    # Plain "what time/date is it" questions are answered directly and
    # deterministically, without involving the LLM at all - see
    # detect_plain_time_query's comment for why.
    time_zone = detect_plain_time_query(user_text)
    if time_zone:
        reply_text = format_time_reply(time_zone)
        assistant_role = "model" if provider == "gemini" else "assistant"
        history.append({"role": assistant_role, "content": reply_text})
        await send_ai_reply(update, context, reply_text)
        return

    if provider == "gemini":
        # Gemini isn't wired up for real tool use here, so it still relies on
        # the lighter approach: if the user explicitly asked for a search
        # inline (e.g. "سرچ کن" / "search it"), do one DuckDuckGo search and
        # hand the snippets over as extra context for this turn only.
        if wants_web_search(user_text):
            search_results = await web_search(user_text)
            messages_for_api = history[:-1] + [
                {
                    "role": "user",
                    "content": (
                        f"{user_text}\n\n"
                        "نتایج جستجوی اینترنتی زیر رو برای پاسخ دقیق‌تر و به‌روزتر در نظر بگیر "
                        "(در صورت نیاز؛ اگه ربطی نداشت نادیده‌اش بگیر):\n"
                        f"{search_results}"
                    ),
                }
            ]
        else:
            messages_for_api = history
        memory_facts = await _get_memory_facts_cached(context, user_id)
        personality = await get_personality(context, user_id)
        pdf_doc = context.user_data.get("pdf_doc")
        if pdf_doc:
            # Gemini has no system-role messages in its history, so the
            # document excerpt rides along on this turn's user message.
            ctx_msg = build_pdf_context_message(pdf_doc, user_text)
            messages_for_api = messages_for_api[:-1] + [
                {"role": "user", "content": f"{ctx_msg['content']}\n\n---\n\n{user_text}"}
            ]
        try:
            reply_text = await call_gemini(
                messages_for_api,
                build_system_prompt(owner, memory_facts, tools_available=False, personality=personality),
            )
        except Exception as e:
            await _handle_ai_failure(update, context, history, provider, e)
            return
        assistant_role = "model"
        image_batches = []
    else:
        # Groq mode has a real tool-use agent: it can decide on its own to
        # search the web, open a specific page, check the actual current
        # time/date, look up images, or set a reminder - no keyword-matching
        # needed. The dedicated buttons (provider == "groq_search" /
        # "groq_images") just nudge the first tool call in one direction.
        force_search = provider == "groq_search"
        force_images = provider == "groq_images"

        # Tools can't reach into the chat by themselves, so hand them what
        # they need: which chat to schedule reminders for, and a list to drop
        # image results into, which gets sent right after the text reply.
        image_batches = []
        tool_ctx = {"chat_id": chat_id, "images": image_batches}

        history_for_call = history
        pdf_doc = context.user_data.get("pdf_doc")
        if pdf_doc:
            # Slip the relevant document excerpt in just before the question,
            # for this one call only - the stored history stays clean, so a
            # long PDF chat doesn't accumulate excerpt after excerpt.
            history_for_call = (
                history[:-1] + [build_pdf_context_message(pdf_doc, user_text), history[-1]]
            )

        try:
            reply_text = await run_groq_agent(
                history_for_call,
                user_id,
                force_search=force_search,
                force_images=force_images,
                tool_ctx=tool_ctx,
            )
        except Exception as e:
            await _handle_ai_failure(update, context, history, provider, e)
            return
        assistant_role = "assistant"

    # Save the model's reply into memory too
    history.append({"role": assistant_role, "content": reply_text})

    if reply_text.strip():
        await send_ai_reply(update, context, reply_text)

    # Any images the model asked for go out after the text, as real photo
    # albums replying to the user's original message. Hard-capped in total
    # so a model that ignores the "call this once" instruction still can't
    # flood the chat with a dozen photos.
    images_sent_this_reply = 0
    for batch in image_batches:
        remaining = TOTAL_IMAGES_PER_REPLY - images_sent_this_reply
        if remaining <= 0:
            break
        batch = dict(batch, count=min(batch.get("count", remaining), remaining))
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)
        sent = await send_image_batch(
            context.bot, chat_id, batch, reply_to_message_id=update.message.message_id
        )
        images_sent_this_reply += sent
        if not sent:
            await update.message.reply_text(ui(context, "images_none", query=batch["query"]))


async def _handle_ai_failure(
    update: Update, context: ContextTypes.DEFAULT_TYPE, history: list, provider: str, error: Exception
) -> None:
    # Roll back the user's turn we optimistically appended before calling the
    # AI, so a failed call doesn't leave two user messages back to back in
    # the saved history (Gemini in particular expects alternating turns).
    if history and history[-1].get("role") == "user":
        history.pop()
    print(f"[ai_handler] {provider} call failed: {error}")
    await update.message.reply_text(ui(context, "ai_call_fail"))


# ---------------------- ADMIN PANEL (owner only) ----------------------
# Everything here is gated by is_owner() at every entry point (command AND
# every button callback) - not just by hiding the /admin command from
# everyone else's /help. Even if another user somehow guesses "/admin" or a
# button's callback_data, they hit the same is_owner() check and get nothing.
ADMIN_USERS_PAGE_SIZE = 15


def build_admin_menu_keyboard() -> InlineKeyboardMarkup:
    maintenance_on = BOT_STATS.get("maintenance", False)
    keyboard = [
        [InlineKeyboardButton("📊 آمار ربات", callback_data="admin_stats")],
        [InlineKeyboardButton("👤 اطلاعات من", callback_data="admin_myinfo")],
        [InlineKeyboardButton("👥 لیست کاربران", callback_data="admin_users:0")],
        [InlineKeyboardButton("📢 پیام همگانی", callback_data="admin_broadcast_start")],
        [
            InlineKeyboardButton(
                "🛑 خاموش کردن حالت تعمیر" if maintenance_on else "🛠 روشن کردن حالت تعمیر",
                callback_data="admin_maintenance_toggle",
            )
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


async def admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Deliberately silent (no "you're not allowed" reply) for non-owners -
    # no reason to even confirm this command does something.
    if not is_owner(update):
        return
    await update.message.reply_text(
        "🔐 پنل مدیریت ربات\nیکی از گزینه‌ها رو انتخاب کن:",
        reply_markup=build_admin_menu_keyboard(),
    )


def _render_admin_stats_text() -> str:
    users = BOT_STATS.get("users", {})
    uptime = _format_duration(datetime.now(ZoneInfo("UTC")) - BOT_BOOTED_AT)
    maintenance = "روشن 🛑" if BOT_STATS.get("maintenance") else "خاموش ✅"
    top = sorted(users.items(), key=lambda kv: kv[1].get("message_count", 0), reverse=True)[:5]
    top_lines = [
        f"  • {('@' + info['username']) if info.get('username') else (info.get('first_name') or uid)} — {info.get('message_count', 0)} پیام"
        for uid, info in top
    ]
    top_block = "\n".join(top_lines) if top_lines else "  (هنوز کسی پیام نداده)"
    return (
        "📊 آمار ربات\n\n"
        f"تعداد کاربران: {len(users)}\n"
        f"کل پیام‌های پردازش‌شده: {BOT_STATS.get('total_messages', 0)}\n"
        f"مدت روشن بودن (از آخرین ری‌استارت): {uptime}\n"
        f"حالت تعمیر: {maintenance}\n\n"
        f"فعال‌ترین کاربران:\n{top_block}"
    )


def _render_admin_myinfo_text(update: Update) -> str:
    user = update.effective_user
    chat = update.effective_chat
    lines = [
        "👤 اطلاعات تلگرام تو (مالک ربات)\n",
        f"آیدی عددی: {user.id}",
        f"یوزرنیم: @{user.username}" if user.username else "یوزرنیم: (نداره)",
        f"نام: {(user.first_name or '') + ' ' + (user.last_name or '')}".strip(),
        f"زبان تلگرام: {user.language_code or 'نامشخص'}",
        f"حساب پرمیوم: {'بله' if getattr(user, 'is_premium', False) else 'خیر'}",
        f"آیدی این چت: {chat.id}",
    ]
    return "\n".join(lines)


def _render_admin_users_page(page: int):
    users = BOT_STATS.get("users", {})
    items = sorted(users.items(), key=lambda kv: kv[1].get("last_seen", ""), reverse=True)
    start = page * ADMIN_USERS_PAGE_SIZE
    page_items = items[start : start + ADMIN_USERS_PAGE_SIZE]
    total_pages = max(1, (len(items) + ADMIN_USERS_PAGE_SIZE - 1) // ADMIN_USERS_PAGE_SIZE)

    if not page_items:
        text = "👥 هنوز هیچ کاربری با ربات صحبت نکرده."
    else:
        lines = [f"👥 لیست کاربران (صفحه {page + 1} از {total_pages})\n"]
        for uid, info in page_items:
            label = f"@{info['username']}" if info.get("username") else (info.get("first_name") or "بدون نام")
            lines.append(f"• {label} — آیدی: {uid} — {info.get('message_count', 0)} پیام")
        text = "\n".join(lines)

    nav_row = []
    if page > 0:
        nav_row.append(InlineKeyboardButton("« قبلی", callback_data=f"admin_users:{page - 1}"))
    if start + ADMIN_USERS_PAGE_SIZE < len(items):
        nav_row.append(InlineKeyboardButton("بعدی »", callback_data=f"admin_users:{page + 1}"))
    keyboard = ([nav_row] if nav_row else []) + [
        [InlineKeyboardButton("« برگشت به پنل", callback_data="admin_menu")]
    ]
    return text, InlineKeyboardMarkup(keyboard)


BACK_TO_ADMIN_KEYBOARD = InlineKeyboardMarkup(
    [[InlineKeyboardButton("« برگشت به پنل", callback_data="admin_menu")]]
)


async def handle_admin_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not is_owner(update):
        # A popup only this (non-owner) user sees, so we're not silently
        # editing someone else's message into an admin screen.
        await query.answer("این بخش فقط برای مالک رباته.", show_alert=True)
        return

    data = query.data

    if data == "admin_menu":
        await query.edit_message_text(
            "🔐 پنل مدیریت ربات\nیکی از گزینه‌ها رو انتخاب کن:",
            reply_markup=build_admin_menu_keyboard(),
        )

    elif data == "admin_stats":
        await query.edit_message_text(_render_admin_stats_text(), reply_markup=BACK_TO_ADMIN_KEYBOARD)

    elif data == "admin_myinfo":
        await query.edit_message_text(_render_admin_myinfo_text(update), reply_markup=BACK_TO_ADMIN_KEYBOARD)

    elif data.startswith("admin_users:"):
        page = int(data.split(":", 1)[1])
        text, markup = _render_admin_users_page(page)
        await query.edit_message_text(text, reply_markup=markup)

    elif data == "admin_maintenance_toggle":
        BOT_STATS["maintenance"] = not BOT_STATS.get("maintenance", False)
        await asyncio.to_thread(_save_stats)
        state_text = (
            "روشنه 🛑 (بقیه‌ی کاربرا فعلاً نمی‌تونن از ربات استفاده کنن)"
            if BOT_STATS["maintenance"]
            else "خاموشه ✅ (همه می‌تونن عادی استفاده کنن)"
        )
        await query.edit_message_text(f"حالت تعمیر الان {state_text}", reply_markup=BACK_TO_ADMIN_KEYBOARD)

    elif data == "admin_broadcast_start":
        context.user_data["awaiting_broadcast"] = True
        await query.edit_message_text(
            "متنی که می‌خوای برای همه‌ی کاربرا "
            f"(فعلاً {len(BOT_STATS.get('users', {}))} نفر) ارسال بشه رو بفرست.\n"
            "برای لغو، دکمه‌ی لغو رو تو مرحله‌ی بعد بزن."
        )

    elif data == "admin_broadcast_send":
        text = context.user_data.pop("broadcast_pending", None)
        if not text:
            await query.edit_message_text("پیامی برای ارسال پیدا نشد - دوباره از پنل شروع کن.")
            return
        await query.edit_message_text("در حال ارسال... ⏳")
        sent, failed = 0, 0
        for uid in list(BOT_STATS.get("users", {}).keys()):
            try:
                await context.bot.send_message(chat_id=int(uid), text=text)
                sent += 1
            except Exception as e:
                failed += 1
                print(f"[broadcast] failed to message {uid}: {e}")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"✅ پیام همگانی ارسال شد.\nموفق: {sent}\nناموفق (مثلاً ربات رو بلاک کردن): {failed}",
        )

    elif data == "admin_broadcast_cancel":
        context.user_data.pop("broadcast_pending", None)
        context.user_data.pop("awaiting_broadcast", None)
        await query.edit_message_text("پیام همگانی لغو شد.")


# ---------------------- MAIN TEXT ROUTER ----------------------
# This decides what to do with a plain text message depending on user's current mode
async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Owner mid-broadcast-flow: this message is the broadcast text itself,
    # not a normal AI-mode message. Checked before ai_mode so it can't be
    # accidentally swallowed by whatever provider is currently active.
    if is_owner(update) and context.user_data.get("awaiting_broadcast"):
        context.user_data["awaiting_broadcast"] = False
        text = update.message.text
        context.user_data["broadcast_pending"] = text
        preview = text if len(text) <= 500 else text[:500] + "…"
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("✅ ارسال", callback_data="admin_broadcast_send"),
                    InlineKeyboardButton("❌ لغو", callback_data="admin_broadcast_cancel"),
                ]
            ]
        )
        await update.message.reply_text(
            f"پیش‌نمایش پیام همگانی برای {len(BOT_STATS.get('users', {}))} کاربر:\n\n{preview}",
            reply_markup=keyboard,
        )
        return

    # The user tapped "write my own personality" and this message is that
    # description, not a question for the AI. Checked before ai_mode for the
    # same reason as the broadcast flow above.
    if context.user_data.get("awaiting_personality"):
        context.user_data["awaiting_personality"] = False
        description = (update.message.text or "").strip()
        if description:
            await set_personality(context, update.effective_user.id, "custom", description)
            await update.message.reply_text(ui(context, "personality_custom_set"))
        else:
            await update.message.reply_text(ui(context, "personality_custom_prompt"))
        return

    # No command, no button, no "mode" needed: if this message is meant for
    # the bot, the gate switches AI mode on and we just answer it.
    if not await ensure_ai_ready(update, context):
        return

    text = _strip_bot_mention(update.message.text or "", context.bot.username)
    if not text:
        return
    await ai_handler(update, context, text)


# ---------------------- INLINE MODE (@thebot <question> in ANY chat) ----------------------
# This is the growth feature: someone types "@MyBigPotatobot what's the
# capital of Chile" inside a group the bot isn't even a member of, and the
# answer gets posted with the bot's name on it. Everyone in that chat sees it.
#
# NOTE: inline mode must also be switched on in BotFather (/setinline), or
# Telegram never sends these updates at all.
INLINE_MIN_QUERY_CHARS = 3
INLINE_CACHE_TTL = 120
_inline_cache: dict = {}  # query -> (timestamp, answer)
_inline_inflight: set = set()

INLINE_SYSTEM_PROMPT = (
    "You are answering a question that will be posted publicly into a Telegram "
    "chat, so keep it self-contained and under about 60 words. Plain text only: "
    "no Markdown, no HTML, no links unless the question is about one. Always "
    "answer in the same language the question was asked in."
)


async def _inline_answer(query: str) -> str:
    cached = _inline_cache.get(query)
    if cached and time.time() - cached[0] < INLINE_CACHE_TTL:
        return cached[1]

    response = await asyncio.to_thread(
        groq_client.chat.completions.create,
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": INLINE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        max_tokens=400,
    )
    answer = (response.choices[0].message.content or "").strip()
    if answer:
        _inline_cache[query] = (time.time(), answer)
        if len(_inline_cache) > 300:  # bounded - this is just a typing-burst cache
            for key in list(_inline_cache)[:100]:
                _inline_cache.pop(key, None)
    return answer


async def inline_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    inline_query = update.inline_query
    query = (inline_query.query or "").strip()

    if len(query) < INLINE_MIN_QUERY_CHARS:
        await inline_query.answer([], cache_time=5, is_personal=True)
        return

    # Telegram fires one update per keystroke. Without this guard a user
    # typing a sentence would kick off a dozen model calls and burn the free
    # tier's rate limit before they even finish the question.
    if query in _inline_inflight:
        return
    _inline_inflight.add(query)
    try:
        answer = await _inline_answer(query)
    except Exception as e:
        print(f"[inline] failed for '{query}': {e}")
        answer = ""
    finally:
        _inline_inflight.discard(query)

    if not answer:
        return

    message_text = f"❓ {query}\n\n{answer}\n\n— via {BOT_USERNAME}"
    results = [
        InlineQueryResultArticle(
            id=uuid.uuid4().hex,
            title=ui(context, "inline_result_title"),
            description=answer[:120],
            input_message_content=InputTextMessageContent(message_text, parse_mode=None),
        )
    ]
    await inline_query.answer(results, cache_time=30, is_personal=True)


# ---------------------- GLOBAL ERROR HANDLER ----------------------
# Catches anything that slips past the try/except blocks above (e.g. errors
# in button_click or start), so the bot logs it and, where possible, tells
# the user something went wrong instead of just going silent.
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    print(f"[global error handler] {context.error}")
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(ui(context, "global_error"))
        except Exception:
            pass  # if we can't even send the error message, just give up quietly


# ---------------------- STARTUP HOOK ----------------------
async def on_startup(application) -> None:
    # Scheduled jobs only live in RAM, so every restart has to re-register the
    # reminders that are still stored in Upstash - otherwise a redeploy would
    # silently kill everybody's reminders while still showing them in
    # /reminders, which is worse than not having the feature at all.
    try:
        restored = await reschedule_all_reminders()
        if restored:
            print(f"[reminders] rescheduled {restored} reminder(s) after startup")
    except Exception as e:
        print(f"[reminders] failed to reschedule at startup: {e}")

    # The "/" menu in Telegram's compose box - free discoverability for the
    # new commands, without the user ever opening /help.
    try:
        await application.bot.set_my_commands(
            [
                BotCommand("start", "منوی اصلی / Main menu"),
                BotCommand("groq", "چت هوش مصنوعی / AI chat"),
                BotCommand("websearch", "جستجوی اینترنتی / Web search"),
                BotCommand("images", "جستجوی عکس / Image search"),
                BotCommand("translate", "ترجمه / Translate"),
                BotCommand("remind", "یادآوری / Set a reminder"),
                BotCommand("reminders", "لیست یادآوری‌ها / My reminders"),
                BotCommand("personality", "شخصیت ربات / Bot personality"),
                BotCommand("language", "زبان رابط / Interface language"),
                BotCommand("help", "راهنما / Help"),
                BotCommand("stop", "خاموش کردن حالت AI / Turn off AI mode"),
            ]
        )
    except Exception as e:
        print(f"[startup] failed to set command menu: {e}")

    if WEBAPP_URL:
        try:
            await application.bot.set_chat_menu_button(
                menu_button=MenuButtonWebApp(
                    text="Open App",
                    web_app=WebAppInfo(url=WEBAPP_URL + "/app"),
                )
            )
            print(f"[webapp] Mini App enabled at {WEBAPP_URL}/app")
        except Exception as e:
            print(f"[webapp] failed to set Open App menu button: {e}")
    else:
        print("[webapp] Set WEBAPP_URL on Render to enable the Telegram Open App button.")


# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    # Newer Python versions (3.14+) no longer auto-create an event loop for the
    # main thread. python-telegram-bot's run_polling() expects one to already
    # exist, so we create and set it ourselves here - this makes the bot work
    # regardless of which Python version Render happens to run it with.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Start the keep-alive web server in the background so Render sees an open port
    threading.Thread(target=run_flask, daemon=True).start()

    # A tuned connection config: more generous timeouts than the library
    # defaults (5s) so slower calls (voice/tool-use turns) don't time out
    # under load, and a dedicated connection pool for get_updates so it
    # doesn't compete with regular bot API calls for connections. This
    # doesn't eliminate the occasional "Server disconnected without sending
    # a response" hiccup (a pooled connection going stale is a normal,
    # self-recovering network event Telegram/httpx both retry around), but
    # it does make it noticeably rarer.
    request = HTTPXRequest(connection_pool_size=20, connect_timeout=15.0, read_timeout=20.0, write_timeout=20.0, pool_timeout=10.0)
    get_updates_request = HTTPXRequest(connection_pool_size=4, connect_timeout=15.0, read_timeout=40.0, pool_timeout=10.0)

    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .request(request)
        .get_updates_request(get_updates_request)
        .post_init(on_startup)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop_ai))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("groq", groq_command))
    app.add_handler(CommandHandler("gemini", gemini_command))
    app.add_handler(CommandHandler("websearch", websearch_command))
    app.add_handler(CommandHandler("translate", translate_command))
    app.add_handler(CommandHandler("language", language_command))
    app.add_handler(CommandHandler("lang", language_command))  # short alias
    app.add_handler(CommandHandler("admin", admin_command))
    app.add_handler(CommandHandler("images", images_command))
    app.add_handler(CommandHandler("image", images_command))  # alias
    app.add_handler(CommandHandler("personality", personality_command))
    app.add_handler(CommandHandler("remind", remind_command))
    app.add_handler(CommandHandler("reminders", reminders_command))
    app.add_handler(CommandHandler("forgetpdf", forget_pdf_command))
    app.add_handler(CallbackQueryHandler(button_click))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))
    app.add_handler(MessageHandler(filters.VOICE, voice_handler))
    app.add_handler(MessageHandler(filters.Document.ALL, document_handler))
    app.add_handler(InlineQueryHandler(inline_query_handler))
    app.add_error_handler(error_handler)

    # Let background helpers (the AI's set_reminder tool, job callbacks) reach
    # the running application without passing it through every call.
    BOT_APP = app

    if app.job_queue is None:
        print(
            "[reminders] JobQueue is not available - reminders are disabled. "
            "Install it with: pip install \"python-telegram-bot[job-queue]\""
        )

    # group=-1 runs before every other handler above, on every update that
    # has an effective_user - this is what feeds real numbers to the admin
    # panel's stats/user-list, without touching any of the actual feature logic.
    app.add_handler(MessageHandler(filters.ALL, track_activity), group=-1)
    app.add_handler(CallbackQueryHandler(track_activity), group=-1)

    app.run_polling()

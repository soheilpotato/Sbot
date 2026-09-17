import os
import json
import io
import uuid
import subprocess
import threading
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from flask import Flask
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackQueryHandler,
)
from telegram.request import HTTPXRequest
from groq import Groq, BadRequestError
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

# ----------------------------------------------------------------------------------------------------------------------
# Load .env locally. On Render this file won't exist, and that's fine -
# Render injects environment variables directly, so os.environ still has them.
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

TOKEN = os.environ.get("TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
BOT_USERNAME = "@MyBigPotatobot"

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

groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "openai/gpt-oss-120b"  # check console.groq.com/docs/models - Groq's free lineup changes often

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"

# How many past messages (user+model combined) to keep per user/provider, to stop memory growing forever
MAX_HISTORY = 20

BASE_SYSTEM_PROMPT = """
You are a helpful, warm, and unfailingly polite AI assistant inside a Telegram bot.

PERSONALITY:
- Your personality has a small, natural touch of girly/cute charm - think a
  friendly, upbeat young woman texting a friend, not an over-the-top anime
  character. Keep it light and occasional, never constant.
- This can show up as: a warm, cheerful tone, an occasional soft emoji
  (like 🌸✨💕😊 - pick one at most per message, and often use none at all),
  or a playful/affectionate word choice here and there.
- Do NOT use baby talk, "uwu"/"owo"-style speech, excessive giggling,
  stretched-out words ("heyyy", "yesss"), or stacks of emoji/kaomoji. That
  reads as cringy, not cute - avoid it entirely.
- Never let this override clarity or usefulness. For serious, technical, or
  sensitive topics, drop the cute touches completely and just be direct and
  helpful.

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


def build_system_prompt(owner: bool) -> str:
    if owner:
        return BASE_SYSTEM_PROMPT + "\n" + OWNER_ADDRESS_INSTRUCTION
    return BASE_SYSTEM_PROMPT
# ----------------------------------------------------------------------------------------------------------------------


# ---------------------- TINY WEB SERVER (keeps Render's free Web Service happy) ----------------------
# Render's free tier only runs "Web Services" that bind to a port - it has no free
# always-on Background Worker. This Flask app just answers health-check pings so
# Render is satisfied, while the real bot logic runs via polling in the background.
flask_app = Flask(__name__)


@flask_app.route("/")
def health_check():
    return "Bot is alive", 200


def run_flask():
    port = int(os.environ.get("PORT", 10000))
    flask_app.run(host="0.0.0.0", port=port)


# ---------------------- START MENU ----------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [
            InlineKeyboardButton("🤖 AI Groq", callback_data="ai_groq"),
            InlineKeyboardButton("🤖 AI Gemini", callback_data="ai_gemini"),
        ],
        [
            InlineKeyboardButton("🔎 جستجوی اینترنتی", callback_data="ai_groq_search"),
        ],
        [
            InlineKeyboardButton("🌐 ترجمه چندزبانه", callback_data="ai_translate_menu"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    greeting = "خوش آمدید! من ربات شخصی شما هستم، هرطور که مایلید ازم استفاده کنید."
    await update.message.reply_text(greeting, reply_markup=reply_markup)


# ---------------------- HELP ----------------------
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "این کارایی هستن که از دستم برمیاد:\n\n"
        "🤖 هوش مصنوعی (Groq یا Gemini) - چت آزاد و پاسخ به سوالات\n"
        "🔎 جستجوی اینترنتی - برای گرفتن اطلاعات به‌روز\n"
        "🌐 ترجمه چندزبانه - ترجمه هر متنی به هر زبونی\n"
        "🎙️ تبدیل متن به صدا و صدا به متن\n\n"
        "دستورات:\n"
        "/start - نمایش منوی اصلی\n"
        "/stop - خاموش کردن حالت هوش مصنوعی\n"
        "/help - همین راهنما\n\n"
        "برای شروع فقط /start رو بزن و از دکمه‌ها انتخاب کن."
    )
    await update.message.reply_text(help_text)


# ---------------------- BUTTON HANDLER ----------------------
async def button_click(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()  # tells telegram the button press was received

    if query.data == "ai_groq":
        context.user_data["ai_mode"] = True
        context.user_data["ai_provider"] = "groq"
        context.user_data.setdefault("history_groq", [])  # init memory for this user/provider
        await query.edit_message_text("حالت هوش مصنوعی (Groq) فعال شد🤖!")

    elif query.data == "ai_gemini":
        context.user_data["ai_mode"] = True
        context.user_data["ai_provider"] = "gemini"
        context.user_data.setdefault("history_gemini", [])  # init memory for this user/provider
        await query.edit_message_text("حالت هوش مصنوعی (Gemini) فعال شد🤖!")

    elif query.data == "ai_groq_search":
        context.user_data["ai_mode"] = True
        context.user_data["ai_provider"] = "groq_search"
        context.user_data.setdefault("history_groq_search", [])  # init memory for this user/provider
        await query.edit_message_text("حالت جستجوی اینترنتی (Groq) فعال شد🔎!")

    elif query.data == "ai_translate_menu":
        await query.edit_message_text(
            "زبان مقصد رو انتخاب کن؛ از این به بعد هر متنی به هر زبونی بفرستی، به همون زبون ترجمه می‌کنم:",
            reply_markup=build_translate_language_keyboard(),
        )

    elif query.data.startswith("translate_lang:"):
        target_code = query.data.split(":", 1)[1]
        context.user_data["ai_mode"] = True
        context.user_data["ai_provider"] = "translate"
        context.user_data["translate_target"] = target_code
        if target_code == "auto":
            msg = "حالت ترجمه خودکار فعال شد🌐 (فارسی⇄انگلیسی). هر متنی بفرستی ترجمه می‌کنم."
        else:
            target_label = LANGUAGES[target_code]["label"]
            msg = (
                f"حالت ترجمه فعال شد🌐! هر متنی به هر زبونی بفرستی، "
                f"به {target_label} ترجمه می‌کنم.\nبرای عوض کردن زبان مقصد دوباره /start رو بزن."
            )
        await query.edit_message_text(msg)

    elif query.data.startswith("tts:"):
        await handle_tts_button(update, context)

    elif query.data.startswith("trmenu:"):
        await handle_translate_menu_button(update, context)

    elif query.data.startswith("trcancel:"):
        await handle_translate_cancel_button(update, context)

    elif query.data.startswith("trlang:"):
        await handle_translate_lang_button(update, context)


# ---------------------- STOP AI MODE (optional command) ----------------------
async def stop_ai(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["ai_mode"] = False
    await update.message.reply_text("حالت هوش مصنوعی خاموش شد.")


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
]


async def execute_tool(tool_call) -> str:
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}

    if name == "web_search":
        return await web_search(args.get("query", ""))
    if name == "fetch_page":
        return await fetch_page(args.get("url", ""))
    if name == "get_current_datetime":
        return get_current_datetime(args.get("timezone", "UTC"))
    return f"ابزار ناشناخته: {name}"


# ---------------------- GROQ CALL (with real tool use) ----------------------
MAX_TOOL_STEPS = 6  # how many search/fetch/time round-trips the model can make per message

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


async def run_groq_agent(history: list, model: str = GROQ_MODEL, force_search: bool = False) -> str:
    # The user's actual question, so we can rebuild a clean prompt later.
    original_user_text = history[-1]["content"] if history and history[-1].get("role") == "user" else ""

    # Work on a local copy: the back-and-forth tool-call turns are only needed
    # to produce this one reply and are never saved into the persisted
    # per-user history (that would bloat it fast and isn't needed later).
    working = list(history)
    if force_search:
        working = working + [FORCE_SEARCH_HINT]

    gathered_info = []  # plain-text notes on what each tool call found

    for _ in range(MAX_TOOL_STEPS):
        try:
            response = await asyncio.to_thread(
                groq_client.chat.completions.create,
                model=model,
                messages=working,
                tools=TOOLS,
                tool_choice="auto",
            )
        except BadRequestError as e:
            # Groq's gpt-oss endpoint can get into a bad state once a
            # transcript has several tool-call turns in it - it may keep
            # trying to emit more tool calls almost regardless of what we
            # ask for next, and any mismatch throws a 400. Once that
            # happens, stop asking it for more tools and go straight to the
            # clean, tool-free final answer below instead of retrying the
            # same broken pattern.
            print(f"[run_groq_agent] tool round failed, giving up on tools: {e}")
            break

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
            result = await execute_tool(tc)
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

    final_messages = history[:-1] + [{"role": "user", "content": final_prompt}]

    try:
        response = await asyncio.to_thread(
            groq_client.chat.completions.create,
            model=model,
            messages=final_messages,
        )
        return response.choices[0].message.content or "ببخشید، نتونستم جواب بدم. دوباره امتحان کن."
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
}
DEFAULT_TTS_VOICE = "en-US-AriaNeural"

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


def build_translate_language_keyboard() -> InlineKeyboardMarkup:
    rows, row = [], []
    for code, info in LANGUAGES.items():
        row.append(InlineKeyboardButton(info["label"], callback_data=f"translate_lang:{code}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("🔄 خودکار (فارسی⇄انگلیسی)", callback_data="translate_lang:auto")])
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


def build_message_translate_keyboard(tts_id: str) -> InlineKeyboardMarkup:
    rows, row = [], []
    for code, info in LANGUAGES.items():
        row.append(InlineKeyboardButton(info["label"], callback_data=f"trlang:{tts_id}:{code}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("❌ لغو", callback_data=f"trcancel:{tts_id}")])
    return InlineKeyboardMarkup(rows)


def build_ai_reply_keyboard(tts_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🔊 تبدیل به صدا", callback_data=f"tts:{tts_id}")],
            [InlineKeyboardButton("🌐 ترجمه", callback_data=f"trmenu:{tts_id}")],
        ]
    )


async def send_ai_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, reply_text: str) -> None:
    # Every AI reply goes out through here so it always gets the "convert to
    # voice" and "translate" buttons underneath it. reply_to_message_id is
    # remembered so the voice version (sent later, from a button press) can
    # still quote the same original user message.
    tts_id = _remember_tts_text(context, reply_text, update.message.message_id)
    keyboard = build_ai_reply_keyboard(tts_id)
    await update.message.reply_text(reply_text, parse_mode=None, reply_markup=keyboard)


async def handle_translate_menu_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    tts_id = query.data.split(":", 1)[1]
    if tts_id not in context.chat_data.get("tts_texts", {}):
        await query.message.reply_text("این پیام دیگه برای ترجمه در دسترس نیست.")
        return
    await query.edit_message_reply_markup(reply_markup=build_message_translate_keyboard(tts_id))


async def handle_translate_cancel_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    tts_id = query.data.split(":", 1)[1]
    await query.edit_message_reply_markup(reply_markup=build_ai_reply_keyboard(tts_id))


async def handle_translate_lang_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    _, tts_id, target_code = query.data.split(":", 2)

    entry = context.chat_data.get("tts_texts", {}).get(tts_id)
    # Restore the normal keyboard on the original message either way, so it
    # doesn't get stuck showing the language picker.
    try:
        await query.edit_message_reply_markup(reply_markup=build_ai_reply_keyboard(tts_id))
    except Exception:
        pass

    if not entry:
        await query.message.reply_text("این پیام دیگه برای ترجمه در دسترس نیست.")
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    try:
        translated = await translate_text(entry["text"], target_code)
    except Exception as e:
        print(f"[handle_translate_lang_button] translation failed: {e}")
        await query.message.reply_text("ترجمه با خطا مواجه شد. لطفاً دوباره امتحان کن.")
        return

    new_tts_id = _remember_tts_text(context, translated, entry["reply_to_message_id"])
    await query.message.reply_text(
        translated, parse_mode=None, reply_markup=build_ai_reply_keyboard(new_tts_id)
    )


async def handle_tts_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    tts_id = query.data.split(":", 1)[1]
    entry = context.chat_data.get("tts_texts", {}).get(tts_id)
    if not entry:
        await query.message.reply_text("این پیام دیگه برای تبدیل به صدا در دسترس نیست.")
        return
    text = entry["text"]
    reply_to_message_id = entry["reply_to_message_id"]

    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.RECORD_VOICE)

    try:
        ogg_bytes = await synthesize_speech(text)
    except Exception as e:
        print(f"[handle_tts_button] TTS failed: {e}")
        await query.message.reply_text("تبدیل به صدا با خطا مواجه شد. لطفاً دوباره امتحان کن.")
        return

    # "Replace" the text message with a real voice-note version of the same
    # content, still replying to the same original user message so the
    # conversation thread doesn't lose context.
    try:
        await query.message.delete()
    except Exception:
        pass  # message may already be gone / too old to delete - not critical

    await context.bot.send_voice(
        chat_id=chat_id,
        voice=io.BytesIO(ogg_bytes),
        filename="voice.ogg",
        reply_to_message_id=reply_to_message_id,
    )



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
    # Same gate as text messages: only act on voice notes while AI mode is on
    if not context.user_data.get("ai_mode"):
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
        await update.message.reply_text(
            "نتونستم صدا رو تبدیل به متن کنم. لطفاً دوباره امتحان کن یا پیامت رو تایپ کن."
        )
        return

    if not user_text:
        await update.message.reply_text("متنی از پیام صوتی استخراج نشد. لطفاً دوباره امتحان کن.")
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
    if target_code == "auto" or target_code not in LANGUAGES:
        content = _AUTO_TRANSLATE_PROMPT_TEXT
    else:
        target_name = LANGUAGES[target_code]["name"]
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


# ---------------------- SHARED AI HANDLER ----------------------
async def ai_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, user_text: str) -> None:
    # If this user hasn't activated AI mode, ignore their message here
    if not context.user_data.get("ai_mode"):
        return

    provider = context.user_data.get("ai_provider", "groq")
    chat_id = update.effective_chat.id
    owner = is_owner(update)

    if provider == "translate":
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        target_code = context.user_data.get("translate_target", "auto")
        try:
            translated = await translate_text(user_text, target_code)
        except Exception as e:
            print(f"[ai_handler] translation failed: {e}")
            msg = "ترجمه با خطا مواجه شد. لطفاً دوباره امتحان کن."
            await update.message.reply_text(msg)
            return
        await send_ai_reply(update, context, translated)
        return

    history_key = f"history_{provider}"
    history = context.user_data.setdefault(history_key, [])

    # Groq (OpenAI-style) wants the system prompt as the first message in the
    # list. Gemini takes its system prompt separately via config, so it's
    # never added to its own history list. groq_search uses the same
    # OpenAI-style chat format as groq, just with a different model. The
    # prompt is built per-user so only the configured owner gets addressed
    # as "پدر"/"Father" - everyone else gets the normal polite prompt.
    if not history and provider in ("groq", "groq_search"):
        history.append({"role": "system", "content": build_system_prompt(owner)})

    # Add the user's new message to this provider's own memory
    history.append({"role": "user", "content": user_text})

    # Trim history so it doesn't grow forever
    if len(history) > MAX_HISTORY:
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
        try:
            reply_text = await call_gemini(messages_for_api, build_system_prompt(owner))
        except Exception as e:
            await _handle_ai_failure(update, history, provider, e)
            return
        assistant_role = "model"
    else:
        # Groq mode has a real tool-use agent: it can decide on its own to
        # search the web, open a specific page, or check the actual current
        # time/date - no keyword-matching needed. The dedicated search button
        # (provider == "groq_search") just forces the first search to happen.
        force_search = provider == "groq_search"
        try:
            reply_text = await run_groq_agent(history, force_search=force_search)
        except Exception as e:
            await _handle_ai_failure(update, history, provider, e)
            return
        assistant_role = "assistant"

    # Save the model's reply into memory too
    history.append({"role": assistant_role, "content": reply_text})

    await send_ai_reply(update, context, reply_text)


async def _handle_ai_failure(update: Update, history: list, provider: str, error: Exception) -> None:
    # Roll back the user's turn we optimistically appended before calling the
    # AI, so a failed call doesn't leave two user messages back to back in
    # the saved history (Gemini in particular expects alternating turns).
    if history and history[-1].get("role") == "user":
        history.pop()
    print(f"[ai_handler] {provider} call failed: {error}")
    await update.message.reply_text(
        "یه خطا توی گرفتن جواب از هوش مصنوعی پیش اومد (ممکنه موقتی باشه). لطفاً دوباره امتحان کن."
    )


# ---------------------- MAIN TEXT ROUTER ----------------------
# This decides what to do with a plain text message depending on user's current mode
async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.user_data.get("ai_mode"):
        await ai_handler(update, context, update.message.text)
        return

    # Not in AI mode: ignore other text messages.
    # Add other non-AI logic here if needed.


# ---------------------- GLOBAL ERROR HANDLER ----------------------
# Catches anything that slips past the try/except blocks above (e.g. errors
# in button_click or start), so the bot logs it and, where possible, tells
# the user something went wrong instead of just going silent.
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    print(f"[global error handler] {context.error}")
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "یه خطای غیرمنتظره پیش اومد. لطفاً دوباره امتحان کن."
            )
        except Exception:
            pass  # if we can't even send the error message, just give up quietly


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
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop_ai))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(button_click))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))
    app.add_handler(MessageHandler(filters.VOICE, voice_handler))
    app.add_error_handler(error_handler)

    app.run_polling()

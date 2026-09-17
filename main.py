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
            InlineKeyboardButton(ui(context, "btn_ai_groq"), callback_data="ai_groq"),
            InlineKeyboardButton(ui(context, "btn_ai_gemini"), callback_data="ai_gemini"),
        ],
        [
            InlineKeyboardButton(ui(context, "btn_web_search"), callback_data="ai_groq_search"),
        ],
        [
            InlineKeyboardButton(ui(context, "btn_translate_menu"), callback_data="ai_translate_menu"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(ui(context, "start_greeting"), reply_markup=reply_markup)


# ---------------------- HELP ----------------------
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(ui(context, "help_text"))


# ---------------------- SHARED MODE-ACTIVATION HELPERS ----------------------
# Used by both the inline buttons (button_click) and the equivalent slash
# commands below, so the two entry points never drift out of sync.
def activate_groq_mode(context: ContextTypes.DEFAULT_TYPE) -> str:
    context.user_data["ai_mode"] = True
    context.user_data["ai_provider"] = "groq"
    context.user_data.setdefault("history_groq", [])
    return ui(context, "mode_groq_on")


def activate_gemini_mode(context: ContextTypes.DEFAULT_TYPE) -> str:
    context.user_data["ai_mode"] = True
    context.user_data["ai_provider"] = "gemini"
    context.user_data.setdefault("history_gemini", [])
    return ui(context, "mode_gemini_on")


def activate_search_mode(context: ContextTypes.DEFAULT_TYPE) -> str:
    context.user_data["ai_mode"] = True
    context.user_data["ai_provider"] = "groq_search"
    context.user_data.setdefault("history_groq_search", [])
    return ui(context, "mode_search_on")


def activate_translate_mode(context: ContextTypes.DEFAULT_TYPE, target: str) -> str:
    """target can be 'auto', a known LANGUAGES code, or an arbitrary
    free-text language name (e.g. from the /translate command) that isn't
    in the LANGUAGES dict at all - the underlying AI translator understands
    language names directly, so this isn't limited to the dict's contents."""
    context.user_data["ai_mode"] = True
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


# ---------------------- STOP AI MODE (optional command) ----------------------
async def stop_ai(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["ai_mode"] = False
    await update.message.reply_text(ui(context, "stop_ai_msg"))


# ---------------------- DIRECT MODE COMMANDS (shortcuts for the /start buttons) ----------------------
async def groq_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(activate_groq_mode(context))


async def gemini_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(activate_gemini_mode(context))


async def websearch_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(activate_search_mode(context))


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
    "en":    {"label": "🇬🇧 English",    "name": "English",    "voice": "en-US-AnaNeural"},
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
DEFAULT_TTS_VOICE = "en-US-AnaNeural"

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
        "stop_ai_msg": "حالت هوش مصنوعی خاموش شد.",
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
        "stop_ai_msg": "AI mode turned off.",
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
    },
    "ar": {
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
    template = table.get(key, UI_STRINGS["fa"].get(key, key))
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


def _tts_style_for_voice(voice: str) -> tuple[str, str]:
    # Voice-specific tuning is intentional: a single pitch/rate setting makes
    # some languages sound unnaturally adult or robotic.
    if voice == "fa-IR-DilaraNeural":
        # Persian has only two standard Microsoft voices (Dilara/Farid), so
        # make Dilara noticeably lighter/younger with a brighter pitch and
        # slightly quicker delivery.
        return "+8%", "+20Hz"
    if voice == "en-US-AnaNeural":
        # Ana is Microsoft's explicitly cute/cartoon-oriented English voice.
        # Keep the processing gentle so she sounds young rather than squeaky.
        return "+4%", "+10Hz"
    # Leave the other language voices essentially natural.
    return "+0%", "+0Hz"


async def _edge_tts_mp3(text: str, voice: str) -> bytes:
    rate, pitch = _tts_style_for_voice(voice)
    communicate = edge_tts.Communicate(
        text,
        voice,
        rate=rate,
        pitch=pitch,
    )
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def _mp3_to_ogg_opus_sync(mp3_bytes: bytes, pitch_lift: float = 1.0) -> bytes:
    # Telegram's round, tap-to-play voice bubble (send_voice) requires OGG
    # container + Opus codec specifically - edge-tts only gives mp3, so this
    # shells out to ffmpeg to convert. ffmpeg must be installed and on PATH.
    process = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", "pipe:0",
            "-vn",
            *(
                ["-filter:a", f"asetrate=24000*{pitch_lift:.4f},aresample=24000,atempo={1.0/pitch_lift:.4f}"]
                if abs(pitch_lift - 1.0) > 0.0001
                else []
            ),
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
    # Persian benefits from a small additional formant-free pitch lift after
    # synthesis. This keeps the speaking speed nearly unchanged while making
    # Dilara less mature-sounding. English Ana already has a cute/cartoon
    # voice profile, so it does not need this extra processing.
    pitch_lift = 1.05 if voice == "fa-IR-DilaraNeural" else 1.0
    # ffmpeg is a blocking subprocess call, so push it to a thread like the
    # other blocking calls in this file, to avoid blocking the bot's event loop.
    return await asyncio.to_thread(_mp3_to_ogg_opus_sync, mp3_bytes, pitch_lift)


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
            await update.message.reply_text(ui(context, "translate_fail"))
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
            await _handle_ai_failure(update, context, history, provider, e)
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
            await _handle_ai_failure(update, context, history, provider, e)
            return
        assistant_role = "assistant"

    # Save the model's reply into memory too
    history.append({"role": assistant_role, "content": reply_text})

    await send_ai_reply(update, context, reply_text)


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
            await update.effective_message.reply_text(ui(context, "global_error"))
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
    app.add_handler(CommandHandler("groq", groq_command))
    app.add_handler(CommandHandler("gemini", gemini_command))
    app.add_handler(CommandHandler("websearch", websearch_command))
    app.add_handler(CommandHandler("translate", translate_command))
    app.add_handler(CommandHandler("language", language_command))
    app.add_handler(CommandHandler("lang", language_command))  # short alias
    app.add_handler(CallbackQueryHandler(button_click))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))
    app.add_handler(MessageHandler(filters.VOICE, voice_handler))
    app.add_error_handler(error_handler)

    app.run_polling()

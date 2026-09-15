import os
import threading
import asyncio
from pathlib import Path

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
from groq import Groq
from google import genai
from google.genai import types as genai_types

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

groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "openai/gpt-oss-120b"  # check console.groq.com/docs/models - Groq's free lineup changes often

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"

# How many past messages (user+model combined) to keep per user/provider, to stop memory growing forever
MAX_HISTORY = 20

SYSTEM_PROMPT = """
You are a helpful AI assistant inside a Telegram bot.

IMPORTANT OUTPUT RULES:
- Always reply using plain text only.
- Never output HTML or HTML tags such as <p>, <br>, <b>, <i>, <div>, <a>, etc.
- Never output XML or other markup/code intended to be interpreted as a web page.
- Do not use Markdown formatting such as **bold**, __bold__, [links](...), or ```code blocks```.
- Use normal text that can be sent directly as a Telegram message.
- You may use normal line breaks, simple numbered lists, and simple bullet points.
- If you need to show code, write it as plain text without Markdown code fences.
"""
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
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "خوش آمدید! من ربات شخصی شما هستم هرطور که مایلید از من استفاده کنید.",
        reply_markup=reply_markup,
    )


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


# ---------------------- STOP AI MODE (optional command) ----------------------
async def stop_ai(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["ai_mode"] = False
    await update.message.reply_text("حالت هوش مصنوعی خاموش شد.")


# ---------------------- GROQ CALL ----------------------
async def call_groq(history: list) -> str:
    # groq_client.chat.completions.create runs synchronously, so push it to a thread
    # so it doesn't block the bot's event loop while waiting for a response.
    response = await asyncio.to_thread(
        groq_client.chat.completions.create,
        model=GROQ_MODEL,
        messages=history,
    )
    return response.choices[0].message.content


# ---------------------- GEMINI CALL ----------------------
async def call_gemini(history: list) -> str:
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
        config=genai_types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return response.text


# ---------------------- SHARED AI HANDLER ----------------------
async def ai_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # If this user hasn't activated AI mode, ignore their message here
    if not context.user_data.get("ai_mode"):
        return

    provider = context.user_data.get("ai_provider", "groq")
    chat_id = update.effective_chat.id
    user_text = update.message.text

    history_key = f"history_{provider}"
    history = context.user_data.setdefault(history_key, [])

    # Groq (OpenAI-style) wants the system prompt as the first message in the
    # list. Gemini takes its system prompt separately via config, so it's
    # never added to its own history list.
    if not history and provider == "groq":
        history.append({"role": "system", "content": SYSTEM_PROMPT})

    # Add the user's new message to this provider's own memory
    history.append({"role": "user", "content": user_text})

    # Trim history so it doesn't grow forever
    if len(history) > MAX_HISTORY:
        history[:] = history[-MAX_HISTORY:]

    # Show "typing..." while we wait on the AI
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    if provider == "gemini":
        reply_text = await call_gemini(history)
        assistant_role = "model"
    else:
        reply_text = await call_groq(history)
        assistant_role = "assistant"

    # Save the model's reply into memory too
    history.append({"role": assistant_role, "content": reply_text})

    await update.message.reply_text(reply_text, parse_mode=None)


# ---------------------- MAIN TEXT ROUTER ----------------------
# This decides what to do with a plain text message depending on user's current mode
async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.user_data.get("ai_mode"):
        await ai_handler(update, context)
        return

    # Not in AI mode: ignore other text messages.
    # Add other non-AI logic here if needed.


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

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop_ai))
    app.add_handler(CallbackQueryHandler(button_click))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    app.run_polling()

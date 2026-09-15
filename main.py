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

# ----------------------------------------------------------------------------------------------------------------------
# Load .env locally. On Render this file won't exist, and that's fine -
# Render injects environment variables directly, so os.environ still has them.
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

TOKEN = os.environ.get("TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
BOT_USERNAME = "@MyBigPotatobot"

if not TOKEN or not GROQ_API_KEY:
    raise RuntimeError("Missing TOKEN or GROQ_API_KEY environment variables.")

groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "openai/gpt-oss-20b"  # fast + smart enough for plain text chat, generous free-tier limits

# How many past messages (user+model combined) to keep per user, to stop memory growing forever
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
        [InlineKeyboardButton("🤖ai", callback_data="option2")],
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

    if query.data == "option2":
        context.user_data["ai_mode"] = True
        context.user_data.setdefault("history", [])  # init memory for this user
        await query.edit_message_text("حالت هوش مصنوعی فعال شد🤖!")


# ---------------------- STOP AI MODE (optional command) ----------------------
async def stop_ai(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["ai_mode"] = False
    await update.message.reply_text("حالت هوش مصنوعی خاموش شد.")


# ---------------------- GROQ AI HANDLER ----------------------
async def groq_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # If this user hasn't activated AI mode, ignore their message here
    if not context.user_data.get("ai_mode"):
        return

    chat_id = update.effective_chat.id
    user_text = update.message.text
    history = context.user_data.setdefault("history", [])

    # Keep the AI output Telegram-safe and plain-text for this user.
    if not history:
        history.append({"role": "system", "content": SYSTEM_PROMPT})

    # Add the user's new message to memory (OpenAI-style format: role + content)
    history.append({"role": "user", "content": user_text})

    # Trim history so it doesn't grow forever
    if len(history) > MAX_HISTORY:
        history[:] = history[-MAX_HISTORY:]

    # Show "typing..." while we wait on Groq
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    # groq_client.chat.completions.create runs synchronously, so push it to a thread
    # so it doesn't block the bot's event loop while waiting for a response.
    response = await asyncio.to_thread(
        groq_client.chat.completions.create,
        model=GROQ_MODEL,
        messages=history,
    )

    reply_text = response.choices[0].message.content

    # Save the model's reply into memory too
    history.append({"role": "assistant", "content": reply_text})

    await update.message.reply_text(reply_text, parse_mode=None)


# ---------------------- MAIN TEXT ROUTER ----------------------
# This decides what to do with a plain text message depending on user's current mode
async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.user_data.get("ai_mode"):
        await groq_handler(update, context)
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

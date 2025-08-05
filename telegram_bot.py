# telegram_bot.py
import requests

TELEGRAM_TOKEN = "7863509137:AAHBuRbtzMAOM_yBbVZASfx-oORubvQYxY8"
TELEGRAM_CHAT_ID = 6587216  # يمكنك تغيير المعرف إن كان مختلفًا

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"❌ خطأ في إرسال رسالة تيليجرام: {e}")

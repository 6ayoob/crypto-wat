# notifier.py
import requests

BOT_TOKEN = "توكن_بوت_تيليجرام_هنا"
CHAT_ID = "معرف_المستخدم_أو_القناة"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("خطأ في إرسال رسالة تيليجرام:", e)

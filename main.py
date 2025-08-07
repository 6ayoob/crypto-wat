import time
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS, MAX_OPEN_POSITIONS
from strategy import check_signal, execute_buy, manage_position
from okx_api import fetch_price
import requests

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
        if not response.ok:
            print(f"Failed to send Telegram message: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Telegram error: {e}")

if __name__ == "__main__":
    current_position = None

    send_telegram_message("🚀 بدأ البوت بمراقبة الأسواق باستخدام استراتيجية EMA + RSI ✅")

    while True:
        try:
            for symbol in SYMBOLS:
                if current_position is None:
                    signal = check_signal(symbol)
                    if signal == "buy":
                        order, position, message = execute_buy(symbol)
                        if message:
                            send_telegram_message(message)
                        if position:
                            current_position = position
                            break  # صفقة واحدة فقط في نفس الوقت
                else:
                    # إدارة الصفقة المفتوحة
                    new_position = manage_position(current_position, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
                    if new_position is None:
                        current_position = None  # الصفقة أغلقت
                        send_telegram_message(f"🛑 لا توجد صفقات مفتوحة حالياً، جاري البحث عن فرص جديدة...")
                        break

        except Exception as e:
            import traceback
            send_telegram_message(f"⚠️ خطأ في main.py:\n{traceback.format_exc()}")

        time.sleep(60)

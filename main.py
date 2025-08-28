import time
import requests
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS
from strategy import check_signal, execute_buy, manage_position, load_position

# جرّب استيراد عدّاد جاهز إن كان متوفراً داخل strategy، وإلا سنعمل بـ fallback
try:
    from strategy import count_open_positions as _count_open_positions
except Exception:
    _count_open_positions = None

MAX_OPEN_POSITIONS = 1  # الحد الأقصى للصفقات المفتوحة في نفس الوقت
LOOP_DELAY_SEC = 60     # زمن الانتظار بين الدورات

def send_telegram_message(text, parse_mode=None):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        r = requests.post(url, data=payload, timeout=10)
        if not r.ok:
            print(f"Failed to send Telegram message: {r.status_code} {r.text}")
    except Exception as e:
        print(f"Telegram error: {e}")

def get_open_positions_count():
    # استخدم الدالة من strategy إن وجدت
    if _count_open_positions:
        try:
            return int(_count_open_positions())
        except Exception:
            pass
    # fallback: عدّ الرموز التي لها صفقة محفوظة
    return sum(1 for s in SYMBOLS if load_position(s) is not None)

if __name__ == "__main__":
    send_telegram_message("🚀 يارب توفيقك  — تشغيل استراتيجية EMA9/EMA21 + RSI (هدف واحد ووقف) ✅")

    while True:
        try:
            open_positions_count = get_open_positions_count()

            for symbol in SYMBOLS:
                try:
                    position = load_position(symbol)

                    # لا توجد صفقة على الرمز
                    if position is None:
                        if open_positions_count >= MAX_OPEN_POSITIONS:
                            continue  # وصلنا للحد الأقصى
                        signal = check_signal(symbol)
                        if signal == "buy":
                            order, message = execute_buy(symbol)
                            if message:
                                send_telegram_message(message)
                            if order:
                                open_positions_count += 1  # ✅ زيادة واحدة فقط
                    # توجد صفقة — نديرها
                    else:
                        closed = manage_position(symbol)
                        if closed:
                            send_telegram_message(f"✅ صفقة {symbol} أُغلقت (هدف/وقف).")
                            open_positions_count = max(0, open_positions_count - 1)  # ✅ إنقاص واحد فقط

                except Exception as e:
                    # خطأ خاص بالرمز الحالي
                    send_telegram_message(f"⚠️ خطأ في معالجة {symbol}: {e}")

        except KeyboardInterrupt:
            send_telegram_message("⏹️ تم إيقاف البوت يدويًا.")
            break
        except Exception as e:
            import traceback
            send_telegram_message(f"⚠️ خطأ في البوت:\n{traceback.format_exc()}")

        time.sleep(LOOP_DELAY_SEC)

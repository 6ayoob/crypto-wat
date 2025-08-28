import time
import threading
import requests
from datetime import datetime, timedelta, timezone

from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS, MAX_OPEN_POSITIONS
from strategy import (
    check_signal, execute_buy, manage_position, load_position
)

# حاول استخدام عدّاد جاهز من الاستراتيجية (إن وُجد)
try:
    from strategy import count_open_positions as _count_open_positions
except Exception:
    _count_open_positions = None

# إعدادات عامة
LOOP_DELAY_SEC = 60  # زمن الانتظار بين الدورات
REPORT_HOUR = 9      # تقرير يومي 09:00 بتوقيت الرياض
REPORT_MINUTE = 0
RIYADH_TZ = timezone(timedelta(hours=3))

# دالة إرسال تيليجرام (كما هي تقريبًا مع خيار HTML)
def send_telegram_message(text, parse_mode=None):
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "CHANGE_ME":
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
        payload["disable_web_page_preview"] = True
    try:
        r = requests.post(url, data=payload, timeout=10)
        if not r.ok:
            print(f"Failed to send Telegram message: {r.status_code} {r.text}")
    except Exception as e:
        print(f"Telegram error: {e}")

# عدّ الصفقات المفتوحة
def get_open_positions_count():
    if _count_open_positions:
        try:
            return int(_count_open_positions())
        except Exception:
            pass
    # fallback: عدّ الرموز التي لها صفقة محفوظة
    return sum(1 for s in SYMBOLS if load_position(s) is not None)

# ====== جدولة التقرير اليومي ======
try:
    from strategy import build_daily_report_text
except Exception:
    build_daily_report_text = None  # لو النسخة القديمة من الاستراتيجية لا تحتوي الدالة

def schedule_daily_report(hour=REPORT_HOUR, minute=REPORT_MINUTE):
    if not build_daily_report_text:
        print("ℹ️ لا توجد دالة build_daily_report_text في strategy — لن يتم إرسال تقرير يومي.")
        return

    def loop():
        sent_for = None
        while True:
            now = datetime.now(RIYADH_TZ)
            key = now.date().isoformat()
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)
            sleep_seconds = (target - now).total_seconds()
            if sleep_seconds > 1:
                time.sleep(sleep_seconds)
            try:
                if sent_for != key:
                    txt = build_daily_report_text()
                    send_telegram_message(txt, parse_mode="HTML")
                    sent_for = key
            except Exception as e:
                send_telegram_message(f"⚠️ فشل إرسال التقرير اليومي: {e}")
            time.sleep(61)  # لتجاوز نفس الدقيقة
    t = threading.Thread(target=loop, daemon=True)
    t.start()

if __name__ == "__main__":
    send_telegram_message("🚀 تشغيل استراتيجية EMA9/EMA21 + RSI (ATR/MTF) — الحد الأقصى 3 صفقات ✅")
    schedule_daily_report()  # تفعيل التقرير اليومي 09:00

    while True:
        try:
            open_positions_count = get_open_positions_count()

            for symbol in SYMBOLS:
                try:
                    position = load_position(symbol)

                    # لا توجد صفقة على الرمز
                    if position is None:
                        if open_positions_count >= MAX_OPEN_POSITIONS:  # ← 3 من config.py
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
                            send_telegram_message(f"✅ صفقة {symbol} أُغلقت (TP/SL/Trailing).")
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

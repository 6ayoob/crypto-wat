# run.py — تشغيل + فلترة رموز + تقرير يومي
import time
import threading
import requests

from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS as RAW_SYMBOLS
from strategy import (
    check_signal, execute_buy, manage_position, load_position, count_open_positions,
    build_daily_report_text
)
from okx_api import exchange

MAX_LOOP_DELAY_SEC = 60  # مهلة بين الدورات
REPORT_HOUR = 9          # توقيت التقرير اليومي (ساعة)
REPORT_MINUTE = 0        # توقيت التقرير اليومي (دقيقة)

def send_telegram(text, html=False):
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "CHANGE_ME":
        print(text);  return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    if html:
        payload["parse_mode"] = "HTML"
        payload["disable_web_page_preview"] = True
    try:
        r = requests.post(url, data=payload, timeout=10)
        if not r.ok: print("Telegram failed:", r.status_code, r.text)
    except Exception as e:
        print("Telegram error:", e)

# خريطة تصحيح أسماء إن احتجت
ALIAS_MAP = {"RENDER/USDT": "RNDR/USDT", "LUNA/USDT": "LUNC/USDT"}

def normalize_symbols(symbols):
    seen, out = set(), []
    for s in symbols:
        s = ALIAS_MAP.get(s, s).replace("-", "/").upper()
        if s not in seen:
            out.append(s); seen.add(s)
    return out

def filter_supported_symbols(symbols):
    exchange.load_markets()
    markets = exchange.markets
    ok, bad = [], []
    for s in symbols:
        (ok if s in markets else bad).append(s)
    if bad:
        print("⚠️ رموز غير مدعومة وسيتم تجاهلها:", bad)
    return ok

SYMBOLS = filter_supported_symbols(normalize_symbols(RAW_SYMBOLS))
print("✅ الرموز المستخدمة:", SYMBOLS)

# ====== جدولة التقرير اليومي (09:00 بتوقيت الرياض) ======
def schedule_daily_report(hour=REPORT_HOUR, minute=REPORT_MINUTE):
    from datetime import datetime, timedelta, timezone
    RIYADH_TZ = timezone(timedelta(hours=3))

    def loop():
        sent_for = None
        while True:
            now = datetime.now(RIYADH_TZ)
            key = now.date().isoformat()
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)
            sleep_s = (target - now).total_seconds()
            if sleep_s > 1:
                time.sleep(sleep_s)
            try:
                if sent_for != key:
                    txt = build_daily_report_text()
                    send_telegram(txt, html=True)
                    sent_for = key
            except Exception as e:
                send_telegram(f"⚠️ فشل إرسال التقرير اليومي: {e}")
            time.sleep(61)  # تجاوز نفس الدقيقة
    t = threading.Thread(target=loop, daemon=True)
    t.start()

if __name__ == "__main__":
    send_telegram("🚀 بدء تشغيل البوت (ATR + MTF + تقرير يومي 09:00)", html=True)
    schedule_daily_report()  # ← يبدأ مجدول التقرير في الخلفية

    while True:
        try:
            open_cnt = count_open_positions()

            for symbol in SYMBOLS:
                try:
                    position = load_position(symbol)

                    if position is None:
                        if open_cnt >= 3:  # الحد الأقصى من config
                            continue
                        sig = check_signal(symbol)
                        if sig == "buy":
                            order, msg = execute_buy(symbol)
                            if msg: send_telegram(msg)
                            if order:
                                open_cnt += 1
                    else:
                        closed = manage_position(symbol)
                        if closed:
                            send_telegram(f"✅ صفقة {symbol} أُغلقت (TP/SL/Trailing)")
                            open_cnt = max(0, open_cnt - 1)

                except Exception as e:
                    send_telegram(f"⚠️ خطأ في {symbol}: {e}")

        except KeyboardInterrupt:
            send_telegram("⏹️ تم إيقاف البوت يدويًا.")
            break
        except Exception as e:
            import traceback
            send_telegram("⚠️ خطأ:\n" + traceback.format_exc())

        time.sleep(MAX_LOOP_DELAY_SEC)

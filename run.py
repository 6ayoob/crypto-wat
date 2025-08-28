# run.py — حلقة التشغيل + فلترة الرموز + تيليجرام
import time
import requests

from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS as RAW_SYMBOLS
from strategy import check_signal, execute_buy, manage_position, load_position, count_open_positions
from okx_api import exchange

MAX_LOOP_DELAY_SEC = 60  # مهلة بين الدورات

def send_telegram(text):
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "CHANGE_ME":
        print(text);  return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
        if not r.ok: print("Telegram failed:", r.status_code, r.text)
    except Exception as e:
        print("Telegram error:", e)

# خريطة تصحيح أسماء إن احتجت
ALIAS_MAP = {
    "RENDER/USDT": "RNDR/USDT",
    "LUNA/USDT":   "LUNC/USDT",
}

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

if __name__ == "__main__":
    send_telegram("🚀 بدء تشغيل البوت (ATR + MTF) — موفقين")

    while True:
        try:
            # عدّ الصفقات الحالية مرة واحدة لكل دورة
            open_cnt = count_open_positions()

            for symbol in SYMBOLS:
                try:
                    position = load_position(symbol)

                    if position is None:
                        # نتحقق من الإشارة فقط إذا لا زال لدينا سعة
                        if open_cnt >= 3:  # الحد الأقصى للصفقات المفتوحة من config
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

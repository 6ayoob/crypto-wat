# main.py — Always-On loop (ignore stop signals + auto-recover on errors)

import os, time, random, traceback, signal
from datetime import datetime, timezone, timedelta
import requests

from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS, STRAT_LTF_TIMEFRAME, STRAT_HTF_TIMEFRAME

from strategy import (
    check_signal, execute_buy, manage_position, load_position,
    count_open_positions, build_daily_report_text
)

# اختياري: دوال تشخيصية إن وُجدت
try:
    from strategy import maybe_emit_reject_summary, check_signal_debug
except Exception:
    def maybe_emit_reject_summary(): pass
    def check_signal_debug(symbol): return None, []

# كاش الأسعار الجماعي (اختياري)
try:
    from okx_api import start_tickers_cache, stop_tickers_cache
    _HAS_CACHE = True
except Exception:
    _HAS_CACHE = False

# ================== إعدادات ==================
MAX_OPEN_POSITIONS_OVERRIDE = None  # أو رقم لتقييد محلي
SCAN_INTERVAL_SEC    = int(os.getenv("SCAN_INTERVAL_SEC", "25"))
MANAGE_INTERVAL_SEC  = int(os.getenv("MANAGE_INTERVAL_SEC", "10"))
LOOP_SLEEP_SEC       = 1.0

ENABLE_DAILY_REPORT  = True
DAILY_REPORT_HOUR    = int(os.getenv("DAILY_REPORT_HOUR", "23"))
DAILY_REPORT_MINUTE  = int(os.getenv("DAILY_REPORT_MINUTE", "58"))

MUTE_NOISEY_ALERTS   = True

# “لا يوقف” = تجاهل الإشارات
IGNORE_SIGNALS       = os.getenv("IGNORE_SIGNALS", "1").lower() in ("1","true","yes","y")
# تحكم برسائل الخطأ
MAX_TG_TRACE_CHARS   = int(os.getenv("MAX_TG_TRACE_CHARS", "900"))

RIYADH_TZ = timezone(timedelta(hours=3))

def _now_riyadh(): return datetime.now(RIYADH_TZ)

# ================== Telegram ==================
def send_telegram_message(text, parse_mode=None, disable_notification=False):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    if parse_mode: payload["parse_mode"] = parse_mode
    if disable_notification: payload["disable_notification"] = True
    try:
        r = requests.post(url, data=payload, timeout=10)
        if not r.ok: print(f"[TG] Failed: {r.status_code} {r.text}")
    except Exception as e:
        print(f"[TG] Error: {e}")

# ================== سلوك الإشارات ==================
def _ignore_system_signals():
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception: pass
    try:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except Exception: pass

# ================== Helpers ==================
def _get_open_positions_count_safe():
    try:
        return int(count_open_positions())
    except Exception:
        try:
            return sum(1 for s in SYMBOLS if load_position(s) is not None)
        except Exception:
            return 0

def _can_open_new_position(current_open: int) -> bool:
    if MAX_OPEN_POSITIONS_OVERRIDE is None:
        return True
    return current_open < int(MAX_OPEN_POSITIONS_OVERRIDE)

# ================== الحلقة الرئيسية ==================
def run_forever():
    if IGNORE_SIGNALS:
        _ignore_system_signals()

    # بدء كاش الأسعار
    if _HAS_CACHE:
        try:
            start_tickers_cache(period=int(os.getenv("OKX_CACHE_PERIOD", "3")), usdt_only=True)
        except Exception:
            pass

    # إشعار تشغيل “وضع عدم التوقف”
    try:
        send_telegram_message(
            f"🚀 تشغيل البوت (Always-On) — {len(SYMBOLS)} رمز | HTF={STRAT_HTF_TIMEFRAME} / LTF={STRAT_LTF_TIMEFRAME} ✅",
            disable_notification=True
        )
    except Exception:
        print("🚀 تشغيل البوت (Always-On)")

    last_scan_ts   = 0.0
    last_manage_ts = 0.0
    last_report_day = None

    # توزيع حمل أولي
    time.sleep(random.uniform(0.5, 1.5))

    while True:
        now = time.time()

        # 1) فحص إشارات الدخول
        if now - last_scan_ts >= SCAN_INTERVAL_SEC + random.uniform(-2, 2):
            try:
                open_positions_count = _get_open_positions_count_safe()

                for symbol in SYMBOLS:
                    # لو امتلأ الحد المحلي، توقف عن محاولة فتح صفقات جديدة
                    if not _can_open_new_position(open_positions_count):
                        break

                    # تخطي إن كان على الرمز صفقة مفتوحة
                    try:
                        if load_position(symbol) is not None:
                            continue
                    except Exception:
                        pass

                    # فحص إشارة
                    try:
                        sig = check_signal(symbol)
                    except Exception as e:
                        if not MUTE_NOISEY_ALERTS:
                            t = (traceback.format_exc() or str(e))[:MAX_TG_TRACE_CHARS]
                            send_telegram_message(f"⚠️ check_signal خطأ في {symbol}:\n{t}")
                        else:
                            print(f"[check_signal] {symbol} error: {e}")
                        continue

                    is_buy = (sig == "buy") or (isinstance(sig, dict) and str(sig.get("decision","")).lower()=="buy")
                    if is_buy:
                        try:
                            order, msg = execute_buy(symbol)
                            if msg: send_telegram_message(msg)
                            open_positions_count = _get_open_positions_count_safe()
                        except Exception as e:
                            t = (traceback.format_exc() or str(e))[:MAX_TG_TRACE_CHARS]
                            send_telegram_message(f"❌ فشل تنفيذ شراء {symbol}:\n{t}")
                    else:
                        # تشخيص اختياري
                        try:
                            _, reasons = check_signal_debug(symbol)
                            if reasons:
                                print(f"[debug] {symbol} reject reasons: {reasons[:5]}")
                        except Exception:
                            pass

                    time.sleep(0.2)

                # ملخص أسباب الرفض (اختياري)
                try:
                    maybe_emit_reject_summary()
                except Exception:
                    pass

            except Exception as outer_scan_e:
                t = (traceback.format_exc() or str(outer_scan_e))[:MAX_TG_TRACE_CHARS]
                send_telegram_message(f"⚠️ خطأ عام أثناء فحص الإشارات:\n{t}")
            finally:
                last_scan_ts = now

        # 2) إدارة الصفقات المفتوحة
        if now - last_manage_ts >= MANAGE_INTERVAL_SEC:
            try:
                for symbol in SYMBOLS:
                    try:
                        closed = manage_position(symbol)
                        if closed:
                            print(f"[manage] {symbol} closed by TP/SL/TIME")
                    except Exception as e:
                        if not MUTE_NOISEY_ALERTS:
                            t = (traceback.format_exc() or str(e))[:MAX_TG_TRACE_CHARS]
                            send_telegram_message(f"⚠️ خطأ إدارة {symbol}:\n{t}")
                        else:
                            print(f"[manage_position] {symbol} error: {e}")
                    time.sleep(0.1)
            except Exception as outer_mng_e:
                t = (traceback.format_exc() or str(outer_mng_e))[:MAX_TG_TRACE_CHARS]
                send_telegram_message(f"⚠️ خطأ عام أثناء إدارة الصفقات:\n{t}")
            finally:
                last_manage_ts = now

        # 3) تقرير يومي تلقائي
        if ENABLE_DAILY_REPORT:
            try:
                now_r = _now_riyadh()
                day_key = now_r.strftime("%Y-%m-%d")
                if (now_r.hour == DAILY_REPORT_HOUR and now_r.minute >= DAILY_REPORT_MINUTE) and (last_report_day != day_key):
                    try:
                        report = build_daily_report_text()
                        if report:
                            send_telegram_message(report, parse_mode="HTML", disable_notification=True)
                    except Exception as e:
                        print(f"[daily_report] build error: {e}")
                    last_report_day = day_key
            except Exception:
                pass

        time.sleep(LOOP_SLEEP_SEC)

# ================== الحارس الخارجي ==================
if __name__ == "__main__":
    # غلاف يحمي من أي انهيار غير متوقع (ويُكمل التشغيل)
    # backoff بسيط عند الأخطاء المتتالية
    consecutive_errors = 0
    while True:
        try:
            run_forever()  # لا ينبغي أن تعود
        except Exception as e:
            consecutive_errors += 1
            wait_s = min(60, 2 ** min(consecutive_errors, 6))  # 1,2,4,8,16,32,60...
            t = (traceback.format_exc() or str(e))[:MAX_TG_TRACE_CHARS]
            send_telegram_message(f"🔥 تعطل غير متوقع — سيُعاد التشغيل تلقائياً بعد {wait_s}s\n{t}")
            time.sleep(wait_s)
            # ثم يُعاد تشغيل الحلقة تلقائياً
        else:
            # لو خرجت run_forever لسبب ما، أعد تشغيلها فوراً
            consecutive_errors = 0
            time.sleep(1)

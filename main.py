# main.py — resilient loop (ignore stop signals + auto-respawn + optional TG stacktraces)
import os
import time
import random
import signal
import traceback
from datetime import datetime, timezone, timedelta

import requests

from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS, STRAT_LTF_TIMEFRAME, STRAT_HTF_TIMEFRAME

# الاستراتيجية
from strategy import (
    check_signal, execute_buy, manage_position, load_position,
    count_open_positions, build_daily_report_text
)

# (اختياري) تشخيصات من الاستراتيجية — لو غير موجودة لا مشكلة
try:
    from strategy import maybe_emit_reject_summary, check_signal_debug
except Exception:
    def maybe_emit_reject_summary(): pass
    def check_signal_debug(symbol): return None, []

# كاش أسعار جماعي من okx_api
try:
    from okx_api import start_tickers_cache, stop_tickers_cache
    _HAS_CACHE = True
except Exception:
    _HAS_CACHE = False

# ================== إعدادات السلوك المرن ==================
# تجاهل إشارات الإيقاف الافتراضيًا (لا توقَف إلا لو غيّرت المتغير)
IGNORE_SIGNALS        = os.getenv("IGNORE_SIGNALS", "1").lower() in ("1","true","yes","y")
# إعادة التشغيل تلقائياً عند الخروج غير المقصود
AUTORESPAWN           = os.getenv("AUTORESPAWN", "1").lower() in ("1","true","yes","y")
RESPAWN_DELAY_SEC     = int(os.getenv("RESPAWN_DELAY_SEC", "3"))

# كتم إشعارات الإيقاف حتى لا تربكك
MUTE_STOP_NOTICES     = os.getenv("MUTE_STOP_NOTICES", "1").lower() in ("1","true","yes","y")

# تقليل ضجيج تيليجرام لبعض الأخطاء المؤقتة
MUTE_NOISEY_ALERTS    = os.getenv("MUTE_NOISEY_ALERTS", "1").lower() in ("1","true","yes","y")

# إرسال تتبّع الاستثناءات المفاجئة إلى تيليجرام (Stacktrace مختصر)
TG_EXCEPTIONS         = os.getenv("TG_EXCEPTIONS", "1").lower() in ("1","true","yes","y")
TG_TRACE_MAX_CHARS    = int(os.getenv("TG_TRACE_MAX_CHARS", "1400"))

# إذا أردت فرض حد مؤقت محلي (غير إلزامي)
MAX_OPEN_POSITIONS_OVERRIDE = os.getenv("MAX_OPEN_POSITIONS_OVERRIDE")
if MAX_OPEN_POSITIONS_OVERRIDE not in (None, "",):
    try:
        MAX_OPEN_POSITIONS_OVERRIDE = int(MAX_OPEN_POSITIONS_OVERRIDE)
    except Exception:
        MAX_OPEN_POSITIONS_OVERRIDE = None

# فواصل التكرار
SCAN_INTERVAL_SEC    = int(os.getenv("SCAN_INTERVAL_SEC", "25"))   # فحص إشارات الدخول
MANAGE_INTERVAL_SEC  = int(os.getenv("MANAGE_INTERVAL_SEC", "10")) # إدارة المراكز
LOOP_SLEEP_SEC       = 1.0

# تقرير يومي تلقائي (بتوقيت الرياض)
ENABLE_DAILY_REPORT  = os.getenv("ENABLE_DAILY_REPORT", "1").lower() in ("1","true","yes","y")
DAILY_REPORT_HOUR    = int(os.getenv("DAILY_REPORT_HOUR", "23"))
DAILY_REPORT_MINUTE  = int(os.getenv("DAILY_REPORT_MINUTE", "58"))

RIYADH_TZ = timezone(timedelta(hours=3))

# ================== Telegram ==================
def send_telegram_message(text, parse_mode=None, disable_notification=False):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    if disable_notification:
        payload["disable_notification"] = True
    try:
        r = requests.post(url, data=payload, timeout=10)
        if not r.ok:
            print(f"[TG] Failed: {r.status_code} {r.text}")
    except Exception as e:
        print(f"[TG] Error: {e}")

def _send_exception_to_tg(context: str):
    if not TG_EXCEPTIONS:
        return
    tb = traceback.format_exc()
    tb_short = tb[-TG_TRACE_MAX_CHARS:]
    try:
        send_telegram_message(f"⚠️ {context}\n{tb_short}", disable_notification=True)
    except Exception:
        pass

# ================== أدوات ==================
def _now_riyadh():
    return datetime.now(RIYADH_TZ)

def _get_open_positions_count_safe():
    """يرجع عدد الصفقات المفتوحة من الاستراتيجية (مع fallback بسيط)."""
    try:
        return int(count_open_positions())
    except Exception:
        try:
            return sum(1 for s in SYMBOLS if load_position(s) is not None)
        except Exception:
            return 0

def _can_open_new_position(current_open: int) -> bool:
    """يقرّر محليًا إن كنا نسمح بإشارات شراء جديدة بناءً على override فقط."""
    if MAX_OPEN_POSITIONS_OVERRIDE is None:
        return True
    return current_open < int(MAX_OPEN_POSITIONS_OVERRIDE)

# ================== حلقة جلسة واحدة ==================
_stop_flag = False
_start_ts  = time.time()

def _handle_stop(signum, frame):
    """
    إن كان IGNORE_SIGNALS=1 → نتجاهل الإشارة ونستمر.
    إن كان =0 → نغلق بهدوء.
    """
    global _stop_flag
    if IGNORE_SIGNALS:
        try:
            send_telegram_message("⏹️ استلمنا إشارة نظام… تم تجاهلها والاستمرار ✅", disable_notification=True)
        except Exception:
            pass
        # لا نغيّر _stop_flag → نستمر
        return
    else:
        _stop_flag = True
        try:
            send_telegram_message("⏹️ تم استلام إشارة إيقاف من النظام… إنهاء بهدوء.", disable_notification=True)
        except Exception:
            pass

signal.signal(signal.SIGINT, _handle_stop)
signal.signal(signal.SIGTERM, _handle_stop)

def run_bot_session():
    global _stop_flag, _start_ts
    _stop_flag = False
    _start_ts = time.time()

    # بدء كاش الأسعار الجماعي (طلب واحد كل عدة ثواني) إن توفر
    if _HAS_CACHE:
        try:
            start_tickers_cache(period=int(os.getenv("OKX_CACHE_PERIOD", "3")), usdt_only=True)
        except Exception:
            pass

    # رسالة تشغيل مختصرة
    try:
        send_telegram_message(
            f"🚀 تشغيل البوت — {len(SYMBOLS)} رمز | HTF={STRAT_HTF_TIMEFRAME} / LTF={STRAT_LTF_TIMEFRAME} ✅",
            disable_notification=True
        )
    except Exception:
        print("🚀 تشغيل البوت")

    last_scan_ts   = 0.0
    last_manage_ts = 0.0
    last_report_day = None

    # Jitter أولي لتوزيع الأحمال إذا كان لديك أكثر من عملية
    time.sleep(random.uniform(0.5, 1.5))

    while not _stop_flag:
        now = time.time()

        # 1) فحص إشارات الدخول
        if now - last_scan_ts >= SCAN_INTERVAL_SEC + random.uniform(-2, 2):
            try:
                open_positions_count = _get_open_positions_count_safe()

                for symbol in SYMBOLS:
                    if _stop_flag:
                        break

                    # إذا امتلأ حدّنا المحلي (إن فُعِّل)، لا نحاول شراء جديد
                    if not _can_open_new_position(open_positions_count):
                        break

                    # لا تفتح صفقة على رمز لديه مركز قائم
                    try:
                        if load_position(symbol) is not None:
                            continue  # يُدار لاحقًا
                    except Exception:
                        pass

                    # فحص الإشارة
                    try:
                        sig = check_signal(symbol)
                    except Exception as e:
                        if not MUTE_NOISEY_ALERTS:
                            send_telegram_message(f"⚠️ check_signal خطأ في {symbol}:\n{e}")
                        else:
                            print(f"[check_signal] {symbol} error: {e}")
                        continue

                    # دعم نوعين من النتيجة
                    is_buy = (sig == "buy") or (isinstance(sig, dict) and str(sig.get("decision", "")).lower() == "buy")

                    if is_buy:
                        try:
                            order, msg = execute_buy(symbol)
                            if msg:
                                send_telegram_message(msg)
                            # تحديث العدّ من المصدر بعد كل محاولة شراء
                            open_positions_count = _get_open_positions_count_safe()
                        except Exception:
                            _send_exception_to_tg(f"execute_buy فشل ({symbol})")
                            continue
                    else:
                        # (اختياري) لماذا رُفضت الإشارة؟
                        try:
                            _, reasons = check_signal_debug(symbol)
                            if reasons:
                                print(f"[debug] {symbol} reject reasons: {reasons[:5]}")
                        except Exception:
                            pass

                    # مهلة قصيرة بين الرموز لتخفيف الضغط
                    time.sleep(0.2)

                # ملخص أسباب الرفض كل ~30 دقيقة كحد أقصى — إن كانت الدالة مفعلة
                try:
                    maybe_emit_reject_summary()
                except Exception:
                    pass

            except Exception:
                _send_exception_to_tg("خطأ عام أثناء فحص الإشارات")
            finally:
                last_scan_ts = now

        # 2) إدارة الصفقات المفتوحة (TP/SL/Trailing)
        if now - last_manage_ts >= MANAGE_INTERVAL_SEC:
            try:
                for symbol in SYMBOLS:
                    if _stop_flag:
                        break
                    try:
                        closed = manage_position(symbol)
                        if closed:
                            print(f"[manage] {symbol} closed by TP/SL/TIME")
                    except Exception as e:
                        if not MUTE_NOISEY_ALERTS:
                            send_telegram_message(f"⚠️ خطأ إدارة {symbol}:\n{e}")
                        else:
                            print(f"[manage_position] {symbol} error: {e}")
                    time.sleep(0.1)
            except Exception:
                _send_exception_to_tg("خطأ عام أثناء إدارة الصفقات")
            finally:
                last_manage_ts = now

        # 3) تقرير يومي تلقائي (23:58 الرياض افتراضيًا)
        if ENABLE_DAILY_REPORT:
            try:
                now_r = _now_riyadh()
                day_key = now_r.strftime("%Y-%m-%d")
                if (now_r.hour == DAILY_REPORT_HOUR and now_r.minute >= DAILY_REPORT_MINUTE) and (last_report_day != day_key):
                    try:
                        report = build_daily_report_text()
                        if report:
                            try:
                                send_telegram_message(report, parse_mode="HTML", disable_notification=True)
                            except Exception as tg_err:
                                print(f"[daily_report] telegram error: {tg_err}")
                    except Exception:
                        _send_exception_to_tg("daily_report build error")
                    last_report_day = day_key
            except Exception:
                pass

        # نوم قصير
        time.sleep(LOOP_SLEEP_SEC)

    # خرجنا من الحلقة (فقط إن IGNORE_SIGNALS=0 وتم استلام إشارة/أو إيقاف يدوي)
    if _HAS_CACHE:
        try:
            stop_tickers_cache()
        except Exception:
            pass
    # لا ترسل “إيقاف” إلا إذا لم نكتمها وكان إيقافًا حقيقيًا وليس إعادة تشغيل
    if not MUTE_STOP_NOTICES:
        if time.time() - _start_ts >= 10:
            send_telegram_message("🛑 تم إيقاف البوت — إلى اللقاء.", disable_notification=True)

# ================== المشرف الخارجي (Auto-Respawn) ==================
if __name__ == "__main__":
    attempt = 0
    while True:
        try:
            run_bot_session()
            # إن خرجنا بشكل طبيعي (_stop_flag=True و IGNORE_SIGNALS=0) ولم نفعّل AUTORESPAWN — انهِ.
            if not AUTORESPAWN:
                break
            # إن خرجنا طبيعيًا مع AUTORESPAWN=1 → أعد التشغيل أيضًا (للاستمرارية التامة)
        except Exception:
            attempt += 1
            _send_exception_to_tg(f"🚨 تعطل غير متوقع — إعادة تشغيل (محاولة {attempt})")
        # مهلة قصيرة قبل إعادة التشغيل
        if AUTORESPAWN:
            try:
                send_telegram_message(f"🔁 إعادة تشغيل تلقائي… (محاولة {attempt})", disable_notification=True)
            except Exception:
                pass
            time.sleep(max(1, RESPAWN_DELAY_SEC))
        else:
            break

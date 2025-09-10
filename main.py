# main.py — Loop for 15m/5m strategy (signals + management + daily report)
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

# (اختياري) دوال تشخيص من الاستراتيجية
try:
    from strategy import maybe_emit_reject_summary, check_signal_debug  # قد لا تكون متوفرة
except Exception:
    def maybe_emit_reject_summary():
        pass
    def check_signal_debug(symbol):
        return None, []

# كاش أسعار جماعي من okx_api لتقليل الضغط
try:
    from okx_api import start_tickers_cache, stop_tickers_cache
    _HAS_CACHE = True
except Exception:
    _HAS_CACHE = False

# ================== إعدادات الحلقة ==================
# حد محلي لعدد الصفقات (اختياري) — مستقل عن حد الاستراتيجية الداخلي
MAX_OPEN_POSITIONS_OVERRIDE = None  # مثال: 2 أو 3 … أو None لإيقافه

# فواصل التكرار
SCAN_INTERVAL_SEC    = int(os.getenv("SCAN_INTERVAL_SEC", "25"))   # فحص إشارات الدخول
MANAGE_INTERVAL_SEC  = int(os.getenv("MANAGE_INTERVAL_SEC", "10")) # إدارة المراكز
LOOP_SLEEP_SEC       = 1.0  # نوم قصير داخل الحلقة

# تقرير يومي تلقائي (بتوقيت الرياض)
ENABLE_DAILY_REPORT  = os.getenv("ENABLE_DAILY_REPORT", "1").lower() in ("1","true","yes")
DAILY_REPORT_HOUR    = int(os.getenv("DAILY_REPORT_HOUR", "23"))
DAILY_REPORT_MINUTE  = int(os.getenv("DAILY_REPORT_MINUTE", "58"))

# تقليل ضجيج التيليجرام + التحكم بإرسال الأخطاء
MUTE_NOISEY_ALERTS        = True
SEND_ERRORS_TO_TELEGRAM   = os.getenv("SEND_ERRORS_TO_TELEGRAM", "0").lower() in ("1","true","yes")
SEND_INFO_TO_TELEGRAM     = os.getenv("SEND_INFO_TO_TELEGRAM", "1").lower() in ("1","true","yes")

# سياسة التعامل مع إشارات الإيقاف (SIGTERM/SIGINT)
# options: "ignore" | "debounce" | "immediate"
STOP_POLICY = os.getenv("STOP_POLICY", "debounce").lower()
STOP_DEBOUNCE_WINDOW_SEC = int(os.getenv("STOP_DEBOUNCE_WINDOW_SEC", "5"))

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

def _is_error_text(text: str) -> bool:
    if not text: return False
    t = str(text).strip()
    return t.startswith("⚠️") or t.startswith("❌") or "خطأ" in t or "Error" in t

def tg_info(text, parse_mode=None, silent=True):
    if SEND_INFO_TO_TELEGRAM:
        try:
            send_telegram_message(text, parse_mode=parse_mode, disable_notification=silent)
        except Exception:
            pass

def tg_error(text, parse_mode=None, silent=True):
    if SEND_ERRORS_TO_TELEGRAM:
        try:
            send_telegram_message(text, parse_mode=parse_mode, disable_notification=silent)
        except Exception:
            pass

# ================== أدوات ==================
_stop_flag = False
_last_stop_signal_ts = 0.0

def _handle_stop(signum, frame):
    """
    سياسة الإيقاف:
      - ignore   : نتجاهل الإشارة (نطبع فقط).
      - debounce : لا نتوقف إلا إذا وصلت إشارتان خلال نافذة قصيرة.
      - immediate: نتوقف فورًا (سلوك قديم).
    """
    global _stop_flag, _last_stop_signal_ts
    now = time.time()

    if STOP_POLICY == "ignore":
        print(f"⏸️ تم استقبال إشارة {signum} وتم تجاهلها حسب STOP_POLICY=ignore.")
        # لا نرسل للتلغرام لتفادي الإرباك
        return

    if STOP_POLICY == "debounce":
        if (now - _last_stop_signal_ts) <= STOP_DEBOUNCE_WINDOW_SEC:
            _stop_flag = True
            msg = "⏹️ تم تأكيد إيقاف البوت بعد إشارة ثانية ضمن النافذة (debounce)."
            print(msg)
            tg_info(msg, disable_notification=True)
        else:
            _last_stop_signal_ts = now
            msg = f"⚠️ استلمت إشارة إيقاف. لن يتم الإيقاف إلا إذا وصلت إشارة ثانية خلال {STOP_DEBOUNCE_WINDOW_SEC}ث."
            print(msg)
            # إشعار معلوماتي فقط
            tg_info(msg, disable_notification=True)
        return

    # immediate
    _stop_flag = True
    try:
        tg_info("⏹️ تم استلام إشارة إيقاف — جاري الإنهاء بهدوء…", silent=True)
    except Exception:
        pass

# ربط الإشارات
try:
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)
except Exception:
    pass  # أنظمة لا تدعم الإشارات (ويندوز/بيئات محدودة)

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

# ================== الحلقة الرئيسية ==================
if __name__ == "__main__":
    # بدء كاش الأسعار الجماعي (طلب واحد كل عدة ثوانٍ) إن توفر
    if _HAS_CACHE:
        try:
            start_tickers_cache(period=int(os.getenv("OKX_CACHE_PERIOD", "3")), usdt_only=True)
        except Exception:
            pass

    # معلومات بدء مع عرض الإطارات الزمنية الفعلية
    try:
        tg_info(
            f"🚀 تشغيل البوت — {len(SYMBOLS)} رمز | HTF={STRAT_HTF_TIMEFRAME} / LTF={STRAT_LTF_TIMEFRAME} ✅",
            silent=True
        )
    except Exception:
        print("🚀 تشغيل البوت")

    last_scan_ts   = 0.0
    last_manage_ts = 0.0
    last_report_day = None

    # Jitter أولي لتوزيع الأحمال إذا كان لديك أكثر من عملية
    time.sleep(random.uniform(0.5, 1.5))

    try:
        while True:
            # في وضع immediate/debounce قد يُطلب الإيقاف
            if _stop_flag and STOP_POLICY in ("immediate", "debounce"):
                break

            now = time.time()

            # 1) فحص إشارات الدخول
            if now - last_scan_ts >= SCAN_INTERVAL_SEC + random.uniform(-2, 2):
                try:
                    open_positions_count = _get_open_positions_count_safe()

                    for symbol in SYMBOLS:
                        # عند طلب إيقاف "مؤكد" نخرج بأمان
                        if _stop_flag and STOP_POLICY in ("immediate", "debounce"):
                            break

                        # إذا امتلأ حدّنا المحلي (إن فُعِّل)، لا نحاول شراء جديد
                        if not _can_open_new_position(open_positions_count):
                            break  # اكتفِ بما لدينا

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
                            if not MUTE_NOISEY_ALERTS and SEND_ERRORS_TO_TELEGRAM:
                                tg_error(f"⚠️ check_signal خطأ في {symbol}:\n{e}")
                            else:
                                print(f"[check_signal] {symbol} error: {e}")
                            continue

                        # دعم نوعين من النتيجة: "buy" أو dict(decision="buy")
                        is_buy = (sig == "buy") or (isinstance(sig, dict) and str(sig.get("decision", "")).lower() == "buy")

                        if is_buy:
                            try:
                                order, msg = execute_buy(symbol)
                                # فقط رسائل النجاح للتلغرام (نمنع إرسال الأخطاء)
                                if msg:
                                    if _is_error_text(msg):
                                        if SEND_ERRORS_TO_TELEGRAM:
                                            tg_error(msg)
                                    else:
                                        tg_info(msg)
                                # تحديث العدّ من المصدر بعد كل محاولة شراء
                                open_positions_count = _get_open_positions_count_safe()
                            except Exception as e:
                                if SEND_ERRORS_TO_TELEGRAM:
                                    tg_error(f"❌ فشل تنفيذ شراء {symbol}:\n{e}")
                                else:
                                    print(f"[execute_buy] {symbol} error: {e}")
                                continue
                        else:
                            # (اختياري) عندما لا توجد إشارة، نفحص أسباب الرفض (لو الدالة موجودة)
                            try:
                                _, reasons = check_signal_debug(symbol)
                                if reasons:
                                    print(f"[debug] {symbol} reject reasons: {reasons[:5]}")
                            except Exception:
                                pass

                        # مهلة قصيرة بين الرموز لتخفيف الضغط
                        time.sleep(0.2)

                    # (اختياري) إرسال ملخص أسباب الرفض كل ~30 دقيقة — إن كانت الدالة مفعلة
                    try:
                        maybe_emit_reject_summary()
                    except Exception:
                        pass

                except Exception:
                    if SEND_ERRORS_TO_TELEGRAM:
                        tg_error(f"⚠️ خطأ عام أثناء فحص الإشارات:\n{traceback.format_exc()}")
                    else:
                        print(f"[scan] general error:\n{traceback.format_exc()}")
                finally:
                    last_scan_ts = now

            # 2) إدارة الصفقات المفتوحة (TP/SL/Trailing)
            if now - last_manage_ts >= MANAGE_INTERVAL_SEC:
                try:
                    for symbol in SYMBOLS:
                        if _stop_flag and STOP_POLICY in ("immediate", "debounce"):
                            break
                        try:
                            closed = manage_position(symbol)
                            if closed:
                                print(f"[manage] {symbol} closed by TP/SL/TIME")
                        except Exception as e:
                            if not MUTE_NOISEY_ALERTS and SEND_ERRORS_TO_TELEGRAM:
                                tg_error(f"⚠️ خطأ إدارة {symbol}:\n{e}")
                            else:
                                print(f"[manage_position] {symbol} error: {e}")
                        time.sleep(0.1)
                except Exception:
                    if SEND_ERRORS_TO_TELEGRAM:
                        tg_error(f"⚠️ خطأ عام أثناء إدارة الصفقات:\n{traceback.format_exc()}")
                    else:
                        print(f"[manage] general error:\n{traceback.format_exc()}")
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
                                tg_info(report, parse_mode="HTML", silent=True)
                        except Exception as e:
                            print(f"[daily_report] build error: {e}")
                        last_report_day = day_key
                except Exception:
                    pass

            # نوم قصير
            time.sleep(LOOP_SLEEP_SEC)

    finally:
        # خرجنا بسلاسة (فقط عند سياسات الإيقاف التي تسمح بالخروج)
        if _HAS_CACHE:
            try:
                stop_tickers_cache()
            except Exception:
                pass
        if _stop_flag and STOP_POLICY in ("immediate", "debounce"):
            tg_info("🛑 تم إيقاف البوت — إلى اللقاء.", silent=True)
        else:
            print("🟢 انتهت الحلقة بدون إيقاف مؤكد.")

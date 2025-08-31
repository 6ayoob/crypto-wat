# run.py — Loop for 15m/5m strategy (signals + management + daily report)
import os
import time
import random
import signal
import traceback
from datetime import datetime, timezone, timedelta

import requests

from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS

# الاعتماد على الدوال من الاستراتيجية (النسخة الكاملة التي جهزناها)
from strategy import (
    check_signal, execute_buy, manage_position, load_position,
    count_open_positions, build_daily_report_text
)

# ================== إعدادات الحلقة ==================
# حد الصفقات المفتوحة يُفضَّل قراءته من strategy.count_open_positions() مباشرة
# لكن إن رغبت بتثبيته محليًا ضع قيمة هنا (None = اعتمد دالة الاستراتيجية)
MAX_OPEN_POSITIONS_OVERRIDE = None   # مثال: 1 أو 2 … أو None

# فواصل التكرار
SCAN_INTERVAL_SEC    = 25   # فحص إشارات الدخول (LTF=5m) — 20-45 ثانية مناسب
MANAGE_INTERVAL_SEC  = 10   # إدارة المراكز (TP/SL/Trailing) — 5-15 ثواني
LOOP_SLEEP_SEC       = 1.0  # نوم قصير داخل الحلقة

# تقرير يومي تلقائي قبل منتصف الليل (بتوقيت الرياض)
ENABLE_DAILY_REPORT  = True
DAILY_REPORT_MINUTE  = 23
DAILY_REPORT_SECOND  = 58

# تقليل ضجيج التيليجرام لبعض الرسائل غير الحرجة
MUTE_NOISEY_ALERTS   = True

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

# ================== أدوات ==================
_stop_flag = False
def _handle_stop(signum, frame):
    global _stop_flag
    _stop_flag = True
    try:
        send_telegram_message("⏹️ تم استلام إشارة إيقاف، جاري الإنهاء بهدوء…", disable_notification=True)
    except Exception:
        pass

signal.signal(signal.SIGINT, _handle_stop)
signal.signal(signal.SIGTERM, _handle_stop)

def _now_riyadh():
    return datetime.now(RIYADH_TZ)

def _get_open_positions_count_safe():
    try:
        if MAX_OPEN_POSITIONS_OVERRIDE is not None:
            # احترم القيمة المخصصة إذا أردت تثبيت سقف مختلف مؤقتًا
            return int(min(MAX_OPEN_POSITIONS_OVERRIDE, count_open_positions()))
        return int(count_open_positions())
    except Exception:
        # fallback: عدّ الرموز التي لها صفقة محفوظة
        try:
            return sum(1 for s in SYMBOLS if load_position(s) is not None)
        except Exception:
            return 0

# ================== الحلقة الرئيسية ==================
if __name__ == "__main__":
    send_telegram_message("🚀 تشغيل البوت — استراتيجية 15m/5m (Pullback/Breakout/Hybrid) بدأت ✅")

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
                    if _stop_flag: break

                    # احترم حد الصفقات المفتوحة
                    if open_positions_count >= _get_open_positions_count_safe() and MAX_OPEN_POSITIONS_OVERRIDE is not None:
                        # إذا كنت تستخدم override ثابت، قارن به أيضًا
                        if open_positions_count >= MAX_OPEN_POSITIONS_OVERRIDE:
                            continue

                    # لا تفتح صفقة على رمز لديه مركز قائم
                    pos = None
                    try:
                        pos = load_position(symbol)
                    except Exception:
                        pos = None

                    if pos is not None:
                        continue  # سيُدار في خطوة الإدارة

                    # فحص الإشارة
                    try:
                        sig = check_signal(symbol)
                    except Exception as e:
                        if not MUTE_NOISEY_ALERTS:
                            send_telegram_message(f"⚠️ check_signal خطأ في {symbol}:\n{e}")
                        else:
                            print(f"[check_signal] {symbol} error: {e}")
                        continue

                    if sig == "buy":
                        try:
                            order, msg = execute_buy(symbol)
                            if msg:
                                send_telegram_message(msg)
                            # تحديث العدّ من المصدر بدل الزيادة اليدوية
                            open_positions_count = _get_open_positions_count_safe()
                        except Exception as e:
                            send_telegram_message(f"❌ فشل تنفيذ شراء {symbol}:\n{e}")
                            continue

                    # مهلة قصيرة بين الرموز لتخفيف الضغط
                    time.sleep(0.2)

            except Exception as e:
                send_telegram_message(f"⚠️ خطأ عام أثناء فحص الإشارات:\n{traceback.format_exc()}")
            finally:
                last_scan_ts = now

        # 2) إدارة الصفقات المفتوحة (TP/SL/Trailing)
        if now - last_manage_ts >= MANAGE_INTERVAL_SEC:
            try:
                for symbol in SYMBOLS:
                    if _stop_flag: break
                    try:
                        closed = manage_position(symbol)
                        if closed:
                            send_telegram_message(f"✅ صفقة {symbol} أُغلقت (هدف/وقف).")
                    except Exception as e:
                        # أخطاء الإدارة قد تكون مؤقتة (شبكة/تنفيذ)
                        if not MUTE_NOISEY_ALERTS:
                            send_telegram_message(f"⚠️ خطأ إدارة {symbol}:\n{e}")
                        else:
                            print(f"[manage_position] {symbol} error: {e}")
                    time.sleep(0.1)
            except Exception as e:
                send_telegram_message(f"⚠️ خطأ عام أثناء إدارة الصفقات:\n{traceback.format_exc()}")
            finally:
                last_manage_ts = now

        # 3) تقرير يومي تلقائي (اختياري)
        if ENABLE_DAILY_REPORT:
            try:
                now_r = _now_riyadh()
                if (now_r.hour == DAILY_REPORT_MINUTE and now_r.minute >= DAILY_REPORT_SECOND) or \
                   (now_r.hour == 23 and now_r.minute >= 58):
                    day_key = now_r.strftime("%Y-%m-%d")
                    if last_report_day != day_key:
                        try:
                            report = build_daily_report_text()
                            if report:
                                send_telegram_message(report, parse_mode="HTML", disable_notification=True)
                        except Exception as e:
                            print(f"[daily_report] error: {e}")
                        last_report_day = day_key
            except Exception:
                pass

        # نوم قصير
        time.sleep(LOOP_SLEEP_SEC)

    # خرجنا بسلاسة
    send_telegram_message("🛑 تم إيقاف البوت — إلى اللقاء.")

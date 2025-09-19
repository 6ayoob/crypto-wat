# -*- coding: utf-8 -*-
# main.py — Sync loop (HTF/LTF) with per-round cache + perf metrics + breadth status
# تحسينات:
# - طبقة تيليجرام أقوى (تجزئة الرسائل الطويلة + إعادة محاولة باك-أوف)
# - قفل عملية مفردة (PID file) اختياري لمنع التشغيل المزدوج
# - توقيتات دقيقة مع تعويض انجراف (drift compensation) + jitter خفيف
# - إيقاف مرن بسياسة ignore/debounce/immediate (كما هي) مع رسائل أوضح
# - تلخيص أداء الجولة وإدارة المراكز كما في نسختك + تحسين طباعة الأخطاء
# - نقاط تكامل محسّنة مع الاستراتيجية (ستستفيد لاحقًا من regime controller في strategy.py)
# - حماية شاملة حول الاستدعاءات + تنظيف (finally) آمن
# - NEW: اكتشاف أرصدة السبوت (Discovery) وإنشاء ملفات مراكز مستوردة لإدارتها تلقائيًا

import os
import sys
import time
import random
import signal
import traceback
from time import perf_counter
from datetime import datetime, timezone, timedelta

import requests

from config import (
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    SYMBOLS, STRAT_LTF_TIMEFRAME, STRAT_HTF_TIMEFRAME
)

# الاستراتيجية
from strategy import (
    check_signal, execute_buy, manage_position, load_position, save_position,
    count_open_positions, build_daily_report_text,
    reset_cycle_cache, metrics_format,
    maybe_emit_reject_summary,     # لو غير موجودة سيتخطّاها try/except
    check_signal_debug,            # لو غير موجودة سيتخطّاها try/except
    breadth_status                 # يُتوقع أن تُرجع dict: {ratio,min,ok}
)

# كاش أسعار جماعي من okx_api لتقليل الضغط (اختياري)
try:
    from okx_api import start_tickers_cache, stop_tickers_cache, fetch_balance, fetch_price
    _HAS_CACHE = True
except Exception:
    # حتى لو ما توفر الكاش، نحتاج على الأقل دوال الرصيد/السعر للاكتشاف
    try:
        from okx_api import fetch_balance, fetch_price
    except Exception:
        fetch_balance = lambda asset: 0.0
        fetch_price = lambda symbol: 0.0
    _HAS_CACHE = False

# ================== إعدادات الحلقة ==================
MAX_OPEN_POSITIONS_OVERRIDE = None  # حد محلي لعدد الصفقات (اختياري)

SCAN_INTERVAL_SEC    = int(os.getenv("SCAN_INTERVAL_SEC", "25"))   # فحص إشارات الدخول
MANAGE_INTERVAL_SEC  = int(os.getenv("MANAGE_INTERVAL_SEC", "10")) # إدارة المراكز
LOOP_SLEEP_SEC       = float(os.getenv("LOOP_SLEEP_SEC", "1.0"))

ENABLE_DAILY_REPORT  = os.getenv("ENABLE_DAILY_REPORT", "1").lower() in ("1","true","yes")
DAILY_REPORT_HOUR    = int(os.getenv("DAILY_REPORT_HOUR", "23"))
DAILY_REPORT_MINUTE  = int(os.getenv("DAILY_REPORT_MINUTE", "58"))

MUTE_NOISEY_ALERTS        = True
SEND_ERRORS_TO_TELEGRAM   = os.getenv("SEND_ERRORS_TO_TELEGRAM", "0").lower() in ("1","true","yes")
SEND_INFO_TO_TELEGRAM     = os.getenv("SEND_INFO_TO_TELEGRAM", "1").lower() in ("1","true","yes")
SEND_METRICS_TO_TELEGRAM  = os.getenv("SEND_METRICS_TO_TELEGRAM", "0").lower() in ("1","true","yes")

STOP_POLICY = os.getenv("STOP_POLICY", "debounce").lower()  # ignore | debounce | immediate
STOP_DEBOUNCE_WINDOW_SEC = int(os.getenv("STOP_DEBOUNCE_WINDOW_SEC", "5"))

# قفل عملية مفردة (اختياري) عبر ملف PID
SINGLETON_PIDFILE = os.getenv("PIDFILE", "").strip()

RIYADH_TZ = timezone(timedelta(hours=3))

# ================== أدوات عامة ==================

def _now_riyadh():
    return datetime.now(RIYADH_TZ)

def _print(s: str):
    # طباعة فورية دون تخزين مؤقت (للدِبلويات)
    try:
        print(s, flush=True)
    except Exception:
        try:
            sys.stdout.write(s + "\n"); sys.stdout.flush()
        except Exception:
            pass

# ================== Telegram ==================

_TELEGRAM_MAX_CHARS = 4096

def _tg_post(url: str, payload: dict, tries: int = 3, timeout=10):
    # إعادة المحاولة مع backoff بسيط
    delay = 0.8
    last_err = None
    for _ in range(max(1, tries)):
        try:
            r = requests.post(url, data=payload, timeout=timeout)
            if r.ok:
                return True
            last_err = f"HTTP {r.status_code} {r.text}"
        except Exception as e:
            last_err = str(e)
        time.sleep(delay)
        delay *= 1.6
    _print(f"[TG] Failed: {last_err}")
    return False

def _tg_split_chunks(text: str, max_chars: int = _TELEGRAM_MAX_CHARS):
    text = text or ""
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        nl = text.rfind("\n", start, end)
        if nl != -1 and nl > start:
            end = nl
        chunks.append(text[start:end])
        start = end
        if start < len(text) and text[start] == "\n":
            start += 1
    return chunks

def send_telegram_message(text, parse_mode=None, disable_notification=False):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for chunk in _tg_split_chunks(str(text)):
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": chunk}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if disable_notification:
            payload["disable_notification"] = True
        _tg_post(url, payload)

def _is_error_text(text: str) -> bool:
    if not text:
        return False
    t = str(text).strip()
    return t.startswith(("⚠️","❌")) or ("خطأ" in t) or ("Error" in t)

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

# ================== قفل مفرد (PID file) اختياري ==================

def _acquire_pidfile(path: str) -> bool:
    if not path:
        return True
    try:
        if os.path.exists(path):
            _print(f"⚠️ PIDFILE موجود: {path}. يبدو أن مثيلاً آخر يعمل. إنهاء.")
            return False
        with open(path, "w") as f:
            f.write(str(os.getpid()))
        return True
    except Exception as e:
        _print(f"⚠️ فشل إنشاء PIDFILE {path}: {e}")
        return True  # لا نمنع التشغيل إذا فشلنا بالكتابة

def _release_pidfile(path: str):
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

# ================== أدوات الحلقة / المراكز ==================
_stop_flag = False
_last_stop_signal_ts = 0.0

def _handle_stop(signum, frame):
    """
    سياسة الإيقاف:
      - ignore   : نتجاهل الإشارة (نطبع فقط).
      - debounce : لا نتوقف إلا إذا وصلت إشارتان خلال نافذة قصيرة.
      - immediate: نتوقف فورًا.
    """
    global _stop_flag, _last_stop_signal_ts
    now = time.time()

    if STOP_POLICY == "ignore":
        _print(f"⏸️ تم استقبال إشارة {signum} وتم تجاهلها حسب STOP_POLICY=ignore.")
        return

    if STOP_POLICY == "debounce":
        if (now - _last_stop_signal_ts) <= STOP_DEBOUNCE_WINDOW_SEC:
            _stop_flag = True
            msg = "⏹️ تم تأكيد إيقاف البوت بعد إشارة ثانية ضمن النافذة (debounce)."
            _print(msg)
            tg_info(msg, silent=True)
        else:
            _last_stop_signal_ts = now
            msg = f"⚠️ استلمت إشارة إيقاف. لن يتم الإيقاف إلا إذا وصلت إشارة ثانية خلال {STOP_DEBOUNCE_WINDOW_SEC}ث."
            _print(msg)
            tg_info(msg, silent=True)
        return

    # immediate
    _stop_flag = True
    try:
        tg_info("⏹️ تم استلام إشارة إيقاف — جاري الإنهاء بهدوء…", silent=True)
    except Exception:
        pass

# ربط الإشارات (قد لا يُدعم على بعض الأنظمة)
try:
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)
except Exception:
    pass

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

# ================ NEW: اكتشاف أرصدة السبوت (Discovery) ================
def _discover_spot_positions(min_usd: float = 5.0):
    """
    ينشئ ملفات مراكز مستوردة لأي رصيد Spot موجود بدون ملف.
    يعتمد على fetch_balance(asset) + سعر السوق الحالي لتقدير قيمة USD.
    """
    try:
        for symbol in SYMBOLS:
            base = symbol.split("/")[0]
            # لو عندنا ملف مركز مسبقاً، نكمل
            if load_position(symbol) is not None:
                continue

            # رصيد السبوت المتاح للأصل (بدون مارجن/اقتراض)
            qty = float(fetch_balance(base) or 0.0)
            if qty <= 0.0:
                continue

            px = float(fetch_price(symbol) or 0.0)
            if px <= 0.0:
                continue

            usd_val = qty * px
            if usd_val < float(min_usd):
                continue  # رصيد صغير جداً — تجاهله

            # أنشئ مركز "Imported" لإدارته لاحقاً في manage_position
            pos = {
                "symbol": symbol,
                "variant": "imported",
                "entry_price": px,     # تقدير: آخر سعر
                "qty": qty,
                "imported": True,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "notes": "auto-imported from spot balance"
            }
            save_position(symbol, pos)
            _print(f"[import] created position for {symbol}: qty={qty}, px={px}, ~${usd_val:.2f}")
    except Exception as e:
        _print(f"[import] discovery error: {e}")
# ================== الحلقة الرئيسية ==================
if __name__ == "__main__":
    # قفل مفرد (اختياري)
    if not _acquire_pidfile(SINGLETON_PIDFILE):
        sys.exit(0)

    # بدء كاش الأسعار الجماعي (طلب واحد كل عدة ثوانٍ) إن توفر
    if _HAS_CACHE:
        try:
            start_tickers_cache(period=int(os.getenv("OKX_CACHE_PERIOD", "3")), usdt_only=True)
        except Exception:
            pass

    # ✅ اكتشاف أي مراكز Spot موجودة بالفعل (Discovery)
    try:
        _discover_spot_positions()
    except Exception as e:
        _print(f"[discovery] error: {e}")

    # معلومات بدء مع عرض الإطارات الزمنية الفعلية + حالة السعة
    try:
        bs0 = breadth_status() or {}
        ratio_txt = "—" if bs0.get("ratio") is None else f"{bs0.get('ratio', 0.0):.2f}"
        min_txt   = f"{bs0.get('min', 0.0):.2f}"
        ok_txt    = "✅" if bs0.get("ok") else "❌"
        bs_line   = f"breadth: {ratio_txt}, min={min_txt}, ok={ok_txt}"

        tg_info(
            f"🚀 تشغيل البوت — {len(SYMBOLS)} رمز | HTF={STRAT_HTF_TIMEFRAME} / LTF={STRAT_LTF_TIMEFRAME}\n"
            f"📡 {bs_line}",
            silent=True
        )
    except Exception:
        _print("🚀 تشغيل البوت")

    # جداول زمنية مع تعويض الانجراف
    start_wall = time.time()
    next_scan  = start_wall + random.uniform(0.5, 1.5) + SCAN_INTERVAL_SEC
    next_manage= start_wall + random.uniform(0.2, 0.8) + MANAGE_INTERVAL_SEC
    last_report_day = None
    time.sleep(random.uniform(0.5, 1.5))  # Jitter أولي

    try:
        while True:
            if _stop_flag and STOP_POLICY in ("immediate", "debounce"):
                break

            now = time.time()

            # 1) فحص إشارات الدخول
            if now >= next_scan:
                t_round_start = perf_counter()
                try:
                    try:
                        reset_cycle_cache()
                    except Exception:
                        pass

                    open_positions_count = _get_open_positions_count_safe()
                    for symbol in SYMBOLS:
                        if _stop_flag and STOP_POLICY in ("immediate", "debounce"):
                            break
                        if not _can_open_new_position(open_positions_count):
                            break
                        if load_position(symbol) is not None:
                            continue

                        try:
                            sig = check_signal(symbol)
                        except Exception as e:
                            _print(f"[check_signal] {symbol} error: {e}")
                            continue

                        is_buy = (sig == "buy") or (isinstance(sig, dict) and str(sig.get("decision", "")).lower() == "buy")
                        if is_buy:
                            try:
                                order, msg = execute_buy(symbol)
                                if msg:
                                    if _is_error_text(msg):
                                        tg_error(msg)
                                    else:
                                        tg_info(msg, parse_mode="HTML", silent=False)
                                elif order:
                                    price = getattr(order, "price", None) or getattr(order, "avg_price", None) or ""
                                    qty   = getattr(order, "amount", None) or getattr(order, "qty", None) or ""
                                    tg_info(f"✅ دخول صفقة\nرمز: <b>{symbol}</b>\nسعر: <b>{price}</b>\nكمية: <b>{qty}</b>", parse_mode="HTML", silent=False)
                                open_positions_count = _get_open_positions_count_safe()
                            except Exception as e:
                                _print(f"[execute_buy] {symbol} error: {e}")
                                continue
                        else:
                            try:
                                _, reasons = check_signal_debug(symbol)
                                if reasons:
                                    _print(f"[debug] {symbol} reject reasons: {reasons[:5]}")
                            except Exception:
                                pass
                        time.sleep(0.15)

                    try:
                        maybe_emit_reject_summary()
                    except Exception:
                        pass

                    # أداء الجولة
                    try:
                        t_round_end = perf_counter()
                        dur_sec = t_round_end - t_round_start
                        avg_per_symbol = (dur_sec / max(1, len(SYMBOLS)))
                        bs = breadth_status() or {}
                        b_ratio_txt = "—" if bs.get("ratio") is None else f"{bs.get('ratio'):.2f}"
                        b_line  = f"breadth: <b>{b_ratio_txt}</b> | min: <b>{bs.get('min',0.0):.2f}</b> | {('✅ OK' if bs.get('ok') else '❌ LOW')}"
                        perf_text = (
                            "⏱️ <b>Round Perf</b>\n"
                            f"- Duration: <b>{dur_sec:.2f}s</b>\n"
                            f"- Avg / symbol: <b>{avg_per_symbol:.3f}s</b>\n"
                            f"- {b_line}\n"
                        )
                        metrics_text = metrics_format()
                        full_report = perf_text + "\n" + metrics_text
                        _print(full_report)
                        if SEND_METRICS_TO_TELEGRAM:
                            tg_info(full_report, parse_mode="HTML", silent=True)
                    except Exception:
                        pass

                except Exception:
                    _print(f"[scan] general error:\n{traceback.format_exc()}")
                finally:
                    next_scan += SCAN_INTERVAL_SEC
                    if now - next_scan > SCAN_INTERVAL_SEC:
                        next_scan = now + SCAN_INTERVAL_SEC + random.uniform(-2, 2)

            # 2) إدارة الصفقات المفتوحة
            if now >= next_manage:
                t_manage_start = perf_counter()
                try:
                    for symbol in SYMBOLS:
                        if _stop_flag and STOP_POLICY in ("immediate", "debounce"):
                            break
                        try:
                            closed = manage_position(symbol)
                            if closed:
                                text = None
                                if isinstance(closed, dict):
                                    text = closed.get("text") or closed.get("msg")
                                elif isinstance(closed, (list, tuple)) and closed:
                                    text = closed[0]
                                if text:
                                    tg_info(text, parse_mode="HTML", silent=False)
                                else:
                                    tg_info(f"✅ إغلاق صفقة: <b>{symbol}</b>", parse_mode="HTML", silent=False)
                                _print(f"[manage] {symbol} closed by TP/SL/TIME")
                        except Exception as e:
                            _print(f"[manage_position] {symbol} error: {e}")
                        time.sleep(0.1)

                    try:
                        dur_mng = perf_counter() - t_manage_start
                        _print(f"⏱️ Manage Perf — Duration: {dur_mng:.2f}s")
                    except Exception:
                        pass
                finally:
                    next_manage += MANAGE_INTERVAL_SEC
                    if now - next_manage > MANAGE_INTERVAL_SEC:
                        next_manage = now + MANAGE_INTERVAL_SEC

            # 3) تقرير يومي تلقائي
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
                            _print(f"[daily_report] build error: {e}")
                        last_report_day = day_key
                except Exception:
                    pass

            time.sleep(LOOP_SLEEP_SEC)

    finally:
        try:
            if _HAS_CACHE:
                stop_tickers_cache()
        finally:
            if _stop_flag and STOP_POLICY in ("immediate", "debounce"):
                tg_info("🛑 تم إيقاف البوت — إلى اللقاء.", silent=True)
            else:
                _print("🟢 انتهت الحلقة بدون إيقاف مؤكد.")
            _release_pidfile(SINGLETON_PIDFILE)

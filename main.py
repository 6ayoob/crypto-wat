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
# - NEW/LIQ: بوابة سيولة محدثة مع هامش رسوم اختياري + سجل تشخيصي واضح
# - NEW/RISK: تكامل اختياري مع RiskBlocker (risk_and_notify.py) لحظر الدخول/كتم التنبيهات تحت ظروف سوق سيئة

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
    try:
        from okx_api import fetch_balance, fetch_price
    except Exception:
        fetch_balance = lambda asset: 0.0
        fetch_price = lambda symbol: 0.0
    _HAS_CACHE = False

# ================== تكامل RiskBlocker (اختياري) ==================
_HAS_RISK = False
try:
    # ملفك: risk_and_notify.py
    from risk_and_notify import RiskBlocker, RiskBlockConfig, ensure_ema21, tg_send
    _HAS_RISK = True
except Exception:
    # بدائل آمنة لا تغيّر السلوك لو الملف غير موجود
    class RiskBlocker:
        def __init__(self, *a, **k): pass
        def on_round_start(self, breadth=None): pass
        def record_reject(self, symbol, reasons): pass
        def should_block_entry(self, symbol=None, breadth=None): return False
        def should_suppress_alert(self, reasons=None, breadth=None): return False
        def round_summary_text(self): return ""
    def ensure_ema21(*a, **k): return None
    def tg_send(*a, **k): pass

# ================== إعدادات الحلقة ==================
_MAX_OVERRIDE_ENV = os.getenv("MAX_OPEN_POSITIONS_OVERRIDE") or os.getenv("MAX_OPEN_POSITIONS")
try:
    MAX_OPEN_POSITIONS_OVERRIDE = int(_MAX_OVERRIDE_ENV) if _MAX_OVERRIDE_ENV is not None else None
except ValueError:
    MAX_OPEN_POSITIONS_OVERRIDE = None

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

# ---- سيولة ----
USDT_MIN_RESERVE   = float(os.getenv("USDT_MIN_RESERVE", "5"))
USDT_BUY_THRESHOLD = float(os.getenv("USDT_BUY_THRESHOLD", "15"))
LIQUIDITY_POLICY   = os.getenv("LIQUIDITY_POLICY", "manage_first") # manage_first | neutral
FEE_BUFFER_PCT     = float(os.getenv("FEE_BUFFER_PCT", "0.0"))

# ---- RiskBlocker ENV (اختياري) ----
RB_ENABLED                = os.getenv("RB_ENABLED", "1").lower() in ("1","true","yes")
RB_BREADTH_MIN            = float(os.getenv("RB_BREADTH_MIN", "0.65"))   # أدنى breadth للسماح بالدخول
RB_BLOCK_HTF_DOWN         = os.getenv("RB_BLOCK_HTF_DOWN", "1").lower() in ("1","true","yes")
RB_REJECTS_WINDOW         = int(os.getenv("RB_REJECTS_WINDOW", "120"))    # ثانية
RB_REJECTS_THRESHOLD      = int(os.getenv("RB_REJECTS_THRESHOLD", "40"))  # كم رفضًا يعتبر موجة سلبية
RB_SUPPRESS_ALERTS        = os.getenv("RB_SUPPRESS_ALERTS", "1").lower() in ("1","true","yes")
RB_SUMMARY_EVERY_ROUND    = os.getenv("RB_SUMMARY_EVERY_ROUND", "0").lower() in ("1","true","yes")

# قفل عملية مفردة (اختياري) عبر ملف PID
SINGLETON_PIDFILE = os.getenv("PIDFILE", "").strip()

RIYADH_TZ = timezone(timedelta(hours=3))

# ================== أدوات عامة ==================

def _now_riyadh():
    return datetime.now(RIYADH_TZ)

def _print(s: str):
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
            # لو كان لدينا tg_send مخصص في risk_and_notify استخدمه:
            if _HAS_RISK:
                tg_send(text, parse_mode=parse_mode, silent=silent, fallback=send_telegram_message)
            else:
                send_telegram_message(text, parse_mode=parse_mode, disable_notification=silent)
        except Exception:
            pass

def tg_error(text, parse_mode=None, silent=True):
    if SEND_ERRORS_TO_TELEGRAM:
        try:
            if _HAS_RISK:
                tg_send(text, parse_mode=parse_mode, silent=silent, fallback=send_telegram_message)
            else:
                send_telegram_message(text, parse_mode=parse_mode, disable_notification=silent)
        except Exception:
            pass

# ================== سيولة: أدوات مساعدة ==================

def _usdt_free() -> float:
    try:
        return float(fetch_balance("USDT") or 0.0)
    except Exception:
        return 0.0

def _has_liquidity_for_new_trade() -> bool:
    fee_buf = max(0.0, float(FEE_BUFFER_PCT)) / 100.0
    min_req = (USDT_BUY_THRESHOLD * (1.0 + fee_buf)) + USDT_MIN_RESERVE
    return _usdt_free() >= min_req

def _balance_gate_debug():
    free_now = _usdt_free()
    fee_buf = max(0.0, float(FEE_BUFFER_PCT)) / 100.0
    min_req = (USDT_BUY_THRESHOLD * (1.0 + fee_buf)) + USDT_MIN_RESERVE
    ok = free_now >= min_req
    _print(
        f"[balance_gate] free={free_now:.2f} min_req={min_req:.2f} "
        f"(thr={USDT_BUY_THRESHOLD}, res={USDT_MIN_RESERVE}, fee_buf={fee_buf*100:.2f}%) => {'OK' if ok else 'BLOCK'}"
    )
    return ok

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
        return True

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

    _stop_flag = True
    try:
        tg_info("⏹️ تم استلام إشارة إيقاف — جاري الإنهاء بهدوء…", silent=True)
    except Exception:
        pass

try:
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)
except Exception:
    pass


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

# ================ NEW: اكتشاف أرصدة السبوت (Discovery) ================

def _discover_spot_positions(min_usd: float = 5.0):
    try:
        for symbol in SYMBOLS:
            base = symbol.split("/")[0]
            if load_position(symbol) is not None:
                continue
            qty = float(fetch_balance(base) or 0.0)
            if qty <= 0.0:
                continue
            px = float(fetch_price(symbol) or 0.0)
            if px <= 0.0:
                continue
            usd_val = qty * px
            if usd_val < float(min_usd):
                continue
            pos = {
                "symbol": symbol,
                "variant": "imported",
                "entry_price": px,
                "amount": qty,
                "imported": True,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "notes": "auto-imported from spot balance"
            }
            save_position(symbol, pos)
            _print(f"[import] created position for {symbol}: qty={qty}, px={px}, ~${usd_val:.2f}")
    except Exception as e:
        _print(f"[import] discovery error: {e}")

# ================== RiskBlocker: تهيئة ==================
RB = None
if RB_ENABLED:
    try:
        RB = RiskBlocker(RiskBlockConfig(
            breadth_min=RB_BREADTH_MIN,
            block_htf_down=RB_BLOCK_HTF_DOWN,
            rejects_window_sec=RB_REJECTS_WINDOW,
            rejects_threshold=RB_REJECTS_THRESHOLD,
            suppress_alerts=RB_SUPPRESS_ALERTS
        ))
    except Exception as e:
        _print(f"[risk] init error: {e}")
        RB = None

# ================== الحلقة الرئيسية ==================
if __name__ == "__main__":
    if not _acquire_pidfile(SINGLETON_PIDFILE):
        sys.exit(0)

    if _HAS_CACHE:
        try:
            start_tickers_cache(period=int(os.getenv("OKX_CACHE_PERIOD", "3")), usdt_only=True)
        except Exception:
            pass

    try:
        _discover_spot_positions()
    except Exception as e:
        _print(f"[discovery] error: {e}")

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

    if MAX_OPEN_POSITIONS_OVERRIDE is not None:
        _print(f"[limits] MAX_OPEN_POSITIONS_OVERRIDE = {MAX_OPEN_POSITIONS_OVERRIDE}")

    start_wall = time.time()
    next_scan  = start_wall + random.uniform(0.5, 1.5) + SCAN_INTERVAL_SEC
    next_manage= start_wall + random.uniform(0.2, 0.8) + MANAGE_INTERVAL_SEC
    last_report_day = None
    time.sleep(random.uniform(0.5, 1.5))

    try:
        while True:
            if _stop_flag and STOP_POLICY in ("immediate", "debounce"):
                break

            now = time.time()
            free_now = _usdt_free()

            # breadth الحالي للجولة (مفيد لـ RiskBlocker)
            try:
                bs_now = breadth_status() or {}
            except Exception:
                bs_now = {}

            # أخبر RiskBlocker ببداية الجولة
            if RB:
                try:
                    RB.on_round_start(breadth=bs_now)
                except Exception as e:
                    _print(f"[risk] on_round_start error: {e}")

            should_manage_now = (now >= next_manage)
            if LIQUIDITY_POLICY == "manage_first":
                if (now >= next_scan) and (not _has_liquidity_for_new_trade()):
                    should_manage_now = True

            # 2) إدارة الصفقات
            if should_manage_now:
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
                                    # كتم تنبيهات مزعجة لو RiskBlocker قرر ذلك (اختياري)
                                    if RB and RB.should_suppress_alert(reasons=None, breadth=bs_now):
                                        pass
                                    else:
                                        tg_info(text, parse_mode="HTML", silent=False)
                                else:
                                    if RB and RB.should_suppress_alert(reasons=None, breadth=bs_now):
                                        pass
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

            # 1) فحص إشارات الدخول
            if now >= next_scan:
                gate_ok = _balance_gate_debug()
                if not gate_ok:
                    _print(f"[scan] skipped — free USDT {free_now:.2f} < threshold+reserve")
                    next_scan += SCAN_INTERVAL_SEC
                    if now - next_scan > SCAN_INTERVAL_SEC:
                        next_scan = now + SCAN_INTERVAL_SEC + random.uniform(-2, 2)
                else:
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

                            # حظر سوقي عام قبل أي محاولة دخول
                            if RB and RB.should_block_entry(symbol=symbol, breadth=bs_now):
                                # سجل الرفض داخلياً فقط
                                _print(f"[risk] entry blocked by regime for {symbol}")
                                continue

                            # فحص سيولة
                            if not _has_liquidity_for_new_trade():
                                break

                            # فحص الإشارة
                            try:
                                sig = check_signal(symbol)
                            except Exception as e:
                                _print(f"[check_signal] {symbol} error: {e}")
                                continue

                            is_buy = (sig == "buy") or (isinstance(sig, dict) and str(sig.get("decision", "")).lower() == "buy")
                            if is_buy:
                                # فحص نهائي من RiskBlocker قبل تنفيذ الشراء
                                if RB and RB.should_block_entry(symbol=symbol, breadth=bs_now):
                                    _print(f"[risk] blocked immediate-entry {symbol}")
                                    continue
                                try:
                                    order, msg = execute_buy(symbol)
                                    if msg:
                                        if _is_error_text(msg):
                                            if not (RB and RB.should_suppress_alert(reasons=None, breadth=bs_now)):
                                                tg_error(msg)
                                        else:
                                            if not (RB and RB.should_suppress_alert(reasons=None, breadth=bs_now)):
                                                tg_info(msg, parse_mode="HTML", silent=False)
                                    elif order:
                                        price = getattr(order, "price", None) or getattr(order, "avg_price", None) or ""
                                        qty   = getattr(order, "amount", None) or getattr(order, "qty", None) or ""
                                        if not (RB and RB.should_suppress_alert(reasons=None, breadth=bs_now)):
                                            tg_info(f"✅ دخول صفقة\nرمز: <b>{symbol}</b>\nسعر: <b>{price}</b>\nكمية: <b>{qty}</b>", parse_mode="HTML", silent=False)
                                    open_positions_count = _get_open_positions_count_safe()
                                except Exception as e:
                                    _print(f"[execute_buy] {symbol} error: {e}")
                                    continue
                            else:
                                # اجلب أسباب الرفض (إن وُجدت) وسجّلها لدى RiskBlocker
                                try:
                                    _, reasons = check_signal_debug(symbol)
                                    if reasons:
                                        # تعليم RiskBlocker بموجات الرفض
                                        if RB:
                                            try:
                                                RB.record_reject(symbol, reasons)
                                            except Exception:
                                                pass
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
                            bs = bs_now or {}
                            b_ratio_txt = "—" if bs.get("ratio") is None else f"{bs.get('ratio'):.2f}"
                            b_line  = f"breadth: <b>{b_ratio_txt}</b> | min: <b>{bs.get('min',0.0):.2f}</b> | {('✅ OK' if bs.get('ok') else '❌ LOW')}"
                            perf_text = (
                                "⏱️ <b>Round Perf</b>\n"
                                f"- Duration: <b>{dur_sec:.2f}s</b>\n"
                                f"- Avg / symbol: <b>{avg_per_symbol:.3f}s</b>\n"
                                f"- {b_line}\n"
                            )
                            metrics_text = metrics_format()
                            # ملخّص RiskBlocker (اختياري)
                            rb_text = ""
                            if RB and RB_SUMMARY_EVERY_ROUND:
                                try:
                                    rb_text = RB.round_summary_text() or ""
                                except Exception:
                                    rb_text = ""
                            full_report = perf_text + "\n" + metrics_text + (("\n" + rb_text) if rb_text else "")
                            _print(full_report)
                            if SEND_METRICS_TO_TELEGRAM and not (RB and RB.should_suppress_alert(reasons=None, breadth=bs_now)):
                                tg_info(full_report, parse_mode="HTML", silent=True)
                        except Exception:
                            pass

                    except Exception:
                        _print(f"[scan] general error:\n{traceback.format_exc()}")
                    finally:
                        next_scan += SCAN_INTERVAL_SEC
                        if now - next_scan > SCAN_INTERVAL_SEC:
                            next_scan = now + SCAN_INTERVAL_SEC + random.uniform(-2, 2)

            # 3) تقرير يومي تلقائي
            if ENABLE_DAILY_REPORT:
                try:
                    now_r = _now_riyadh()
                    day_key = now_r.strftime("%Y-%m-%d")
                    if (now_r.hour == DAILY_REPORT_HOUR and now_r.minute >= DAILY_REPORT_MINUTE) and (last_report_day != day_key):
                        try:
                            report = build_daily_report_text()
                            if report and not (RB and RB.should_suppress_alert(reasons=None, breadth=bs_now)):
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

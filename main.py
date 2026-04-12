# -*- coding: utf-8 -*-
"""
main.py — نسخة محسّنة v2.0
التغييرات الرئيسية:
- إصلاح loop الإدارة: كل base يُدار مرة واحدة فقط (بدل 5 مرات)
- تفعيل variants بشكل صحيح في check_signal
- SCAN_INTERVAL مناسب لعدد الرموز الجديد (35 بدل 300)
- إضافة break-even notification
- تنظيف الكود وتبسيط المنطق
"""

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
    SYMBOLS, STRAT_LTF_TIMEFRAME, STRAT_HTF_TIMEFRAME,
    TRADE_BASE_USDT, MAX_OPEN_POSITIONS,
)

from strategy import (
    check_signal, check_signal_debug, manage_position,
    load_position, save_position,
    count_open_positions, build_daily_report_text,
    reset_cycle_cache, metrics_format,
    maybe_emit_reject_summary, breadth_status,
    execute_buy,
)

try:
    from okx_api import start_tickers_cache, stop_tickers_cache, fetch_balance, fetch_price
    _HAS_CACHE = True
except Exception:
    try:
        from okx_api import fetch_balance, fetch_price
    except Exception:
        fetch_balance = lambda asset=None: 0.0
        fetch_price   = lambda symbol=None: 0.0
    _HAS_CACHE = False

# ================== إعدادات الحلقة ==================
# مناسبة لـ 35 رمز بدل 300
SCAN_INTERVAL_SEC   = int(os.getenv("SCAN_INTERVAL_SEC",   "30"))   # كل 30 ثانية كافٍ
MANAGE_INTERVAL_SEC = int(os.getenv("MANAGE_INTERVAL_SEC", "15"))   # كل 15 ثانية
LOOP_SLEEP_SEC      = float(os.getenv("LOOP_SLEEP_SEC",    "1.0"))
SYMBOL_SLEEP_SEC    = float(os.getenv("SYMBOL_SLEEP_SEC",  "0.10")) # تأخير بين الرموز

ENABLE_DAILY_REPORT = os.getenv("ENABLE_DAILY_REPORT", "1").lower() in ("1","true","yes")
DAILY_REPORT_HOUR   = int(os.getenv("DAILY_REPORT_HOUR",   "23"))
DAILY_REPORT_MINUTE = int(os.getenv("DAILY_REPORT_MINUTE", "58"))

SEND_ERRORS_TO_TELEGRAM  = os.getenv("SEND_ERRORS_TO_TELEGRAM",  "0").lower() in ("1","true","yes")
SEND_INFO_TO_TELEGRAM    = os.getenv("SEND_INFO_TO_TELEGRAM",    "1").lower() in ("1","true","yes")
SEND_METRICS_TO_TELEGRAM = os.getenv("SEND_METRICS_TO_TELEGRAM", "0").lower() in ("1","true","yes")

NOTIFY_ENTRIES = os.getenv("NOTIFY_ENTRIES", "1").lower() in ("1","true","yes")
NOTIFY_EXITS   = os.getenv("NOTIFY_EXITS",   "1").lower() in ("1","true","yes")

STOP_POLICY              = os.getenv("STOP_POLICY", "debounce").lower()
STOP_DEBOUNCE_WINDOW_SEC = int(os.getenv("STOP_DEBOUNCE_WINDOW_SEC", "5"))

SINGLETON_PIDFILE = os.getenv("PIDFILE", "").strip()

# بوابة السيولة
EXCHANGE_MIN_NOTIONAL_USDT = float(os.getenv("EXCHANGE_MIN_NOTIONAL_USDT", "5.0"))
BALANCE_RESERVE_USDT       = float(os.getenv("BALANCE_RESERVE_USDT",       "5.0"))
FEE_BUF_BPS                = float(os.getenv("FEE_BUF_BPS", "20"))

RIYADH_TZ = timezone(timedelta(hours=3))

# ================== أدوات عامة ==================
def _now_riyadh() -> datetime:
    return datetime.now(RIYADH_TZ)

def _print(s: str) -> None:
    try:
        print(s, flush=True)
    except Exception:
        pass

# ================== Telegram ==================
_TELEGRAM_MAX_CHARS = 4096

def _tg_post(url: str, payload: dict, tries: int = 3, timeout: int = 10) -> bool:
    delay = 0.8
    for _ in range(max(1, tries)):
        try:
            r = requests.post(url, data=payload, timeout=timeout)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(delay)
        delay *= 1.6
    return False

def _tg_chunks(text: str, max_chars: int = _TELEGRAM_MAX_CHARS):
    text = text or ""
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        nl  = text.rfind("\n", start, end)
        if nl != -1 and nl > start:
            end = nl
        chunks.append(text[start:end])
        start = end + (1 if start < len(text) and text[start:start+1] == "\n" else 0)
    return chunks

def send_telegram_message(text: str, parse_mode: str = None, silent: bool = False) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for chunk in _tg_chunks(str(text)):
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": chunk}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if silent:
            payload["disable_notification"] = True
        _tg_post(url, payload)

def tg_info(text: str, parse_mode: str = None, silent: bool = True) -> None:
    if SEND_INFO_TO_TELEGRAM:
        try:
            send_telegram_message(text, parse_mode=parse_mode, silent=silent)
        except Exception:
            pass

def tg_error(text: str) -> None:
    if SEND_ERRORS_TO_TELEGRAM:
        try:
            send_telegram_message(f"❌ {text}", parse_mode=None, silent=True)
        except Exception:
            pass

# ================== سيولة ==================
def _usdt_free() -> float:
    try:
        return float(fetch_balance("USDT") or 0.0)
    except Exception:
        return 0.0

def _min_required() -> float:
    fee_buf = (FEE_BUF_BPS / 10000.0) * max(TRADE_BASE_USDT, EXCHANGE_MIN_NOTIONAL_USDT)
    return max(EXCHANGE_MIN_NOTIONAL_USDT, TRADE_BASE_USDT * 0.5) + BALANCE_RESERVE_USDT + fee_buf

def _has_liquidity() -> bool:
    return _usdt_free() >= _min_required()

# ================== قفل مفرد ==================
def _acquire_pidfile(path: str) -> bool:
    if not path:
        return True
    try:
        if os.path.exists(path):
            _print(f"⚠️ PIDFILE موجود: {path}. مثيل آخر يعمل.")
            return False
        with open(path, "w") as f:
            f.write(str(os.getpid()))
        return True
    except Exception:
        return True

def _release_pidfile(path: str) -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

# ================== إيقاف مرن ==================
_stop_flag = False
_last_stop_ts = 0.0

def _handle_stop(signum, frame) -> None:
    global _stop_flag, _last_stop_ts
    now = time.time()

    if STOP_POLICY == "ignore":
        _print(f"⏸️ إشارة {signum} تم تجاهلها.")
        return

    if STOP_POLICY == "debounce":
        if (now - _last_stop_ts) <= STOP_DEBOUNCE_WINDOW_SEC:
            _stop_flag = True
            tg_info("⏹️ تم تأكيد الإيقاف.", silent=True)
        else:
            _last_stop_ts = now
            tg_info(f"⚠️ إشارة إيقاف. أرسل مرة ثانية خلال {STOP_DEBOUNCE_WINDOW_SEC}ث للتأكيد.", silent=True)
        return

    _stop_flag = True
    tg_info("⏹️ جاري الإيقاف…", silent=True)

try:
    signal.signal(signal.SIGINT,  _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)
except Exception:
    pass

# ================== مساعدات المراكز ==================
def _open_count() -> int:
    try:
        return int(count_open_positions())
    except Exception:
        return 0

def _can_open() -> bool:
    return _open_count() < MAX_OPEN_POSITIONS

# ================== اكتشاف أرصدة Spot ==================
def _discover_spot_positions(min_usd: float = 5.0) -> None:
    """
    ينشئ ملفات مراكز مستوردة لأي رصيد Spot موجود.
    يعمل على BASE فريد فقط (لا تكرار).
    """
    seen_bases = set()
    for symbol in SYMBOLS:
        base = symbol.split("#")[0]
        if base in seen_bases:
            continue
        seen_bases.add(base)

        # تخطّى إن وجد مركز بالفعل
        if load_position(base) is not None:
            continue

        coin = base.split("/")[0]
        try:
            qty = float(fetch_balance(coin) or 0.0)
            if qty <= 0.0:
                continue
            px = float(fetch_price(base) or 0.0)
            if px <= 0.0 or qty * px < min_usd:
                continue
            pos = {
                "symbol":     base,
                "variant":    "imported",
                "entry_price": px,
                "amount":      qty,
                "imported":    True,
                "created_at":  datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "notes":       "auto-imported from spot balance",
            }
            save_position(base, pos)
            _print(f"[import] {base}: qty={qty:.6f}, px={px:.4f}, ~${qty*px:.2f}")
        except Exception as e:
            _print(f"[import] {base} error: {e}")

# ================== إدارة المراكز — مُصلَحة ==================
def _manage_all_positions() -> None:
    """
    إدارة كل المراكز المفتوحة.
    الإصلاح الرئيسي: كل base يُدار مرة واحدة فقط (بدل 5 مرات مع variants).
    """
    seen_bases = set()
    for symbol in SYMBOLS:
        if _stop_flag:
            break
        base = symbol.split("#")[0]
        if base in seen_bases:
            continue              # ← تجنب التكرار
        seen_bases.add(base)

        pos = load_position(base)
        if pos is None:
            continue              # لا مركز مفتوح

        try:
            result = manage_position(base)
            if result:
                _print(f"[manage] {base} → تم التصرف (TP/SL/TIME)")
                if NOTIFY_EXITS:
                    text = None
                    if isinstance(result, dict):
                        text = result.get("text") or result.get("msg")
                    tg_info(text or f"✅ إغلاق: <b>{base}</b>", parse_mode="HTML", silent=False)
        except Exception as e:
            _print(f"[manage] {base} error: {e}")

        time.sleep(0.05)

# ================== فحص الإشارات — مُصلَح ==================
def _scan_signals() -> None:
    """
    فحص إشارات الدخول.
    الإصلاح الرئيسي: يمرر الـ symbol الكامل (مع #variant) لـ check_signal
    حتى تعمل الـ variants بشكل صحيح.
    """
    seen_bases = set()

    for symbol in SYMBOLS:
        if _stop_flag:
            break

        base = symbol.split("#")[0]

        # تجنب فتح أكثر من مركز لنفس الـ base
        if base in seen_bases:
            continue
        if load_position(base) is not None:
            seen_bases.add(base)
            continue

        if not _can_open():
            _print(f"[scan] MAX_OPEN_POSITIONS={MAX_OPEN_POSITIONS} reached.")
            break

        if not _has_liquidity():
            _print(f"[scan] رصيد غير كافٍ — إيقاف الفحص.")
            break

        try:
            # ← الإصلاح: نمرر symbol الكامل (مع variant) لا base فقط
            sig = check_signal(symbol)
        except Exception as e:
            _print(f"[check_signal] {symbol} error: {e}")
            time.sleep(SYMBOL_SLEEP_SEC)
            continue

        is_buy = (
            sig == "buy" or
            (isinstance(sig, dict) and str(sig.get("decision", "")).lower() == "buy")
        )

        if is_buy:
            try:
                order, msg = execute_buy(base, sig if isinstance(sig, dict) else None)
                if order:
                    seen_bases.add(base)  # منع تكرار الدخول في نفس الجولة
                    notify_text = msg if (msg and isinstance(msg, str)) else None
                    tg_info(
                        notify_text or f"🚀 دخول: <b>{symbol}</b>",
                        parse_mode="HTML",
                        silent=False
                    )
                    _print(f"[buy] ✅ {symbol} | {msg}")
                else:
                    _print(f"[buy] ❌ {symbol}: {msg}")
            except Exception as e:
                _print(f"[execute_buy] {symbol} error: {e}")
        else:
            # طباعة سبب الرفض في وضع الـ debug
            try:
                _, reasons = check_signal_debug(symbol)
                if reasons:
                    _print(f"[skip] {symbol}: {reasons[0]}")
            except Exception:
                pass

        time.sleep(SYMBOL_SLEEP_SEC)

# ================== الحلقة الرئيسية ==================
def main() -> None:
    global _stop_flag

    if not _acquire_pidfile(SINGLETON_PIDFILE):
        sys.exit(0)

    # بدء كاش الأسعار الجماعي
    if _HAS_CACHE:
        try:
            start_tickers_cache(
                period=int(os.getenv("OKX_CACHE_PERIOD", "6")),
                usdt_only=True
            )
        except Exception:
            pass

    # اكتشاف أرصدة Spot موجودة
    try:
        _discover_spot_positions()
    except Exception as e:
        _print(f"[discovery] error: {e}")

    # رسالة البدء
    try:
        bs   = breadth_status() or {}
        b_ok = "✅" if bs.get("ok") else "❌"
        b_r  = "—" if bs.get("ratio") is None else f"{bs['ratio']:.2f}"
        tg_info(
            f"🚀 <b>تشغيل البوت v2.0</b>\n"
            f"📊 رموز: <b>{len(SYMBOLS)}</b> | HTF: <b>{STRAT_HTF_TIMEFRAME}</b> | LTF: <b>{STRAT_LTF_TIMEFRAME}</b>\n"
            f"📡 Breadth: <b>{b_r}</b> {b_ok} | رأس المال: <b>{_usdt_free():.2f}$</b>",
            parse_mode="HTML", silent=True
        )
    except Exception:
        _print("🚀 تشغيل البوت v2.0")

    _print(f"[init] SYMBOLS={len(SYMBOLS)} | SCAN={SCAN_INTERVAL_SEC}s | MANAGE={MANAGE_INTERVAL_SEC}s")

    now         = time.time()
    next_scan   = now + random.uniform(1.0, 3.0)
    next_manage = now + random.uniform(0.5, 1.5)
    last_report_day = None

    try:
        while not _stop_flag:
            now = time.time()

            # ── إدارة المراكز ──
            if now >= next_manage:
                t0 = perf_counter()
                try:
                    _manage_all_positions()
                except Exception:
                    _print(f"[manage] error:\n{traceback.format_exc()}")
                finally:
                    next_manage = time.time() + MANAGE_INTERVAL_SEC
                _print(f"[manage] ⏱ {perf_counter()-t0:.2f}s")

            # ── فحص الإشارات ──
            if now >= next_scan:
                if not _has_liquidity():
                    _print(f"[scan] skip — رصيد {_usdt_free():.2f}$ < {_min_required():.2f}$")
                else:
                    t0 = perf_counter()
                    try:
                        reset_cycle_cache()
                        _scan_signals()
                        maybe_emit_reject_summary()
                    except Exception:
                        _print(f"[scan] error:\n{traceback.format_exc()}")
                    finally:
                        dur = perf_counter() - t0

                    # أداء الجولة
                    try:
                        bs      = breadth_status() or {}
                        b_r     = "—" if bs.get("ratio") is None else f"{bs['ratio']:.2f}"
                        b_ok    = "✅" if bs.get("ok") else "❌"
                        mf      = metrics_format()
                        summary = (
                            f"⏱ <b>جولة انتهت</b> في {dur:.1f}s\n"
                            f"📡 Breadth: {b_r} {b_ok}\n"
                            f"{mf}"
                        )
                        _print(summary)
                        if SEND_METRICS_TO_TELEGRAM:
                            tg_info(summary, parse_mode="HTML", silent=True)
                    except Exception:
                        pass

                next_scan = time.time() + SCAN_INTERVAL_SEC

            # ── تقرير يومي ──
            if ENABLE_DAILY_REPORT:
                try:
                    now_r   = _now_riyadh()
                    day_key = now_r.strftime("%Y-%m-%d")
                    if (
                        now_r.hour   == DAILY_REPORT_HOUR and
                        now_r.minute >= DAILY_REPORT_MINUTE and
                        last_report_day != day_key
                    ):
                        report = build_daily_report_text()
                        if report:
                            tg_info(report, parse_mode="HTML", silent=True)
                        last_report_day = day_key
                except Exception as e:
                    _print(f"[daily_report] error: {e}")

            time.sleep(LOOP_SLEEP_SEC)

    finally:
        try:
            if _HAS_CACHE:
                stop_tickers_cache()
        except Exception:
            pass
        _release_pidfile(SINGLETON_PIDFILE)
        tg_info("🛑 <b>البوت أوقف</b>.", parse_mode="HTML", silent=True)
        _print("🛑 البوت أوقف.")

if __name__ == "__main__":
    main()

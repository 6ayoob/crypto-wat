# main.py — Loop for 15m/5m strategy (signals + management + daily report)
import os
import time
import random
import signal
import traceback
from datetime import datetime, timezone, timedelta
import asyncio
import requests

from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS, STRAT_LTF_TIMEFRAME, STRAT_HTF_TIMEFRAME

# الاستراتيجية
from strategy import (
    check_signal, execute_buy, manage_position, load_position,
    count_open_positions, build_daily_report_text, reset_cycle_cache
)

# (اختياري) دوال تشخيص من الاستراتيجية
try:
    from strategy import maybe_emit_reject_summary, check_signal_debug
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
MAX_OPEN_POSITIONS_OVERRIDE = None  # مثال: 2 أو 3 … أو None لإيقافه

SCAN_INTERVAL_SEC    = int(os.getenv("SCAN_INTERVAL_SEC", "25"))
MANAGE_INTERVAL_SEC  = int(os.getenv("MANAGE_INTERVAL_SEC", "10"))
LOOP_SLEEP_SEC       = 1.0

ENABLE_DAILY_REPORT  = os.getenv("ENABLE_DAILY_REPORT", "1").lower() in ("1","true","yes")
DAILY_REPORT_HOUR    = int(os.getenv("DAILY_REPORT_HOUR", "23"))
DAILY_REPORT_MINUTE  = int(os.getenv("DAILY_REPORT_MINUTE", "58"))

MUTE_NOISEY_ALERTS        = True
SEND_ERRORS_TO_TELEGRAM   = os.getenv("SEND_ERRORS_TO_TELEGRAM", "0").lower() in ("1","true","yes")
SEND_INFO_TO_TELEGRAM     = os.getenv("SEND_INFO_TO_TELEGRAM", "1").lower() in ("1","true","yes")

STOP_POLICY = os.getenv("STOP_POLICY", "debounce").lower()
STOP_DEBOUNCE_WINDOW_SEC = int(os.getenv("STOP_DEBOUNCE_WINDOW_SEC", "5"))

RIYADH_TZ = timezone(timedelta(hours=3))

# تحكم بالتوازي
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "8"))
TASK_TIMEOUT_SEC = int(os.getenv("TASK_TIMEOUT_SEC", "25"))

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
    global _stop_flag, _last_stop_signal_ts
    now = time.time()

    if STOP_POLICY == "ignore":
        print(f"⏸️ تم استقبال إشارة {signum} وتم تجاهلها.")
        return

    if STOP_POLICY == "debounce":
        if (now - _last_stop_signal_ts) <= STOP_DEBOUNCE_WINDOW_SEC:
            _stop_flag = True
            msg = "⏹️ تم تأكيد إيقاف البوت بعد إشارة ثانية."
            print(msg)
            tg_info(msg, silent=True)
        else:
            _last_stop_signal_ts = now
            msg = f"⚠️ استلمت إشارة إيقاف. لن يتم الإيقاف إلا إذا وصلت إشارة ثانية خلال {STOP_DEBOUNCE_WINDOW_SEC}ث."
            print(msg)
            tg_info(msg, silent=True)
        return

    _stop_flag = True
    tg_info("⏹️ تم استلام إشارة إيقاف — جاري الإنهاء بهدوء…", silent=True)

try:
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)
except Exception:
    pass

def _now_riyadh():
    return datetime.now(RIYADH_TZ)

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

# ================== مهام Async ==================
async def _scan_one_symbol(symbol: str, sem: asyncio.Semaphore):
    async with sem:
        try:
            if load_position(symbol) is not None:
                return
            sig = check_signal(symbol)
            is_buy = (sig == "buy") or (isinstance(sig, dict) and str(sig.get("decision", "")).lower() == "buy")
            if is_buy:
                order, msg = execute_buy(symbol)
                if msg:
                    if _is_error_text(msg):
                        if SEND_ERRORS_TO_TELEGRAM: tg_error(msg)
                    else:
                        tg_info(msg)
            else:
                try:
                    _, reasons = check_signal_debug(symbol)
                    if reasons:
                        print(f"[debug] {symbol} reject reasons: {reasons[:5]}")
                except Exception:
                    pass
        except Exception as e:
            print(f"[scan_one] {symbol} error: {e}")

async def scan_symbols_round(symbols):
    reset_cycle_cache()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [asyncio.wait_for(_scan_one_symbol(sym, sem), timeout=TASK_TIMEOUT_SEC) for sym in symbols]
    await asyncio.gather(*tasks, return_exceptions=True)

async def manage_positions_round(symbols):
    for symbol in symbols:
        try:
            closed = manage_position(symbol)
            if closed:
                print(f"[manage] {symbol} closed by TP/SL/TIME")
        except Exception as e:
            print(f"[manage] {symbol} error: {e}")
        await asyncio.sleep(0)

# ================== الحلقة الرئيسية ==================
async def main_loop():
    if _HAS_CACHE:
        try:
            start_tickers_cache(period=int(os.getenv("OKX_CACHE_PERIOD", "3")), usdt_only=True)
        except Exception:
            pass

    tg_info(f"🚀 تشغيل البوت — {len(SYMBOLS)} رمز | HTF={STRAT_HTF_TIMEFRAME} / LTF={STRAT_LTF_TIMEFRAME} ✅", silent=True)

    last_scan_ts   = 0.0
    last_manage_ts = 0.0
    last_report_day = None

    await asyncio.sleep(random.uniform(0.5, 1.5))

    try:
        while True:
            if _stop_flag and STOP_POLICY in ("immediate", "debounce"):
                break
            now = time.time()

            if now - last_scan_ts >= SCAN_INTERVAL_SEC:
                try:
                    await scan_symbols_round(SYMBOLS)
                    maybe_emit_reject_summary()
                except Exception as e:
                    tg_error(f"[scan] error:\n{traceback.format_exc()}")
                finally:
                    last_scan_ts = now

            if now - last_manage_ts >= MANAGE_INTERVAL_SEC:
                try:
                    await manage_positions_round(SYMBOLS)
                except Exception:
                    tg_error(f"[manage] error:\n{traceback.format_exc()}")
                finally:
                    last_manage_ts = now

            if ENABLE_DAILY_REPORT:
                try:
                    now_r = _now_riyadh()
                    day_key = now_r.strftime("%Y-%m-%d")
                    if (now_r.hour == DAILY_REPORT_HOUR and now_r.minute >= DAILY_REPORT_MINUTE) and (last_report_day != day_key):
                        report = build_daily_report_text()
                        if report:
                            tg_info(report, parse_mode="HTML", silent=True)
                        last_report_day = day_key
                except Exception:
                    pass

            await asyncio.sleep(LOOP_SLEEP_SEC)
    finally:
        if _HAS_CACHE:
            try: stop_tickers_cache()
            except Exception: pass
        if _stop_flag and STOP_POLICY in ("immediate", "debounce"):
            tg_info("🛑 تم إيقاف البوت — إلى اللقاء.", silent=True)
        else:
            print("🟢 انتهت الحلقة بدون إيقاف مؤكد.")

if __name__ == "__main__":
    asyncio.run(main_loop())

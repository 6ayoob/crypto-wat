# -*- coding: utf-8 -*-
# main.py — Sync loop (HTF/LTF) with per-round cache + perf metrics + breadth status
# تحسينات:
# - طبقة تيليجرام أقوى (تجزئة الرسائل الطويلة + إعادة محاولة باك-أوف)
# - قفل عملية مفردة (PID file) اختياري لمنع التشغيل المزدوج
# - توقيتات دقيقة مع تعويض انجراف (drift compensation) + jitter خفيف
# - إيقاف مرن بسياسة ignore/debounce/immediate (كما هي) مع رسائل أوضح
# - تلخيص أداء الجولة وإدارة المراكز + تحسين طباعة الأخطاء
# - تكامل محسّن مع الاستراتيجية (execute_buy / check_signal_debug)
# - حماية شاملة حول الاستدعاءات + تنظيف (finally) آمن
# - NEW: اكتشاف أرصدة السبوت (Discovery) وإنشاء ملفات مراكز مستوردة للـ base فقط
# - NEW/LIQ: بوابة سيولة مبسّطة وغير خانقة — تسمح بالعمل حتى مع أرصدة صغيرة
# - NEW/RISK: تم تعطيل التكامل الخارجي RiskBlocker نهائيًا (نعتمد ريسك الاستراتيجية)
# - NEW/TG: إشعار دخول «مؤكّد» حتى لو لم تُرجع strategy رسالة (fallback افتراضي)

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
    TRADE_BASE_USDT,  # حجم الأساس (تُستعمل داخل الاستراتيجية أيضًا)
)

# من strategy — لاحظ استعمال execute_buy (بديل open_trade) + check_signal_debug
from strategy import (
    check_signal, check_signal_debug, manage_position, load_position, save_position,
    count_open_positions, build_daily_report_text,
    reset_cycle_cache, metrics_format,
    maybe_emit_reject_summary, breadth_status,
    execute_buy,            # ← فتح الصفقة داخل الاستراتيجية (هي تتحكم بالحجم والتدرّج)
)

# كاش أسعار جماعي من okx_api لتقليل الضغط (اختياري)
try:
    from okx_api import start_tickers_cache, stop_tickers_cache, fetch_balance, fetch_price
    _HAS_CACHE = True
except Exception:
    try:
        from okx_api import fetch_balance, fetch_price
    except Exception:
        fetch_balance = lambda asset=None: 0.0
        fetch_price = lambda symbol=None: 0.0
    _HAS_CACHE = False

# ===== RiskBlocker معطّل نهائيًا (stub آمن) =====
_risk = None
def _risk_is_blocked() -> bool:
    # لا يوجد حظر خارجي، نعتمد فقط على نظام الريسك الداخلي في strategy.py
    return False

# ================== إعدادات الحلقة ==================
_MAX_OVERRIDE_ENV = os.getenv("MAX_OPEN_POSITIONS_OVERRIDE") or os.getenv("MAX_OPEN_POSITIONS")
try:
    MAX_OPEN_POSITIONS_OVERRIDE = int(_MAX_OVERRIDE_ENV) if _MAX_OVERRIDE_ENV is not None else None
except ValueError:
    MAX_OPEN_POSITIONS_OVERRIDE = None

SCAN_INTERVAL_SEC    = int(os.getenv("SCAN_INTERVAL_SEC", "25"))
MANAGE_INTERVAL_SEC  = int(os.getenv("MANAGE_INTERVAL_SEC", "10"))
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

# ---- سيولة (مبسّطة وغير خانقة) ----
EXCHANGE_MIN_NOTIONAL_USDT = float(os.getenv("EXCHANGE_MIN_NOTIONAL_USDT", "5.0"))  # حد المنصّة الأدنى
BALANCE_THRESHOLD_USDT     = float(os.getenv("BALANCE_THRESHOLD_USDT", "3.0"))      # سماح بمحاولات صغيرة
BALANCE_RESERVE_USDT       = float(os.getenv("BALANCE_RESERVE_USDT", "0.0"))        # احتياطي بسيط
FEE_BUF_BPS                = float(os.getenv("FEE_BUF_BPS", "20"))                  # 0.20% افتراضيًا
LIQUIDITY_POLICY           = os.getenv("LIQUIDITY_POLICY", "manage_first")          # manage_first | neutral

# ---- إشعارات تيليجرام (تحكم دقيق) ----
NOTIFY_ENTRIES = os.getenv("NOTIFY_ENTRIES", "1").lower() in ("1","true","yes")
NOTIFY_EXITS   = os.getenv("NOTIFY_EXITS",   "1").lower() in ("1","true","yes")

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

# ===== تنسيقات سريعة لإشعارات الدخول/الخروج =====
def _fmt_num(v, nd=4):
    try:
        return (f"{float(v):.{nd}f}")
    except Exception:
        return str(v)

def tg_trade_open(symbol: str, order: dict | None, extra: dict | None = None):
    if not NOTIFY_ENTRIES:
        return
    # محاولة قراءة حقول شائعة من order
    p = None
    amt = None
    cost = None
    if isinstance(order, dict):
        p = order.get("price") or order.get("avg") or order.get("average")
        amt = order.get("amount") or order.get("filled") or order.get("qty")
        cost = order.get("cost")
    parts = [f"🚀 <b>دخول صفقة</b> — <b>{symbol}</b>"]
    if p is not None:   parts.append(f"السعر: <code>{_fmt_num(p, 6)}</code>")
    if amt is not None: parts.append(f"الكمية: <code>{_fmt_num(amt, 6)}</code>")
    if cost is not None:parts.append(f"القيمة: <code>{_fmt_num(cost, 2)}</code> USDT")
    if extra and extra.get("note"):
        parts.append(str(extra["note"]))
    msg = "\n".join(parts)
    tg_info(msg, parse_mode="HTML", silent=False)

def tg_trade_close(symbol: str, text: str | None = None):
    if not NOTIFY_EXITS:
        return
    if text:
        tg_info(text, parse_mode="HTML", silent=False)
    else:
        tg_info(f"✅ إغلاق صفقة: <b>{symbol}</b>", parse_mode="HTML", silent=False)

# ================== سيولة: أدوات مساعدة (بوابة غير خانقة) ==================

def _usdt_free() -> float:
    try:
        # يدعم صيغ fetch_balance المختلفة: fetch_balance("USDT") أو قاموس
        val = fetch_balance("USDT")
        return float(val or 0.0)
    except Exception:
        return 0.0


def _min_req_now(want_usdt: float | None = None) -> float:
    """الحد الأدنى الواقعي المطلوب لمحاولة شراء (مع هامش رسوم بسيط)."""
    if want_usdt is None:
        want_usdt = float(TRADE_BASE_USDT or 0.0)
    fee_buf = (FEE_BUF_BPS / 10000.0) * max(want_usdt, EXCHANGE_MIN_NOTIONAL_USDT)
    base_need = max(EXCHANGE_MIN_NOTIONAL_USDT, BALANCE_THRESHOLD_USDT)
    return base_need + BALANCE_RESERVE_USDT + fee_buf


def _has_liquidity_for_new_trade() -> bool:
    """تسمح بالمحاولة حتى مع رصيد صغير إن كان فوق الحد الأدنى الواقعي."""
    free_now = _usdt_free()
    need = _min_req_now(TRADE_BASE_USDT)
    return free_now >= need


def _balance_gate_debug():
    free_now = _usdt_free()
    need = _min_req_now(TRADE_BASE_USDT)
    ok = free_now >= need
    _print(
        f"[balance_gate] free={free_now:.2f} need>={need:.2f} "
        f"(min_notional={EXCHANGE_MIN_NOTIONAL_USDT}, thr={BALANCE_THRESHOLD_USDT}, res={BALANCE_RESERVE_USDT}, fee_buf≈{FEE_BUF_BPS/100:.2f}%) => {'OK' if ok else 'BLOCK'}"
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


# ================ NEW: اكتشاف أرصدة السبوت (Discovery) بدون تكرار للـ variants ================

def _discover_spot_positions(min_usd: float = 5.0):
    """
    ينشئ ملفات مراكز مستوردة لأي رصيد Spot موجود بدون تكرار للـ variants.
    مثال: لو يوجد DOGE سبوت، ينشئ ملفًا واحدًا لـ 'DOGE/USDT' فقط.
    كما يتخطّى الإنشاء إذا وجد أي ملف مركز موجود للـ base أو لأي variant.
    """
    try:
        seen_bases = set()
        for symbol in SYMBOLS:
            base = symbol.split("#")[0]
            if base in seen_bases:
                continue
            seen_bases.add(base)

            if load_position(base) is not None:
                continue
            has_variant_file = any(
                load_position(f"{base}#" + v) is not None for v in ("old", "srr", "brt", "vbr")
            )
            if has_variant_file:
                continue

            coin = base.split("/")[0]
            qty = float(fetch_balance(coin) or 0.0)
            if qty <= 0.0:
                continue

            px = float(fetch_price(base) or 0.0)
            if px <= 0.0:
                continue

            usd_val = qty * px
            if usd_val < float(min_usd):
                continue

            pos = {
                "symbol": base,
                "variant": "imported",
                "entry_price": px,
                "amount": qty,
                "imported": True,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "notes": "auto-imported from spot balance",
            }
            save_position(base, pos)
            _print(f"[import] created position for {base}: qty={qty}, px={px}, ~${usd_val:.2f}")
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
            silent=True,
        )
    except Exception:
        _print("🚀 تشغيل البوت")

    if MAX_OPEN_POSITIONS_OVERRIDE is not None:
        _print(f"[limits] MAX_OPEN_POSITIONS_OVERRIDE = {MAX_OPEN_POSITIONS_OVERRIDE}")

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

            # NEW/LIQ: قياس السيولة الحرة كل لفة (لأغراض الطباعة فقط)
            free_now = _usdt_free()

            # NEW/RISK: (RiskBlocker خارجي معطّل) — نعتمد ريسك الاستراتيجية فقط
            risk_blocked = _risk_is_blocked()
            if risk_blocked:
                _print("[risk] blocked — skipping new entries this cycle.")

            # NEW/LIQ: لو السياسة manage_first — قدّم الإدارة عندما لا توجد سيولة كافية للشراء
            should_manage_now = (now >= next_manage)
            if LIQUIDITY_POLICY == "manage_first":
                if (now >= next_scan) and (not _has_liquidity_for_new_trade()):
                    should_manage_now = True

            # 2) إدارة الصفقات المفتوحة — قد تُنفّذ أولًا عند نقص السيولة
            if should_manage_now:
                t_manage_start = perf_counter()
                try:
                    for symbol in SYMBOLS:
                        if _stop_flag and STOP_POLICY in ("immediate", "debounce"):
                            break
                        try:
                            closed = manage_position(symbol.split("#")[0])  # إدارة على base دائمًا
                            if closed:
                                # نص الإغلاق من الاستراتيجية إن وجد، وإلا نص افتراضي
                                text = None
                                if isinstance(closed, dict):
                                    text = closed.get("text") or closed.get("msg")
                                elif isinstance(closed, (list, tuple)) and closed:
                                    text = closed[0]
                                tg_trade_close(symbol.split('#')[0], text)
                                _print(f"[manage] {symbol.split('#')[0]} closed by TP/SL/TIME")
                        except Exception as e:
                            _print(f"[manage_position] {symbol.split('#')[0]} error: {e}")
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

            # 1) فحص إشارات الدخول — مع بوابة سيولة مخففة + (بدون RiskBlocker خارجي)
            if now >= next_scan:
                gate_ok = _balance_gate_debug()
                if (not gate_ok) or risk_blocked:
                    if not gate_ok:
                        _print(f"[scan] skipped — free USDT {free_now:.2f} < min_required")
                    if risk_blocked:
                        _print("[scan] skipped — risk blocker active.")
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

                            base = symbol.split("#")[0]

                            # حد فتح المراكز
                            if MAX_OPEN_POSITIONS_OVERRIDE is not None and open_positions_count >= MAX_OPEN_POSITIONS_OVERRIDE:
                                break

                            # تجنّب التكرار إن كان هناك مركز مفتوح للـ base
                            if load_position(base) is not None:
                                continue

                            # بوابة سيولة لكل محاولة (خفيفة)
                            if not _has_liquidity_for_new_trade():
                                break  # أوقف محاولات الدخول في هذه الجولة

                            try:
                                sig = check_signal(base)
                            except Exception as e:
                                _print(f"[check_signal] {base} error: {e}")
                                continue

                            is_buy = (sig == "buy") or (isinstance(sig, dict) and str(sig.get("decision", "")).lower() == "buy")
                            if is_buy:
                                # فتح الصفقة عبر execute_buy — الاستراتيجية تتحكم بالحجم والتدرج داخليًا
                                try:
                                    order, msg = execute_buy(base, sig)
                                    if order:
                                        # إشعار مضمون حتى لو msg فارغ
                                        if msg and isinstance(msg, str):
                                            tg_info(msg, parse_mode="HTML", silent=False)
                                        else:
                                            tg_trade_open(base, order)
                                        open_positions_count = _get_open_positions_count_safe()
                                    else:
                                        _print(f"[execute_buy] failed to open {base}: {msg}")
                                except Exception as e:
                                    _print(f"[execute_buy] {base} error: {e}")
                                    continue
                            else:
                                try:
                                    _, reasons = check_signal_debug(base)
                                    if reasons:
                                        _print(f"[debug] {base} reject reasons: {reasons[:5]}")
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

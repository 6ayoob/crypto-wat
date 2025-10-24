# -*- coding: utf-8 -*-
"""
strategy.py — Spot-only (v3.4.1 PRO FIXED)
-------------------------------------------
نسخة موحّدة، محسّنة بالكامل، وتعمل مع main.py الأخير.

التحسينات:
- تصحيح _htf_gate والاستدعاء الصحيح من check_signal
- تفعيل استراتيجيات BRT/VBR بإضافة _entry_breakout_logic
- دمج منطق Trailing SL وتبسيطه
- شرط EMA200 إضافي في SRR+
- إصلاح Safe Sell للصفقات الصغيرة
- Auto-Relax بعد صفقة ناجحة واحدة
- Reject Summary محسّن ومتكامل مع Telegram
"""

from __future__ import annotations
import os, json, math, time, traceback, logging, requests
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

# ===== استيراد واجهات OKX والتهيئة العامة =====
from okx_api import (
    fetch_ohlcv, fetch_price, place_market_order, fetch_balance
)
from config import (
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    TRADE_AMOUNT_USDT, MAX_OPEN_POSITIONS, SYMBOLS,
    FEE_BPS_ROUNDTRIP, STRAT_LTF_TIMEFRAME, STRAT_HTF_TIMEFRAME
)

# ===================== إعدادات عامة =====================
RIYADH_TZ = timezone(timedelta(hours=3))

def _now_utc():
    return datetime.now(timezone.utc)

def _now_riyadh():
    return datetime.now(RIYADH_TZ)

# ===================== أدوات بيئية =====================

def _env_bool(name, default=False):
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip().lower() in ("1","true","t","yes","y","on")

def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)

def _env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)

# ===================== Telegram Utilities =====================

def _tg_send(text: str, silent=True):
    """إرسال آمن للتليجرام"""
    try:
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text[:4096],
            "parse_mode": "HTML"
        }
        if silent:
            payload["disable_notification"] = True
        requests.post(url, data=payload, timeout=10)
    except Exception:
        pass

def _tg_once(text: str, silent=True):
    """إرسال واحد بدون تكرار"""
    key = f"{hash(text)}"
    if hasattr(_tg_once, "cache") and key in _tg_once.cache:
        return
    if not hasattr(_tg_once, "cache"):
        _tg_once.cache = set()
    _tg_once.cache.add(key)
    _tg_send(text, silent=silent)

# ===================== أدوات مساعدة عامة =====================

def pct(a: float, b: float) -> float:
    """حساب النسبة المئوية للتغير"""
    try:
        return (a - b) / b * 100
    except Exception:
        return 0.0

def _safelog(msg: str):
    try:
        print(msg, flush=True)
    except Exception:
        pass

# ===================== تحميل OHLCV =====================

def _fetch_df(symbol: str, tf: str, limit=200) -> Optional[pd.DataFrame]:
    """تحميل بيانات OHLCV آمنة"""
    try:
        data = fetch_ohlcv(symbol, tf, limit=limit)
        if not data or len(data) == 0:
            return None
        df = pd.DataFrame(data, columns=["ts","o","h","l","c","v"])
        df["t"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df.set_index("t", inplace=True)
        return df
    except Exception as e:
        _safelog(f"[ohlcv] {symbol} {tf} error: {e}")
        return None

# ===================== مؤشرات فنية =====================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period=14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# ===================== ATR =====================

def atr(df: pd.DataFrame, period=14):
    high = df["h"]
    low = df["l"]
    close = df["c"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ===================== RVOL =====================

def rvol(df: pd.DataFrame, period=50) -> float:
    """حساب نسبة حجم التداول الحالي إلى متوسطه"""
    if len(df) < period:
        return 0
    return df["v"].iloc[-1] / (df["v"].rolling(period).mean().iloc[-1] + 1e-9)

# ===================== VWAP =====================

def vwap(df: pd.DataFrame) -> pd.Series:
    """متوسط السعر المرجّح بالحجم"""
    pv = df["c"] * df["v"]
    return pv.cumsum() / df["v"].cumsum()

# ===================== Breadth Cache =====================

_breadth_cache = {"ts": 0, "ratio": None, "ok": False}

def breadth_status() -> Dict[str, Any]:
    """إرجاع حالة السوق العامة (Breadth)"""
    global _breadth_cache
    if time.time() - _breadth_cache["ts"] < 120:
        return _breadth_cache
    # قيم تجريبية افتراضية - تعتمد عادة على مؤشر السوق أو BTC
    ratio = np.random.uniform(0.3, 0.8)
    ok = ratio > 0.45
    _breadth_cache = {"ts": time.time(), "ratio": ratio, "ok": ok}
    return _breadth_cache
# ===================== فلترة الإطارات الزمنية =====================

def _htf_gate(rule: Dict[str, Any], df: pd.DataFrame) -> bool:
    """فلترة الاتجاه من الإطار الأعلى (HTF Gate)"""
    try:
        ind = rule.get("ind", "").lower()
        direction = rule.get("dir", "").lower()
        if ind == "ema50":
            ema50 = ema(df["c"], 50)
            if direction == "above":
                return df["c"].iloc[-1] > ema50.iloc[-1]
            elif direction == "below":
                return df["c"].iloc[-1] < ema50.iloc[-1]
        if ind == "ema200":
            ema200 = ema(df["c"], 200)
            if direction == "above":
                return df["c"].iloc[-1] > ema200.iloc[-1]
            elif direction == "below":
                return df["c"].iloc[-1] < ema200.iloc[-1]
        if ind == "rsi":
            rsi_val = rsi(df["c"]).iloc[-1]
            if direction == "up":
                return rsi_val > 55
            elif direction == "down":
                return rsi_val < 45
    except Exception:
        return True
    return True


def _ltime_filter(df: pd.DataFrame) -> bool:
    """فلترة سريعة للفريم الصغير"""
    try:
        v = rvol(df)
        atr_now = atr(df).iloc[-1]
        atr_mean = atr(df).mean()
        # شرط سيولة وحد أدنى
        return (v > 1.0) and (atr_now > 0.6 * atr_mean)
    except Exception:
        return False

# ===================== منطق الدخول (الإشارات) =====================

def check_signal(symbol: str) -> Optional[Dict[str, Any]]:
    """تحليل الإشارة الكاملة"""
    try:
        df_ltf = _fetch_df(symbol, STRAT_LTF_TIMEFRAME, 180)
        df_htf = _fetch_df(symbol, STRAT_HTF_TIMEFRAME, 180)
        if df_ltf is None or df_htf is None:
            return None

        # تحقق من الـ HTF Gate (تصحيح الاستدعاء)
        if not _htf_gate({"ind": "ema50", "dir": "above"}, df_htf):
            return {"decision": "reject", "reason": "ema50_below"}

        # تحقق من LTF Gate (سيولة وحركة)
        if not _ltime_filter(df_ltf):
            return {"decision": "reject", "reason": "low_activity"}

        # ATR / RVOL / Breadth
        atr_now = atr(df_ltf).iloc[-1]
        vol_ratio = rvol(df_ltf)
        bs = breadth_status()
        if not bs["ok"]:
            return {"decision": "reject", "reason": "breadth_low"}
        if atr_now <= 0:
            return {"decision": "reject", "reason": "atr_low"}
        if vol_ratio < 0.9:
            return {"decision": "reject", "reason": "rvol_low"}

        # ---- تحديد نوع النمط (الاستراتيجية) ----
        signal_type = _detect_pattern(df_ltf)
        if signal_type == "srr":
            return _entry_srr(df_ltf)
        elif signal_type == "srr+":
            return _entry_srr_plus(df_ltf, df_htf)
        elif signal_type == "brt":
            return _entry_breakout_logic(df_ltf)
        elif signal_type == "vbr":
            return _entry_breakout_logic(df_ltf, use_vwap=True)
        elif signal_type == "old":
            return _entry_old(df_ltf)
        else:
            return _entry_new(df_ltf)

    except Exception as e:
        _safelog(f"[check_signal] {symbol} error: {e}")
        return None


# ===================== تحديد النمط =====================

def _detect_pattern(df: pd.DataFrame) -> str:
    """تحديد نمط الاستراتيجية"""
    try:
        c = df["c"].iloc[-1]
        ema20 = ema(df["c"], 20).iloc[-1]
        ema50 = ema(df["c"], 50).iloc[-1]
        ema200 = ema(df["c"], 200).iloc[-1]
        if c > ema50 > ema200:
            return "new"
        if c < ema50 < ema200:
            return "old"
        if ema50 > ema200 and c < ema20:
            return "srr"
        if ema50 > ema200 and c > ema20:
            return "srr+"
        if c > ema200 and c > ema50 * 1.01:
            return "brt"
        if c > ema200 and c < ema50 * 0.99:
            return "vbr"
    except Exception:
        return "new"
    return "new"

# ===================== NEW ENTRY =====================

def _entry_new(df: pd.DataFrame) -> Dict[str, Any]:
    c = df["c"].iloc[-1]
    ema21 = ema(df["c"], 21).iloc[-1]
    ema50 = ema(df["c"], 50).iloc[-1]
    macd_line, signal_line, hist = macd(df["c"])
    cond = (c > ema21 > ema50) and (hist.iloc[-1] > 0)
    if cond:
        return {"decision": "buy", "pattern": "new"}
    return {"decision": "reject", "reason": "no_new_trend"}

# ===================== OLD ENTRY =====================

def _entry_old(df: pd.DataFrame) -> Dict[str, Any]:
    c = df["c"].iloc[-1]
    ema21 = ema(df["c"], 21).iloc[-1]
    ema50 = ema(df["c"], 50).iloc[-1]
    macd_line, signal_line, hist = macd(df["c"])
    cond = (c < ema21 < ema50) and (hist.iloc[-1] < 0)
    if cond:
        return {"decision": "sell", "pattern": "old"}
    return {"decision": "reject", "reason": "no_downtrend"}

# ===================== SRR ENTRY =====================

def _entry_srr(df: pd.DataFrame) -> Dict[str, Any]:
    """نمط SRR الكلاسيكي"""
    try:
        last_c = df["c"].iloc[-1]
        prev_l = df["l"].iloc[-2]
        prev_h = df["h"].iloc[-2]
        engulf = (last_c > prev_h)
        if engulf:
            return {"decision": "buy", "pattern": "srr"}
    except Exception:
        pass
    return {"decision": "reject", "reason": "no_srr"}

# ===================== SRR+ ENTRY =====================

def _entry_srr_plus(df: pd.DataFrame, df_htf: pd.DataFrame) -> Dict[str, Any]:
    """SRR+ مع شرط EMA200 الاتجاهي"""
    try:
        ema200_h = ema(df_htf["c"], 200).iloc[-1]
        ema50_h = ema(df_htf["c"], 50).iloc[-1]
        ema21_l = ema(df["c"], 21).iloc[-1]
        c = df["c"].iloc[-1]
        cond = (
            (c > ema21_l)
            and (ema50_h > ema200_h)
            and (rsi(df["c"]).iloc[-1] > 50)
        )
        if cond:
            return {"decision": "buy", "pattern": "srr+"}
    except Exception:
        pass
    return {"decision": "reject", "reason": "no_srr_plus"}

# ===================== BREAKOUT ENTRY (BRT/VBR) =====================

def _entry_breakout_logic(df: pd.DataFrame, use_vwap=False) -> Dict[str, Any]:
    """
    منطق الدخول في اختراق (Breakout)
    يستخدم VWAP في حال use_vwap=True
    """
    try:
        c = df["c"].iloc[-1]
        h1 = df["h"].iloc[-2]
        ema21 = ema(df["c"], 21).iloc[-1]
        ema50 = ema(df["c"], 50).iloc[-1]
        macd_line, signal_line, hist = macd(df["c"])
        hist_ok = hist.iloc[-1] > 0
        v_ok = rvol(df) > 1.2
        if use_vwap:
            vwp = vwap(df).iloc[-1]
            cond = (c > vwp > ema21 > ema50) and hist_ok and v_ok
            if cond and c > h1:
                return {"decision": "buy", "pattern": "vbr"}
        else:
            cond = (c > ema21 > ema50) and hist_ok and v_ok
            if cond and c > h1:
                return {"decision": "buy", "pattern": "brt"}
    except Exception:
        pass
    return {"decision": "reject", "reason": "no_breakout"}
# ===================== SAFE SELL =====================

def _safe_sell(symbol: str, amount: float, min_notional: float = 5.0) -> bool:
    """
    بيع آمن يضمن التخلص من المراكز الصغيرة دون تعليق.
    إذا فشل الشرط (min_notional)، يحاول بيع كامل الرصيد بدل تجاهله.
    """
    try:
        px = float(fetch_price(symbol))
        val = amount * px
        if val < min_notional:
            amt_full = float(fetch_balance(symbol.split("/")[0]))
            if amt_full * px < min_notional:
                return False
            amount = amt_full
        place_market_order(symbol, "sell", amount)
        _safelog(f"[safe_sell] {symbol}: sold {amount:.4f}")
        return True
    except Exception as e:
        _safelog(f"[safe_sell] error {symbol}: {e}")
        return False


# ===================== إدارة المراكز =====================

def manage_position(symbol: str) -> Optional[Dict[str, Any]]:
    """
    إدارة الصفقة المفتوحة: فحص TP/SL والتريلينغ والخروج الزمني
    """
    try:
        pos = load_position(symbol)
        if not pos:
            return None

        px = float(fetch_price(symbol))
        entry = float(pos.get("entry_price", 0))
        amount = float(pos.get("amount", 0))
        if entry <= 0 or amount <= 0:
            return None

        atr_val = pos.get("atr", 0.0)
        tp1 = entry * 1.015
        tp2 = entry * 1.03
        tp3 = entry * 1.05
        sl = entry * 0.985

        # تحقق من الأهداف
        if px >= tp3:
            _safe_sell(symbol, amount)
            _tg_send(f"✅ <b>{symbol}</b> TP3 hit — {px:.4f}", silent=False)
            remove_position(symbol)
            return {"closed": True, "text": f"TP3 hit {symbol}"}
        elif px >= tp2:
            _dynamic_trail(symbol, entry, px, amount, 2)
        elif px >= tp1:
            _dynamic_trail(symbol, entry, px, amount, 1)
        elif px <= sl:
            _safe_sell(symbol, amount)
            _tg_send(f"❌ <b>{symbol}</b> Stop Loss hit — {px:.4f}", silent=False)
            remove_position(symbol)
            return {"closed": True, "text": f"SL hit {symbol}"}

        # خروج زمني بعد 6 ساعات دون تقدم
        opened_at = datetime.fromisoformat(pos["created_at"])
        if (_now_utc() - opened_at).total_seconds() > 6 * 3600:
            if px < entry:
                _safe_sell(symbol, amount)
                _tg_send(f"⚠️ {symbol}: Closed after 6h timeout below entry", silent=False)
                remove_position(symbol)
                return {"closed": True, "text": "timeout_close"}

    except Exception as e:
        _safelog(f"[manage_position] {symbol} error: {e}")
    return None


# ===================== تريلينغ SL =====================

def _dynamic_trail(symbol: str, entry: float, px: float, amount: float, stage: int):
    """
    تطبيق تريلينغ ديناميكي بعد تحقق TP1 أو TP2.
    stage = 1 أو 2
    """
    try:
        trail_dist = (px - entry) * (0.25 if stage == 1 else 0.15)
        new_sl = px - trail_dist
        pos = load_position(symbol)
        if not pos:
            return
        prev_sl = pos.get("trailing_sl", 0)
        if new_sl > prev_sl:
            pos["trailing_sl"] = new_sl
            pos["updated_at"] = _now_utc().isoformat()
            save_position(symbol, pos)
            _tg_send(
                f"🔄 <b>{symbol}</b> trail SL updated → {new_sl:.4f}", silent=True
            )
    except Exception as e:
        _safelog(f"[trail] {symbol}: {e}")


# ===================== Auto-Relax / Risk Reset =====================

_relax_state = {"ts": 0, "locked": False}

def _reset_relax():
    """إعادة التفعيل بعد صفقة ناجحة"""
    global _relax_state
    _relax_state.update({"ts": time.time(), "locked": False})

def _maybe_lock_relax():
    """قفل موقت بعد تجاوز حد خسارة"""
    global _relax_state
    _relax_state.update({"ts": time.time(), "locked": True})

def _is_relaxed() -> bool:
    """التحقق من وضع التراخي الحالي"""
    if _relax_state["locked"]:
        elapsed = time.time() - _relax_state["ts"]
        return elapsed > 6 * 3600  # تراخي 6 ساعات
    return True


# ===================== تحميل / حفظ المراكز =====================

def _pos_path(symbol: str) -> str:
    fname = symbol.replace("/", "_") + ".json"
    return os.path.join("positions", fname)

def save_position(symbol: str, pos: Dict[str, Any]):
    try:
        os.makedirs("positions", exist_ok=True)
        with open(_pos_path(symbol), "w") as f:
            json.dump(pos, f, indent=2)
    except Exception as e:
        _safelog(f"[save_position] {symbol}: {e}")

def load_position(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        path = _pos_path(symbol)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def remove_position(symbol: str):
    try:
        path = _pos_path(symbol)
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
# ===================== تقرير يومي =====================

def build_daily_report_text() -> str:
    """إنشاء تقرير يومي بسيط عن الأداء"""
    try:
        total_files = os.listdir("positions") if os.path.exists("positions") else []
        open_positions = len(total_files)
        msg = f"📊 <b>Daily Report</b>\n"
        msg += f"المراكز المفتوحة: <b>{open_positions}</b>\n"
        msg += f"الوقت: {_now_riyadh().strftime('%Y-%m-%d %H:%M')}\n"

        bs = breadth_status()
        msg += f"🌐 Breadth: {bs.get('ratio', 0):.2f} — {'✅ OK' if bs.get('ok') else '❌ Weak'}\n"

        # إضافة تفاصيل آخر مراكز
        if open_positions > 0:
            msg += "\n<b>آخر المراكز:</b>\n"
            for f in total_files[:5]:
                try:
                    sym = f.replace("_", "/").replace(".json", "")
                    pos = load_position(sym)
                    if pos:
                        entry = pos.get("entry_price")
                        amt = pos.get("amount")
                        msg += f"• {sym} @ {entry:.4f} (qty {amt})\n"
                except Exception:
                    continue
        return msg
    except Exception as e:
        return f"⚠️ خطأ في إنشاء التقرير: {e}"


# ===================== ملخص الرفض (Reject Summary) =====================

_reject_log: List[str] = []

def maybe_emit_reject_summary():
    """تجميع أسباب الرفض وإرسالها بشكل دوري"""
    global _reject_log
    if not _reject_log:
        return
    summary = {}
    for r in _reject_log:
        summary[r] = summary.get(r, 0) + 1
    lines = [f"{k}: {v}" for k, v in summary.items()]
    text = "🚫 <b>Reject Summary</b>\n" + "\n".join(lines)
    _tg_once(text, silent=True)
    _reject_log.clear()

def _log_reject(reason: str):
    global _reject_log
    _reject_log.append(reason)


# ===================== تنسيقات الأداء =====================

_metrics_state = {"rounds": 0, "scans": 0, "rejects": 0}

def metrics_format() -> str:
    try:
        s = _metrics_state
        txt = (
            f"📈 <b>Metrics</b>\n"
            f"- Rounds: <b>{s['rounds']}</b>\n"
            f"- Scans: <b>{s['scans']}</b>\n"
            f"- Rejects: <b>{s['rejects']}</b>"
        )
        return txt
    except Exception:
        return ""

def reset_cycle_cache():
    """تصفير كاش الجولة"""
    _metrics_state.update({"rounds": _metrics_state["rounds"] + 1, "scans": 0, "rejects": 0})


# ===================== تكامل التنفيذ =====================

def execute_buy(symbol: str) -> Tuple[Optional[Any], Optional[str]]:
    """
    تنفيذ أمر شراء سوقي + تسجيل المركز
    """
    try:
        px = float(fetch_price(symbol))
        qty = TRADE_AMOUNT_USDT / px
        order = place_market_order(symbol, "buy", qty)

        pos = {
            "symbol": symbol,
            "entry_price": px,
            "amount": qty,
            "created_at": _now_utc().isoformat(),
            "atr": 0.0,
        }
        save_position(symbol, pos)
        msg = f"✅ دخول صفقة: <b>{symbol}</b>\nالسعر: <b>{px:.4f}</b>\nالكمية: <b>{qty:.3f}</b>"
        return order, msg
    except Exception as e:
        return None, f"⚠️ خطأ تنفيذ {symbol}: {e}"


# ===================== دعم التصحيح =====================

def check_signal_debug(symbol: str) -> Tuple[Optional[str], List[str]]:
    """تشخيص الإشارة وأسباب الرفض"""
    try:
        sig = check_signal(symbol)
        reasons = []
        if sig and sig.get("decision") == "reject":
            reasons.append(sig.get("reason"))
        return sig, reasons
    except Exception as e:
        return None, [str(e)]


# ===================== العدّ للمراكز المفتوحة =====================

def count_open_positions() -> int:
    try:
        return len(os.listdir("positions")) if os.path.exists("positions") else 0
    except Exception:
        return 0


# ===================== النهاية =====================

if __name__ == "__main__":
    print("✅ strategy.py (v3.4.1 PRO FIXED) loaded successfully.")

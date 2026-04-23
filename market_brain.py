# -*- coding: utf-8 -*-
# market_brain.py - العقل المدبر للبوت (v1.0)
#
# يعمل كل 30 دقيقة ويحلل:
# 1. حالة السوق (Trending/Ranging/Volatile/Crash/Recovery)
# 2. أداء الأنماط (أي pattern يشتغل هذا الأسبوع)
# 3. أداء البوت (win rate، drawdown، consecutive losses)
# 4. يكتب توجيهات مباشرة في brain_state.json
# 5. strategy.py يقرأ هذه التوجيهات في كل جولة

from __future__ import annotations

import os, json, math, time, logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple

# ===== Logger =====
logger = logging.getLogger("market_brain")
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

# ===== Constants =====
RIYADH_TZ        = timezone(timedelta(hours=3))
BRAIN_STATE_FILE = "brain_state.json"
BRAIN_LOG_FILE   = "brain_log.json"

# ===== ENV helpers =====
def _ef(name, default):
    try:    return float(os.getenv(name, str(default)))
    except: return float(default)

def _ei(name, default):
    try:    return int(float(os.getenv(name, str(default))))
    except: return int(default)

def _eb(name, default=False):
    v = os.getenv(name)
    if v is None: return default
    return str(v).strip().lower() in ("1","true","yes","on","t","y")

# ===== Tunables =====
BRAIN_ENABLED            = _eb("BRAIN_ENABLED", True)
BRAIN_INTERVAL_MIN       = _ef("BRAIN_INTERVAL_MIN", 30)
BRAIN_LOOKBACK_TRADES    = _ei("BRAIN_LOOKBACK_TRADES", 20)
BRAIN_MIN_TRADES_ANALYZE = _ei("BRAIN_MIN_TRADES_ANALYZE", 3)

# حالة السوق — العتبات
CRASH_RSI_THRESHOLD      = _ef("BRAIN_CRASH_RSI", 30.0)
VOLATILE_ATR_MULT        = _ef("BRAIN_VOLATILE_ATR_MULT", 2.0)
TRENDING_ADX_MIN         = _ef("BRAIN_TRENDING_ADX_MIN", 25.0)
RANGING_ATR_MAX          = _ef("BRAIN_RANGING_ATR_MAX", 0.8)

# تعديلات الأداء
WINRATE_HIGH             = _ef("BRAIN_WINRATE_HIGH", 0.60)
WINRATE_LOW              = _ef("BRAIN_WINRATE_LOW",  0.35)
PATTERN_CB_LOSSES        = _ei("BRAIN_PATTERN_CB_LOSSES", 3)

# حدود التعديل التلقائي
SCORE_MIN_FLOOR          = _ei("BRAIN_SCORE_MIN_FLOOR", 40)
SCORE_MAX_CEILING        = _ei("BRAIN_SCORE_MAX_CEILING", 70)
SIZE_MIN_MULT            = _ef("BRAIN_SIZE_MIN_MULT", 0.40)
SIZE_MAX_MULT            = _ef("BRAIN_SIZE_MAX_MULT", 1.50)

def _now(): return datetime.now(RIYADH_TZ)
def _now_iso(): return _now().isoformat(timespec="seconds")

def _read_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except: pass
    return default

def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ══════════════════════════════════════════════════════
#  القسم ١: قراءة البيانات
# ══════════════════════════════════════════════════════

def _load_closed_trades() -> List[Dict]:
    """تحميل آخر N صفقة مغلقة"""
    trades = _read_json("closed_positions.json", [])
    return trades[-BRAIN_LOOKBACK_TRADES:] if trades else []


def _load_ohlcv_local(symbol: str, tf: str, bars: int) -> Optional[List]:
    """
    يحاول جلب OHLCV من okx_api إذا كان متاحاً،
    وإلا يرجع None ليستخدم البيانات الجاهزة.
    """
    try:
        from okx_api import fetch_ohlcv
        data = fetch_ohlcv(symbol, tf, bars)
        return data if data and len(data) >= bars // 2 else None
    except:
        return None


# ══════════════════════════════════════════════════════
#  القسم ٢: تحليل السوق
# ══════════════════════════════════════════════════════

def _calc_atr(highs, lows, closes, period=14) -> float:
    """حساب ATR بسيط"""
    try:
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)
        if not trs: return 0.0
        # EMA بسيط
        atr = sum(trs[:period]) / period
        for tr in trs[period:]:
            atr = atr * (1 - 1/period) + tr * (1/period)
        return float(atr)
    except:
        return 0.0


def _calc_rsi(closes, period=14) -> float:
    """حساب RSI"""
    try:
        if len(closes) < period + 2: return 50.0
        gains, losses = [], []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i-1]
            gains.append(max(diff, 0))
            losses.append(max(-diff, 0))
        ag = sum(gains[-period:]) / period
        al = sum(losses[-period:]) / period
        if al == 0: return 100.0
        rs = ag / al
        return float(100 - 100 / (1 + rs))
    except:
        return 50.0


def _calc_adx(highs, lows, closes, period=14) -> float:
    """حساب ADX تقريبي لقياس قوة الترند"""
    try:
        if len(closes) < period * 2: return 20.0
        plus_dm, minus_dm, tr_list = [], [], []
        for i in range(1, len(closes)):
            h_diff = highs[i] - highs[i-1]
            l_diff = lows[i-1] - lows[i]
            plus_dm.append(h_diff if h_diff > l_diff and h_diff > 0 else 0)
            minus_dm.append(l_diff if l_diff > h_diff and l_diff > 0 else 0)
            tr_list.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            ))

        def smooth(arr, p):
            s = sum(arr[:p])
            result = [s]
            for x in arr[p:]:
                s = s - s/p + x
                result.append(s)
            return result

        atr_s    = smooth(tr_list, period)
        plus_s   = smooth(plus_dm, period)
        minus_s  = smooth(minus_dm, period)
        dx_list  = []
        for i in range(len(atr_s)):
            if atr_s[i] == 0: continue
            pdi = 100 * plus_s[i] / atr_s[i]
            mdi = 100 * minus_s[i] / atr_s[i]
            dx  = 100 * abs(pdi - mdi) / (pdi + mdi + 1e-9)
            dx_list.append(dx)

        if not dx_list: return 20.0
        return float(sum(dx_list[-period:]) / min(period, len(dx_list)))
    except:
        return 20.0


def _calc_volatility_ratio(highs, lows, closes, period=14) -> float:
    """
    نسبة التذبذب: ATR الحالي ÷ متوسط ATR التاريخي
    > 2.0 = متذبذب جداً
    < 0.7 = هادئ جداً
    """
    try:
        if len(closes) < period * 3: return 1.0
        current_atr = _calc_atr(highs[-period*2:], lows[-period*2:], closes[-period*2:], period)
        hist_atr    = _calc_atr(highs[:-period], lows[:-period], closes[:-period], period)
        if hist_atr <= 0: return 1.0
        return float(current_atr / hist_atr)
    except:
        return 1.0


def analyze_market_regime(btc_data: Optional[List]) -> Dict[str, Any]:
    """
    يحلل بيانات BTC ويحدد حالة السوق الحالية.

    يرجع:
    {
        "regime": "TRENDING_UP" | "TRENDING_DOWN" | "RANGING" | "VOLATILE" | "CRASH" | "RECOVERY",
        "strength": 0.0-1.0,
        "btc_rsi": float,
        "btc_adx": float,
        "volatility_ratio": float,
        "btc_trend": "up" | "down" | "neutral",
        "description": str
    }
    """
    default = {
        "regime": "RANGING",
        "strength": 0.5,
        "btc_rsi": 50.0,
        "btc_adx": 20.0,
        "volatility_ratio": 1.0,
        "btc_trend": "neutral",
        "description": "بيانات غير كافية — وضع محايد"
    }

    if not btc_data or len(btc_data) < 50:
        return default

    try:
        closes = [float(c[4]) for c in btc_data]
        highs  = [float(c[2]) for c in btc_data]
        lows   = [float(c[3]) for c in btc_data]

        rsi      = _calc_rsi(closes)
        adx      = _calc_adx(highs, lows, closes)
        vol_ratio= _calc_volatility_ratio(highs, lows, closes)

        # EMA50 بسيط
        ema50 = sum(closes[-50:]) / 50
        ema20 = sum(closes[-20:]) / 20
        price_now = closes[-1]

        # اتجاه BTC
        if price_now > ema50 and ema20 > ema50:
            btc_trend = "up"
        elif price_now < ema50 and ema20 < ema50:
            btc_trend = "down"
        else:
            btc_trend = "neutral"

        # تحديد الحالة
        if rsi < 20:
            regime = "CRASH"
            strength = 0.95
            desc = f"انهيار حاد — RSI={rsi:.0f} — لا دخول"

        elif rsi < CRASH_RSI_THRESHOLD and btc_trend == "down":
            regime = "CRASH"
            strength = 0.8
            desc = f"سوق هابط قوي — RSI={rsi:.0f} — تقليل مخاطرة"

        elif rsi > 70 and vol_ratio > VOLATILE_ATR_MULT:
            regime = "VOLATILE"
            strength = 0.7
            desc = f"تذبذب مرتفع — RSI={rsi:.0f} Volatility={vol_ratio:.1f}x"

        elif vol_ratio > VOLATILE_ATR_MULT * 1.3:
            regime = "VOLATILE"
            strength = 0.6
            desc = f"تذبذب شديد — {vol_ratio:.1f}x المعدل الطبيعي"

        elif rsi < 40 and btc_trend == "down":
            regime = "TRENDING_DOWN"
            strength = 0.65
            desc = f"ترند هبوطي — RSI={rsi:.0f} — صفقات أقل"

        elif adx > TRENDING_ADX_MIN and btc_trend == "up" and rsi > 50:
            regime = "TRENDING_UP"
            strength = min(adx / 50, 1.0)
            desc = f"ترند صاعد قوي — ADX={adx:.0f} RSI={rsi:.0f} ✅"

        elif adx > TRENDING_ADX_MIN and btc_trend == "down":
            regime = "TRENDING_DOWN"
            strength = min(adx / 50, 1.0)
            desc = f"ترند هبوطي — ADX={adx:.0f}"

        elif 30 < rsi < 70 and btc_trend == "up" and rsi > 40:
            regime = "RECOVERY"
            strength = 0.5
            desc = f"انتعاش — RSI={rsi:.0f} BTC فوق EMA50"

        else:
            regime = "RANGING"
            strength = 0.4
            desc = f"سوق جانبي — ADX={adx:.0f} RSI={rsi:.0f}"

        return {
            "regime": regime,
            "strength": round(strength, 3),
            "btc_rsi": round(rsi, 1),
            "btc_adx": round(adx, 1),
            "volatility_ratio": round(vol_ratio, 2),
            "btc_trend": btc_trend,
            "btc_price": round(price_now, 2),
            "btc_ema50": round(ema50, 2),
            "description": desc
        }

    except Exception as e:
        logger.warning(f"[brain] analyze_market_regime error: {e}")
        return default


# ══════════════════════════════════════════════════════
#  القسم ٣: تحليل الأداء
# ══════════════════════════════════════════════════════

def analyze_performance(trades: List[Dict]) -> Dict[str, Any]:
    """
    يحلل آخر N صفقة ويرجع:
    - win_rate عام
    - أداء كل pattern
    - أداء كل mode (breakout/pullback/early_scout)
    - drawdown
    - توصيات: أي patterns تُوقَف، أي modes تُفضَّل
    """
    if len(trades) < BRAIN_MIN_TRADES_ANALYZE:
        return {
            "win_rate": None,
            "total_pnl": 0.0,
            "pattern_stats": {},
            "mode_stats": {},
            "blocked_patterns": [],
            "preferred_modes": [],
            "drawdown_pct": 0.0,
            "note": "صفقات غير كافية للتحليل"
        }

    wins  = [t for t in trades if float(t.get("profit", 0)) > 0]
    total_pnl = sum(float(t.get("profit", 0)) for t in trades)
    win_rate  = len(wins) / len(trades)

    # تحليل الأنماط
    pattern_stats: Dict[str, Dict] = {}
    for t in trades:
        pt  = str(t.get("pattern", "Generic"))
        pnl = float(t.get("profit", 0))
        if pt not in pattern_stats:
            pattern_stats[pt] = {"wins": 0, "losses": 0, "total_pnl": 0.0}
        pattern_stats[pt]["total_pnl"] += pnl
        if pnl > 0: pattern_stats[pt]["wins"] += 1
        else:       pattern_stats[pt]["losses"] += 1

    # أنماط تُوقَف (3+ خسائر متتالية أو win_rate < 25%)
    blocked_patterns = []
    for pt, stats in pattern_stats.items():
        total_pt = stats["wins"] + stats["losses"]
        if total_pt < 2: continue
        pt_wr = stats["wins"] / total_pt
        if stats["losses"] >= PATTERN_CB_LOSSES and pt_wr < 0.30:
            blocked_patterns.append(pt)
            logger.info(f"[brain] 🚫 نمط محظور: {pt} | WR={pt_wr:.0%} | PnL={stats['total_pnl']:.2f}$")

    # تحليل الـ Mode
    mode_stats: Dict[str, Dict] = {}
    for t in trades:
        mode = str(t.get("entry_reason", "") or "unknown")
        # استخرج mode من entry_reason
        if "breakout" in mode.lower():   m = "breakout"
        elif "pullback" in mode.lower():  m = "pullback"
        elif "early_scout" in mode.lower(): m = "early_scout"
        else: m = "other"

        pnl = float(t.get("profit", 0))
        if m not in mode_stats:
            mode_stats[m] = {"wins": 0, "losses": 0, "total_pnl": 0.0}
        mode_stats[m]["total_pnl"] += pnl
        if pnl > 0: mode_stats[m]["wins"] += 1
        else:       mode_stats[m]["losses"] += 1

    # أفضل modes
    preferred_modes = []
    for m, stats in mode_stats.items():
        total_m = stats["wins"] + stats["losses"]
        if total_m < 2: continue
        m_wr = stats["wins"] / total_m
        if m_wr >= 0.50 and stats["total_pnl"] > 0:
            preferred_modes.append(m)

    # Drawdown
    equity_curve = []
    running = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        running += float(t.get("profit", 0))
        equity_curve.append(running)
        if running > peak: peak = running
        dd = (peak - running) / max(abs(peak), 1)
        if dd > max_dd: max_dd = dd

    return {
        "win_rate":        round(win_rate, 3),
        "total_pnl":       round(total_pnl, 2),
        "trades_analyzed": len(trades),
        "pattern_stats":   pattern_stats,
        "mode_stats":      mode_stats,
        "blocked_patterns": blocked_patterns,
        "preferred_modes": preferred_modes,
        "drawdown_pct":    round(max_dd, 3),
    }


# ══════════════════════════════════════════════════════
#  القسم ٤: توليد التوجيهات
# ══════════════════════════════════════════════════════

def generate_directives(
    regime_info: Dict[str, Any],
    perf_info:   Dict[str, Any],
    current_state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    يجمع تحليل السوق + الأداء ويولّد توجيهات للـ strategy.

    التوجيهات:
    - score_threshold_override: رفع/خفض عتبة الدخول
    - size_multiplier: تضخيم/تصغير الحجم
    - blocked_patterns: أنماط محظورة
    - preferred_modes: modes مفضّلة
    - max_open_positions_override: تعديل الحد الأقصى للمراكز
    - entry_allowed: هل الدخول مسموح أصلاً
    - notes: شرح سبب كل قرار
    """
    regime   = regime_info.get("regime", "RANGING")
    strength = regime_info.get("strength", 0.5)
    win_rate = perf_info.get("win_rate")
    drawdown = perf_info.get("drawdown_pct", 0.0)

    # القيم الافتراضية (محايدة)
    score_thr  = int(os.getenv("SCORE_THRESHOLD", "50"))
    size_mult  = 1.0
    max_pos    = int(os.getenv("MAX_OPEN_POSITIONS", "5"))
    entry_ok   = True
    notes      = []

    # ─── تعديلات بناءً على حالة السوق ───

    if regime == "CRASH":
        entry_ok  = False
        size_mult = 0.0
        notes.append(f"🚫 إيقاف تام — السوق في انهيار ({regime_info.get('description','')})")

    elif regime == "TRENDING_UP":
        score_thr = max(SCORE_MIN_FLOOR, score_thr - 5)
        size_mult = 1.0 + (strength * 0.3)  # حتى +30% في الترند القوي
        max_pos   = min(max_pos + 1, 8)
        notes.append(f"✅ ترند صاعد قوي (ADX={regime_info.get('btc_adx',0):.0f}) — رفع الحجم {size_mult:.1f}x")

    elif regime == "TRENDING_DOWN":
        score_thr = min(SCORE_MAX_CEILING, score_thr + 8)
        size_mult = 0.60
        max_pos   = max(max_pos - 2, 2)
        notes.append(f"⚠️ ترند هبوطي — تقليل الحجم 40% ورفع عتبة الدخول")

    elif regime == "VOLATILE":
        score_thr = min(SCORE_MAX_CEILING, score_thr + 5)
        size_mult = 0.55
        max_pos   = max(max_pos - 1, 3)
        notes.append(f"⚡ تذبذب عالٍ ({regime_info.get('volatility_ratio',1):.1f}x) — تقليل الحجم والمراكز")

    elif regime == "RECOVERY":
        score_thr = score_thr  # محايد
        size_mult = 0.85
        notes.append(f"📈 انتعاش — حذر مع إمكانية الدخول")

    else:  # RANGING
        score_thr = min(SCORE_MAX_CEILING, score_thr + 3)
        size_mult = 0.80
        notes.append(f"↔️ سوق جانبي — تقليل الحجم وتفضيل Pullback")

    # ─── تعديلات بناءً على الأداء ───

    if win_rate is not None:
        if win_rate >= WINRATE_HIGH:
            size_mult = min(size_mult * 1.15, SIZE_MAX_MULT)
            notes.append(f"🏆 Win Rate مرتفع ({win_rate:.0%}) — رفع الحجم")

        elif win_rate <= WINRATE_LOW:
            size_mult = max(size_mult * 0.75, SIZE_MIN_MULT)
            score_thr = min(SCORE_MAX_CEILING, score_thr + 5)
            notes.append(f"📉 Win Rate منخفض ({win_rate:.0%}) — تقليل الحجم ورفع العتبة")

    if drawdown >= 0.15:
        size_mult = max(size_mult * 0.70, SIZE_MIN_MULT)
        entry_ok  = entry_ok and (drawdown < 0.30)
        notes.append(f"🛡️ Drawdown {drawdown:.0%} — تفعيل الحماية")

    # ─── تطبيق الحدود ───
    score_thr = max(SCORE_MIN_FLOOR, min(SCORE_MAX_CEILING, int(score_thr)))
    size_mult = max(SIZE_MIN_MULT, min(SIZE_MAX_MULT, round(size_mult, 2)))

    # ─── الأنماط والـ Modes ───
    blocked_patterns = perf_info.get("blocked_patterns", [])
    preferred_modes  = perf_info.get("preferred_modes", [])

    # في السوق الجانبي → تفضيل pullback
    if regime == "RANGING" and "pullback" not in preferred_modes:
        preferred_modes = ["pullback"] + preferred_modes

    # في الترند → تفضيل breakout
    if regime == "TRENDING_UP" and "breakout" not in preferred_modes:
        preferred_modes = ["breakout"] + preferred_modes

    return {
        "entry_allowed":               bool(entry_ok),
        "score_threshold_override":    int(score_thr),
        "size_multiplier":             float(size_mult),
        "max_open_positions_override": int(max_pos),
        "blocked_patterns":            blocked_patterns,
        "preferred_modes":             preferred_modes[:3],
        "notes":                       notes,
        "regime":                      regime,
        "regime_strength":             strength,
        "win_rate":                    win_rate,
        "drawdown_pct":                drawdown,
    }


# ══════════════════════════════════════════════════════
#  القسم ٥: الحفظ والقراءة
# ══════════════════════════════════════════════════════

def save_brain_state(state: Dict[str, Any]):
    """حفظ حالة العقل في brain_state.json"""
    state["updated_at"] = _now_iso()
    _write_json(BRAIN_STATE_FILE, state)
    logger.info(f"[brain] 💾 تم حفظ الحالة: {state.get('regime','?')} | size×{state.get('size_multiplier',1):.2f} | score≥{state.get('score_threshold_override','?')}")

    # سجل تاريخي
    log = _read_json(BRAIN_LOG_FILE, [])
    log.append({k: v for k, v in state.items() if k != "pattern_stats"})
    _write_json(BRAIN_LOG_FILE, log[-200:])


def load_brain_state() -> Dict[str, Any]:
    """تحميل آخر حالة محفوظة"""
    return _read_json(BRAIN_STATE_FILE, _default_state())


def _default_state() -> Dict[str, Any]:
    """الحالة الافتراضية عند بدء التشغيل"""
    return {
        "entry_allowed":               True,
        "score_threshold_override":    50,
        "size_multiplier":             1.0,
        "max_open_positions_override": 5,
        "blocked_patterns":            [],
        "preferred_modes":             [],
        "regime":                      "UNKNOWN",
        "regime_strength":             0.5,
        "win_rate":                    None,
        "drawdown_pct":                0.0,
        "notes":                       ["لم يتم تشغيل العقل بعد"],
        "updated_at":                  None,
        "btc_rsi":                     50.0,
        "btc_adx":                     20.0,
        "btc_trend":                   "neutral",
        "volatility_ratio":            1.0,
    }


def get_brain_directives() -> Dict[str, Any]:
    """
    الدالة الرئيسية التي تستدعيها strategy.py.
    تُرجع التوجيهات الحالية من brain_state.json.
    إذا لم تكن الحالة محدّثة (> ساعتين) → تُرجع الافتراضية.
    """
    if not BRAIN_ENABLED:
        return _default_state()

    state = load_brain_state()

    # تحقق من الصلاحية
    updated_at = state.get("updated_at")
    if updated_at:
        try:
            dt = datetime.fromisoformat(updated_at)
            age_hours = (datetime.now(dt.tzinfo) - dt).total_seconds() / 3600
            if age_hours > 2.0:
                logger.warning(f"[brain] ⚠️ الحالة قديمة ({age_hours:.1f} ساعة) — استخدام الافتراضي")
                return _default_state()
        except: pass

    return state


# ══════════════════════════════════════════════════════
#  القسم ٦: الحلقة الرئيسية
# ══════════════════════════════════════════════════════

def run_brain_cycle():
    """
    دورة تحليل واحدة.
    تُستدعى كل 30 دقيقة من brain_scheduler.py.
    """
    logger.info("[brain] 🧠 بدء دورة التحليل...")
    start = time.time()

    # ── ١. جلب بيانات BTC ──
    btc_data = _load_ohlcv_local("BTC/USDT", "1h", 100)

    # ── ٢. تحليل السوق ──
    regime_info = analyze_market_regime(btc_data)
    logger.info(f"[brain] 📊 {regime_info['regime']} | {regime_info['description']}")

    # ── ٣. تحليل الأداء ──
    trades    = _load_closed_trades()
    perf_info = analyze_performance(trades)
    if perf_info.get("win_rate") is not None:
        logger.info(
            f"[brain] 📈 Win Rate={perf_info['win_rate']:.0%} | "
            f"PnL={perf_info['total_pnl']:.2f}$ | "
            f"DD={perf_info['drawdown_pct']:.0%}"
        )

    # ── ٤. توليد التوجيهات ──
    current = load_brain_state()
    directives = generate_directives(regime_info, perf_info, current)

    # ── ٥. دمج كل شيء وحفظ ──
    full_state = {
        **directives,
        "btc_rsi":          regime_info.get("btc_rsi", 50.0),
        "btc_adx":          regime_info.get("btc_adx", 20.0),
        "btc_trend":        regime_info.get("btc_trend", "neutral"),
        "btc_price":        regime_info.get("btc_price", 0.0),
        "volatility_ratio": regime_info.get("volatility_ratio", 1.0),
        "pattern_stats":    perf_info.get("pattern_stats", {}),
        "mode_stats":       perf_info.get("mode_stats", {}),
    }
    save_brain_state(full_state)

    elapsed = time.time() - start
    logger.info(f"[brain] ✅ دورة انتهت في {elapsed:.1f}s")
    logger.info(f"[brain] 📋 التوجيهات: entry={directives['entry_allowed']} | "
                f"size×{directives['size_multiplier']} | "
                f"score≥{directives['score_threshold_override']} | "
                f"blocked={directives['blocked_patterns']}")

    return full_state


# ══════════════════════════════════════════════════════
#  القسم ٧: التشغيل المستقل (للاختبار)
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    print("🧠 Market Brain v1.0 — تشغيل مباشر")
    print("=" * 50)

    if "--once" in sys.argv:
        # تشغيل دورة واحدة
        state = run_brain_cycle()
        print("\n📋 التوجيهات الحالية:")
        print(json.dumps({
            k: v for k, v in state.items()
            if k not in ("pattern_stats", "mode_stats")
        }, ensure_ascii=False, indent=2))

    elif "--status" in sys.argv:
        # عرض الحالة الحالية
        state = load_brain_state()
        print(json.dumps({
            k: v for k, v in state.items()
            if k not in ("pattern_stats", "mode_stats")
        }, ensure_ascii=False, indent=2))

    else:
        print("الاستخدام:")
        print("  python market_brain.py --once    # تشغيل دورة واحدة")
        print("  python market_brain.py --status  # عرض الحالة الحالية")

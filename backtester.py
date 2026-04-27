# -*- coding: utf-8 -*-
# backtester.py - اختبار الاستراتيجية على البيانات التاريخية
#
# يشغّل نفس منطق strategy.py على بيانات حقيقية من الماضي
# ويحسب: Win Rate, RR Ratio, Drawdown, أفضل الرموز والأنماط

from __future__ import annotations

import os, json, math, logging, time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

logger = logging.getLogger("backtester")

RIYADH_TZ = timezone(timedelta(hours=3))

# ===== إعدادات =====
BT_INITIAL_CAPITAL = float(os.getenv("BT_INITIAL_CAPITAL", "100.0"))
BT_RISK_PCT        = float(os.getenv("BT_RISK_PCT",        "5.0"))
BT_FEE_PCT         = float(os.getenv("BT_FEE_PCT",         "0.08"))
BT_SLIPPAGE_PCT    = float(os.getenv("BT_SLIPPAGE_PCT",    "0.05"))
BT_MIN_BARS_HOLD   = int(os.getenv("BT_MIN_BARS_HOLD",     "2"))
BT_MAX_BARS_HOLD   = int(os.getenv("BT_MAX_BARS_HOLD",     "20"))
BT_SL_ATR_MULT     = float(os.getenv("BT_SL_ATR_MULT",     "1.0"))
BT_TP1_ATR_MULT    = float(os.getenv("BT_TP1_ATR_MULT",    "1.5"))
BT_TP2_ATR_MULT    = float(os.getenv("BT_TP2_ATR_MULT",    "2.5"))
BT_SCORE_MIN       = int(os.getenv("BT_SCORE_MIN",         "65"))  # رُفع من 45 بناءً على Backtesting
BT_MAX_CONCURRENT  = int(os.getenv("BT_MAX_CONCURRENT",    "5"))
RESULTS_FILE       = os.getenv("BT_RESULTS_FILE",
                    "/opt/render/project/data/backtest_results.json")

def _df(data: List) -> pd.DataFrame:
    df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
    try:
        if len(df) and df["timestamp"].iloc[-1] < 10**12:
            df["timestamp"] = df["timestamp"] * 1000
    except: pass
    for c in ("open","high","low","close","volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()

def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """إضافة المؤشرات الأساسية"""
    # EMAs
    for n in [9, 21, 50, 100, 200]:
        df[f"ema{n}"] = df["close"].ewm(span=n, adjust=False).mean()

    # RSI
    d    = df["close"].diff()
    gain = d.where(d > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-d.where(d < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))

    # ATR
    c     = df["close"].shift(1)
    tr    = pd.concat([(df["high"]-df["low"]).abs(),
                       (df["high"]-c).abs(),
                       (df["low"]-c).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()

    # RVOL
    vol_ma      = df["volume"].rolling(24, min_periods=6).mean()
    df["rvol"]  = df["volume"] / vol_ma.replace(0, 1e-9)

    # VWAP يومي
    df["vwap"] = (df["close"] * df["volume"]).rolling(24).sum() / \
                  df["volume"].rolling(24).sum()

    # NR (Narrow Range)
    rng    = df["high"] - df["low"]
    rng_ma = rng.rolling(10, min_periods=3).mean()
    df["is_nr"] = rng < (0.75 * rng_ma)

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_hist"] = (ema12 - ema26) - \
                      (ema12 - ema26).ewm(span=9, adjust=False).mean()

    return df

def _score(df: pd.DataFrame, i: int) -> Tuple[int, str]:
    """
    حساب السكور لشمعة معينة —
    نفس منطق _opportunity_score في strategy.py
    """
    score = 0
    reasons = []
    try:
        row  = df.iloc[i]
        prev = df.iloc[i-1]
        c    = float(row["close"])
        o    = float(row["open"])

        # هيكل EMA
        if c > row["ema50"] and c > row["ema100"]:
            score += 12; reasons.append("AboveEMA50+100")
        elif c > row["ema50"]:
            score += 6;  reasons.append("AboveEMA50")
        if c > row["ema200"]:
            score += 8;  reasons.append("AboveEMA200")

        # RVOL
        rv = float(row["rvol"])
        if rv >= 2.0:   score += 25; reasons.append("RVOL≥2.0")
        elif rv >= 1.5: score += 15; reasons.append("RVOL≥1.5")
        elif rv >= 1.2: score += 8;  reasons.append("RVOL≥1.2")

        # NR Breakout
        nr_recent = bool(df["is_nr"].iloc[i-2:i].all())
        hi_range  = float(df["high"].iloc[i-12:i-1].max())
        if nr_recent and c > hi_range:
            score += 25; reasons.append("NR_Breakout")
        # Bullish Engulf — فلتر أقوى: يجب أن يكون الحجم مرتفعاً
        elif (c > o and prev["close"] < prev["open"] and
              c >= prev["open"] and o <= prev["close"]):
            # تأكيد بالحجم: حجم الشمعة الحالية > المتوسط
            vol_ma_check = float(df["volume"].iloc[i-10:i].mean())
            if float(row["volume"]) > vol_ma_check * 1.2:
                score += 25; reasons.append("BullishEngulf")
            else:
                score += 10; reasons.append("BullishEngulf_weak")

        # جسم الشمعة
        rng = max(float(row["high"]) - float(row["low"]), 1e-9)
        if abs(c - o) / rng >= 0.65 and c > o:
            score += 8; reasons.append("StrongBody")

        # RSI صحي
        rsi = float(row["rsi"])
        if 50 <= rsi <= 68:   score += 10; reasons.append(f"RSI_OK({rsi:.0f})")
        elif 68 < rsi <= 75:  score += 4

        # قرب VWAP
        vwap = float(row["vwap"]) if not math.isnan(float(row["vwap"])) else c
        atr  = float(row["atr"])  if not math.isnan(float(row["atr"]))  else c * 0.002
        if atr > 0 and abs(c - vwap) <= 0.3 * atr:
            score += 10; reasons.append("NearVWAP")

    except Exception as e:
        pass

    pattern = "NR_Breakout" if "NR_Breakout" in reasons else \
              "BullishEngulf" if "BullishEngulf" in reasons else "Generic"

    return max(0, score), pattern

def _check_entry(df: pd.DataFrame, i: int) -> Optional[Dict]:
    """
    يتحقق من شروط الدخول عند شمعة i
    يرجع dict أو None
    """
    if i < 50: return None
    try:
        row  = df.iloc[i]
        c    = float(row["close"])
        o    = float(row["open"])
        atr  = float(row["atr"])
        ema100 = float(row["ema100"])
        ema50  = float(row["ema50"])
        rsi    = float(row["rsi"])

        if atr <= 0 or c <= 0: return None

        # شروط أساسية
        if c < ema100: return None          # تحت EMA100
        if rsi < 28:   return None          # oversold
        if rsi > 78:   return None          # exhausted

        # حساب السكور
        score, pattern = _score(df, i)
        if score < BT_SCORE_MIN: return None

        # [FIX] رفض Generic — يجب أن يكون النمط واضحاً
        if pattern == "Generic":
            return None

        # فلتر RVOL — رُفع من 0.85 إلى 1.1 لجودة أعلى
        rvol = float(row["rvol"])
        if rvol < 1.1: return None

        # تحديد نوع الدخول
        hi_range = float(df["high"].iloc[i-10:i-1].max())
        mode = "breakout" if c > hi_range else "pullback"

        # SL و TP
        sl  = c - BT_SL_ATR_MULT  * atr
        tp1 = c + BT_TP1_ATR_MULT * atr
        tp2 = c + BT_TP2_ATR_MULT * atr

        if sl >= c: return None

        return {
            "entry_price": c,
            "sl":          sl,
            "tp1":         tp1,
            "tp2":         tp2,
            "score":       score,
            "pattern":     pattern,
            "mode":        mode,
            "atr":         atr,
            "rvol":        rvol,
            "rsi":         rsi,
        }
    except:
        return None

def backtest_symbol(symbol: str, data: List,
                    capital: float = BT_INITIAL_CAPITAL) -> Dict:
    """
    يختبر رمزاً واحداً ويرجع إحصاءاته
    """
    df = _df(data)
    if len(df) < 100:
        return {"symbol": symbol, "error": "بيانات غير كافية"}

    df = _add_indicators(df)
    df = df.reset_index(drop=True)

    trades     = []
    equity     = capital
    peak       = capital
    max_dd     = 0.0
    in_trade   = False
    entry_data = {}
    entry_bar  = 0

    for i in range(60, len(df) - 1):
        row = df.iloc[i]
        c   = float(row["close"])
        h   = float(row["high"])
        l   = float(row["low"])

        if in_trade:
            bars_held = i - entry_bar
            ep  = entry_data["entry_price"]
            sl  = entry_data["sl"]
            tp1 = entry_data["tp1"]
            tp2 = entry_data["tp2"]
            size_usdt = entry_data["size_usdt"]
            qty = entry_data["qty"]

            exit_price = None
            exit_reason = ""

            # SL
            if l <= sl:
                exit_price  = sl * (1 - BT_SLIPPAGE_PCT/100)
                exit_reason = "SL"

            # TP2
            elif h >= tp2:
                exit_price  = tp2
                exit_reason = "TP2"

            # TP1 (جزئي 50%)
            elif h >= tp1 and not entry_data.get("tp1_hit"):
                entry_data["tp1_hit"] = True
                pnl_partial = (tp1 - ep) * qty * 0.5
                fee = tp1 * qty * 0.5 * BT_FEE_PCT / 100
                equity += pnl_partial - fee
                entry_data["qty"] *= 0.5  # أبقِ نصف الكمية
                # SL → breakeven بعد TP1
                entry_data["sl"] = ep * 1.002

            # انتهاء الوقت
            elif bars_held >= BT_MAX_BARS_HOLD:
                exit_price  = c
                exit_reason = "TIME"

            if exit_price:
                pnl  = (exit_price - ep) * entry_data["qty"]
                fee  = (ep + exit_price) * entry_data["qty"] * BT_FEE_PCT / 100
                net  = pnl - fee
                equity += net

                trades.append({
                    "symbol":      symbol,
                    "entry":       ep,
                    "exit":        exit_price,
                    "pnl":         round(net, 4),
                    "pnl_pct":     round(net / size_usdt * 100, 2),
                    "reason":      exit_reason,
                    "bars":        bars_held,
                    "score":       entry_data["score"],
                    "pattern":     entry_data["pattern"],
                    "mode":        entry_data["mode"],
                    "win":         net > 0,
                    "bar_idx":     i,
                })

                if equity > peak: peak = equity
                dd = (peak - equity) / peak
                if dd > max_dd: max_dd = dd

                in_trade = False
                entry_data = {}

        else:
            # البحث عن إشارة دخول
            sig = _check_entry(df, i)
            if sig:
                size_usdt = equity * BT_RISK_PCT / 100
                if size_usdt < 1.0: continue

                entry_price = sig["entry_price"] * (1 + BT_SLIPPAGE_PCT/100)
                qty = size_usdt / entry_price
                fee = entry_price * qty * BT_FEE_PCT / 100
                equity -= fee

                entry_data = {
                    **sig,
                    "entry_price": entry_price,
                    "size_usdt":   size_usdt,
                    "qty":         qty,
                    "tp1_hit":     False,
                }
                entry_bar = i
                in_trade  = True

    # إحصاءات
    if not trades:
        return {
            "symbol":    symbol,
            "trades":    0,
            "win_rate":  0.0,
            "total_pnl": 0.0,
            "max_dd":    0.0,
            "rr_ratio":  0.0,
        }

    wins   = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]

    avg_win  = sum(t["pnl"] for t in wins)   / max(len(wins),   1)
    avg_loss = sum(t["pnl"] for t in losses) / max(len(losses), 1)
    rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # أفضل الأنماط
    pattern_stats = {}
    for t in trades:
        pt = t["pattern"]
        if pt not in pattern_stats:
            pattern_stats[pt] = {"wins": 0, "losses": 0, "pnl": 0.0}
        pattern_stats[pt]["pnl"] += t["pnl"]
        if t["win"]: pattern_stats[pt]["wins"] += 1
        else:        pattern_stats[pt]["losses"] += 1

    return {
        "symbol":        symbol,
        "trades":        len(trades),
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      round(len(wins) / len(trades), 3),
        "total_pnl":     round(sum(t["pnl"] for t in trades), 2),
        "avg_win":       round(avg_win, 4),
        "avg_loss":      round(avg_loss, 4),
        "rr_ratio":      round(rr_ratio, 2),
        "max_dd":        round(max_dd, 3),
        "final_equity":  round(equity, 2),
        "return_pct":    round((equity - capital) / capital * 100, 1),
        "pattern_stats": pattern_stats,
        "all_trades":    trades,
    }


def run_full_backtest(symbols_data: Dict[str, List],
                      capital: float = BT_INITIAL_CAPITAL) -> Dict:
    """
    يشغّل Backtest على كل الرموز ويجمع النتائج
    """
    all_results = []
    total_symbols = len(symbols_data)

    logger.info(f"[backtest] بدء اختبار {total_symbols} رمز...")

    for i, (symbol, data) in enumerate(symbols_data.items()):
        logger.info(f"[backtest] {i+1}/{total_symbols}: {symbol}")
        result = backtest_symbol(symbol, data, capital)
        all_results.append(result)

    # فلتر الرموز التي فيها بيانات كافية
    valid = [r for r in all_results if r.get("trades", 0) >= 5]

    if not valid:
        return {"error": "لا توجد بيانات كافية", "results": all_results}

    # إحصاءات إجمالية
    total_trades = sum(r["trades"] for r in valid)
    total_wins   = sum(r["wins"]   for r in valid)
    total_pnl    = sum(r["total_pnl"] for r in valid)
    avg_wr       = total_wins / max(total_trades, 1)
    avg_rr       = sum(r["rr_ratio"] for r in valid) / len(valid)
    max_dd       = max(r["max_dd"] for r in valid)

    # ترتيب الرموز حسب الأداء
    valid.sort(key=lambda x: x["total_pnl"], reverse=True)

    # أفضل وأسوأ الأنماط
    all_patterns: Dict[str, Dict] = {}
    for r in valid:
        for pt, stats in r.get("pattern_stats", {}).items():
            if pt not in all_patterns:
                all_patterns[pt] = {"wins": 0, "losses": 0, "pnl": 0.0}
            all_patterns[pt]["wins"]   += stats["wins"]
            all_patterns[pt]["losses"] += stats["losses"]
            all_patterns[pt]["pnl"]    += stats["pnl"]

    summary = {
        "run_at":          datetime.now(RIYADH_TZ).isoformat(timespec="seconds"),
        "symbols_tested":  total_symbols,
        "symbols_valid":   len(valid),
        "total_trades":    total_trades,
        "total_wins":      total_wins,
        "win_rate":        round(avg_wr, 3),
        "avg_rr_ratio":    round(avg_rr, 2),
        "total_pnl":       round(total_pnl, 2),
        "max_drawdown":    round(max_dd, 3),
        "initial_capital": capital,
        "pattern_stats":   all_patterns,
        "top_symbols":     valid[:10],
        "worst_symbols":   valid[-5:],
        "all_results":     valid,
    }

    # حفظ النتائج
    try:
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            # حفظ بدون all_trades لتوفير المساحة
            compact = {k: v for k, v in summary.items() if k != "all_results"}
            compact["all_results"] = [
                {k: v for k, v in r.items() if k != "all_trades"}
                for r in valid
            ]
            json.dump(compact, f, ensure_ascii=False, indent=2)
        logger.info(f"[backtest] ✅ النتائج محفوظة: {RESULTS_FILE}")
    except Exception as e:
        logger.warning(f"[backtest] تعذّر حفظ النتائج: {e}")

    return summary


def print_report(summary: Dict):
    """طباعة تقرير واضح"""
    print("\n" + "="*55)
    print("📊 نتائج Backtesting")
    print("="*55)
    print(f"الرموز المختبرة : {summary.get('symbols_tested', 0)}")
    print(f"الرموز الصالحة  : {summary.get('symbols_valid', 0)}")
    print(f"إجمالي الصفقات  : {summary.get('total_trades', 0)}")
    print(f"Win Rate        : {summary.get('win_rate', 0):.1%}")
    print(f"RR Ratio        : {summary.get('avg_rr_ratio', 0):.2f}")
    print(f"إجمالي PnL      : {summary.get('total_pnl', 0):+.2f}$")
    print(f"Max Drawdown    : {summary.get('max_drawdown', 0):.1%}")
    print("-"*55)

    print("\n🏆 أفضل 5 رموز:")
    for r in summary.get("top_symbols", [])[:5]:
        wr  = r.get("win_rate", 0)
        pnl = r.get("total_pnl", 0)
        tr  = r.get("trades", 0)
        rr  = r.get("rr_ratio", 0)
        print(f"  {r['symbol']:<15} WR={wr:.0%} PnL={pnl:+.2f}$ "
              f"RR={rr:.1f} ({tr} صفقة)")

    print("\n📉 أسوأ 3 رموز:")
    for r in summary.get("worst_symbols", [])[-3:]:
        wr  = r.get("win_rate", 0)
        pnl = r.get("total_pnl", 0)
        tr  = r.get("trades", 0)
        print(f"  {r['symbol']:<15} WR={wr:.0%} PnL={pnl:+.2f}$ ({tr} صفقة)")

    print("\n🎯 أداء الأنماط:")
    for pt, stats in summary.get("pattern_stats", {}).items():
        total = stats["wins"] + stats["losses"]
        if total < 3: continue
        wr  = stats["wins"] / total
        pnl = stats["pnl"]
        print(f"  {pt:<20} WR={wr:.0%} PnL={pnl:+.2f}$ ({total} صفقة)")

    print("="*55 + "\n")


# ==================== تشغيل مستقل ====================
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    print("🧪 Backtester — تشغيل")
    print("="*40)

    # جلب الرموز من config
    try:
        from config import SYMBOLS
        from data_loader import fetch_all_symbols
    except ImportError as e:
        print(f"❌ {e}")
        sys.exit(1)

    force = "--refresh" in sys.argv

    # جلب البيانات
    print(f"\n📥 جلب البيانات{'  (تحديث إجباري)' if force else ''}...")
    data = fetch_all_symbols(SYMBOLS, force_refresh=force)

    if not data:
        print("❌ لا توجد بيانات")
        sys.exit(1)

    print(f"✅ تم جلب {len(data)} رمز\n")

    # تشغيل الاختبار
    print("🔄 جاري الاختبار...")
    results = run_full_backtest(data)

    # عرض النتائج
    print_report(results)

    print(f"💾 النتائج محفوظة في: {RESULTS_FILE}")

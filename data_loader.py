# -*- coding: utf-8 -*-
# data_loader.py - جلب البيانات التاريخية (v3.1)
#
# الحل الاحترافي:
# 1. جلب 4h من OKX (8 أشهر = 1440 شمعة، الحد الأقصى المتاح)
# 2. تحويل 4h → 15m اصطناعي (كل شمعة = 16 شمعة)
# النتيجة: 23,040 شمعة 15m = 8 أشهر حقيقية

from __future__ import annotations
import os, json, time, logging, requests, math
from typing import Dict, List, Optional

logger = logging.getLogger("data_loader")

DATA_DIR     = os.getenv("BACKTEST_DATA_DIR", "/opt/render/project/data/backtest")
DELAY_SEC    = float(os.getenv("BACKTEST_DELAY_SEC", "0.4"))
BARS_PER_REQ = 300
TIMEFRAME    = os.getenv("BACKTEST_TF", "15m")
OKX_BASE     = "https://www.okx.com"

os.makedirs(DATA_DIR, exist_ok=True)

def _cache_path(symbol, tf):
    safe = symbol.replace("/","_").replace("#","_")
    return os.path.join(DATA_DIR, f"{safe}_{tf}.json")

def _is_fresh(path, max_age_hours=12.0):
    try:
        return (time.time() - os.path.getmtime(path)) / 3600 < max_age_hours
    except:
        return False

def _fetch_batch_1h(symbol, after_ts=None):
    """يجلب دفعة واحدة من بيانات 1h"""
    try:
        params = {
            "instId": symbol.replace("/","-"),
            "bar":    "4H",
            "limit":  str(BARS_PER_REQ),
        }
        if after_ts:
            params["after"] = str(after_ts)
        r = requests.get(f"{OKX_BASE}/api/v5/market/candles",
                         params=params, timeout=15)
        if r.status_code != 200: return []
        data = r.json()
        if data.get("code") != "0": return []
        result = []
        for c in data.get("data", []):
            try:
                result.append([
                    int(c[0]), float(c[1]), float(c[2]),
                    float(c[3]), float(c[4]), float(c[5])
                ])
            except: continue
        return result
    except Exception as e:
        logger.warning(f"[loader] {symbol} 1h: {e}")
        return []

def _convert_4h_to_15m(candles_4h):
    """
    يحوّل شموع 4h إلى 15m اصطناعية.
    كل شمعة 4h → 16 شمعة 15m بتوزيع واقعي.
    """
    result = []
    interval = 15 * 60 * 1000  # 15 دقيقة بالمللي ثانية

    for c in candles_4h:
        ts, o, h, l, close, vol = c[0], c[1], c[2], c[3], c[4], c[5]
        n = 16  # عدد شموع 15m في كل 4h

        # توزيع الحركة: من Open إلى Close بشكل تدريجي مع تذبذب
        prices = []
        for i in range(n + 1):
            t = i / n
            # حركة تدريجية مع تذبذب بسيط
            trend = o + (close - o) * t
            wave  = (h - l) * 0.15 * math.sin(t * math.pi * 2)
            prices.append(trend + wave)

        # توزيع الحجم (أعلى في المنتصف)
        vols = []
        for i in range(n):
            w = 1.0 + 0.5 * math.sin((i / n) * math.pi)
            vols.append(w)
        total_w = sum(vols)
        vols = [v / total_w * vol for v in vols]

        for i in range(n):
            bar_o = prices[i]
            bar_c = prices[i + 1]
            # High/Low مع هامش بسيط
            margin = abs(h - l) * 0.05
            bar_h = max(bar_o, bar_c) + margin
            bar_l = min(bar_o, bar_c) - margin
            # تأكد من عدم تجاوز حدود الشمعة الأصلية
            bar_h = min(bar_h, h)
            bar_l = max(bar_l, l)
            result.append([ts + interval * i, bar_o, bar_h, bar_l, bar_c, vols[i]])

    result.sort(key=lambda x: x[0])
    return result

def fetch_symbol_4h(symbol, total_bars_4h=1440, force_refresh=False):
    """يجلب بيانات 1h كاملة (سنة)"""
    base  = symbol.split("#")[0]
    cpath = _cache_path(base, "4h_raw")

    if not force_refresh and os.path.exists(cpath) and _is_fresh(cpath, 12):
        try:
            with open(cpath) as f:
                data = json.load(f)
            if data and len(data) >= total_bars_1h * 0.8:
                return data
        except: pass

    num_batches = (total_bars_1h + BARS_PER_REQ - 1) // BARS_PER_REQ
    all_candles = []
    after_ts    = None

    logger.info(f"[loader] {base}: جلب {total_bars_1h} شمعة 1h ({num_batches} دفعة)...")

    for batch_num in range(num_batches):
        batch = _fetch_batch_1h(base, after_ts)
        if not batch:
            logger.warning(f"[loader] {base}: 4h توقف عند دفعة {batch_num+1} ({len(all_candles)} شمعة)")
            break
        all_candles.extend(batch)
        after_ts = min(c[0] for c in batch)
        time.sleep(DELAY_SEC)
        if len(all_candles) >= total_bars_1h:
            break

    if not all_candles:
        return None

    all_candles.sort(key=lambda x: x[0])
    all_candles = all_candles[-total_bars_1h:]

    try:
        with open(cpath, "w") as f:
            json.dump(all_candles, f)
    except: pass

    return all_candles

def fetch_symbol_data(symbol, tf=TIMEFRAME, total_bars=8640, force_refresh=False):
    """
    الدالة الرئيسية:
    - تجلب 1h (سنة كاملة)
    - تحوّل إلى 15m اصطناعي
    - ترجع شموع 15m
    """
    base  = symbol.split("#")[0]
    cpath = _cache_path(base, "15m_from_4h")

    # قراءة من الكاش
    if not force_refresh and os.path.exists(cpath) and _is_fresh(cpath, 12):
        try:
            with open(cpath) as f:
                data = json.load(f)
            if data and len(data) >= 1000:
                logger.debug(f"[loader] {base}: كاش 15m ({len(data)} شمعة)")
                return data
        except: pass

    # جلب 1h
    bars_4h = 1440  # على الأقل 3 أشهر
    data_4h = fetch_symbol_4h(base, bars_4h, force_refresh)

    if not data_4h or len(data_4h) < 50:
        logger.warning(f"[loader] {base}: بيانات 1h غير كافية")
        return None

    # تحويل لـ 15m
    data_15m = _convert_4h_to_15m(data_4h)

    if not data_15m:
        return None

    logger.info(f"[loader] {base}: {len(data_4h)} شمعة 4h → {len(data_15m)} شمعة 15m")

    # حفظ
    try:
        with open(cpath, "w") as f:
            json.dump(data_15m, f)
    except Exception as e:
        logger.warning(f"[loader] {base}: تعذر الحفظ — {e}")

    return data_15m

def fetch_all_symbols(symbols, tf=TIMEFRAME, total_bars=8640, force_refresh=False):
    """يجلب كل الرموز"""
    results = {}
    seen    = set()
    success = 0
    failed  = 0

    unique = []
    for sym in symbols:
        base = sym.split("#")[0]
        if base not in seen:
            unique.append(base)
            seen.add(base)

    logger.info(f"[loader] جلب {len(unique)} رمز (1h → 15m)...")

    for i, base in enumerate(unique):
        data = fetch_symbol_data(base, tf, total_bars, force_refresh)
        if data and len(data) >= 100:
            results[base] = data
            success += 1
        else:
            failed += 1

        if (i + 1) % 5 == 0:
            logger.info(f"[loader] {i+1}/{len(unique)} | نجح={success} فشل={failed}")

    logger.info(f"[loader] انتهى: نجح={success} فشل={failed}")
    return results

def get_cache_info():
    info = {}
    try:
        for f in os.listdir(DATA_DIR):
            if not f.endswith(".json"): continue
            path = os.path.join(DATA_DIR, f)
            try:
                size = os.path.getsize(path) / 1024
                age  = (time.time() - os.path.getmtime(path)) / 3600
                with open(path) as fp:
                    d = json.load(fp)
                bars = len(d) if isinstance(d, list) else 0
                sym  = f.replace("","").replace("","").replace("_","/",1)
                info[f] = {"bars": bars, "size_kb": round(size,1), "age_hrs": round(age,1)}
            except: pass
    except: pass
    return info

def clear_cache():
    import shutil
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    logger.info("[loader] تم حذف الكاش")

if __name__ == "__main__":
    import sys
    from datetime import datetime
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if "--info" in sys.argv:
        info = get_cache_info()
        if not info:
            print("الكاش فارغ")
        else:
            for fname, d in sorted(info.items()):
                print(f"{fname:<40} {d['bars']:>8} شمعة | {d['size_kb']:>8}KB | {d['age_hrs']:.1f}h")

    elif "--clear" in sys.argv:
        clear_cache()
        print("تم حذف الكاش")

    elif "--test" in sys.argv:
        sym = "BTC/USDT"
        if len(sys.argv) > sys.argv.index("--test") + 1:
            sym = sys.argv[sys.argv.index("--test") + 1]
        data = fetch_symbol_data(sym, force_refresh=True)
        if data:
            print(f"OK {sym}: {len(data)} شمعة 15m")
            print(f"   من:   {datetime.fromtimestamp(data[0][0]/1000)}")
            print(f"   حتى:  {datetime.fromtimestamp(data[-1][0]/1000)}")
            days = (data[-1][0] - data[0][0]) / 1000 / 86400
            print(f"   مدة:  {days:.0f} يوم")
        else:
            print(f"فشل {sym}")

    else:
        print("الاستخدام:")
        print("  python data_loader.py --test BTC/USDT")
        print("  python data_loader.py --info")
        print("  python data_loader.py --clear")

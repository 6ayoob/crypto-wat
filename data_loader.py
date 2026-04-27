# -*- coding: utf-8 -*-
# data_loader.py v3.1 - 4h to 15m conversion
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

def _cache_path(symbol, suffix):
    safe = symbol.replace("/","_").replace("#","_")
    return os.path.join(DATA_DIR, f"{safe}_{suffix}.json")

def _is_fresh(path, max_age_hours=12.0):
    try:
        return (time.time() - os.path.getmtime(path)) / 3600 < max_age_hours
    except:
        return False

def _fetch_batch_4h(symbol, after_ts=None):
    try:
        params = {
            "instId": symbol.replace("/","-"),
            "bar":    "4H",
            "limit":  str(BARS_PER_REQ),
        }
        if after_ts:
            params["after"] = str(after_ts)
        r = requests.get(f"{OKX_BASE}/api/v5/market/candles", params=params, timeout=15)
        if r.status_code != 200: return []
        data = r.json()
        if data.get("code") != "0": return []
        result = []
        for c in data.get("data", []):
            try:
                result.append([int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])])
            except: continue
        return result
    except Exception as e:
        logger.warning(f"[loader] {symbol} 4h: {e}")
        return []

def _convert_4h_to_15m(candles_4h):
    result = []
    interval = 15 * 60 * 1000
    n = 16
    for c in candles_4h:
        ts, o, h, l, close, vol = c[0], c[1], c[2], c[3], c[4], c[5]
        prices = []
        for i in range(n + 1):
            t = i / n
            trend = o + (close - o) * t
            wave  = (h - l) * 0.12 * math.sin(t * math.pi * 2)
            prices.append(trend + wave)
        vols = []
        for i in range(n):
            w = 1.0 + 0.5 * math.sin((i / n) * math.pi)
            vols.append(w)
        total_w = sum(vols)
        vols = [v / total_w * vol for v in vols]
        for i in range(n):
            bar_o = prices[i]
            bar_c = prices[i + 1]
            margin = abs(h - l) * 0.04
            bar_h  = min(max(bar_o, bar_c) + margin, h)
            bar_l  = max(min(bar_o, bar_c) - margin, l)
            result.append([ts + interval * i, bar_o, bar_h, bar_l, bar_c, vols[i]])
    result.sort(key=lambda x: x[0])
    return result

def fetch_symbol_4h(symbol, total_bars=1440, force_refresh=False):
    base  = symbol.split("#")[0]
    cpath = _cache_path(base, "4h_raw")
    if not force_refresh and os.path.exists(cpath) and _is_fresh(cpath):
        try:
            with open(cpath) as f:
                data = json.load(f)
            if data and len(data) >= total_bars * 0.8:
                return data
        except: pass
    num_batches = (total_bars + BARS_PER_REQ - 1) // BARS_PER_REQ
    all_candles = []
    after_ts    = None
    logger.info(f"[loader] {base}: جلب 4h ({num_batches} دفعة)...")
    for b in range(num_batches):
        batch = _fetch_batch_4h(base, after_ts)
        if not batch:
            logger.warning(f"[loader] {base}: توقف عند دفعة {b+1} ({len(all_candles)} شمعة)")
            break
        all_candles.extend(batch)
        after_ts = min(c[0] for c in batch)
        time.sleep(DELAY_SEC)
        if len(all_candles) >= total_bars:
            break
    if not all_candles:
        return None
    all_candles.sort(key=lambda x: x[0])
    all_candles = all_candles[-total_bars:]
    try:
        with open(cpath, "w") as f:
            json.dump(all_candles, f)
    except: pass
    return all_candles

def fetch_symbol_data(symbol, tf=TIMEFRAME, total_bars=8640, force_refresh=False):
    base  = symbol.split("#")[0]
    cpath = _cache_path(base, "15m_from_4h")
    if not force_refresh and os.path.exists(cpath) and _is_fresh(cpath):
        try:
            with open(cpath) as f:
                data = json.load(f)
            if data and len(data) >= 1000:
                logger.debug(f"[loader] {base}: كاش ({len(data)} شمعة)")
                return data
        except: pass
    data_4h = fetch_symbol_4h(base, 1440, force_refresh)
    if not data_4h or len(data_4h) < 50:
        logger.warning(f"[loader] {base}: بيانات 4h غير كافية")
        return None
    data_15m = _convert_4h_to_15m(data_4h)
    if not data_15m:
        return None
    logger.info(f"[loader] {base}: {len(data_4h)} شمعة 4h → {len(data_15m)} شمعة 15m")
    try:
        with open(cpath, "w") as f:
            json.dump(data_15m, f)
    except: pass
    return data_15m

def fetch_all_symbols(symbols, tf=TIMEFRAME, total_bars=8640, force_refresh=False):
    results = {}
    seen    = set()
    success = 0
    failed  = 0
    unique  = []
    for sym in symbols:
        base = sym.split("#")[0]
        if base not in seen:
            unique.append(base)
            seen.add(base)
    logger.info(f"[loader] جلب {len(unique)} رمز (4h → 15m)...")
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
        if not info: print("الكاش فارغ")
        else:
            for fname, d in sorted(info.items()):
                print(f"{fname:<45} {d['bars']:>8} شمعة | {d['size_kb']:>8}KB")
    elif "--clear" in sys.argv:
        clear_cache(); print("تم حذف الكاش")
    elif "--test" in sys.argv:
        sym = sys.argv[sys.argv.index("--test")+1] if len(sys.argv) > sys.argv.index("--test")+1 else "BTC/USDT"
        data = fetch_symbol_data(sym, force_refresh=True)
        if data:
            print(f"OK {sym}: {len(data)} شمعة 15m")
            print(f"   من:   {datetime.fromtimestamp(data[0][0]/1000)}")
            print(f"   حتى:  {datetime.fromtimestamp(data[-1][0]/1000)}")
            print(f"   مدة:  {(data[-1][0]-data[0][0])/1000/86400:.0f} يوم")
        else: print(f"فشل {sym}")
    else:
        print("الاستخدام:")
        print("  python data_loader.py --test BTC/USDT")
        print("  python data_loader.py --info")
        print("  python data_loader.py --clear")

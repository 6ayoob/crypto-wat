# -*- coding: utf-8 -*-
# data_loader.py - جلب البيانات التاريخية بأمان (v2.0)
#
# الإصلاح الرئيسي: pagination لجلب 3 أشهر كاملة
# OKX يحدد 300 شمعة لكل طلب → نجلب عدة دفعات

from __future__ import annotations
import os, json, time, logging, requests
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

def _is_fresh(path, max_age_hours=6.0):
    try:
        return (time.time() - os.path.getmtime(path)) / 3600 < max_age_hours
    except:
        return False

def _fetch_batch(symbol, tf, after_ts=None):
    try:
        params = {"instId": symbol.replace("/","-"), "bar": tf, "limit": str(BARS_PER_REQ)}
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
        logger.warning(f"[loader] {symbol}: {e}")
        return []

def fetch_symbol_data(symbol, tf=TIMEFRAME, total_bars=8640, force_refresh=False):
    base  = symbol.split("#")[0]
    cpath = _cache_path(base, tf)

    if not force_refresh and os.path.exists(cpath) and _is_fresh(cpath):
        try:
            with open(cpath) as f:
                data = json.load(f)
            if data and len(data) >= total_bars * 0.8:
                logger.debug(f"[loader] {base}: كاش ({len(data)} شمعة)")
                return data
        except: pass

    num_batches = (total_bars + BARS_PER_REQ - 1) // BARS_PER_REQ
    all_candles = []
    after_ts    = None

    logger.info(f"[loader] جلب {base}: {total_bars} شمعة ({num_batches} دفعة)...")

    for batch_num in range(num_batches):
        batch = _fetch_batch(base, tf, after_ts)
        if not batch:
            logger.warning(f"[loader] {base}: دفعة {batch_num+1} فارغة")
            break
        all_candles.extend(batch)
        after_ts = min(c[0] for c in batch)
        time.sleep(DELAY_SEC)
        if len(all_candles) >= total_bars:
            break

    if not all_candles:
        logger.warning(f"[loader] {base}: لا بيانات")
        return None

    all_candles.sort(key=lambda x: x[0])
    all_candles = all_candles[-total_bars:]

    try:
        with open(cpath, "w") as f:
            json.dump(all_candles, f)
    except Exception as e:
        logger.warning(f"[loader] {base}: تعذر الحفظ — {e}")

    logger.info(f"[loader] OK {base}: {len(all_candles)} شمعة")
    return all_candles

def fetch_all_symbols(symbols, tf=TIMEFRAME, total_bars=8640, force_refresh=False):
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

    logger.info(f"[loader] جلب {len(unique)} رمز × {total_bars} شمعة...")

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
        from datetime import datetime
        for f in os.listdir(DATA_DIR):
            if not f.endswith(".json"): continue
            path = os.path.join(DATA_DIR, f)
            try:
                size = os.path.getsize(path) / 1024
                age  = (time.time() - os.path.getmtime(path)) / 3600
                with open(path) as fp:
                    d = json.load(fp)
                bars = len(d) if isinstance(d, list) else 0
                sym  = f.replace("_" + TIMEFRAME + ".json","").replace("_","/",1)
                info[sym] = {"bars": bars, "size_kb": round(size,1), "age_hrs": round(age,1)}
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
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if "--info" in sys.argv:
        info = get_cache_info()
        if not info:
            print("الكاش فارغ")
        else:
            print(f"\n{'الرمز':<20} {'شموع':>8} {'حجم(KB)':>10} {'عمر(ساعة)':>12}")
            print("-" * 55)
            for sym, d in sorted(info.items()):
                print(f"{sym:<20} {d['bars']:>8} {d['size_kb']:>10} {d['age_hrs']:>12}")

    elif "--clear" in sys.argv:
        clear_cache()
        print("تم حذف الكاش")

    elif "--test" in sys.argv:
        sym = "BTC/USDT"
        if len(sys.argv) > sys.argv.index("--test") + 1:
            sym = sys.argv[sys.argv.index("--test") + 1]
        from datetime import datetime
        data = fetch_symbol_data(sym, force_refresh=True)
        if data:
            print(f"OK {sym}: {len(data)} شمعة")
            print(f"   من: {datetime.fromtimestamp(data[0][0]/1000)}")
            print(f"   حتى: {datetime.fromtimestamp(data[-1][0]/1000)}")
        else:
            print(f"فشل {sym}")

    else:
        print("الاستخدام:")
        print("  python data_loader.py --info")
        print("  python data_loader.py --clear")
        print("  python data_loader.py --test BTC/USDT")

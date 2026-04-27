# -*- coding: utf-8 -*-
# data_loader.py - جلب البيانات التاريخية بأمان
#
# يجلب OHLCV من OKX بتأخير آمن ويحفظها محلياً
# المرة الثانية يقرأ من الملف مباشرة (لا طلبات)

from __future__ import annotations

import os, json, time, logging, math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("data_loader")

# ===== إعدادات =====
DATA_DIR     = os.getenv("BACKTEST_DATA_DIR",
               "/opt/render/project/data/backtest")
DELAY_SEC    = float(os.getenv("BACKTEST_DELAY_SEC", "0.5"))  # تأخير آمن بين الطلبات
MAX_BARS     = int(os.getenv("BACKTEST_MAX_BARS",    "2000")) # ~20 يوم على 15m
TIMEFRAME    = os.getenv("BACKTEST_TF",              "15m")

os.makedirs(DATA_DIR, exist_ok=True)

def _cache_path(symbol: str, tf: str) -> str:
    safe = symbol.replace("/","_").replace("#","_")
    return os.path.join(DATA_DIR, f"{safe}_{tf}.json")

def _is_fresh(path: str, max_age_hours: float = 6.0) -> bool:
    """هل الملف محدّث (أقل من 6 ساعات)؟"""
    try:
        mtime = os.path.getmtime(path)
        age   = (time.time() - mtime) / 3600
        return age < max_age_hours
    except:
        return False

def fetch_symbol_data(symbol: str, tf: str = TIMEFRAME,
                      bars: int = MAX_BARS,
                      force_refresh: bool = False) -> Optional[List]:
    """
    يجلب بيانات رمز واحد.
    إذا موجودة ومحدّثة → يقرأ من الملف
    إذا لا → يجلب من OKX ويحفظ
    """
    base   = symbol.split("#")[0]
    cpath  = _cache_path(base, tf)

    # قراءة من الكاش إذا متاح
    if not force_refresh and os.path.exists(cpath) and _is_fresh(cpath):
        try:
            with open(cpath) as f:
                data = json.load(f)
            if data and len(data) >= bars // 2:
                logger.debug(f"[loader] {base}: كاش ({len(data)} شمعة)")
                return data
        except:
            pass

    # جلب من OKX
    try:
        from okx_api import fetch_ohlcv
        logger.info(f"[loader] جلب {base} {tf} ({bars} شمعة)...")
        data = fetch_ohlcv(base, tf, bars)
        if data and len(data) > 50:
            with open(cpath, "w") as f:
                json.dump(data, f)
            logger.info(f"[loader] ✅ {base}: {len(data)} شمعة محفوظة")
            time.sleep(DELAY_SEC)  # تأخير آمن
            return data
    except Exception as e:
        logger.warning(f"[loader] {base}: فشل الجلب — {e}")

    return None


def fetch_all_symbols(symbols: List[str],
                      tf: str = TIMEFRAME,
                      bars: int = MAX_BARS,
                      force_refresh: bool = False) -> Dict[str, List]:
    """
    يجلب بيانات كل الرموز بتأخير آمن.
    يرجع dict: {symbol: [بيانات]}
    """
    results  = {}
    seen     = set()
    total    = 0
    failed   = 0

    # إزالة التكرار (BTC/USDT و BTC/USDT#brt → نفس البيانات)
    unique_bases = []
    for sym in symbols:
        base = sym.split("#")[0]
        if base not in seen:
            unique_bases.append(base)
            seen.add(base)

    logger.info(f"[loader] جلب {len(unique_bases)} رمز فريد...")

    for i, base in enumerate(unique_bases):
        data = fetch_symbol_data(base, tf, bars, force_refresh)
        if data:
            results[base] = data
            total += 1
        else:
            failed += 1

        # تقرير التقدم كل 10 رموز
        if (i + 1) % 10 == 0:
            logger.info(f"[loader] تقدم: {i+1}/{len(unique_bases)} | نجح={total} فشل={failed}")

    logger.info(f"[loader] ✅ انتهى: {total} رمز | فشل: {failed}")
    return results


def get_cached_symbols() -> List[str]:
    """يرجع قائمة الرموز المحفوظة محلياً"""
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    return [f.replace(f"_{TIMEFRAME}.json","").replace("_","/",1) for f in files]


def clear_cache():
    """حذف كل البيانات المحفوظة"""
    import shutil
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    logger.info("[loader] تم حذف الكاش")

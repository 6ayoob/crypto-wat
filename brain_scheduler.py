# -*- coding: utf-8 -*-
# brain_scheduler.py - مُجدوِل العقل المدبر (v1.0)
#
# يشغّل market_brain.py كل 30 دقيقة في الخلفية.
# أضفه في نفس الـ process أو كـ thread منفصل.
#
# طريقة التشغيل:
#   python brain_scheduler.py          # تشغيل مستقل
#   أو استدعاء start_brain_scheduler() من main.py

from __future__ import annotations

import threading, time, logging, os
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("brain_scheduler")

RIYADH_TZ = timezone(timedelta(hours=3))

def _now(): return datetime.now(RIYADH_TZ)

def _ef(name, default):
    try:    return float(os.getenv(name, str(default)))
    except: return float(default)

BRAIN_INTERVAL_SEC = _ef("BRAIN_INTERVAL_MIN", 30) * 60
BRAIN_STARTUP_DELAY= _ef("BRAIN_STARTUP_DELAY_SEC", 10)  # انتظر 10 ثواني قبل أول دورة

_scheduler_thread: threading.Thread = None
_stop_event = threading.Event()


def _brain_loop():
    """الحلقة الرئيسية للعقل"""
    logger.info(f"[scheduler] 🧠 العقل المدبر بدأ — دورة كل {BRAIN_INTERVAL_SEC/60:.0f} دقيقة")

    # انتظر قليلاً قبل أول دورة
    time.sleep(BRAIN_STARTUP_DELAY)

    while not _stop_event.is_set():
        try:
            from market_brain import run_brain_cycle
            run_brain_cycle()
        except Exception as e:
            logger.error(f"[scheduler] ❌ خطأ في دورة العقل: {e}")

        # انتظر للدورة القادمة (مع إمكانية الإيقاف)
        next_run = _now() + timedelta(seconds=BRAIN_INTERVAL_SEC)
        logger.info(f"[scheduler] ⏰ الدورة القادمة: {next_run.strftime('%H:%M:%S')}")

        _stop_event.wait(timeout=BRAIN_INTERVAL_SEC)

    logger.info("[scheduler] 🛑 العقل المدبر توقف")


def start_brain_scheduler() -> threading.Thread:
    """
    يبدأ العقل المدبر في thread منفصل.
    استدعيها من main.py عند بدء التشغيل.

    مثال:
        from brain_scheduler import start_brain_scheduler
        start_brain_scheduler()
    """
    global _scheduler_thread

    if _scheduler_thread and _scheduler_thread.is_alive():
        logger.warning("[scheduler] العقل يعمل بالفعل")
        return _scheduler_thread

    _stop_event.clear()
    _scheduler_thread = threading.Thread(
        target=_brain_loop,
        name="MarketBrain",
        daemon=True  # يتوقف تلقائياً عند إيقاف البرنامج الرئيسي
    )
    _scheduler_thread.start()
    logger.info("[scheduler] ✅ تم تشغيل العقل المدبر في الخلفية")
    return _scheduler_thread


def stop_brain_scheduler():
    """إيقاف العقل المدبر"""
    _stop_event.set()
    if _scheduler_thread:
        _scheduler_thread.join(timeout=5)
    logger.info("[scheduler] 🛑 تم إيقاف العقل المدبر")


def is_brain_running() -> bool:
    return _scheduler_thread is not None and _scheduler_thread.is_alive()


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    print("🧠 Brain Scheduler — تشغيل مستقل")
    print("اضغط Ctrl+C للإيقاف")
    print("=" * 40)

    start_brain_scheduler()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n⛔ إيقاف...")
        stop_brain_scheduler()

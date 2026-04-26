# -*- coding: utf-8 -*-
# paths.py - مسارات التخزين الدائم
#
# يكتشف تلقائياً إذا كان الـ Disk موجوداً ويستخدمه
# إذا لا → يستخدم المجلد الحالي (للتطوير المحلي)

import os

# مسار الـ Disk على Render
_RENDER_DISK = "/opt/render/project/data"

# هل الـ Disk موجود ومكتوب؟
def _disk_available() -> bool:
    try:
        os.makedirs(_RENDER_DISK, exist_ok=True)
        test = os.path.join(_RENDER_DISK, ".write_test")
        with open(test, "w") as f:
            f.write("ok")
        os.remove(test)
        return True
    except:
        return False

# اختر المجلد الجذر
if _disk_available():
    DATA_DIR = _RENDER_DISK
    print(f"[paths] ✅ Disk دائم: {DATA_DIR}", flush=True)
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"[paths] ⚠️ Disk غير متاح — استخدام المجلد المحلي: {DATA_DIR}", flush=True)

# ===== المسارات الدائمة =====
POSITIONS_DIR         = os.path.join(DATA_DIR, "positions")
CLOSED_POSITIONS_FILE = os.path.join(DATA_DIR, "closed_positions.json")
RISK_STATE_FILE       = os.path.join(DATA_DIR, "risk_state.json")
BRAIN_STATE_FILE      = os.path.join(DATA_DIR, "brain_state.json")
BRAIN_LOG_FILE        = os.path.join(DATA_DIR, "brain_log.json")
DUST_LOG_FILE         = os.path.join(DATA_DIR, "dust_cleaned.json")
TRADES_LOG_FILE       = os.path.join(DATA_DIR, "trades.log")

# إنشاء المجلدات المطلوبة
os.makedirs(POSITIONS_DIR, exist_ok=True)

# ===== دالة مساعدة =====
def get_path(filename: str) -> str:
    """يرجع المسار الكامل لأي ملف في مجلد البيانات"""
    return os.path.join(DATA_DIR, filename)

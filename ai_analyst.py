# -*- coding: utf-8 -*-
# ai_analyst.py - Claude AI كمحلل استراتيجي (المرحلة ٣)
#
# يعمل كل ساعة ويقرأ:
# - brain_state.json (العقل المدبر)
# - closed_positions.json (آخر الصفقات)
# - اللوق الأخير
#
# ثم يرسل توجيهات مباشرة للبوت عبر ai_directives.json
# وإشعار تيليغرام بتحليله

from __future__ import annotations

import os, json, time, logging, requests
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List

logger = logging.getLogger("ai_analyst")

RIYADH_TZ = timezone(timedelta(hours=3))

# ===== إعدادات =====
AI_ENABLED          = os.getenv("AI_ANALYST_ENABLED", "1") in ("1","true","yes")
AI_INTERVAL_MIN     = float(os.getenv("AI_INTERVAL_MIN", "60"))  # كل ساعة
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
TELEGRAM_TOKEN      = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")
AI_DIRECTIVES_FILE  = os.getenv("AI_DIRECTIVES_FILE",
                                 "/opt/render/project/data/ai_directives.json")
AI_LOG_FILE         = os.getenv("AI_LOG_FILE",
                                 "/opt/render/project/data/ai_log.json")

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

def _tg(text: str):
    try:
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": text,
                  "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=10
        )
    except: pass


# ══════════════════════════════════════════════════════
# بناء السياق للـ AI
# ══════════════════════════════════════════════════════

def _build_context() -> str:
    """يجمع كل المعلومات المهمة ويبنيها كنص للـ AI"""

    # العقل المدبر
    brain_path = os.getenv("BRAIN_STATE_FILE",
                           "/opt/render/project/data/brain_state.json")
    brain = _read_json(brain_path, {})

    # آخر الصفقات
    closed_path = os.getenv("CLOSED_POSITIONS_FILE",
                            "/opt/render/project/data/closed_positions.json")
    closed = _read_json(closed_path, [])
    last_trades = closed[-10:] if closed else []

    # التوجيهات الحالية
    current_dir = _read_json(AI_DIRECTIVES_FILE, {})

    # إحصاءات الأداء
    wins   = [t for t in last_trades if float(t.get("profit", 0)) > 0]
    losses = [t for t in last_trades if float(t.get("profit", 0)) <= 0]
    total_pnl = sum(float(t.get("profit", 0)) for t in last_trades)
    win_rate  = len(wins) / len(last_trades) if last_trades else 0

    # تفاصيل الصفقات الأخيرة
    trades_summary = ""
    for t in last_trades[-5:]:
        pnl    = float(t.get("profit", 0))
        symbol = t.get("symbol", "?")
        reason = t.get("reason", "?")
        mode   = t.get("entry_reason", "?")
        pattern= t.get("pattern", "?")
        trades_summary += f"  • {symbol}: {pnl:+.2f}$ | {reason} | {pattern} | {mode}\n"

    context = f"""
=== حالة البوت الحالية ===
الوقت: {_now_iso()}

=== العقل المدبر (market_brain) ===
الحالة: {brain.get('regime', '?')}
BTC: {brain.get('btc_price', 0):.0f}$ | RSI: {brain.get('btc_rsi', 50):.0f} | ADX: {brain.get('btc_adx', 20):.0f}
الاتجاه: {brain.get('btc_trend', '?')}
التذبذب: {brain.get('volatility_ratio', 1):.2f}x
الإعدادات الحالية: score≥{brain.get('score_threshold_override', 55)} | size×{brain.get('size_multiplier', 1):.2f}
الدخول مسموح: {brain.get('entry_allowed', True)}
الأنماط المحظورة: {brain.get('blocked_patterns', [])}

=== أداء آخر 10 صفقات ===
عدد الصفقات: {len(last_trades)}
الرابحة: {len(wins)} | الخاسرة: {len(losses)}
Win Rate: {win_rate:.0%}
إجمالي PnL: {total_pnl:+.2f}$

الصفقات الأخيرة:
{trades_summary if trades_summary else "لا توجد صفقات"}

=== التوجيهات الحالية من AI ===
{json.dumps(current_dir, ensure_ascii=False, indent=2) if current_dir else "لا توجد توجيهات سابقة"}
""".strip()

    return context


# ══════════════════════════════════════════════════════
# استدعاء Claude API
# ══════════════════════════════════════════════════════

def _call_claude(context: str) -> Optional[str]:
    """يستدعي Claude API ويرجع التحليل"""

    if not ANTHROPIC_API_KEY:
        logger.warning("[AI] ANTHROPIC_API_KEY غير موجود")
        return None

    system_prompt = """أنت محلل استراتيجي لبوت تداول عملات رقمية على منصة OKX.
مهمتك: تحليل الوضع الحالي وإعطاء توجيهات دقيقة ومختصرة.

قواعد مهمة:
- كن دقيقاً وعملياً — لا كلام عام
- ركز على ما يجب تغييره الآن
- إذا الوضع جيد → قل ذلك واذكر ما يمكن تحسينه
- إذا هناك خطر → كن صريحاً وواضحاً

يجب أن يكون ردك بتنسيق JSON فقط، هكذا:
{
  "market_assessment": "تقييم السوق في جملة واحدة",
  "performance_assessment": "تقييم الأداء في جملة واحدة",
  "directives": {
    "entry_allowed": true/false,
    "score_threshold_override": 50-70,
    "size_multiplier": 0.5-1.5,
    "preferred_modes": ["breakout", "pullback"],
    "avoid_symbols": ["رموز يجب تجنبها إن وجدت"],
    "max_open_positions_override": 3-8,
    "special_notes": "ملاحظات خاصة"
  },
  "telegram_message": "رسالة قصيرة للمستخدم (بالعربي، 3-5 أسطر)",
  "confidence": 0.0-1.0
}"""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-sonnet-4-6",
                "max_tokens": 1000,
                "system":     system_prompt,
                "messages":   [{"role": "user", "content": context}],
            },
            timeout=30
        )

        if response.status_code != 200:
            logger.warning(f"[AI] Claude API error: {response.status_code}")
            return None

        data    = response.json()
        content = data["content"][0]["text"]
        return content

    except Exception as e:
        logger.error(f"[AI] استدعاء Claude فشل: {e}")
        return None


# ══════════════════════════════════════════════════════
# معالجة الرد وحفظ التوجيهات
# ══════════════════════════════════════════════════════

def _parse_and_save(raw_response: str) -> Optional[Dict]:
    """يحلّل رد Claude ويحفظ التوجيهات"""
    try:
        # تنظيف JSON
        text = raw_response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # إيجاد أول { وآخر }
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        parsed = json.loads(text)

        # إضافة metadata
        parsed["updated_at"]  = _now_iso()
        parsed["source"]      = "claude_ai"

        # حفظ التوجيهات
        _write_json(AI_DIRECTIVES_FILE, parsed)
        logger.info(f"[AI] ✅ توجيهات محفوظة: confidence={parsed.get('confidence', 0):.0%}")

        # حفظ في السجل
        log = _read_json(AI_LOG_FILE, [])
        log.append({
            "ts":          _now_iso(),
            "assessment":  parsed.get("market_assessment", ""),
            "confidence":  parsed.get("confidence", 0),
            "directives":  parsed.get("directives", {}),
        })
        _write_json(AI_LOG_FILE, log[-50:])

        return parsed

    except Exception as e:
        logger.error(f"[AI] خطأ في تحليل الرد: {e}")
        logger.debug(f"[AI] الرد الخام: {raw_response[:500]}")
        return None


# ══════════════════════════════════════════════════════
# الدورة الرئيسية
# ══════════════════════════════════════════════════════

def run_ai_cycle() -> Optional[Dict]:
    """دورة تحليل AI واحدة"""

    if not AI_ENABLED:
        return None

    if not ANTHROPIC_API_KEY:
        logger.warning("[AI] ANTHROPIC_API_KEY غير موجود — تعطيل AI Analyst")
        return None

    logger.info("[AI] 🤖 بدء دورة التحليل...")
    start = time.time()

    # ١. بناء السياق
    context = _build_context()

    # ٢. استدعاء Claude
    raw = _call_claude(context)
    if not raw:
        return None

    # ٣. تحليل وحفظ
    result = _parse_and_save(raw)
    if not result:
        return None

    elapsed = time.time() - start
    logger.info(f"[AI] ✅ دورة انتهت في {elapsed:.1f}s | confidence={result.get('confidence', 0):.0%}")

    # ٤. إرسال تيليغرام
    tg_msg = result.get("telegram_message", "")
    if tg_msg:
        d = result.get("directives", {})
        full_msg = (
            f"🤖 <b>تحليل Claude AI</b>\n"
            f"━━━━━━━━━━━━━━\n"
            f"{tg_msg}\n"
            f"━━━━━━━━━━━━━━\n"
            f"⚙️ score≥{d.get('score_threshold_override','?')} | "
            f"size×{d.get('size_multiplier','?')} | "
            f"entry={'✅' if d.get('entry_allowed', True) else '🚫'}\n"
            f"🎯 ثقة: {result.get('confidence', 0):.0%} | {_now().strftime('%H:%M')}"
        )
        _tg(full_msg)

    return result


# ══════════════════════════════════════════════════════
# قراءة التوجيهات (تستدعيها strategy.py)
# ══════════════════════════════════════════════════════

def get_ai_directives() -> Dict[str, Any]:
    """
    تُستدعى من strategy.py للحصول على توجيهات AI.
    إذا الملف قديم (> ساعتين) → تُرجع فارغة.
    """
    if not AI_ENABLED:
        return {}

    data = _read_json(AI_DIRECTIVES_FILE, {})
    if not data:
        return {}

    # تحقق من الصلاحية
    updated_at = data.get("updated_at")
    if updated_at:
        try:
            dt = datetime.fromisoformat(updated_at)
            age_hours = (datetime.now(dt.tzinfo) - dt).total_seconds() / 3600
            if age_hours > 2.0:
                return {}
        except: pass

    return data.get("directives", {})


# ══════════════════════════════════════════════════════
# Scheduler للـ AI
# ══════════════════════════════════════════════════════

def start_ai_scheduler():
    """يشغّل AI Analyst في thread منفصل كل ساعة"""
    import threading

    def _loop():
        # انتظر دقيقتين قبل أول دورة
        time.sleep(120)
        while True:
            try:
                run_ai_cycle()
            except Exception as e:
                logger.error(f"[AI] خطأ في الدورة: {e}")
            time.sleep(AI_INTERVAL_MIN * 60)

    t = threading.Thread(target=_loop, name="AIAnalyst", daemon=True)
    t.start()
    logger.info(f"[AI] 🤖 AI Analyst بدأ — دورة كل {AI_INTERVAL_MIN:.0f} دقيقة")
    return t


# ══════════════════════════════════════════════════════
# تشغيل مستقل للاختبار
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    print("🤖 AI Analyst — تشغيل مباشر")
    print("=" * 50)

    if "--once" in sys.argv:
        result = run_ai_cycle()
        if result:
            print("\n📋 التوجيهات:")
            print(json.dumps(result.get("directives", {}),
                             ensure_ascii=False, indent=2))
            print(f"\n💬 {result.get('telegram_message', '')}")
        else:
            print("❌ فشل التحليل — تحقق من ANTHROPIC_API_KEY")

    elif "--context" in sys.argv:
        print(_build_context())

    elif "--status" in sys.argv:
        d = get_ai_directives()
        print(json.dumps(d, ensure_ascii=False, indent=2) if d else "لا توجد توجيهات")

    else:
        print("الاستخدام:")
        print("  python ai_analyst.py --once     # تشغيل دورة واحدة")
        print("  python ai_analyst.py --context  # عرض السياق")
        print("  python ai_analyst.py --status   # التوجيهات الحالية")

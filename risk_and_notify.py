# risk_and_notify.py
from __future__ import annotations
import time, math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# ========= 1) Telegram Dedupe / Cooldown =========
_last_tg: Dict[str, float] = {}  # مفتاح → آخر إرسال

def tg_send(msg: str, key: str = "general", ttl_sec: int = 900, send_fn=None):
    """
    منع التكرار: لن يُرسل نفس المفتاح خلال ttl_sec ثانية.
    - key: مفتاح منطقي (مثل: 'risk_block.active', 'risk_block.lifted', 'smart_exit.info')
    - ttl_sec: زمن التهدئة (افتراضي 15 دقيقة)
    - send_fn: دالة الإرسال الحقيقية لديك (مثل _tg)
    """
    now = time.time()
    last = _last_tg.get(key, 0.0)
    if now - last >= ttl_sec:
        _last_tg[key] = now
        if send_fn:
            try:
                send_fn(msg)
            except Exception:
                pass
        return True
    return False

# ========= 2) Risk Block Manager =========
@dataclass
class RiskBlockConfig:
    max_daily_loss_usdt: float = 0.0      # 0 = تعطيل
    max_consec_losses: int = 0            # 0 = تعطيل
    cooloff_minutes: int = 60             # حظر زمني مبدئي
    lift_on_recovery_rr: float = 1.5      # فك الحظر إذا تحقق RR تراكمي ≥ هذا الرقم بعد الحظر
    lift_on_wins: int = 2                  # أو اذا حقق صفقتين رابحتين متتاليتين
    tg_ttl_sec: int = 900                 # تهدئة رسائل التلغرام
    enable_reason_echo: bool = True       # إظهار سبب الحظر في الرسالة

@dataclass
class RiskBlockState:
    is_active: bool = False
    reason: str = ""
    since_ts: float = 0.0
    cooloff_sec: int = 0
    post_block_rr_acc: float = 0.0
    post_block_consec_wins: int = 0

class RiskBlocker:
    """
    واجهة مبسطة لإدارة الحظر:
    - call check_and_maybe_block(...) بعد كل إغلاق صفقة أو تحديث يومي.
    - call can_trade() قبل توليد أي صفقة.
    - call on_trade_closed(pnl_usdt) عند إغلاق كل صفقة (لتحديث معايير فك الحظر).
    """
    def __init__(self, cfg: RiskBlockConfig, tg_fn=None):
        self.cfg = cfg
        self.tg_fn = tg_fn
        self.state = RiskBlockState()

        # متغيرات يومية/جلسة — اربطها بنظامك الفعلي
        self.daily_realized_pnl: float = 0.0
        self.daily_consec_losses: int = 0

    def _now(self) -> float:
        return time.time()

    def _say(self, message: str, key: str):
        tg_send(message, key=key, ttl_sec=self.cfg.tg_ttl_sec, send_fn=self.tg_fn)

    def reset_daily_counters(self):
        self.daily_realized_pnl = 0.0
        self.daily_consec_losses = 0

    def on_trade_closed(self, pnl_usdt: float, entry: float = 0.0, exit: float = 0.0, qty: float = 0.0):
        # تحديت العدادات اليومية
        self.daily_realized_pnl += pnl_usdt
        if pnl_usdt < 0:
            self.daily_consec_losses += 1
        else:
            self.daily_consec_losses = 0

        # إذا كنا تحت الحظر — راقب معايير فك الحظر
        if self.state.is_active:
            # RR تقريبي: ربح/خسارة مقسوم على مخاطرة تقديرية؛ إن لم تكن متوفرة، استخدم USDT مباشرة
            rr = pnl_usdt  # عدِّلها لو عندك المخاطرة الأساسية للصفقة
            self.state.post_block_rr_acc += max(0.0, rr)
            if pnl_usdt > 0:
                self.state.post_block_consec_wins += 1
            else:
                self.state.post_block_consec_wins = 0

            self._try_lift()

    def _try_lift(self):
        if not self.state.is_active:
            return

        time_ok = (self._now() - self.state.since_ts) >= self.state.cooloff_sec
        wins_ok = self.state.post_block_consec_wins >= self.cfg.lift_on_wins > 0
        rr_ok   = self.state.post_block_rr_acc >= self.cfg.lift_on_recovery_rr > 0

        if time_ok or wins_ok or rr_ok:
            self.state = RiskBlockState(is_active=False)
            self._say("✅ تم فك الحظر: الشروط تحققت (وقت/تعافٍ بالأداء).", key="risk_block.lifted")

    def can_trade(self) -> bool:
        if self.state.is_active:
            # رسالة خفيفة كل فترة لمنع الإزعاج
            self._say("⏸️ النظام في حالة حظر مؤقت (إدارة مخاطر).", key="risk_block.active")
            return False
        return True

    def check_and_maybe_block(self) -> Optional[str]:
        """ نادِها بعد تحديث الإحصاءات (مثلاً عند نهاية كل صفقة أو كل N دقائق). """
        if self.state.is_active:
            self._try_lift()
            return None

        reasons = []
        if self.cfg.max_daily_loss_usdt and self.daily_realized_pnl <= -abs(self.cfg.max_daily_loss_usdt):
            reasons.append(f"تجاوز حد الخسارة اليومي: {self.daily_realized_pnl:.2f} USDT")

        if self.cfg.max_consec_losses and self.daily_consec_losses >= self.cfg.max_consec_losses:
            reasons.append(f"عدد خسائر متتالية: {self.daily_consec_losses}")

        if reasons:
            self.state.is_active = True
            self.state.reason = " | ".join(reasons)
            self.state.since_ts = self._now()
            self.state.cooloff_sec = int(self.cfg.cooloff_minutes * 60)
            self.state.post_block_rr_acc = 0.0
            self.state.post_block_consec_wins = 0

            reason_txt = f"⏸️ تم تفعيل الحظر المؤقت.\n🧾 السبب: {self.state.reason}" if self.cfg.enable_reason_echo else "⏸️ تم تفعيل الحظر المؤقت."
            self._say(reason_txt + f"\n⏳ مدة التهدئة: {self.cfg.cooloff_minutes} دقيقة.", key="risk_block.trigger")
            return self.state.reason
        return None


# ========= 3) سلامة المؤشرات (ema21) =========
def ensure_ema21(df, period: int = 21):
    """
    يضمن وجود عمود ema21 في df.
    - إذا كان df يحتوي ema21 يمرّ.
    - إذا لا، يُحسب EMA بسيط بدون مكتبات خارجية.
    """
    if "ema21" in df.columns:
        return df
    if "close" not in df.columns:
        return df

    alpha = 2.0 / (period + 1.0)
    ema_vals = []
    ema = None
    for c in df["close"].tolist():
        ema = c if ema is None else (alpha * c + (1 - alpha) * ema)
        ema_vals.append(ema)
    df = df.copy()
    df["ema21"] = ema_vals
    return df

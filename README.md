# crypto-wat

بوت تداول تلقائي على منصة Bybit باستخدام استراتيجية Breakout + Volume

## الإعداد

1. ضع مفاتيح API في ملف `config.py`
2. تأكد من تنصيب المكتبات: `pip install -r requirements.txt`
3. ارفع المشروع على GitHub
4. أنشئ Background Worker على Render مع أمر التشغيل:
   `python crypto-wat.py`

## ملاحظات

- هذا البوت ينفذ صفقات حقيقية تلقائيًا، استخدمه بحذر.
- يفضل تجربة مفاتيح API على حساب تجريبي (Testnet) أولًا.
- إضافة أوامر وقف خسارة وهدف ربح ستتم لاحقًا.

# main.py

import time
from strategy import check_signal, execute_buy, manage_position, load_position
from telegram_bot import send_message

# ✅ رموز OKX بصيغة صحيحة
SYMBOLS = [
    "CRV-USDT",    # Store of Value
    "GALA-USDT",   # Gaming
    "BNB-USDT",    # Exchange Chain
    "SOL-USDT",    # Layer 1
    "ADA-USDT",    # Layer 1
    "AVAX-USDT",   # Layer 1
    "ATOM-USDT",   # Interoperability
    "DOT-USDT",    # Parachains
    "PEPE-USDT",
    "LINK-USDT",   # Oracle
    "UNI-USDT",    # DeFi
    "AAVE-USDT",   # DeFi Lending
    "SUSHI-USDT",  # DEX
    "LDO-USDT",    # Staking
    "INJ-USDT",    # DeFi Trading
    "XRP-USDT",   
    "FET-USDT",    # AI
    "APE-USDT",    # Metaverse
    "TIA-USDT",    # Modular Blockchain
    "OP-USDT",     # Optimism (L2)
]

def run_bot():
    send_message("🤖 بدأ تشغيل بوت التداول اللحظي!")

    while True:
        try:
            for symbol in SYMBOLS:
                # إدارة الصفقة إذا كانت موجودة
                if load_position(symbol):
                    manage_position(symbol, send_message)
                    continue

                # إذا لا يوجد صفقة، افحص للإشارة
                signal = check_signal(symbol)

                if signal == "buy":
                    order, msg = execute_buy(symbol)
                    if order:
                        send_message(msg)

            time.sleep(60)  # انتظر دقيقة واحدة قبل الجولة التالية

        except Exception as e:
            send_message(f"❌ حدث خطأ: {e}")
            time.sleep(30)

if __name__ == "__main__":
    run_bot()

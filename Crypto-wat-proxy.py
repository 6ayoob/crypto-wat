import time
from binance_api import get_balance

def main():
    while True:
        balance = get_balance("USDT")
        print(f"🟢 الرصيد المتاح: {balance} USDT")
        time.sleep(60)  # انتظر 60 ثانية

if __name__ == "__main__":
    main()

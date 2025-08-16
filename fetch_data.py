from binance_historical_data import BinanceDataDumper
from datetime import date

def fetch_data():
    # 初始化 dumper
    dumper = BinanceDataDumper(
        path_dir_where_to_dump="./data",   # 数据保存路径
        asset_class="um",                  # um = USDT-M Futures 合约
        data_type="klines",                # K线
        data_frequency="1m"                # 1 分钟级别
    )

    # 拉 2023 年 8 月 1 日到 8 月 3 日的数据
    dumper.dump_data(
        tickers=["BNBUSDT"],
        date_start=date(2025, 7, 1),
        date_end=date(2025, 7, 31)
    )

    print("✅ 数据下载完成！文件在 ./data 目录下")

# 保护 Windows 环境启动问题
if __name__ == "__main__":
    fetch_data()

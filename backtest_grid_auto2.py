# grid_order_backtester.py
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


class GridOrderBacktester:
    def __init__(self, df, grid_spacing, config):
        self.df = df.reset_index(drop=True)
        self.grid_spacing = grid_spacing
        self.config = config

        self.balance = config["initial_balance"]
        self.max_drawdown = config["max_drawdown"]
        self.fee = config["fee_pct"]
        self.direction = config.get("direction", "both")
        self.leverage = config["leverage"]

        self.long_positions = []
        self.short_positions = []
        self.trade_history = []
        self.equity_curve = []
        self.max_equity = self.balance

        self.orders = {"long": [], "short": []}
        self.last_refresh_time = None  # 用来记录上次刷新挂单的时间
        self._init_orders(self.df['close'].iloc[0])

    def _init_orders(self, price):
        if self.direction in ["long", "both"]:
            self._place_long_orders(price)
        if self.direction in ["short", "both"]:
            self._place_short_orders(price)

    def _place_long_orders(self, center_price):
        self.orders["long"] = [
            (center_price - center_price * self.grid_spacing, "BUY"),
            (center_price + center_price * self.grid_spacing, "SELL")
        ]

    def _place_short_orders(self, center_price):
        self.orders["short"] = [
            (center_price + center_price * self.grid_spacing, "SELL_SHORT"),
            (center_price - center_price * self.grid_spacing, "COVER_SHORT")
        ]

    def _update_orders_after_trade(self, side, fill_price):
        if side == "long":
            self._place_long_orders(fill_price)
        elif side == "short":
            self._place_short_orders(fill_price)

    def _refresh_orders_if_needed(self, price, current_time):
        # 用数据时间判断挂单刷新（而不是系统时间）
        if not self.long_positions and not self.short_positions:
            if self.last_refresh_time is None or (current_time - self.last_refresh_time) >= timedelta(
                    minutes=self.config["grid_refresh_interval"]):
                self._init_orders(price)
                self.last_refresh_time = current_time

    def _calculate_unrealized_pnl(self, price):
        # 多头：市值 - 成本
        # long_pnl = sum(qty * (price - entry_price) for entry_price, qty, _ in self.long_positions)
        long_pnl = sum((price - entry_price) * qty for entry_price, qty, _ in self.long_positions)
        # 空头：(开仓价-现价)*数量
        # short_pnl = sum(qty * (entry_price - price) for entry_price, qty, _ in self.short_positions)
        short_pnl = sum((entry_price - price) * qty for entry_price, qty, _ in self.short_positions)
        return long_pnl + short_pnl

    def run(self):
        for _, row in self.df.iterrows():
            # 检查最大持仓限制（多头+空头）
            if len(self.short_positions) + len(self.long_positions) >= self.config["max_positions"]:
                print("⚠️ 达到最大持仓限制")
                break

            price = row['close']
            timestamp = row['open_time']
            effective_order_value = self.config["order_value"] * self.leverage

            # 刷新挂单价格（如果没持仓）
            self._refresh_orders_if_needed(price, timestamp)

            # 计算已用保证金和可用保证金
            used_margin = sum(pos[2] for pos in self.long_positions + self.short_positions)
            available_margin = self.balance - used_margin

            # LONG SIDE
            if self.direction in ["long", "both"]:
                for order_price, action in self.orders["long"]:
                    if action == "BUY" and price <= order_price:
                        # 计算开仓参数
                        qty = effective_order_value / price
                        notional_value = qty * price
                        margin_required = notional_value / self.leverage
                        fee_cost = qty * price * (self.fee / 2)  # 开仓手续费

                        # 检查资金是否足够
                        if (margin_required + fee_cost) > available_margin:
                            continue

                        # 执行开仓
                        self.balance -= (margin_required + fee_cost)
                        self.long_positions.append((price, qty, margin_required))

                        # 记录交易
                        unrealized_pnl = self._calculate_unrealized_pnl(price)
                        total_equity = self.balance + unrealized_pnl
                        self.trade_history.append((
                            timestamp, "BUY", price, qty, "LONG",
                            0.0, fee_cost, 0.0, unrealized_pnl, total_equity
                        ))

                        # 更新订单
                        self._update_orders_after_trade("long", price)
                        break

                    elif action == "SELL" and self.long_positions and price >= order_price:
                        # 平仓操作
                        entry_price, qty, margin_required = self.long_positions.pop(0)
                        fee_cost = qty * price * (self.fee / 2)  # 平仓手续费
                        gross_pnl = (price - entry_price) * qty
                        net_pnl = gross_pnl - fee_cost

                        # 资金变动：返还保证金 + 净盈亏
                        self.balance += margin_required + net_pnl

                        # 记录交易
                        unrealized_pnl = self._calculate_unrealized_pnl(price)
                        total_equity = self.balance + unrealized_pnl
                        self.trade_history.append((
                            timestamp, "SELL", price, qty, "LONG",
                            net_pnl, fee_cost, gross_pnl, unrealized_pnl, total_equity
                        ))

                        # 更新订单
                        self._update_orders_after_trade("long", price)
                        break

            # SHORT SIDE
            if self.direction in ["short", "both"]:
                for order_price, action in self.orders["short"]:
                    if action == "SELL_SHORT" and price >= order_price:
                        # 计算开仓参数
                        qty = effective_order_value / price
                        notional_value = qty * price
                        margin_required = notional_value / self.leverage
                        fee_cost = qty * price * (self.fee / 2)  # 开仓手续费

                        # 检查资金是否足够
                        if (margin_required + fee_cost) > available_margin:
                            continue

                        # 执行开仓
                        self.balance -= (margin_required + fee_cost)
                        self.short_positions.append((price, qty, margin_required))

                        # 记录交易
                        unrealized_pnl = self._calculate_unrealized_pnl(price)
                        total_equity = self.balance + unrealized_pnl
                        self.trade_history.append((
                            timestamp, "SELL_SHORT", price, qty, "SHORT",
                            0.0, fee_cost, 0.0, unrealized_pnl, total_equity
                        ))

                        print(f"📉 [做空开仓] 时间: {timestamp} | 价格: {price:.4f} | 数量: {qty:.4f} | "
                              f"冻结保证金: {margin_required:.2f} | 可用余额: {self.balance:.2f}")
                        self._update_orders_after_trade("short", price)
                        break

                    elif action == "COVER_SHORT" and self.short_positions and price <= order_price:
                        # 平仓操作
                        entry_price, qty, margin_required = self.short_positions.pop(0)
                        fee_cost = qty * price * (self.fee / 2)  # 平仓手续费
                        gross_pnl = (entry_price - price) * qty
                        net_pnl = gross_pnl - fee_cost

                        # 资金变动：返还保证金 + 净盈亏
                        self.balance += margin_required + net_pnl

                        # 记录交易
                        unrealized_pnl = self._calculate_unrealized_pnl(price)
                        total_equity = self.balance + unrealized_pnl
                        self.trade_history.append((
                            timestamp, "COVER_SHORT", price, qty, "SHORT",
                            net_pnl, fee_cost, gross_pnl, unrealized_pnl, total_equity
                        ))

                        print(f"📈 [做空平仓] 时间: {timestamp} | 开仓价: {entry_price:.4f} | 平仓价: {price:.4f} | "
                              f"数量: {qty:.4f} | 盈亏: {gross_pnl:.2f} | "
                              f"返还保证金: {margin_required:.2f} | 账户余额: {self.balance:.2f}")
                        self._update_orders_after_trade("short", price)
                        break

            # 计算当前盈亏和净值
            long_pnl = sum((price - entry_price) * qty for entry_price, qty, _ in self.long_positions)
            short_pnl = sum((entry_price - price) * qty for entry_price, qty, _ in self.short_positions)
            unrealized_pnl = long_pnl + short_pnl
            equity = self.balance + unrealized_pnl

            # 更新最大净值和回撤
            self.max_equity = max(self.max_equity, equity)
            drawdown = 1 - (equity / self.max_equity) if self.max_equity > 0 else 0

            # 计算已实现盈亏
            realized_pnl_so_far = sum(trade[5] for trade in self.trade_history)

            # 记录净值曲线
            self.equity_curve.append((
                timestamp, price, equity,
                realized_pnl_so_far, unrealized_pnl
            ))

            # 检查最大回撤
            if drawdown >= self.max_drawdown:
                print(f"⚠️ 达到最大回撤限制 {drawdown * 100:.2f}%，停止回测")
                break

            # # 打印持仓状态
            # if self.long_positions or self.short_positions:
            #     print(f"\n🔄 [持仓状态] 时间: {timestamp} | 当前价格: {price:.4f}")
            #     if self.short_positions:
            #         short_avg_price = sum(p[0] * p[1] for p in self.short_positions) / sum(
            #             p[1] for p in self.short_positions)
            #         print(
            #             f"    🏷️ 做空持仓: {len(self.short_positions)}笔 | 均价: {short_avg_price:.4f} | 浮动盈亏: {short_pnl:.2f}")
            #     if self.long_positions:
            #         long_avg_price = sum(p[0] * p[1] for p in self.long_positions) / sum(
            #             p[1] for p in self.long_positions)
            #         print(
            #             f"    🏷️ 做多持仓: {len(self.long_positions)}笔 | 均价: {long_avg_price:.4f} | 浮动盈亏: {long_pnl:.2f}")
            #     print(f"    💰 账户净值: {equity:.2f} (余额: {self.balance:.2f} | 浮动盈亏: {unrealized_pnl:.2f})\n")

        return self.summary(price)

    def summary(self, final_price):
        long_pnl = sum((final_price - entry_price) * qty for entry_price, qty, _ in self.long_positions)
        short_pnl = sum((entry_price - final_price) * qty for entry_price, qty, _ in self.short_positions)
        # ✅ 新增：浮动盈亏
        unrealized_pnl = long_pnl + short_pnl

        # ✅ 已实现盈亏（成交后记录的 net_pnl）
        realized_pnl = sum(row[5] for row in self.trade_history if row[5] != 0.0)

        final_equity = self.balance + unrealized_pnl

        return {
            "final_equity": final_equity,
            "return_pct": (final_equity - self.config["initial_balance"]) / self.config["initial_balance"],
            "max_drawdown": 1 - final_equity / self.max_equity,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,  # ✅ 新增字段
            "total_pnl": realized_pnl + unrealized_pnl,  # ✅ 新增字段
            "trades": len(self.trade_history),
            "direction": self.direction
        }

    def export_trades(self, filename="grid_orders_trades.csv"):
        df = pd.DataFrame(self.trade_history, columns=[
            "time", "action", "price", "quantity", "direction", "pnl", "fee_cost", "gross_pnl", "unrealized_pnl", "total_equity"
        ])

        df.to_csv(filename, index=False)

    # 修改 export_positions 方法：
    def export_positions(self, filename="positions_snapshot.csv"):
        # 多头持仓
        long_df = pd.DataFrame(self.long_positions,
                               columns=["entry_price", "quantity", "margin"])
        long_df["type"] = "LONG"

        # 空头持仓
        short_df = pd.DataFrame(self.short_positions,
                                columns=["entry_price", "quantity", "margin"])
        short_df["type"] = "SHORT"

        pd.concat([long_df, short_df]).to_csv(filename, index=False)

    def export_equity_curve(self, filename="equity_curve.csv"):
        df = pd.DataFrame(self.equity_curve, columns=[
            "time", "price", "equity", "realized_pnl", "unrealized_pnl"
        ])
        df.to_csv(filename, index=False)


# -------- 🔁 回测框架（可选） -------- #

def load_data_for_date(date_str):
    try:
        path = f"data/futures/um/daily/klines/BNBUSDT/1m/BNBUSDT-1m-{date_str}.csv"
        print(f"读取文件: {path}")
        df = pd.read_csv(path)

        if df.empty or df.columns.size == 0:
            raise ValueError("文件为空或无列")

        df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')

        return df
    except Exception as e:
        print(f"❌ 读取失败 {date_str}: {e}")
        return None


def run_backtest_for_params(spacing):
    current = CONFIG["start_date"]
    all_data = []
    while current <= CONFIG["end_date"]:
        df = load_data_for_date(current.strftime("%Y-%m-%d"))
        if df is not None:
            all_data.append(df)
        current += timedelta(days=1)

    if not all_data:
        return None

    full_df = pd.concat(all_data, ignore_index=True)
    bt = GridOrderBacktester(full_df, spacing, CONFIG)
    result = bt.run()
    return result


def visualize_results(df_results):
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_results, x="spacing", y="return_pct", palette="Blues_d")
    plt.title("Return by Grid Spacing")
    plt.xlabel("Grid Spacing")
    plt.ylabel("Return (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_equity_curve(bt):
    #  图表函数
    # 读取 equity 曲线，包括浮动盈亏
    df = pd.DataFrame(bt.equity_curve, columns=["time", "price", "equity", "realized_pnl", "unrealized_pnl"])
    df["time"] = pd.to_datetime(df["time"], errors='coerce')

    # 读取交易记录
    trades_df = pd.DataFrame(bt.trade_history, columns=[
        "time", "action", "price", "quantity", "direction", "pnl", "fee_cost", "gross_pnl", "unrealized_pnl", "total_equity"
    ])
    trades_df["time"] = pd.to_datetime(trades_df["time"], errors='coerce')

    # 创建3个子图，高度比例调整为3:1:1
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})

    # --- 上图：价格与买卖信号 ---
    ax1.plot(df["time"], df["price"], label="Price", color="blue", alpha=0.5)

    if not trades_df.empty:
        # 做多信号
        buy_trades = trades_df[trades_df["action"] == "BUY"]
        sell_trades = trades_df[trades_df["action"] == "SELL"]
        ax1.scatter(buy_trades["time"], buy_trades["price"], marker="^", color="green", label="BUY", s=60, zorder=3)
        ax1.scatter(sell_trades["time"], sell_trades["price"], marker="v", color="red", label="SELL", s=60, zorder=3)

        # 做空信号
        sell_short_trades = trades_df[trades_df["action"] == "SELL_SHORT"]
        cover_short_trades = trades_df[trades_df["action"] == "COVER_SHORT"]
        ax1.scatter(sell_short_trades["time"], sell_short_trades["price"], marker="v", color="purple",
                    label="SELL_SHORT", s=60, zorder=3)
        ax1.scatter(cover_short_trades["time"], cover_short_trades["price"], marker="^", color="orange",
                    label="COVER_SHORT", s=60, zorder=3)

    ax1.set_ylabel("Price", color="blue")
    ax1.set_title("Price Curve with Trade Signals")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # --- 中图：净值与盈亏曲线 ---
    ax2.plot(df["time"], df["equity"], color="green", label="Equity")
    ax2.plot(df["time"], df["realized_pnl"], color="blue", linestyle="--", label="Realized PnL")
    ax2.plot(df["time"], df["unrealized_pnl"], color="red", linestyle=":", label="Unrealized PnL")
    ax2.set_ylabel("Account Value")
    ax2.set_title("Equity, Realized and Unrealized PnL Over Time")
    ax2.legend(loc="upper left")
    ax2.grid(True)

    # --- 下图：Total Equity (从交易记录中获取) ---
    if not trades_df.empty:
        ax3.plot(trades_df["time"], trades_df["total_equity"], color="purple", label="Total Equity")
        ax3.set_ylabel("Total Equity")
        ax3.set_title("Total Account Equity Over Time")
        ax3.legend(loc="upper left")
        ax3.grid(True)

        # 时间格式
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()


def grid_search_backtest():
    results = []
    best_result = None
    best_params = None

    for spacing in CONFIG["grid_spacing_range"]:
        print(f"🚀 回测 Grid Spacing: {spacing}")
        result = run_backtest_for_params(spacing)
        if result:
            results.append({
                "spacing": spacing,
                **result
            })
            if not best_result or result["return_pct"] > best_result["return_pct"]:
                best_result = result
                best_params = spacing

    df_results = pd.DataFrame(results)
    df_results.to_csv("grid_order_results.csv", index=False)

    if best_params:
        print("\n✅ 最优参数:")
        print(f"Grid Spacing: {best_params}")
        visualize_results(df_results)

        current = CONFIG["start_date"]
        all_data = []
        while current <= CONFIG["end_date"]:
            df = load_data_for_date(current.strftime("%Y-%m-%d"))
            if df is not None:
                all_data.append(df)
            current += timedelta(days=1)

        full_df = pd.concat(all_data, ignore_index=True)
        best_bt = GridOrderBacktester(full_df, best_params, CONFIG)
        best_bt.run()
        # ✅ 提前导出当前持仓（run 后立刻）
        best_bt.export_positions("best_grid_positions.csv")
        plot_equity_curve(best_bt)
    else:
        print("❌ 没有找到任何可用的参数结果")

    best_bt.export_trades("best_grid_trade.csv")
    best_bt.export_equity_curve("best_grid_equity_curve.csv")
    best_bt.export_positions("best_grid_positions.csv")

    return df_results


# -------- 🧪 配置示例 -------- #

CONFIG = {
    "initial_balance": 1000,
    "order_value": 10,  # 每次固定用 50 美金下单
    "max_drawdown": 0.9,   # 超过该回撤比例时停止回测
    "max_positions": 100,  # 最大持仓数
    "fee_pct": 0.0002,  # 手续费万二
    "direction": "long",  # or "long" / "short" 网格方向
    "leverage": 1,
    "start_date": datetime(2025, 7, 1),
    "end_date": datetime(2025, 7, 31),
    "grid_spacing_range": [0.003, 0.004, 0.001],  # 表示 0.5%, 1%, 1.5% 间距
    "grid_refresh_interval": 500  # 每 10 分钟刷新一次挂单
}

# -------- 🔁 启动 -------- #
if __name__ == "__main__":
    grid_search_backtest()

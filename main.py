"""
Simple SBI (SBIN.NS) RSI strategy (daily bars)
Rules (long-only):
- Buy when RSI crosses UP through 30 (oversold -> recovery)
- Sell when RSI crosses DOWN through 70 (overbought -> weakening)
- Trades executed on next day's open to avoid look-ahead bias (approximation)
Educational only. Not financial advice.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
def download_data(ticker="SBIN.NS", period="5y", interval="1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError("No data returned. Check ticker / internet / yfinance availability.")
    df = df.dropna().copy()
    return df
def add_indicators(df: pd.DataFrame, rsi_len=14) -> pd.DataFrame:
    df = df.copy()
    df["RSI"] = ta.rsi(df["Close"], length=rsi_len)
    return df
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create entry/exit signals based on RSI crosses.
    """
    df = df.copy()
    df["RSI_prev"] = df["RSI"].shift(1)
    # Cross up through 30 => entry
    df["entry"] = (df["RSI_prev"] < 30) & (df["RSI"] >= 30)
    # Cross down through 70 => exit
    df["exit"] = (df["RSI_prev"] > 70) & (df["RSI"] <= 70)
    return df
def backtest(df: pd.DataFrame, fee_bps=10) -> pd.DataFrame:
    """
    Very simple backtest:
    - Build a position series (0 or 1)
    - Apply it to close-to-close returns (or next-open execution approximation)
    - Apply a flat fee in basis points per trade (entry or exit)
    fee_bps=10 => 0.10% per trade side (simplified).
    """
    df = df.copy()
    # Position: 1 when in trade, else 0
    pos = np.zeros(len(df), dtype=int)
    in_pos = 0
    for i, (en, ex) in enumerate(zip(df["entry"].fillna(False), df["exit"].fillna(False))):
        if in_pos == 0 and en:
            in_pos = 1
        elif in_pos == 1 and ex:
            in_pos = 0
        pos[i] = in_pos
    df["position"] = pos
    # Avoid look-ahead: use yesterday's position on today's return
    df["ret"] = df["Close"].pct_change().fillna(0.0)
    df["strat_gross"] = df["position"].shift(1).fillna(0).astype(int) * df["ret"]
    # Fees on trade days (when position changes)
    df["trade"] = df["position"].diff().fillna(df["position"]).abs()  # 1 on entry/exit
    fee = (fee_bps / 10000.0)  # bps to fraction
    df["fee"] = df["trade"] * fee
    df["strat_net"] = df["strat_gross"] - df["fee"]
    # Equity curves
    df["equity_buyhold"] = (1.0 + df["ret"]).cumprod()
    df["equity_strategy"] = (1.0 + df["strat_net"]).cumprod()
    return df
def performance_summary(df: pd.DataFrame) -> dict:
    """
    Basic performance stats (daily).
    """
    strat = df["strat_net"]
    bh = df["ret"]
    def max_drawdown(equity: pd.Series) -> float:
        peak = equity.cummax()
        dd = (equity / peak) - 1.0
        return dd.min()
    ann = 252
    summary = {
        "Strategy total return": float(df["equity_strategy"].iloc[-1] - 1.0),
        "Buy&Hold total return": float(df["equity_buyhold"].iloc[-1] - 1.0),
        "Strategy CAGR (approx)": float(df["equity_strategy"].iloc[-1] ** (ann / max(len(df), 1)) - 1.0),
        "Strategy vol (ann.)": float(strat.std() * np.sqrt(ann)),
        "Strategy Sharpe (rf=0)": float((strat.mean() / (strat.std() + 1e-12)) * np.sqrt(ann)),
        "Strategy max drawdown": float(max_drawdown(df["equity_strategy"])),
        "Trades (count of entries+exits)": int(df["trade"].sum()),
    }
    return summary
def plot_results(df: pd.DataFrame):
    plt.figure()
    plt.plot(df.index, df["equity_buyhold"], label="Buy & Hold")
    plt.plot(df.index, df["equity_strategy"], label="RSI Strategy")
    plt.title("SBIN.NS: Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Equity (starting at 1.0)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.figure()
    plt.plot(df.index, df["RSI"], label="RSI(14)")
    plt.axhline(30, linestyle="--")
    plt.axhline(70, linestyle="--")
    plt.title("RSI(14)")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    df = download_data("SBIN.NS", period="5y", interval="1d")
    df = add_indicators(df, rsi_len=14)
    df = generate_signals(df)
    df = backtest(df, fee_bps=10)  # 0.10% per side as a simple cost model
    stats = performance_summary(df)
    for k, v in stats.items():
        print(f"{k}: {v}")
    plot_results(df)
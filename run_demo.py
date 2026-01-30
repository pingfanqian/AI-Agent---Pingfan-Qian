#!/usr/bin/env python3
"""
run_demo.py — Reproducible demo script for the Technical Analyst Agent notebook.

What it does (end-to-end):
1) Downloads OHLCV data via yfinance
2) Computes technical indicators (SMA, RSI, MACD, Bollinger Bands, ATR)
3) Builds a simple "risk overlay" exposure process (trend + vol regime + optional vol targeting; weekly rebalance)
4) Backtests with transaction costs and slippage
5) Exports key artifacts (CSV/JSON/PNG)
6) Optionally generates an LLM trade note (OpenAI) if OPENAI_API_KEY is set

Example:
  python run_demo.py --ticker GOOGL --period 10y --output_dir outputs --trade_note

Requirements:
  pip install -U yfinance pandas numpy matplotlib openai
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional dependency for trade note generation
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    import yfinance as yf  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("Missing dependency: yfinance. Install with `pip install yfinance`.") from e


# -----------------------------
# Helpers
# -----------------------------
def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns (common in yfinance).

    yfinance may return columns like:
      MultiIndex([('Open','GOOGL'), ('High','GOOGL'), ...])
    or sometimes the levels are swapped. This function picks the level that
    contains OHLCV field names. If it cannot identify it, it falls back to
    joining levels with underscores.
    """
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        ohlc = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        best_level = None
        best_hits = -1
        for i in range(out.columns.nlevels):
            lvl = [str(x).strip() for x in out.columns.get_level_values(i)]
            hits = sum(v in ohlc for v in lvl)
            if hits > best_hits:
                best_hits = hits
                best_level = i

        if best_level is not None and best_hits > 0:
            out.columns = [str(x).strip() for x in out.columns.get_level_values(best_level)]
        else:
            out.columns = [
                "_".join([str(x).strip() for x in tup if str(x).strip() != ""])
                for tup in out.columns.to_list()
            ]
    else:
        out.columns = [str(c).strip() for c in out.columns]

    return out


def pick_col(df: pd.DataFrame, names: list[str]) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise ValueError(f"Missing columns. Expected one of: {names}. Found: {df.columns.tolist()}")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Indicators
# -----------------------------
def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    prices = pd.to_numeric(prices, errors="coerce")
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    prices = pd.to_numeric(prices, errors="coerce")
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    prices = pd.to_numeric(prices, errors="coerce")
    mid = prices.rolling(window, min_periods=window).mean()
    std = prices.rolling(window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window, min_periods=window).mean()
    return atr


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = flatten_cols(df).copy()
    close_col = pick_col(df, ["Adj Close", "Close", "close", "adj_close", "adj close"])
    high_col = pick_col(df, ["High", "high"])
    low_col = pick_col(df, ["Low", "low"])

    close = pd.to_numeric(df[close_col], errors="coerce")
    df["SMA_20"] = close.rolling(20, min_periods=20).mean()
    df["SMA_50"] = close.rolling(50, min_periods=50).mean()
    df["SMA_200"] = close.rolling(200, min_periods=200).mean()
    df["RSI"] = calculate_rsi(close, 14)

    macd_line, macd_signal, macd_hist = calculate_macd(close)
    df["MACD"] = macd_line
    df["MACD_signal"] = macd_signal
    df["MACD_hist"] = macd_hist

    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    df["BB_upper"] = bb_upper
    df["BB_middle"] = bb_mid
    df["BB_lower"] = bb_lower

    df["ATR_14"] = calculate_atr(df[high_col], df[low_col], close, 14)

    return df


# -----------------------------
# Risk overlay exposure (from notebook cell "GENERATE TRADING SIGNALS (OPTIMIZED: RISK OVERLAY)")
# -----------------------------
@dataclass
class RiskOverlayParams:
    SMA_SLOW: int = 200
    VOL_LOOKBACK: int = 20
    VOL_Q: float = 0.70
    REB_FREQ: str = "W-FRI"
    VOL_TARGET_ANN: Optional[float] = 0.18  # set None to disable vol targeting
    MAX_LEV: float = 1.25

    # Exposure scalars by regime
    EXP_BULL_LOWVOL: float = 1.10
    EXP_BULL_HIGHVOL: float = 0.60
    EXP_BEAR_LOWVOL: float = 0.30
    EXP_BEAR_HIGHVOL: float = 0.00


def apply_risk_overlay(df: pd.DataFrame, params: RiskOverlayParams) -> pd.DataFrame:
    df = flatten_cols(df).copy()

    close_col = pick_col(df, ["Adj Close", "Close", "close", "adj_close", "adj close"])
    close = pd.to_numeric(df[close_col], errors="coerce")

    df["ret"] = close.pct_change().fillna(0.0)
    df["SMA_200"] = close.rolling(params.SMA_SLOW, min_periods=params.SMA_SLOW).mean()

    # Trend regime: 1=bull, 0=bear
    df["trend"] = (close > df["SMA_200"]).astype(int)

    # Vol regime: compare current vol vs rolling quantile of vol
    vol_d = df["ret"].rolling(params.VOL_LOOKBACK, min_periods=params.VOL_LOOKBACK).std()
    df["vol_ann"] = (vol_d * np.sqrt(252)).replace(0, np.nan)

    vol_thr = df["vol_ann"].rolling(252, min_periods=60).quantile(params.VOL_Q)
    df["high_vol"] = (df["vol_ann"] > vol_thr).astype(int)  # 1=high vol

    def regime_scalar(trend: int, high_vol: int) -> float:
        if trend == 1 and high_vol == 0:
            return params.EXP_BULL_LOWVOL
        if trend == 1 and high_vol == 1:
            return params.EXP_BULL_HIGHVOL
        if trend == 0 and high_vol == 0:
            return params.EXP_BEAR_LOWVOL
        return params.EXP_BEAR_HIGHVOL

    df["regime_scalar"] = [regime_scalar(int(t), int(hv)) for t, hv in zip(df["trend"], df["high_vol"])]

    # Optional vol targeting (stabilize risk). If disabled -> base_lev=1.
    if params.VOL_TARGET_ANN is None:
        df["base_lev"] = 1.0
    else:
        df["base_lev"] = (params.VOL_TARGET_ANN / df["vol_ann"]).clip(upper=params.MAX_LEV).fillna(0.0)

    # Target exposure before rebalance
    df["target_exposure"] = (df["base_lev"] * df["regime_scalar"]).clip(0.0, params.MAX_LEV)

    # Weekly rebalance: only change position on rebalance dates
    reb_dates = df.resample(params.REB_FREQ).last().index
    df["rebalance_flag"] = df.index.isin(reb_dates).astype(int)

    pos = np.zeros(len(df), dtype=float)
    for i in range(1, len(df)):
        pos[i] = pos[i - 1]
        if df["rebalance_flag"].iloc[i] == 1:
            pos[i] = float(df["target_exposure"].iloc[i])

    df["position"] = pos
    df["position_lag"] = df["position"].shift(1).fillna(0.0)

    return df


# -----------------------------
# Backtest
# -----------------------------
@dataclass
class BacktestParams:
    cost_bps: float = 5.0
    slippage_bps: float = 2.0


@dataclass
class BacktestSummary:
    total_return: float
    annualized_return: float
    sharpe: float
    max_drawdown: float
    num_rebalances: int
    bh_total_return: float
    start_date: str
    end_date: str
    n_obs: int


def run_backtest(df: pd.DataFrame, bt_params: BacktestParams) -> Tuple[pd.DataFrame, BacktestSummary]:
    df = df.copy()

    # Transaction cost on turnover (change in exposure)
    df["turnover"] = (df["position"].fillna(0.0) - df["position_lag"].fillna(0.0)).abs()
    df["tcost"] = df["turnover"] * ((bt_params.cost_bps + bt_params.slippage_bps) / 10_000.0)

    # Strategy daily return (no lookahead): use lagged position
    df["BT_ret_net"] = df["position_lag"] * df["ret"] - df["tcost"]
    df["BT_equity"] = (1.0 + df["BT_ret_net"]).cumprod()
    df["equity_bh"] = (1.0 + df["ret"]).cumprod()

    # Drawdown
    df["dd_strat"] = df["BT_equity"] / df["BT_equity"].cummax() - 1.0
    df["dd_bh"] = df["equity_bh"] / df["equity_bh"].cummax() - 1.0

    r = df["BT_ret_net"].dropna()
    years = len(r) / 252.0 if len(r) else np.nan
    total = float(df["BT_equity"].iloc[-1] - 1.0) if len(df) else np.nan
    bh_total = float(df["equity_bh"].iloc[-1] - 1.0) if len(df) else np.nan
    ann = float((1.0 + total) ** (1.0 / years) - 1.0) if np.isfinite(years) and years > 0 else np.nan
    sharpe = float((r.mean() / r.std()) * np.sqrt(252)) if r.std() != 0 else np.nan
    mdd = float(df["dd_strat"].min()) if "dd_strat" in df.columns else np.nan
    num_reb = int((df.get("rebalance_flag", pd.Series(index=df.index, data=0)) > 0).sum())

    summary = BacktestSummary(
        total_return=total,
        annualized_return=ann,
        sharpe=sharpe,
        max_drawdown=mdd,
        num_rebalances=num_reb,
        bh_total_return=bh_total,
        start_date=str(df.index.min().date()) if len(df) else "",
        end_date=str(df.index.max().date()) if len(df) else "",
        n_obs=int(len(df)),
    )
    return df, summary


# -----------------------------
# Exports
# -----------------------------
def build_trade_log(df: pd.DataFrame) -> pd.DataFrame:
    """Simple entry/exit log from exposure changes (0 -> >0 is entry; >0 -> 0 is exit)."""
    if "position_lag" not in df.columns:
        return pd.DataFrame(columns=["entry_date", "exit_date", "entry_exposure", "exit_exposure"])

    ex = pd.to_numeric(df["position_lag"], errors="coerce").fillna(0.0)
    prev = ex.shift(1).fillna(0.0)

    entries = df.index[(ex > 0) & (prev == 0)]
    exits = df.index[(ex == 0) & (prev > 0)]

    # If last trade is still open, close it at last date
    if len(entries) > len(exits) and len(df):
        exits = exits.append(pd.Index([df.index[-1]]))

    n = min(len(entries), len(exits))
    trade_log = pd.DataFrame(
        {
            "entry_date": entries[:n].astype(str),
            "exit_date": exits[:n].astype(str),
            "entry_exposure": ex.loc[entries[:n]].values if n > 0 else [],
            "exit_exposure": ex.loc[exits[:n]].values if n > 0 else [],
        }
    )
    return trade_log


def export_artifacts(
    df: pd.DataFrame,
    summary: BacktestSummary,
    output_dir: Path,
    ticker: str,
) -> None:
    safe_mkdir(output_dir)
    ticker_str = ticker.lower()

    # Full dataset
    df.to_csv(output_dir / f"{ticker_str}_data_full.csv")

    # "Signals" slice similar to notebook export
    cols_pref = [
        "Adj Close", "Close", "SMA_20", "SMA_50", "SMA_200", "RSI", "MACD", "MACD_signal", "MACD_hist",
        "BB_upper", "BB_middle", "BB_lower", "ATR_14",
        "trend", "high_vol", "regime_scalar", "base_lev", "target_exposure", "rebalance_flag",
        "position", "position_lag", "ret", "BT_ret_net", "BT_equity", "equity_bh", "dd_strat", "dd_bh",
    ]
    cols_present = [c for c in cols_pref if c in df.columns]
    df[cols_present].tail(100).to_csv(output_dir / f"{ticker_str}_trading_signals.csv", index=True)

    # Trade log
    trade_log = build_trade_log(df)
    trade_log.to_csv(output_dir / f"{ticker_str}_trade_log.csv", index=False)

    # Summary stats JSON
    with open(output_dir / f"{ticker_str}_summary_stats.json", "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    # Plots
    export_plots(df, output_dir, ticker)


# -----------------------------
# Plots
# -----------------------------
def export_plots(df: pd.DataFrame, output_dir: Path, ticker: str) -> None:
    ticker_str = ticker.upper()

    # Equity curve
    if {"BT_equity", "equity_bh"}.issubset(df.columns):
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df["BT_equity"], label="Strategy")
        plt.plot(df.index, df["equity_bh"], label="Buy & Hold")
        plt.title(f"Equity Curve: {ticker_str} Strategy vs Buy & Hold")
        plt.xlabel("Date")
        plt.ylabel("Equity (normalized to 1.0)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{ticker.lower()}_equity_curve.png", dpi=160)
        plt.close()

    # Drawdown
    if {"dd_strat", "dd_bh"}.issubset(df.columns):
        plt.figure(figsize=(12, 4))
        plt.plot(df.index, df["dd_strat"], label="Strategy DD")
        plt.plot(df.index, df["dd_bh"], label="Buy & Hold DD")
        plt.title(f"Drawdown: {ticker_str}")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{ticker.lower()}_drawdown.png", dpi=160)
        plt.close()

    # Price + exposure markers
    close_col = None
    for c in ["Adj Close", "Close", "close", "adj_close", "adj close"]:
        if c in df.columns:
            close_col = c
            break

    if close_col and "position_lag" in df.columns:
        close = pd.to_numeric(df[close_col], errors="coerce")
        ex = pd.to_numeric(df["position_lag"], errors="coerce").fillna(0.0)
        prev = ex.shift(1).fillna(0.0)
        entries = df.index[(ex > 0) & (prev == 0)]
        exits = df.index[(ex == 0) & (prev > 0)]

        plt.figure(figsize=(12, 6))
        plt.plot(df.index, close, label=f"{ticker_str} Price", linewidth=1.8, alpha=0.8)
        if "SMA_50" in df.columns:
            plt.plot(df.index, df["SMA_50"], label="SMA_50", linestyle="--", alpha=0.8)
        if "SMA_200" in df.columns:
            plt.plot(df.index, df["SMA_200"], label="SMA_200", linestyle="--", alpha=0.8)

        if len(entries):
            plt.scatter(entries, close.loc[entries], label="Entry", marker="^")
        if len(exits):
            plt.scatter(exits, close.loc[exits], label="Exit", marker="v")

        plt.title(f"Price + Regime Entries/Exits: {ticker_str}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{ticker.lower()}_price_signals.png", dpi=160)
        plt.close()


# -----------------------------
# Optional: LLM trade note
# -----------------------------
def generate_trade_note_openai(
    df: pd.DataFrame,
    summary: BacktestSummary,
    ticker: str,
    output_dir: Path,
    model: str,
) -> Path:
    out_path = output_dir / f"{ticker.lower()}_trade_note.md"

    ds_api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    oa_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    
    api_key = ds_api_key if ds_api_key else oa_api_key
    base_url = None
    
    if ds_api_key:
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        # If the model is still the default OpenAI one, switch to a Qwen model
        if "gpt" in model:
            model = "qwen-plus"

    if not api_key or OpenAI is None:
        # Write a placeholder so the repo always has an artifact
        note = (
            f"# Trade Note: {ticker.upper()}\n\n"
            "Reference API KEY not set (or openai package not installed). "
            "Run with `DASHSCOPE_API_KEY=...` or `OPENAI_API_KEY=... python run_demo.py --trade_note` to generate a full note.\n\n"
            "## Backtest snapshot\n"
            f"- Total return (strategy): {summary.total_return:.3f}\n"
            f"- Annualized return: {summary.annualized_return:.3f}\n"
            f"- Sharpe: {summary.sharpe:.3f}\n"
            f"- Max drawdown: {summary.max_drawdown:.3f}\n"
            f"- Buy & hold total return: {summary.bh_total_return:.3f}\n"
        )
        out_path.write_text(note, encoding="utf-8")
        return out_path

    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)

    # Latest state (use columns if present)
    last = df.iloc[-1].copy()
    state = {
        "date": str(df.index[-1].date()),
        "trend": int(last.get("trend", np.nan)) if pd.notna(last.get("trend", np.nan)) else None,
        "high_vol": int(last.get("high_vol", np.nan)) if pd.notna(last.get("high_vol", np.nan)) else None,
        "target_exposure": float(last.get("target_exposure", np.nan)) if pd.notna(last.get("target_exposure", np.nan)) else None,
        "position_lag": float(last.get("position_lag", np.nan)) if pd.notna(last.get("position_lag", np.nan)) else None,
        "RSI": float(last.get("RSI", np.nan)) if pd.notna(last.get("RSI", np.nan)) else None,
        "MACD_hist": float(last.get("MACD_hist", np.nan)) if pd.notna(last.get("MACD_hist", np.nan)) else None,
    }

    prompt = f"""
You are an investment technical analyst. Write a concise trade note for {ticker.upper()} based ONLY on the provided computed metrics.
Do not invent data. If something is missing, say so.

Include sections:
1) Current regime & exposure recommendation (based on target_exposure / position_lag)
2) Key indicator read (SMA trend, RSI, MACD histogram, Bollinger/ATR if present)
3) Backtest snapshot (total return, annualized return, Sharpe, max drawdown, buy&hold total return)
4) Risks & triggers (what would change the view)

Computed state (latest):
{json.dumps(state, indent=2)}

Backtest summary:
{json.dumps(asdict(summary), indent=2)}
""".strip()

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful financial analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=900,
        )
        text = resp.choices[0].message.content or ""
    except Exception as e:
        text = f"Error generating trade note with model {model}: {str(e)}"
        print(text)

    out_path.write_text(text, encoding="utf-8")
    return out_path


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Technical Analyst Agent demo pipeline.")
    p.add_argument("--ticker", type=str, default="GOOGL", help="Ticker symbol (default: GOOGL)")
    p.add_argument("--period", type=str, default="10y", help="yfinance period (e.g., 1y, 5y, 10y, max). Default: 10y")
    p.add_argument("--interval", type=str, default="1d", help="yfinance interval (default: 1d)")
    p.add_argument("--output_dir", type=str, default="outputs", help="Output directory (default: outputs)")
    p.add_argument("--no_plots", action="store_true", help="Disable saving plots")
    p.add_argument("--trade_note", action="store_true", help="Generate an LLM trade note if OPENAI_API_KEY is set")
    p.add_argument("--openai_model", type=str, default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), help="OpenAI model for trade note")
    # Allow overriding key strategy parameters
    p.add_argument("--vol_target", type=float, default=0.18, help="Annual vol target (set negative to disable)")
    p.add_argument("--max_lev", type=float, default=1.25, help="Max leverage/exposure cap")
    p.add_argument("--reb_freq", type=str, default="W-FRI", help="Rebalance frequency (pandas offset alias), e.g. W-FRI")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ticker = args.ticker.strip().upper()
    output_dir = Path(args.output_dir)

    # 1) Download
    print(f"Downloading data for {ticker}...")
    try:
        df = yf.download(ticker, period=args.period, interval=args.interval, auto_adjust=False, progress=False)
    except Exception as e:
        print(f"yfinance download failed: {e}")
        df = None

    if df is None or df.empty:
        # Fallback: check for local file
        local_file = output_dir / f"{ticker.lower()}_data_full.csv"
        print(f"Checking fallback: {local_file}")
        if local_file.exists():
            print(f"Fallback: loading data from {local_file}...")
            df = pd.read_csv(local_file, index_col=0, parse_dates=True)
        else:
            raise RuntimeError("yfinance returned empty data and no local backup found.")

    df = flatten_cols(df).sort_index()

    # 2) Indicators (optional but useful for plots / note)
    df = add_indicators(df)

    # 3) Risk overlay (core exposure engine)
    vol_target = None if args.vol_target is None or args.vol_target < 0 else float(args.vol_target)
    ro_params = RiskOverlayParams(VOL_TARGET_ANN=vol_target, MAX_LEV=float(args.max_lev), REB_FREQ=str(args.reb_freq))
    df = apply_risk_overlay(df, ro_params)

    # 4) Backtest
    df, summary = run_backtest(df, BacktestParams())

    # 5) Export artifacts
    safe_mkdir(output_dir)

    if args.no_plots:
        # Temporarily disable plot export
        global export_plots
        def export_plots(*_a, **_k):  # type: ignore
            return None

    export_artifacts(df, summary, output_dir, ticker)

    # 6) Optional LLM trade note
    note_path = None
    if args.trade_note:
        note_path = generate_trade_note_openai(df, summary, ticker, output_dir, args.openai_model)

    # Console summary
    print("=" * 72)
    print(f"AI TECHNICAL ANALYST AGENT — DEMO RUN ({ticker})")
    print("=" * 72)
    print(f"Data range: {summary.start_date} → {summary.end_date}  (n={summary.n_obs})")
    print(f"Total return (strategy):   {summary.total_return*100:.1f}%")
    print(f"Annualized return:         {summary.annualized_return*100:.1f}%")
    print(f"Sharpe:                    {summary.sharpe:.2f}")
    print(f"Max drawdown:              {summary.max_drawdown*100:.1f}%")
    print(f"Buy & hold total return:   {summary.bh_total_return*100:.1f}%")
    print(f"Rebalance count:           {summary.num_rebalances}")
    print(f"Artifacts written to:      {output_dir.resolve()}")
    if note_path:
        print(f"Trade note:                {note_path.resolve()}")

    # Sample last rows for quick sanity
    sample_cols = [c for c in ["Close", "Adj Close", "SMA_50", "SMA_200", "RSI", "MACD_hist", "trend", "high_vol", "target_exposure", "position_lag"] if c in df.columns]
    if sample_cols:
        print("\nSAMPLE (last 5 rows):")
        print(df[sample_cols].tail())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

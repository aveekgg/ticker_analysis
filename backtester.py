"""
Strategy backtester — simulates a Stage-2 entry / Stage-3-or-4 exit strategy
across a universe of stocks, with capital rotation. Kept separate from the
UI and from stage_classifier.py so the simulation logic is testable on its
own; it consumes stage transitions produced by `stage_classifier.scan_universe`.

Strategy rules:
    Entry:      `entry_delay_days` trading days after a stock is confirmed
                in Stage 2, buy at that day's close. Only the top
                `participation_pct` of all entry signals in the backtest,
                ranked by momentum (the 30w MA slope %, in %/week, at the
                week Stage 2 was confirmed — see stage_classifier.get_transitions),
                are taken at all. This is a single global cutoff applied
                once up front, not a per-day filter.
    Exit:       the EARLIEST of:
                  - stop-loss: the day's close is >= `stop_loss_pct` below
                    entry price
                  - stage exit: `exit_delay_days` trading days after the
                    FIRST Stage 3 or Stage 4 confirmation that occurs after
                    entry, sell at that day's close
                Stop-loss is checked first if both trigger the same day.
    Sizing:     fixed rupee amount per trade (`position_size`); fractional
                shares are allowed to keep the simulation simple (real
                lot-size constraints are not modeled).
    Rotation:   if a new entry signal is due but no cash is free, and at
                least one open position is currently at an unrealized loss,
                force-sell the biggest loser at that day's close to fund
                the new entry. If no position is at a loss, the signal is
                skipped (logged as a missed trade).
    Fill order: when multiple entry signals are due on the same day and
                capital is scarce, older signals (by entry-due date) fill
                first.
"""

import bisect

import numpy as np
import pandas as pd


DEFAULTS = dict(
    entry_delay_days=5,       # trading days after Stage 2 confirmation to buy
    participation_pct=50.0,   # % of entry signals taken, ranked by momentum
    exit_delay_days=5,        # trading days after Stage 3/4 confirmation to sell
    stop_loss_pct=5.0,
    position_size=50_000.0,   # rupees per trade
    total_capital=1_000_000.0,
    enable_rotation=True,
)


def _trading_day_offset(index: pd.DatetimeIndex, signal_date: pd.Timestamp, offset: int):
    """First trading day at/after signal_date, shifted `offset` trading days
    further. Returns None if that lands beyond the available price history."""
    pos = index.searchsorted(signal_date)
    target = pos + offset
    if target >= len(index):
        return None
    return index[target]


def _select_candidates(transitions: pd.DataFrame, price_frames: dict, params: dict,
                        start_date, end_date) -> pd.DataFrame:
    """Stage-2 signals -> concrete entry dates, filtered to the top
    participation_pct by momentum (a single global cutoff)."""
    s2 = transitions[transitions["stage"] == 2].copy()
    if s2.empty:
        return pd.DataFrame(columns=["symbol", "signal_date", "entry_date", "momentum"])

    rows = []
    for _, r in s2.iterrows():
        symbol = r["symbol"]
        df = price_frames.get(symbol)
        if df is None or df.empty:
            continue
        entry_date = _trading_day_offset(df.index, r["date"], params["entry_delay_days"])
        if entry_date is None or not (start_date <= entry_date <= end_date):
            continue
        rows.append({"symbol": symbol, "signal_date": r["date"], "entry_date": entry_date,
                      "momentum": r["ma30w_slope_pct"]})

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates

    pct = params["participation_pct"]
    if pct < 100:
        cutoff = candidates["momentum"].quantile(1 - pct / 100)
        candidates = candidates[candidates["momentum"] >= cutoff]

    return candidates.sort_values("entry_date").reset_index(drop=True)


def _build_stage34_index(transitions: pd.DataFrame) -> dict:
    """symbol -> sorted list of Stage 3/4 confirmation dates, for O(log n) lookup."""
    stage34 = transitions[transitions["stage"].isin([3, 4])].sort_values("date")
    return {sym: g["date"].tolist() for sym, g in stage34.groupby("symbol")}


def _first_stage34_after(stage34_index: dict, symbol: str, after_date):
    dates = stage34_index.get(symbol)
    if not dates:
        return None
    i = bisect.bisect_right(dates, after_date)
    return dates[i] if i < len(dates) else None


def run_backtest(price_frames: dict, transitions: pd.DataFrame, params: dict = None) -> dict:
    """
    price_frames: dict[symbol] -> daily DataFrame indexed by date (close at least)
    transitions:  output of stage_classifier.scan_universe (symbol, date, stage, ...)
    params:       see DEFAULTS, plus required start_date/end_date (pd.Timestamp)
    """
    p = {**DEFAULTS, **(params or {})}
    start_date, end_date = p["start_date"], p["end_date"]

    # Approximate the trading calendar with business days (a handful of NSE
    # holidays/year aren't excluded) -- far cheaper than unioning 1000+ per-
    # symbol indices, and harmless since no symbol has data on those days anyway.
    all_dates = pd.bdate_range(start=start_date, end=end_date)

    # Forward-fill each symbol's close onto that calendar so a lookup on any
    # date returns the latest known price (handles holidays and short-history
    # symbols alike) instead of requiring None-checks everywhere.
    close_lookup = {sym: df["close"].reindex(all_dates).ffill() for sym, df in price_frames.items()}

    candidates = _select_candidates(transitions, price_frames, p, start_date, end_date)
    candidates_by_date = {d: g for d, g in candidates.groupby("entry_date")} if not candidates.empty else {}
    stage34_index = _build_stage34_index(transitions)

    cash = p["total_capital"]
    open_positions = {}   # symbol -> dict(entry_date, entry_price, shares, stop_price, exit_due_date)
    trades = []
    missed = []
    equity_curve = []

    def mark_to_market(today):
        value = cash
        for sym, pos in open_positions.items():
            px = close_lookup[sym].get(today)
            value += pos["shares"] * (pos["entry_price"] if pd.isna(px) else px)
        return value

    def close_position(sym, today, price, reason):
        nonlocal cash
        pos = open_positions.pop(sym)
        proceeds = pos["shares"] * price
        cash += proceeds
        pnl = proceeds - pos["shares"] * pos["entry_price"]
        trades.append({
            "symbol": sym, "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
            "exit_date": today, "exit_price": price, "exit_reason": reason,
            "pnl": pnl, "pnl_pct": (price / pos["entry_price"] - 1) * 100,
            "holding_days": (today - pos["entry_date"]).days,
        })

    for today in all_dates:
        # 1. exits — stop-loss first, then stage-based exit due today
        for sym in list(open_positions.keys()):
            pos = open_positions[sym]
            px = close_lookup[sym].get(today)
            if pd.isna(px):
                continue
            if px <= pos["stop_price"]:
                close_position(sym, today, px, "stop_loss")
                continue
            if pos["exit_due_date"] is None:
                stage34_date = _first_stage34_after(stage34_index, sym, pos["entry_date"])
                if stage34_date is not None:
                    due = _trading_day_offset(price_frames[sym].index, stage34_date, p["exit_delay_days"])
                    pos["exit_due_date"] = due
            if pos["exit_due_date"] is not None and today >= pos["exit_due_date"]:
                close_position(sym, today, px, "stage_exit")

        # 2. entries due today, oldest signal first
        todays_candidates = candidates_by_date.get(today)
        if todays_candidates is not None:
            for _, cand in todays_candidates.iterrows():
                sym = cand["symbol"]
                if sym in open_positions:
                    continue  # already holding this stock
                px = close_lookup[sym].get(today)
                if pd.isna(px) or px <= 0:
                    continue

                if cash < p["position_size"] and p["enable_rotation"] and open_positions:
                    losers = {}
                    for s, pos in open_positions.items():
                        cur_px = close_lookup[s].get(today)
                        cur_px = pos["entry_price"] if pd.isna(cur_px) else cur_px
                        losers[s] = cur_px - pos["entry_price"]
                    worst_sym = min(losers, key=losers.get)
                    if losers[worst_sym] < 0:
                        worst_px = close_lookup[worst_sym].get(today)
                        if not pd.isna(worst_px):
                            close_position(worst_sym, today, worst_px, "rotated_out")

                if cash >= p["position_size"]:
                    shares = p["position_size"] / px
                    cash -= p["position_size"]
                    open_positions[sym] = {
                        "entry_date": today, "entry_price": px, "shares": shares,
                        "stop_price": px * (1 - p["stop_loss_pct"] / 100), "exit_due_date": None,
                    }
                else:
                    missed.append({"symbol": sym, "signal_date": cand["signal_date"], "entry_date": today,
                                    "reason": "no_capital"})

        equity_curve.append({"date": today, "value": mark_to_market(today)})

    # liquidate anything still open at the end of the window
    if len(all_dates):
        last_day = all_dates[-1]
        for sym in list(open_positions.keys()):
            px = close_lookup[sym].get(last_day)
            if not pd.isna(px):
                close_position(sym, last_day, px, "end_of_backtest")

    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades)
    missed_df = pd.DataFrame(missed)

    metrics = _compute_metrics(equity_df, trades_df, p["total_capital"])

    return {"equity_curve": equity_df, "trades": trades_df, "missed_signals": missed_df, "metrics": metrics}


def _compute_metrics(equity_df: pd.DataFrame, trades_df: pd.DataFrame, start_capital: float) -> dict:
    if equity_df.empty:
        return dict(total_return_pct=0.0, cagr_pct=0.0, max_drawdown_pct=0.0,
                    win_rate_pct=0.0, num_trades=0, avg_holding_days=0.0)

    end_value = equity_df["value"].iloc[-1]
    total_return_pct = (end_value / start_capital - 1) * 100

    days = (equity_df["date"].iloc[-1] - equity_df["date"].iloc[0]).days
    cagr_pct = ((end_value / start_capital) ** (365 / days) - 1) * 100 if days > 0 else 0.0

    running_peak = equity_df["value"].cummax()
    drawdown = (equity_df["value"] - running_peak) / running_peak
    max_drawdown_pct = drawdown.min() * 100

    if not trades_df.empty:
        win_rate_pct = (trades_df["pnl"] > 0).mean() * 100
        avg_holding_days = trades_df["holding_days"].mean()
    else:
        win_rate_pct, avg_holding_days = 0.0, 0.0

    return dict(
        total_return_pct=total_return_pct, cagr_pct=cagr_pct, max_drawdown_pct=max_drawdown_pct,
        win_rate_pct=win_rate_pct, num_trades=len(trades_df), avg_holding_days=avg_holding_days,
    )

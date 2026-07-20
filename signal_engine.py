"""
OHLCV-derived scoring signal engine — a second, independent way to arrive at
buy/sell decisions alongside Weinstein stage analysis (stage_classifier.py).

Rather than binary pass/fail rules, each stock gets a 0-100 score built from
weighted sub-scores across six factor families (all derivable from plain
OHLCV, no fundamentals needed):

    1M Return           -- momentum over the lookback window
    EMA Alignment        -- close > EMA20 > EMA50 > EMA200, partial credit
    Volume Dry-up        -- fast/slow volume MA ratio contracting (basing)
    ATR Compression       -- volatility contracting vs N days ago
    Near High             -- proximity to the rolling N-day high
    Relative Strength     -- return vs a broad-market benchmark index

A buy signal fires when the score crosses above `buy_score_threshold`.
Exits are position-dependent (stop-loss / trailing-stop / max holding period)
and are evaluated in `run_score_backtest`'s simulation loop, not per-day in
isolation -- mirrors backtester.py's mechanics (capital, position sizing,
rotation out of the worst loser when full) so the two systems are directly
comparable.
"""

import bisect

import numpy as np
import pandas as pd

DEFAULTS = dict(
    return_lookback_days=20,
    return_target_pct=20.0,       # % return that earns full credit on the return sub-score
    ema_fast=20, ema_mid=50, ema_slow=200,
    volume_fast_days=5, volume_slow_days=20,
    volume_dryup_threshold=0.7,   # fast/slow volume ratio at/below this earns full credit
    atr_window=14,
    atr_compression_lookback_days=20,
    near_high_lookback_days=60,
    near_high_max_pct=10.0,       # distance from the rolling high beyond which score is 0
    weight_return=20.0,
    weight_ema_alignment=15.0,
    weight_volume_dryup=20.0,
    weight_atr_compression=15.0,
    weight_near_high=15.0,
    weight_relative_strength=15.0,
    buy_score_threshold=60.0,
    participation_pct=100.0,      # % of buy signals taken, ranked by score (like backtester.py)
    stop_loss_pct=8.0,
    trailing_stop_pct=15.0,
    max_holding_days=60,
    position_size=50_000.0,
    total_capital=1_000_000.0,
    enable_rotation=True,
)


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def compute_features(daily: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """daily: columns [date, open, high, low, close, volume]. Returns a
    date-indexed frame with all the OHLCV-derived features used by the score."""
    p = {**DEFAULTS, **(params or {})}
    d = daily.sort_values("date").set_index("date").copy()

    d["ret_lookback_pct"] = d["close"].pct_change(p["return_lookback_days"]) * 100

    d["ema_fast"] = d["close"].ewm(span=p["ema_fast"], adjust=False).mean()
    d["ema_mid"] = d["close"].ewm(span=p["ema_mid"], adjust=False).mean()
    d["ema_slow"] = d["close"].ewm(span=p["ema_slow"], adjust=False).mean()

    d["vol_ma_fast"] = d["volume"].rolling(p["volume_fast_days"]).mean()
    d["vol_ma_slow"] = d["volume"].rolling(p["volume_slow_days"]).mean()
    d["vol_dryup_ratio"] = d["vol_ma_fast"] / d["vol_ma_slow"]

    prev_close = d["close"].shift(1)
    true_range = pd.concat([
        d["high"] - d["low"], (d["high"] - prev_close).abs(), (d["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    d["atr"] = true_range.rolling(p["atr_window"]).mean()
    d["atr_ref"] = d["atr"].shift(p["atr_compression_lookback_days"])

    rolling_high = d["close"].rolling(p["near_high_lookback_days"]).max()
    d["dist_from_high_pct"] = (rolling_high - d["close"]) / rolling_high * 100

    d["daily_range_pct"] = (d["high"] - d["low"]) / d["open"] * 100
    d["gap_pct"] = (d["open"] - prev_close) / prev_close * 100
    candle_range = (d["high"] - d["low"]).replace(0, np.nan)
    d["body_pct"] = (d["close"] - d["open"]).abs() / candle_range * 100
    d["upper_wick_pct"] = (d["high"] - d[["open", "close"]].max(axis=1)) / candle_range * 100
    d["lower_wick_pct"] = (d[["open", "close"]].min(axis=1) - d["low"]) / candle_range * 100

    d["obv"] = (np.sign(d["close"].diff()).fillna(0) * d["volume"]).cumsum()

    return d


def compute_score(features: pd.DataFrame, params: dict = None, benchmark_close: pd.Series = None) -> pd.DataFrame:
    """Adds per-factor sub-scores and a total 'score' column (0 to sum of
    the weights) to an already-computed features frame."""
    p = {**DEFAULTS, **(params or {})}
    f = features.copy()

    f["score_return"] = p["weight_return"] * _clip01(f["ret_lookback_pct"] / p["return_target_pct"])

    aligned = (
        (f["close"] > f["ema_fast"]).astype(float) +
        (f["ema_fast"] > f["ema_mid"]).astype(float) +
        (f["ema_mid"] > f["ema_slow"]).astype(float)
    ) / 3
    f["score_ema_alignment"] = p["weight_ema_alignment"] * aligned

    dryup_threshold = p["volume_dryup_threshold"]
    f["score_volume_dryup"] = p["weight_volume_dryup"] * _clip01(
        (1 - f["vol_dryup_ratio"]) / (1 - dryup_threshold)
    )

    f["score_atr_compression"] = p["weight_atr_compression"] * _clip01(
        (f["atr_ref"] - f["atr"]) / f["atr_ref"]
    )

    f["score_near_high"] = p["weight_near_high"] * _clip01(
        (p["near_high_max_pct"] - f["dist_from_high_pct"]) / p["near_high_max_pct"]
    )

    if benchmark_close is not None:
        bench_ret = benchmark_close.reindex(f.index, method="ffill").pct_change(p["return_lookback_days"]) * 100
        outperformance = f["ret_lookback_pct"] - bench_ret
        f["score_relative_strength"] = p["weight_relative_strength"] * _clip01(
            0.5 + outperformance / (2 * p["return_target_pct"])
        )
    else:
        f["score_relative_strength"] = p["weight_relative_strength"] * 0.5  # neutral without a benchmark

    score_cols = ["score_return", "score_ema_alignment", "score_volume_dryup",
                  "score_atr_compression", "score_near_high", "score_relative_strength"]
    f["score"] = f[score_cols].sum(axis=1)
    return f


def generate_signals(daily: pd.DataFrame, params: dict = None, benchmark_close: pd.Series = None) -> pd.DataFrame:
    """Full pipeline: features -> score -> discrete buy-trigger dates (where
    score crosses above the threshold, not every day it stays elevated)."""
    p = {**DEFAULTS, **(params or {})}
    features = compute_features(daily, p)
    scored = compute_score(features, p, benchmark_close)
    scored["buy_signal"] = (scored["score"] >= p["buy_score_threshold"]) & \
                            (scored["score"].shift(1) < p["buy_score_threshold"])
    return scored


def build_market_index(price_frames: dict) -> pd.Series:
    """Equal-weighted daily index across an entire universe, normalized to
    100 at each member's own first date -- serves as the 'broad market'
    benchmark for the relative-strength sub-score when no better index exists."""
    normalized = []
    for daily in price_frames.values():
        if "date" not in daily.columns:
            daily = daily.reset_index()
        d = daily.sort_values("date").set_index("date")
        if d.empty or not d["close"].iloc[0]:
            continue
        normalized.append(d["close"] / d["close"].iloc[0] * 100)
    if not normalized:
        return pd.Series(dtype=float)
    return pd.concat(normalized, axis=1).mean(axis=1)


def get_buy_triggers(scored: pd.DataFrame) -> pd.DataFrame:
    """Convenience: just the buy-trigger rows (date, close, score) for chart
    annotation, mirroring stage_classifier.get_transitions' shape."""
    hits = scored[scored["buy_signal"]].copy()
    return hits[["close", "score"]].reset_index().rename(columns={"index": "date"})


def simulate_symbol_trades(daily: pd.DataFrame, params: dict = None,
                            benchmark_close: pd.Series = None) -> pd.DataFrame:
    """Non-overlapping buy/sell trades for a SINGLE symbol, with no capital
    constraint -- used for the chart overlay so every signal's trade is
    plotted (the capital-constrained `run_score_backtest` would hide trades
    once cash is tied up, which is wrong for visualizing one stock's history).

    Walks buy signals in order: when flat, enter at that day's close; then
    exit at the earliest of stop-loss / trailing-stop / max-holding; resume
    scanning for the next buy signal after the exit.
    """
    p = {**DEFAULTS, **(params or {})}
    scored = generate_signals(daily, p, benchmark_close)
    closes = scored["close"]
    dates = list(scored.index)
    buy_flags = scored["buy_signal"].to_numpy()

    trades = []
    i = 0
    n = len(dates)
    while i < n:
        if not buy_flags[i]:
            i += 1
            continue
        entry_date = dates[i]
        entry_price = closes.iloc[i]
        if entry_price <= 0 or pd.isna(entry_price):
            i += 1
            continue
        stop_price = entry_price * (1 - p["stop_loss_pct"] / 100)
        peak = entry_price
        exit_idx = None
        exit_reason = None
        j = i + 1
        while j < n:
            px = closes.iloc[j]
            peak = max(peak, px)
            trailing_price = peak * (1 - p["trailing_stop_pct"] / 100)
            holding_days = (dates[j] - entry_date).days
            if px <= stop_price:
                exit_idx, exit_reason = j, "stop_loss"; break
            if px <= trailing_price:
                exit_idx, exit_reason = j, "trailing_stop"; break
            if holding_days >= p["max_holding_days"]:
                exit_idx, exit_reason = j, "max_holding"; break
            j += 1
        if exit_idx is None:
            exit_idx, exit_reason = n - 1, "end_of_data"
        exit_price = closes.iloc[exit_idx]
        trades.append({
            "entry_date": entry_date, "entry_price": entry_price,
            "exit_date": dates[exit_idx], "exit_price": exit_price, "exit_reason": exit_reason,
            "pnl_pct": (exit_price / entry_price - 1) * 100,
            "holding_days": (dates[exit_idx] - entry_date).days,
        })
        i = exit_idx + 1  # resume after this trade's exit

    return pd.DataFrame(trades)


def scan_universe(price_frames: dict, params: dict = None, benchmark_close: pd.Series = None,
                   progress_cb=None) -> pd.DataFrame:
    """Buy triggers for every symbol in price_frames, concatenated with a
    `symbol` column -- the score-engine analogue of stage_classifier.scan_universe."""
    p = {**DEFAULTS, **(params or {})}
    rows = []
    total = len(price_frames)
    for i, (symbol, daily) in enumerate(price_frames.items()):
        try:
            if "date" not in daily.columns:
                daily = daily.reset_index()
            scored = generate_signals(daily, p, benchmark_close)
            triggers = get_buy_triggers(scored)
            if not triggers.empty:
                triggers.insert(0, "symbol", symbol)
                rows.append(triggers)
        except Exception:
            pass
        if progress_cb:
            progress_cb((i + 1) / total)

    if not rows:
        return pd.DataFrame(columns=["symbol", "date", "close", "score"])
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def _trading_day_index(index: pd.DatetimeIndex, date) -> int:
    return index.searchsorted(date)


def run_score_backtest(price_frames: dict, params: dict = None, benchmark_close: pd.Series = None,
                        symbols: list = None) -> dict:
    """Simulate: buy at the close on a buy-trigger date, sell at the
    earliest of a stop-loss, a trailing-stop, or the max holding period.
    Mirrors backtester.run_backtest's mechanics (fixed position size,
    capital rotation out of the worst loser when full) so the two systems'
    results are directly comparable.

    params must include start_date/end_date (pd.Timestamp).
    """
    p = {**DEFAULTS, **(params or {})}
    start_date, end_date = p["start_date"], p["end_date"]
    universe = {s: price_frames[s] for s in (symbols or price_frames) if s in price_frames}

    all_dates = pd.bdate_range(start=start_date, end=end_date)
    close_lookup = {}
    scored_by_symbol = {}
    for sym, daily in universe.items():
        df = daily if "date" in daily.columns else daily.reset_index()
        close_lookup[sym] = df.set_index("date")["close"].reindex(all_dates).ffill()
        scored_by_symbol[sym] = generate_signals(df, p, benchmark_close)

    candidates = []
    for sym, scored in scored_by_symbol.items():
        hits = scored[scored["buy_signal"] & (scored.index >= start_date) & (scored.index <= end_date)]
        for date, row in hits.iterrows():
            candidates.append({"symbol": sym, "signal_date": date, "score": row["score"]})
    candidates = pd.DataFrame(candidates)
    if not candidates.empty:
        pct = p["participation_pct"]
        if pct < 100:
            cutoff = candidates["score"].quantile(1 - pct / 100)
            candidates = candidates[candidates["score"] >= cutoff]
        candidates_by_date = {d: g for d, g in candidates.groupby("signal_date")}
    else:
        candidates_by_date = {}

    cash = p["total_capital"]
    open_positions = {}  # symbol -> dict(entry_date, entry_price, shares, peak_price)
    trades, equity_curve = [], []

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
        for sym in list(open_positions.keys()):
            pos = open_positions[sym]
            px = close_lookup[sym].get(today)
            if pd.isna(px):
                continue
            pos["peak_price"] = max(pos["peak_price"], px)
            stop_price = pos["entry_price"] * (1 - p["stop_loss_pct"] / 100)
            trailing_price = pos["peak_price"] * (1 - p["trailing_stop_pct"] / 100)
            holding_days = (today - pos["entry_date"]).days
            if px <= stop_price:
                close_position(sym, today, px, "stop_loss")
            elif px <= trailing_price:
                close_position(sym, today, px, "trailing_stop")
            elif holding_days >= p["max_holding_days"]:
                close_position(sym, today, px, "max_holding")

        todays_candidates = candidates_by_date.get(today)
        if todays_candidates is not None:
            for _, cand in todays_candidates.sort_values("score", ascending=False).iterrows():
                sym = cand["symbol"]
                if sym in open_positions:
                    continue
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
                        "entry_date": today, "entry_price": px, "shares": shares, "peak_price": px,
                    }

        equity_curve.append({"date": today, "value": mark_to_market(today)})

    if len(all_dates):
        last_day = all_dates[-1]
        for sym in list(open_positions.keys()):
            px = close_lookup[sym].get(last_day)
            if not pd.isna(px):
                close_position(sym, last_day, px, "end_of_backtest")

    equity_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades)
    metrics = _compute_metrics(equity_df, trades_df, p["total_capital"])
    return {"equity_curve": equity_df, "trades": trades_df, "metrics": metrics}


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

"""
Stage Classifier — Weinstein-style Stage 1/2/3/4 labeling per stock.

Kept separate from the UI so the rules can be tuned/tested independently of
how they're displayed. Classification runs on WEEKLY bars (Weinstein stages
are a weekly-regime concept — daily-bar classification flickers constantly
and produces useless noise), using a mix of the 30-week MA and daily-based
200d/50d MAs resampled onto the weekly index:

    Stage 1 (Base):     30w MA slope ~ flat; price whipsawing within a band
                         of the 30w MA; volume contracting vs its 6-month avg.
    Stage 2 (Advance):  price > rising 30w MA and > rising 200d MA; 50d MA
                         also rising and price respecting it; breakout volume
                         well above its 50d average.
    Stage 3 (Top):      price closes below 10w/50d MA on elevated volume
                         while 30w/200d MA are still flat-to-up; or a failed
                         breakout (new high, closes back inside prior range
                         within ~3 weeks).
    Stage 4 (Decline):  price breaks Stage-3 support; 30w and 200d MA slope
                         turns negative.

Note: delivery% (mentioned in the Stage 2 breakout rule) isn't available
from Kite's historical data API, so that leg of the rule is omitted — the
volume-multiple check is used on its own.

A minimum-run-length filter is applied after the raw weekly classification:
a candidate stage must persist for `min_run_weeks` consecutive weeks to be
confirmed, otherwise it's folded back into the prior confirmed stage. This
is what keeps the annotated transitions to genuine regime changes rather
than every noisy week.
"""

import numpy as np
import pandas as pd

STAGE_NAMES = {1: "Stage 1 (Base)", 2: "Stage 2 (Advance)", 3: "Stage 3 (Top)", 4: "Stage 4 (Decline)"}
STAGE_COLORS = {1: "#9e9e9e", 2: "#2ecc71", 3: "#f39c12", 4: "#e74c3c"}

DEFAULTS = dict(
    weekly_ma_window=30,        # 30-week MA
    weekly_slope_window=12,     # trailing regression window, within the 10-15wk spec
    weekly_flat_pct=0.3,        # %/week slope magnitude below which the 30w MA counts as "flat"
    weekly_vol_avg_weeks=26,    # ~6 months
    whipsaw_band_pct=15.0,      # price within +/- this % of the 30w MA during Stage 1
    daily_ma200_window=200,
    daily_ma50_window=50,       # also stands in for the "10-week" MA (10wk * 5d = 50d)
    daily_slope_window=50,      # ~10 weeks of trading days
    daily_flat_pct=0.05,        # %/day slope magnitude below which a daily MA counts as "flat"
    breakout_vol_mult=2.0,      # Stage 2 breakout volume vs 50d avg
    distribution_vol_mult=1.5,  # Stage 3 distribution-week volume vs 50d avg
    failed_breakout_lookback_weeks=12,
    failed_breakout_window_weeks=3,
    failed_breakout_giveback_pct=3.0,  # % below the recent high that counts as "failed"
    min_run_weeks=3,            # a stage must hold this many weeks to be confirmed
)


def _rolling_slope_pct(series: pd.Series, window: int) -> pd.Series:
    """% change per bar implied by a linear regression fit over the trailing window."""
    def slope(y):
        if np.isnan(y).any():
            return np.nan
        x = np.arange(len(y))
        m, _ = np.polyfit(x, y, 1)
        mean_y = y.mean()
        return (m / mean_y) * 100 if mean_y else np.nan
    return series.rolling(window).apply(slope, raw=True)


def _enforce_min_run(stage: pd.Series, min_run: int) -> pd.Series:
    """Merge any run shorter than min_run weeks into the preceding confirmed run."""
    vals = stage.to_numpy(copy=True)
    n = len(vals)
    if n == 0:
        return stage

    segments = []
    start = 0
    for j in range(1, n + 1):
        if j == n or not _same(vals[j], vals[start]):
            segments.append([start, j, vals[start]])
            start = j

    changed = True
    while changed and len(segments) > 1:
        changed = False
        for idx, (s, e, v) in enumerate(segments):
            if e - s < min_run:
                if idx == 0:
                    nxt = segments[idx + 1]
                    segments[idx + 1] = [s, nxt[1], nxt[2]]
                    segments.pop(idx)
                else:
                    prev = segments[idx - 1]
                    segments[idx - 1] = [prev[0], e, prev[2]]
                    segments.pop(idx)
                changed = True
                break

    out = np.empty(n)
    for s, e, v in segments:
        out[s:e] = v
    return pd.Series(out, index=stage.index)


def _same(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    return a == b


def build_weekly_frame(daily: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    p = {**DEFAULTS, **(params or {})}
    d = daily.sort_values("date").set_index("date").copy()

    weekly = d.resample("W-FRI").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
    ).dropna()
    weekly["ma30w"] = weekly["close"].rolling(p["weekly_ma_window"]).mean()
    weekly["ma30w_slope_pct"] = _rolling_slope_pct(weekly["ma30w"], p["weekly_slope_window"])
    weekly["vol26w_avg"] = weekly["volume"].rolling(p["weekly_vol_avg_weeks"]).mean()
    weekly["vol4w_avg"] = weekly["volume"].rolling(4).mean()

    d["ma200"] = d["close"].rolling(p["daily_ma200_window"]).mean()
    d["ma200_slope_pct"] = _rolling_slope_pct(d["ma200"], p["daily_slope_window"])
    d["ma50"] = d["close"].rolling(p["daily_ma50_window"]).mean()
    d["ma50_slope_pct"] = _rolling_slope_pct(d["ma50"], p["daily_slope_window"])
    d["vol50d_avg"] = d["volume"].rolling(p["daily_ma50_window"]).mean()
    d["vol_ratio_50d"] = d["volume"] / d["vol50d_avg"]

    daily_resampled = d[["ma200", "ma200_slope_pct", "ma50", "ma50_slope_pct", "vol_ratio_50d"]].resample("W-FRI").last()
    weekly = weekly.join(daily_resampled)

    weekly["rolling_high"] = weekly["close"].rolling(p["failed_breakout_lookback_weeks"]).max()

    return weekly


def classify_weekly(weekly: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    p = {**DEFAULTS, **(params or {})}
    w = weekly.copy()

    stage2 = (
        (w["close"] > w["ma30w"]) & (w["ma30w_slope_pct"] > p["weekly_flat_pct"]) &
        (w["close"] > w["ma200"]) & (w["ma200_slope_pct"] > p["daily_flat_pct"]) &
        (w["close"] > w["ma50"]) & (w["ma50_slope_pct"] > p["daily_flat_pct"])
    )

    stage4 = (
        (w["ma30w_slope_pct"] < -p["weekly_flat_pct"]) &
        (w["ma200_slope_pct"] < -p["daily_flat_pct"]) &
        (w["close"] < w["ma30w"]) & (w["close"] < w["ma200"])
    )

    distribution_week = (
        (w["close"] < w["ma50"]) & (w["vol_ratio_50d"] >= p["distribution_vol_mult"]) &
        (w["ma30w_slope_pct"] >= -p["weekly_flat_pct"]) & (w["ma200_slope_pct"] >= -p["daily_flat_pct"])
    )

    made_recent_high = (w["close"] >= w["rolling_high"])
    within_window = made_recent_high[::-1].rolling(p["failed_breakout_window_weeks"], min_periods=1).max()[::-1].fillna(0).astype(bool)
    failed_breakout = within_window & (w["close"] < w["rolling_high"] * (1 - p["failed_breakout_giveback_pct"] / 100))

    stage3 = distribution_week | failed_breakout

    stage1 = (
        (w["ma30w_slope_pct"].abs() <= p["weekly_flat_pct"]) &
        ((w["close"] - w["ma30w"]).abs() / w["ma30w"] * 100 <= p["whipsaw_band_pct"]) &
        (w["vol4w_avg"] < w["vol26w_avg"])
    )

    stage = pd.Series(np.nan, index=w.index)
    stage[stage1] = 1
    stage[stage2] = 2   # a clear advance overrides a marginal "flat" read
    stage[stage3] = 3   # a distribution/failed-breakout signal is decisive
    stage[stage4] = 4   # breakdown is the most decisive signal, evaluated last so it always wins

    has_indicators = w["ma30w"].notna() & w["ma200"].notna()
    stage = stage.where(has_indicators).ffill()
    stage = _enforce_min_run(stage, p["min_run_weeks"])

    w["stage"] = stage
    return w


def classify(daily: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """Convenience wrapper: build the weekly frame and classify it in one call."""
    weekly = build_weekly_frame(daily, params)
    return classify_weekly(weekly, params)


def get_transitions(classified_weekly: pd.DataFrame) -> pd.DataFrame:
    """First week each stage was newly confirmed (i.e. where the smoothed label changes)."""
    w = classified_weekly.dropna(subset=["stage"]).copy()
    if w.empty:
        return pd.DataFrame(columns=["date", "stage", "close", "stage_name"])
    changed = w["stage"] != w["stage"].shift(1)
    out = w.loc[changed, ["close"]].reset_index().rename(columns={"index": "date"})
    out["stage"] = w.loc[changed, "stage"].astype(int).values
    out["stage_name"] = out["stage"].map(STAGE_NAMES)
    return out

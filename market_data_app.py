"""
Local UI for querying nse_market_data.duckdb, viewing price charts, screening
stocks by Weinstein stage, and backtesting two independent trading systems.

Run:
    streamlit run market_data_app.py

Tabs:
    Charts             - candlestick + volume + MA overlay for one symbol,
                          with stage annotations and score-engine buy/sell markers
    Stage screener     - which stocks entered a given stage in a date window
    Strategy backtest  - simulate the Stage-2 entry / Stage-3-4-or-stop-loss
                          exit strategy across the full universe
    Signal backtest    - simulate the OHLCV-scoring entry / stop-trailing-
                          stop-max-holding exit strategy across the universe
    Query / Tables     - run arbitrary SQL against the DuckDB file
"""

import duckdb
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import backtester as bt
import signal_engine as se
import stage_classifier as sc

DB_FILE = "nse_market_data.duckdb"

st.set_page_config(page_title="NSE Market Data", layout="wide")


@st.cache_resource
def get_connection():
    return duckdb.connect(DB_FILE, read_only=True)


@st.cache_data
def get_symbols():
    con = get_connection()
    return con.execute(
        "SELECT tradingsymbol FROM instruments ORDER BY tradingsymbol"
    ).df()["tradingsymbol"].tolist()


@st.cache_data
def get_prices(symbol: str) -> pd.DataFrame:
    con = get_connection()
    df = con.execute(
        "SELECT date, open, high, low, close, volume FROM daily_prices "
        "WHERE tradingsymbol = ? ORDER BY date", [symbol]
    ).df()
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_resource(show_spinner="Loading full price history (first time only)...")
def get_price_frames() -> dict:
    """symbol -> daily DataFrame indexed by date. Loaded once for the whole
    universe; only triggered when the Screener or Backtest tab is used."""
    frames = {}
    for symbol in get_symbols():
        df = get_prices(symbol)
        if df.empty:
            continue
        frames[symbol] = df.set_index("date").sort_index()
    return frames


@st.cache_data
def get_sector_map() -> pd.DataFrame:
    con = get_connection()
    return con.execute(
        "SELECT tradingsymbol AS symbol, macro_sector, sector, industry, basic_industry "
        "FROM instruments WHERE macro_sector IS NOT NULL"
    ).df()


@st.cache_data
def get_group_classified_for_symbol(symbol: str, group_level: str, params: dict) -> pd.DataFrame:
    """The classified peer-group index for one symbol's group -- used so the
    Charts tab's single-symbol view can apply the same group-strength/RS
    checks without loading the full 1149-symbol universe."""
    sector_map = get_sector_map()
    row = sector_map[sector_map["symbol"] == symbol]
    if row.empty:
        return pd.DataFrame()
    group_name = row.iloc[0][group_level]
    members = sector_map[sector_map[group_level] == group_name]["symbol"].tolist()
    member_frames = {m: get_prices(m) for m in members}
    member_frames = {m: df for m, df in member_frames.items() if not df.empty}
    return sc.classify_group(member_frames, {**params, "group_level": group_level})


@st.cache_data
def get_stage_transitions(symbol: str, params: dict) -> pd.DataFrame:
    daily = get_prices(symbol)
    group_classified = None
    if params["require_group_strength"] or params["require_rs_rising"]:
        group_classified = get_group_classified_for_symbol(symbol, params["group_level"], params)
    classified = sc.classify(daily, params, group_classified=group_classified)
    return sc.get_transitions(classified)


def render_sector_filters(key_prefix: str) -> set:
    """Cascading Macro Sector -> Sector -> Industry -> Basic Industry filters.
    Returns the set of matching symbols, or None if nothing is selected
    (meaning "no sector filter, include everything")."""
    sector_map = get_sector_map()
    with st.expander(":material/filter_alt: Filter by sector / industry"):
        c1, c2, c3, c4 = st.columns(4)
        macro_sel = c1.multiselect("Macro sector", sorted(sector_map["macro_sector"].unique()), key=f"{key_prefix}_macro")
        scoped = sector_map[sector_map["macro_sector"].isin(macro_sel)] if macro_sel else sector_map

        sector_sel = c2.multiselect("Sector", sorted(scoped["sector"].unique()), key=f"{key_prefix}_sector")
        scoped = scoped[scoped["sector"].isin(sector_sel)] if sector_sel else scoped

        industry_sel = c3.multiselect("Industry", sorted(scoped["industry"].unique()), key=f"{key_prefix}_industry")
        scoped = scoped[scoped["industry"].isin(industry_sel)] if industry_sel else scoped

        basic_sel = c4.multiselect("Basic industry", sorted(scoped["basic_industry"].unique()), key=f"{key_prefix}_basic")
        scoped = scoped[scoped["basic_industry"].isin(basic_sel)] if basic_sel else scoped

        any_filter = bool(macro_sel or sector_sel or industry_sel or basic_sel)
        if any_filter:
            st.caption(f"{len(scoped)} symbols match this sector filter.")
        return set(scoped["symbol"]) if any_filter else None


def render_universe_selector(key_prefix: str, all_symbols: list) -> set:
    """Choose which stocks a backtest runs over. Returns a set of symbols,
    or None meaning "the full universe". Three modes:
      - Full universe
      - By sector / industry (cascading filter)
      - Specific stocks (multiselect dropdown and/or comma-separated list)
    """
    mode = st.radio(
        "Universe", ["Full universe", "By sector / industry", "Specific stocks"],
        horizontal=True, key=f"{key_prefix}_mode",
    )
    if mode == "Full universe":
        return None
    if mode == "By sector / industry":
        return render_sector_filters(key_prefix)

    # Specific stocks: union of the multiselect and the comma-separated list.
    picked = st.multiselect("Pick stocks", all_symbols, key=f"{key_prefix}_pick")
    typed = st.text_input(
        "…or paste a comma-separated list", key=f"{key_prefix}_typed",
        placeholder="e.g. KIRLOSENG, TMPV, INFY, RELIANCE",
    )
    valid = set(all_symbols)
    typed_syms = [s.strip().upper() for s in typed.split(",") if s.strip()]
    unknown = [s for s in typed_syms if s not in valid]
    selected = set(picked) | (set(typed_syms) & valid)

    if unknown:
        st.warning(f"Not in the universe, ignored: {', '.join(unknown)}")
    if selected:
        st.caption(f"{len(selected)} stock(s) selected.")
        return selected
    st.info("Pick at least one stock (or switch to Full universe).")
    return set()  # empty -> nothing to run, distinct from None (full universe)


@st.cache_data(show_spinner="Scanning the full universe for stage transitions (can take a couple minutes the first time)...")
def get_universe_transitions(params: dict) -> pd.DataFrame:
    frames = get_price_frames()
    sector_map = get_sector_map() if (params["require_group_strength"] or params["require_rs_rising"]) else None
    return sc.scan_universe(frames, params, sector_map=sector_map)


@st.cache_resource(show_spinner="Building the broad-market benchmark index (first time only)...")
def get_market_benchmark() -> pd.Series:
    frames = get_price_frames()
    return se.build_market_index(frames)


@st.cache_data
def get_symbol_score_trades(symbol: str, params: dict) -> pd.DataFrame:
    """Every non-overlapping score-engine buy/sell trade for one symbol, for
    the Charts tab overlay -- uses the capital-free single-symbol simulator so
    all signals plot (a capital-constrained backtest would hide trades once
    the position is funded)."""
    daily = get_prices(symbol)
    if daily.empty:
        return pd.DataFrame()
    benchmark = get_market_benchmark()
    return se.simulate_symbol_trades(daily, params, benchmark_close=benchmark)


@st.cache_data(show_spinner="Running the signal strategy on each group... (can take a couple minutes)")
def get_sector_comparison(group_level: str, params: dict) -> pd.DataFrame:
    """Run the score backtest independently on every group at `group_level`,
    one row of metrics per group -- a leaderboard of which sector/industry the
    signal strategy performed best in over the period. Each group gets the same
    capital settings; the benchmark stays the full-universe index so relative
    strength is measured against the broad market, not each subset."""
    frames = get_price_frames()
    sector_map = get_sector_map()
    benchmark = get_market_benchmark()
    rows = []
    for group_name, grp in sector_map.groupby(group_level):
        members = {s: frames[s] for s in grp["symbol"] if s in frames}
        if not members:
            continue
        result = se.run_score_backtest(members, params, benchmark_close=benchmark)
        m = result["metrics"]
        rows.append({
            "group": group_name, "stocks": len(members), "trades": m["num_trades"],
            "total_return_%": round(m["total_return_pct"], 1),
            "cagr_%": round(m["cagr_pct"], 1),
            "max_drawdown_%": round(m["max_drawdown_pct"], 1),
            "win_rate_%": round(m["win_rate_pct"], 0),
            "avg_holding_days": round(m["avg_holding_days"], 0),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("total_return_%", ascending=False).reset_index(drop=True) if not df.empty else df


def render_stage_help(params: dict):
    """Plain-language explainer for the Stage 1-4 framework and what the
    current slider settings mean."""
    st.markdown("#### :material/school: What are Stage 1-4?")
    st.markdown(
        "This labels a stock's long-term trend using **Stan Weinstein's stage "
        "analysis** — a way of reading where a stock is in its cycle by looking "
        "at its 30-week moving average (a smoothed line of the last ~30 weeks "
        "of prices). Think of it like four seasons a stock moves through:"
    )
    st.markdown(
        f"- :gray[**Stage 1 — Base**] :gray[(quiet)]: the stock goes nowhere. "
        f"Price chops sideways, the 30-week average is flat, and volume dries up. "
        f"The calm before a move.\n"
        f"- :green[**Stage 2 — Advance**] :green[(uptrend)]: the stock breaks out "
        f"and climbs. Price stays above a *rising* 30-week average, with strong "
        f"volume on the up-moves. This is the phase trend-followers want to catch.\n"
        f"- :orange[**Stage 3 — Top**] :orange[(warning)]: the rally loses steam. "
        f"Price starts closing below its short-term average on heavy volume, or a "
        f"new high fails and price falls back — often a sign big holders are selling.\n"
        f"- :red[**Stage 4 — Decline**] :red[(downtrend)]: the trend turns down. "
        f"Price sits below a *falling* 30-week average. This is the correction phase."
    )
    st.caption("Stocks typically cycle 1 → 2 → 3 → 4 → back to 1, though they can skip or revisit stages.")

    st.divider()
    st.markdown("#### :material/tune: What your current settings mean")
    st.markdown(
        f"- A trend counts as **\"flat\"** if the 30-week average moves less than "
        f"**{params['weekly_flat_pct']}% per week** (or a daily average moves less "
        f"than **{params['daily_flat_pct']}% per day**). Anything faster than that "
        f"counts as clearly rising or falling — lower this to call fewer trends \"flat\"."
    )
    st.markdown(
        f"- **Stage 1** requires price to stay within **{params['whipsaw_band_pct']}%** "
        f"of the 30-week average. A wider band lets more sideways chop still count as \"basing\"."
    )
    st.markdown(
        f"- **Stage 2** needs breakout volume at least **{params['breakout_vol_mult']}x** "
        f"its 50-day average — raise this to only flag stronger, more convincing breakouts."
    )
    st.markdown(
        f"- **Stage 3** fires on a down week with volume at least "
        f"**{params['distribution_vol_mult']}x** its 50-day average, or on a "
        f"**failed breakout**: a new high that gives back more than "
        f"**{params['failed_breakout_giveback_pct']}%** within a few weeks."
    )
    st.markdown(
        f"- A stage must hold for **{params['min_run_weeks']} consecutive week"
        f"{'s' if params['min_run_weeks'] != 1 else ''}** before it's confirmed and "
        f"annotated on the chart. This is what filters out noisy single-week wiggles — "
        f"raise it for fewer, more confident transitions; lower it to catch changes earlier."
    )

    st.divider()
    st.markdown("#### :material/groups: Why sector/peer group matters")
    st.markdown(
        "Weinstein didn't just look at a stock in isolation — he checked its **industry "
        "group** first. A breakout is much more trustworthy when the stock's peers are "
        "also doing well, and when the stock is outperforming those peers, not just "
        "riding the same wave as everyone else."
    )
    group_label = params["group_level"].replace("_", " ")
    st.markdown(
        f"- Peer group is currently defined at the **{group_label}** level "
        f"(a synthetic index built by averaging every stock in that group)."
    )
    if params["require_group_strength"]:
        st.markdown(
            "- **Require peer group not in decline** is ON: a stock's Stage 2 won't confirm "
            "if its own peer group is in active decline (Stage 4) — don't buy strength in a "
            "sinking industry. This deliberately doesn't require the group to already be "
            "Stage 2 itself, since leadership stocks break out *before* their group does."
        )
    else:
        st.markdown("- **Require peer group not in decline** is OFF: group weakness is ignored.")
    if params["require_rs_rising"]:
        st.markdown(
            "- **Require rising strength vs. peer group** is ON: a stock's Stage 2 won't "
            "confirm unless it's *outperforming* its group (a rising relative-strength line), "
            "not just moving up with it."
        )
    else:
        st.markdown("- **Require rising strength vs. peer group** is OFF: relative strength is ignored.")


def render_about():
    """Full explainer for the two independent strategies, shown in the
    sidebar 'About' pane."""
    st.markdown("### :material/menu_book: About this app")
    st.markdown(
        "Two **independent** ways to decide buy/sell, both from OHLCV data only:"
    )
    st.markdown(
        "1. **Weinstein Stage 1-4** — a trend-regime label from the 30-week moving "
        "average (base → advance → top → decline), with peer-group confirmation.\n"
        "2. **Signal score** — a 0-100 score from six price/volume factors; a buy "
        "fires when it crosses your threshold."
    )
    st.caption("Use the 'What does this mean?' button for the Stage framework. The Signal score system is explained below.")

    st.divider()
    st.markdown("#### :material/query_stats: How the Signal score works")
    st.markdown(
        "Every day, each stock gets a **0-100 score** = the sum of six weighted factors. "
        "A high score is a stock that's **trending up, tightly based on drying volume, "
        "near its highs, and beating the market** — the classic accumulation setup."
    )
    st.markdown(
        "| Factor | Full credit when… |\n"
        "|---|---|\n"
        "| **1-month return** | 20-day return ≥ 20% |\n"
        "| **EMA alignment** | close > EMA20 > EMA50 > EMA200 |\n"
        "| **Volume dry-up** | 5-day avg volume ≤ 0.7× the 20-day avg |\n"
        "| **ATR compression** | volatility well below 20 days ago |\n"
        "| **Near high** | price at its 60-day high |\n"
        "| **Relative strength** | beats the market benchmark by 20% |"
    )
    st.markdown(
        "- Each **factor weight** sets how much that trait counts toward the score "
        "(defaults sum to 100). Raise a weight to make that trait matter more; set it to 0 to ignore it.\n"
        "- The **buy threshold** is the score a stock must cross *up through* to trigger a buy. "
        "Higher = fewer, higher-conviction signals.\n"
        "- **Exit rules** close a trade at the earliest of the **stop loss**, a **trailing stop** "
        "(from the highest price reached), or the **max holding period**."
    )
    st.caption(
        "The market benchmark for relative strength is an equal-weighted index of the whole "
        "universe — so 'outperformance' always means vs. the broad market, even when you filter "
        "to a few stocks."
    )
    st.markdown(
        "These settings drive the **Signal buy/sell markers** on Charts, the **Signal backtest**, "
        "and the **sector leaderboard**."
    )


st.title("NSE Market Data")

symbols = get_symbols()

# --- Sidebar: stage classifier settings, shared across Charts / Screener / Backtest ---
SECTIONS = ["Charts", "Stage screener", "Strategy backtest", "Signal backtest", "Query / Tables"]

with st.sidebar:
    section = st.radio("Go to", SECTIONS, key="nav_section")

    # Which strategy's settings are relevant here. The relevant group renders
    # inline; the other collapses to a single expander row. (Streamlit tabs
    # can't drive the sidebar, so navigation lives here in the left pane. And
    # inline-vs-expander is more reliable than toggling an expander's
    # `expanded=` param, which sticks once a user has interacted with it.)
    stage_relevant = section in {"Charts", "Stage screener", "Strategy backtest"}
    signal_relevant = section in {"Charts", "Signal backtest"}

    def settings_group(title: str, relevant: bool):
        """Inline container when relevant, collapsed expander row when not.
        Either way the contained widgets always render, so the derived params
        stay valid on every section."""
        if relevant:
            st.markdown(f"##### {title}")
            return st.container()
        return st.expander(title, expanded=False)

    with st.expander(":material/menu_book: About / how this works", expanded=False):
        render_about()

    st.divider()
    with settings_group(":material/tune: Stage classifier settings", stage_relevant):
        weekly_flat_pct = st.slider("30w MA flat threshold (%/week)", 0.05, 1.0, sc.DEFAULTS["weekly_flat_pct"], 0.05)
        daily_flat_pct = st.slider("200d/50d MA flat threshold (%/day)", 0.01, 0.3, sc.DEFAULTS["daily_flat_pct"], 0.01)
        whipsaw_band_pct = st.slider("Stage 1 whipsaw band (% of 30w MA)", 5.0, 25.0, sc.DEFAULTS["whipsaw_band_pct"], 1.0)
        distribution_vol_mult = st.slider("Stage 3 distribution volume (x 50d avg)", 1.0, 3.0, sc.DEFAULTS["distribution_vol_mult"], 0.1)
        breakout_vol_mult = st.slider("Stage 2 breakout volume (x 50d avg)", 1.5, 4.0, sc.DEFAULTS["breakout_vol_mult"], 0.1)
        failed_breakout_giveback_pct = st.slider("Failed-breakout giveback (%)", 1.0, 10.0, sc.DEFAULTS["failed_breakout_giveback_pct"], 0.5)
        min_run_weeks = st.slider("Min. weeks to confirm a stage", 1, 8, sc.DEFAULTS["min_run_weeks"], 1)

        st.caption(":material/groups: Group / relative-strength confirmation (Weinstein's group analysis)")
        group_level = st.selectbox(
            "Peer group defined by", ["macro_sector", "sector", "industry", "basic_industry"],
            index=["macro_sector", "sector", "industry", "basic_industry"].index(sc.DEFAULTS["group_level"]),
            format_func=lambda s: s.replace("_", " ").title(),
        )
        require_group_strength = st.checkbox(
            "Require peer group not in decline", value=sc.DEFAULTS["require_group_strength"],
            help="A stock's Stage 2 only confirms if its peer group isn't itself in active decline (Stage 4).",
        )
        require_rs_rising = st.checkbox(
            "Require rising strength vs. peer group", value=sc.DEFAULTS["require_rs_rising"],
            help="A stock's Stage 2 only confirms if it's outperforming its peer group, not just moving with it.",
        )
        if "show_help" not in st.session_state:
            st.session_state.show_help = False
        if st.button(":material/help: What does this mean?", width="stretch"):
            st.session_state.show_help = not st.session_state.show_help

    stage_params = {
        **sc.DEFAULTS,
        "weekly_flat_pct": weekly_flat_pct,
        "daily_flat_pct": daily_flat_pct,
        "whipsaw_band_pct": whipsaw_band_pct,
        "distribution_vol_mult": distribution_vol_mult,
        "breakout_vol_mult": breakout_vol_mult,
        "failed_breakout_giveback_pct": failed_breakout_giveback_pct,
        "min_run_weeks": min_run_weeks,
        "group_level": group_level,
        "require_group_strength": require_group_strength,
        "require_rs_rising": require_rs_rising,
    }
    if st.session_state.show_help:
        with st.container(border=True):
            render_stage_help(stage_params)

    with settings_group(":material/query_stats: Signal score settings", signal_relevant):
        st.caption("A second, independent buy/sell system: a 0-100 OHLCV-derived score. See **About** above for the full explainer.")
        st.markdown("**Factor weights** — how much each trait counts toward the 0-100 score.")
        weight_return = st.slider(
            "1-month return", 0.0, 40.0, se.DEFAULTS["weight_return"], 1.0,
            help="Rewards recent price momentum (20-day return). Higher weight = fast movers dominate the score; 0 = ignore momentum.")
        weight_ema_alignment = st.slider(
            "EMA alignment", 0.0, 40.0, se.DEFAULTS["weight_ema_alignment"], 1.0,
            help="Rewards a clean uptrend stack (close > EMA20 > EMA50 > EMA200). Higher = trend structure matters more.")
        weight_volume_dryup = st.slider(
            "Volume dry-up", 0.0, 40.0, se.DEFAULTS["weight_volume_dryup"], 1.0,
            help="Rewards shrinking volume (quiet basing / accumulation). Higher = quiet bases count for more.")
        weight_atr_compression = st.slider(
            "ATR compression", 0.0, 40.0, se.DEFAULTS["weight_atr_compression"], 1.0,
            help="Rewards falling volatility vs 20 days ago (a tightening coil). Higher = tight consolidations count for more.")
        weight_near_high = st.slider(
            "Near high", 0.0, 40.0, se.DEFAULTS["weight_near_high"], 1.0,
            help="Rewards price sitting near its 60-day high. Higher = only near-breakout stocks score.")
        weight_relative_strength = st.slider(
            "Relative strength vs. market", 0.0, 40.0, se.DEFAULTS["weight_relative_strength"], 1.0,
            help="Rewards beating the broad-market benchmark. Higher = only true outperformers score.")
        total_weight = (weight_return + weight_ema_alignment + weight_volume_dryup +
                         weight_atr_compression + weight_near_high + weight_relative_strength)
        st.caption(f"Max possible score: {total_weight:.0f}")

        buy_score_threshold = st.slider(
            "Buy score threshold", 20.0, min(100.0, total_weight) if total_weight else 100.0,
            min(se.DEFAULTS["buy_score_threshold"], total_weight) if total_weight else 60.0, 1.0,
            help="Score a stock must cross UP through to trigger a buy. Higher = fewer, higher-conviction signals; lower = more, weaker signals.")
        st.markdown("**Exit rules** — a trade closes at the earliest of these.")
        se_stop_loss_pct = st.slider(
            "Stop loss (%)", 1.0, 20.0, se.DEFAULTS["stop_loss_pct"], 0.5, key="se_sl",
            help="Sell if price falls this far below your entry. Lower = tighter risk, cut losers faster; higher = more room, fewer whipsaws.")
        trailing_stop_pct = st.slider(
            "Trailing stop (%)", 1.0, 30.0, se.DEFAULTS["trailing_stop_pct"], 0.5,
            help="Sell if price falls this far below the highest price reached since entry (locks in gains as it runs). Lower = protect profits sooner; higher = let winners breathe.")
        max_holding_days = st.slider(
            "Max holding period (days)", 5, 180, se.DEFAULTS["max_holding_days"], 5,
            help="Force an exit after this many days regardless. Lower = faster capital turnover; higher = give trades more time to work.")

    score_params = {
        **se.DEFAULTS,
        "weight_return": weight_return, "weight_ema_alignment": weight_ema_alignment,
        "weight_volume_dryup": weight_volume_dryup, "weight_atr_compression": weight_atr_compression,
        "weight_near_high": weight_near_high, "weight_relative_strength": weight_relative_strength,
        "buy_score_threshold": buy_score_threshold, "stop_loss_pct": se_stop_loss_pct,
        "trailing_stop_pct": trailing_stop_pct, "max_holding_days": max_holding_days,
    }

if section == "Charts":
    col1, col2 = st.columns([1, 3])
    with col1:
        symbol = st.selectbox("Symbol", symbols, key="symbol_select",
                               index=symbols.index("KIRLOSENG") if "KIRLOSENG" in symbols else 0)
        interval = st.segmented_control("Interval", ["Daily", "Weekly"], default="Daily")
        show_stages = st.checkbox("Show stage annotations", value=True)
        show_score_signals = st.checkbox("Show score-engine buy/sell markers", value=False)
        st.caption("Moving average overlay")
        overlay_choices = st.pills(
            "Moving average overlay", ["30w MA", "200d MA", "50d MA", "30w band"],
            selection_mode="multi", default=["30w MA"], label_visibility="collapsed",
            help="30w band = the Stage 1 whipsaw threshold shaded around the 30w MA",
        )
        show_ma30w = "30w MA" in overlay_choices
        show_ma200 = "200d MA" in overlay_choices
        show_ma50 = "50d MA" in overlay_choices
        show_band = "30w band" in overlay_choices

    df = get_prices(symbol)
    if df.empty:
        st.warning(f"No data for {symbol}")
    else:
        plot_df = df
        if interval == "Weekly":
            plot_df = (
                df.set_index("date")
                .resample("W-FRI")
                .agg({"open": "first", "high": "max", "low": "min",
                      "close": "last", "volume": "sum"})
                .dropna()
                .reset_index()
            )

        min_d, max_d = plot_df["date"].min().date(), plot_df["date"].max().date()
        date_range = st.slider("Date range", min_value=min_d, max_value=max_d,
                                value=(min_d, max_d))
        mask = (plot_df["date"].dt.date >= date_range[0]) & (plot_df["date"].dt.date <= date_range[1])
        plot_df = plot_df[mask]

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25], vertical_spacing=0.03,
        )
        fig.add_trace(go.Candlestick(
            x=plot_df["date"], open=plot_df["open"], high=plot_df["high"],
            low=plot_df["low"], close=plot_df["close"], name=symbol,
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=plot_df["date"], y=plot_df["volume"], name="Volume",
            marker_color="rgba(100,100,100,0.4)",
        ), row=2, col=1)

        if show_ma30w or show_ma200 or show_ma50 or show_band:
            overlay = sc.daily_overlay(df, stage_params)
            overlay = overlay[(overlay["date"].dt.date >= date_range[0]) & (overlay["date"].dt.date <= date_range[1])]
            if show_band:
                fig.add_trace(go.Scatter(
                    x=overlay["date"], y=overlay["ma30w_upper"], line=dict(width=0),
                    showlegend=False, hoverinfo="skip",
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=overlay["date"], y=overlay["ma30w_lower"], line=dict(width=0),
                    fill="tonexty", fillcolor="rgba(158,158,158,0.15)",
                    name="30w band", hoverinfo="skip",
                ), row=1, col=1)
            if show_ma30w:
                fig.add_trace(go.Scatter(
                    x=overlay["date"], y=overlay["ma30w"], line=dict(color="#9e9e9e", width=1.5),
                    name="30w MA",
                ), row=1, col=1)
            if show_ma200:
                fig.add_trace(go.Scatter(
                    x=overlay["date"], y=overlay["ma200"], line=dict(color="#42a5f5", width=1.2),
                    name="200d MA",
                ), row=1, col=1)
            if show_ma50:
                fig.add_trace(go.Scatter(
                    x=overlay["date"], y=overlay["ma50"], line=dict(color="#ab47bc", width=1.2),
                    name="50d MA",
                ), row=1, col=1)

        transitions = get_stage_transitions(symbol, stage_params)
        if show_stages and not transitions.empty:
            visible = transitions[
                (transitions["date"].dt.date >= date_range[0]) & (transitions["date"].dt.date <= date_range[1])
            ]
            for _, t in visible.iterrows():
                color = sc.STAGE_COLORS[t["stage"]]
                fig.add_vline(
                    x=t["date"], line_width=1, line_dash="dash", line_color=color, row=1, col=1,
                )
                fig.add_annotation(
                    x=t["date"], y=1, yref="y domain", yanchor="bottom", row=1, col=1,
                    text=f"S{t['stage']}", showarrow=False, font=dict(color=color, size=11),
                )

        score_trades = pd.DataFrame()
        if show_score_signals:
            score_trades = get_symbol_score_trades(symbol, score_params)
            if not score_trades.empty:
                visible_trades = score_trades[
                    (score_trades["entry_date"].dt.date >= date_range[0]) & (score_trades["entry_date"].dt.date <= date_range[1])
                ]
                BUY_COLOR, SELL_COLOR = "#2ecc71", "#e74c3c"
                # Buys and sells as full-height vertical lines, each tagged with
                # an up/down arrow at the BOTTOM of the price panel. The stage
                # annotations own the top (S1-S4 labels), so keeping the score
                # arrows at the bottom cleanly separates the two systems.
                for _, t in visible_trades.iterrows():
                    fig.add_vline(x=t["entry_date"], line_width=1.5, line_dash="dot",
                                  line_color=BUY_COLOR, row=1, col=1)
                    fig.add_annotation(x=t["entry_date"], y=0.01, yref="y domain", yanchor="bottom",
                                       row=1, col=1, text="▲", showarrow=False,
                                       font=dict(color=BUY_COLOR, size=15))
                    fig.add_vline(x=t["exit_date"], line_width=1.5, line_dash="dot",
                                  line_color=SELL_COLOR, row=1, col=1)
                    fig.add_annotation(x=t["exit_date"], y=0.01, yref="y domain", yanchor="bottom",
                                       row=1, col=1, text="▼", showarrow=False,
                                       font=dict(color=SELL_COLOR, size=15))
                # Invisible traces purely to keep Score buy / Score sell in the legend.
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers", name="Score buy",
                    marker=dict(symbol="triangle-up", size=11, color=BUY_COLOR),
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers", name="Score sell",
                    marker=dict(symbol="triangle-down", size=11, color=SELL_COLOR),
                ), row=1, col=1)

        fig.update_layout(
            height=650, xaxis_rangeslider_visible=False,
            title=f"{symbol} — {interval}",
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, width="stretch")

        latest = plot_df.iloc[-1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Last Close", f"{latest['close']:.2f}")
        m2.metric("Period High", f"{plot_df['high'].max():.2f}")
        m3.metric("Period Low", f"{plot_df['low'].min():.2f}")
        pct_change = (plot_df["close"].iloc[-1] / plot_df["close"].iloc[0] - 1) * 100
        m4.metric("Period Change", f"{pct_change:+.1f}%")

        if show_stages:
            st.subheader("Stage transitions")
            if transitions.empty:
                st.caption("Not enough history yet to classify stages for this symbol.")
            else:
                display_transitions = transitions.sort_values("date", ascending=False).copy()
                display_transitions["date"] = display_transitions["date"].dt.date
                st.dataframe(
                    display_transitions[["date", "stage_name", "close"]]
                    .rename(columns={"stage_name": "stage", "close": "close at transition"}),
                    width="stretch", height=250,
                )

        if show_score_signals:
            st.subheader("Score-engine trades")
            if score_trades.empty:
                st.caption("No score-engine buy signals fired for this symbol at the current threshold.")
            else:
                display_trades = score_trades.sort_values("entry_date", ascending=False).copy()
                display_trades["entry_date"] = display_trades["entry_date"].dt.date
                display_trades["exit_date"] = display_trades["exit_date"].dt.date
                st.dataframe(
                    display_trades[["entry_date", "entry_price", "exit_date", "exit_price", "exit_reason", "pnl_pct"]],
                    width="stretch", height=250,
                )

if section == "Stage screener":
    st.caption(
        "Find stocks that entered a given stage within a date window, using the classifier "
        "settings in the sidebar. First scan takes ~a minute across the full universe; "
        "cached after that until you change the settings."
    )
    s1, s2, s3 = st.columns([1.2, 1, 1])
    with s1:
        stage_filter = st.multiselect(
            "Stage(s)", [1, 2, 3, 4], default=[2],
            format_func=lambda s: sc.STAGE_NAMES[s],
        )
    with s2:
        screen_start = st.date_input("From", value=pd.Timestamp.today() - pd.Timedelta(days=180))
    with s3:
        screen_end = st.date_input("To", value=pd.Timestamp.today())

    screener_sector_symbols = render_sector_filters("screener")

    if st.button(":material/search: Scan universe", type="primary"):
        st.session_state["screener_transitions"] = get_universe_transitions(stage_params)

    if "screener_transitions" in st.session_state:
        universe_transitions = st.session_state["screener_transitions"]
        results = universe_transitions[
            universe_transitions["stage"].isin(stage_filter) &
            (universe_transitions["date"].dt.date >= screen_start) &
            (universe_transitions["date"].dt.date <= screen_end)
        ].sort_values("date", ascending=False).copy()
        if screener_sector_symbols is not None:
            results = results[results["symbol"].isin(screener_sector_symbols)]

        st.write(f"**{len(results)} matches**")
        if results.empty:
            st.caption("No stocks entered the selected stage(s) in this window.")
        else:
            results = results.merge(get_sector_map(), on="symbol", how="left")
            results["days_since"] = (pd.Timestamp.today().normalize() - results["date"]).dt.days
            results["date"] = results["date"].dt.date
            st.dataframe(
                results[["symbol", "stage_name", "date", "close", "days_since", "ma30w_slope_pct",
                          "sector", "basic_industry"]]
                .rename(columns={"stage_name": "stage", "close": "close at transition",
                                  "ma30w_slope_pct": "30w slope %/wk"}),
                width="stretch", height=450,
            )
            st.caption("Pick any symbol above in the Charts tab's symbol dropdown to see its chart.")
    else:
        st.info("Click **Scan universe** to run the screen.")

if section == "Strategy backtest":
    st.caption(
        "Simulate: buy N days after a Stage 2 confirmation, sell at the earliest of a stop-loss "
        "or N days after the first Stage 3/4 confirmation after entry. Uses the same universe "
        "and stage settings as the screener."
    )
    st.info(
        ":material/info: Runs across the full ~1149-stock market-cap>₹2000cr universe by default. "
        "Filtering by market cap / ROCE / sales growth isn't wired in yet — that needs a "
        "fundamentals data source beyond what Kite provides — but sector/industry filtering below works.",
        icon=":material/info:",
    )

    backtest_symbols = render_universe_selector("backtest", symbols)

    with st.form("backtest_form"):
        b1, b2, b3 = st.columns(3)
        with b1:
            bt_start = st.date_input("Start date", value=pd.Timestamp.today() - pd.Timedelta(days=3 * 365))
            bt_end = st.date_input("End date", value=pd.Timestamp.today())
            total_capital = st.number_input("Total starting capital (₹)", min_value=10_000.0,
                                             value=bt.DEFAULTS["total_capital"], step=50_000.0)
        with b2:
            entry_delay_days = st.slider("Days after Stage 2 to enter", 0, 20, bt.DEFAULTS["entry_delay_days"])
            exit_delay_days = st.slider("Days after Stage 3/4 to exit", 0, 20, bt.DEFAULTS["exit_delay_days"])
            stop_loss_pct = st.slider("Stop loss (%)", 1.0, 20.0, bt.DEFAULTS["stop_loss_pct"], 0.5)
        with b3:
            participation_pct = st.slider("% of signals taken (top by momentum)", 5, 100,
                                           int(bt.DEFAULTS["participation_pct"]), 5)
            position_size = st.number_input("Position size per trade (₹)", min_value=1_000.0,
                                             value=bt.DEFAULTS["position_size"], step=5_000.0)
            enable_rotation = st.checkbox("Rotate out of the worst loser when capital is full",
                                           value=bt.DEFAULTS["enable_rotation"])

        run_backtest = st.form_submit_button(":material/play_arrow: Run backtest", type="primary")

    if run_backtest and backtest_symbols == set():
        st.warning("No stocks selected — pick at least one stock or switch the universe to Full universe.")
    elif run_backtest:
        backtest_params = {
            "start_date": pd.Timestamp(bt_start), "end_date": pd.Timestamp(bt_end),
            "entry_delay_days": entry_delay_days, "exit_delay_days": exit_delay_days,
            "stop_loss_pct": stop_loss_pct, "participation_pct": participation_pct,
            "position_size": position_size, "total_capital": total_capital,
            "enable_rotation": enable_rotation,
        }
        transitions = get_universe_transitions(stage_params)
        with st.spinner("Running simulation..."):
            frames = get_price_frames()
            if backtest_symbols is not None:
                transitions = transitions[transitions["symbol"].isin(backtest_symbols)]
                frames = {sym: df for sym, df in frames.items() if sym in backtest_symbols}
            result = bt.run_backtest(frames, transitions, backtest_params)
        st.session_state["backtest_result"] = result

    if "backtest_result" in st.session_state:
        result = st.session_state["backtest_result"]
        metrics = result["metrics"]

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total return", f"{metrics['total_return_pct']:+.1f}%")
        k2.metric("CAGR", f"{metrics['cagr_pct']:+.1f}%")
        k3.metric("Max drawdown", f"{metrics['max_drawdown_pct']:.1f}%")
        k4.metric("Win rate", f"{metrics['win_rate_pct']:.0f}%")
        k5.metric("Trades", f"{metrics['num_trades']}")

        if not result["equity_curve"].empty:
            st.line_chart(result["equity_curve"].set_index("date")["value"], height=350)

        st.subheader("Trade log")
        if result["trades"].empty:
            st.caption("No trades were taken with these settings.")
        else:
            trades = result["trades"].sort_values("entry_date", ascending=False)
            st.dataframe(trades, width="stretch", height=350)
            st.download_button(
                "Download trade log as CSV", trades.to_csv(index=False),
                file_name="backtest_trades.csv", mime="text/csv",
            )

        if not result["missed_signals"].empty:
            with st.expander(f":material/report: {len(result['missed_signals'])} signals skipped (no free capital)"):
                st.dataframe(result["missed_signals"], width="stretch", height=200)
    else:
        st.info("Set your strategy parameters above and click **Run backtest**.")

if section == "Signal backtest":
    st.caption(
        "A second, independent system: buy when the OHLCV-derived score crosses above the "
        "threshold (see sidebar), sell at the earliest of a stop-loss, a trailing stop, or the "
        "max holding period. No Weinstein stages involved -- pure price/volume scoring."
    )
    st.info(
        f":material/info: Score built from: 1-month return, EMA alignment, volume dry-up, "
        f"ATR compression, proximity to highs, and relative strength vs. a broad-market benchmark "
        f"(an equal-weighted index of the full universe). Current max score: "
        f"{sum(score_params[k] for k in ['weight_return','weight_ema_alignment','weight_volume_dryup','weight_atr_compression','weight_near_high','weight_relative_strength']):.0f}.",
        icon=":material/info:",
    )

    signal_symbols = render_universe_selector("signal_backtest", symbols)

    with st.form("signal_backtest_form"):
        sb1, sb2 = st.columns(2)
        with sb1:
            sig_start = st.date_input("Start date", value=pd.Timestamp.today() - pd.Timedelta(days=3 * 365), key="sig_start")
            sig_end = st.date_input("End date", value=pd.Timestamp.today(), key="sig_end")
            sig_total_capital = st.number_input("Total starting capital (₹)", min_value=10_000.0,
                                                 value=se.DEFAULTS["total_capital"], step=50_000.0, key="sig_capital")
        with sb2:
            sig_position_size = st.number_input("Position size per trade (₹)", min_value=1_000.0,
                                                 value=se.DEFAULTS["position_size"], step=5_000.0, key="sig_position")
            sig_participation_pct = st.slider("% of signals taken (top by score)", 5, 100,
                                               int(se.DEFAULTS["participation_pct"]), 5, key="sig_participation")
            sig_enable_rotation = st.checkbox("Rotate out of the worst loser when capital is full",
                                               value=se.DEFAULTS["enable_rotation"], key="sig_rotation")

        run_signal_backtest = st.form_submit_button(":material/play_arrow: Run backtest", type="primary")

    if run_signal_backtest and signal_symbols == set():
        st.warning("No stocks selected — pick at least one stock or switch the universe to Full universe.")
    elif run_signal_backtest:
        signal_backtest_params = {
            **score_params,
            "start_date": pd.Timestamp(sig_start), "end_date": pd.Timestamp(sig_end),
            "total_capital": sig_total_capital, "position_size": sig_position_size,
            "participation_pct": sig_participation_pct, "enable_rotation": sig_enable_rotation,
        }
        with st.spinner("Running simulation (scoring the selected stocks day by day)..."):
            frames = get_price_frames()
            if signal_symbols is not None:
                frames = {sym: df for sym, df in frames.items() if sym in signal_symbols}
            benchmark = get_market_benchmark()
            result = se.run_score_backtest(frames, signal_backtest_params, benchmark_close=benchmark)
        st.session_state["signal_backtest_result"] = result

    if "signal_backtest_result" in st.session_state:
        result = st.session_state["signal_backtest_result"]
        metrics = result["metrics"]

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total return", f"{metrics['total_return_pct']:+.1f}%")
        k2.metric("CAGR", f"{metrics['cagr_pct']:+.1f}%")
        k3.metric("Max drawdown", f"{metrics['max_drawdown_pct']:.1f}%")
        k4.metric("Win rate", f"{metrics['win_rate_pct']:.0f}%")
        k5.metric("Trades", f"{metrics['num_trades']}")

        if not result["equity_curve"].empty:
            st.line_chart(result["equity_curve"].set_index("date")["value"], height=350)

        st.subheader("Trade log")
        if result["trades"].empty:
            st.caption("No trades were taken with these settings.")
        else:
            trades = result["trades"].sort_values("entry_date", ascending=False)
            st.dataframe(trades, width="stretch", height=350)
            st.download_button(
                "Download trade log as CSV", trades.to_csv(index=False),
                file_name="signal_backtest_trades.csv", mime="text/csv",
            )
    else:
        st.info("Set your strategy parameters above and click **Run backtest**.")

    st.divider()
    st.subheader(":material/leaderboard: Compare across sectors / industries")
    st.caption(
        "Runs this same signal strategy (current sidebar score settings + the date/capital "
        "settings above) independently on every group, so you can rank which sector/industry "
        "it performed best in over the selected period. Click any column header to re-sort."
    )
    cmp_col1, cmp_col2 = st.columns([1, 1])
    with cmp_col1:
        cmp_level = st.selectbox(
            "Group by", ["macro_sector", "sector", "industry", "basic_industry"],
            index=1, format_func=lambda s: s.replace("_", " ").title(), key="cmp_level",
        )
    with cmp_col2:
        st.caption(
            "Coarser levels (macro sector, sector) run in seconds; finer levels "
            "(industry, basic industry) have more groups and take longer."
        )
    if st.button(":material/play_arrow: Run sector comparison", key="run_sector_cmp"):
        comparison_params = {
            **score_params,
            "start_date": pd.Timestamp(sig_start), "end_date": pd.Timestamp(sig_end),
            "total_capital": sig_total_capital, "position_size": sig_position_size,
            "participation_pct": sig_participation_pct, "enable_rotation": sig_enable_rotation,
        }
        st.session_state["sector_comparison"] = get_sector_comparison(cmp_level, comparison_params)
        st.session_state["sector_comparison_level"] = cmp_level

    if "sector_comparison" in st.session_state:
        comp_df = st.session_state["sector_comparison"].rename(
            columns={"group": st.session_state.get("sector_comparison_level", "group")}
        )
        st.caption(f"{len(comp_df)} groups, ranked by total return (click headers to re-sort).")
        st.dataframe(comp_df, width="stretch", height=430, hide_index=True)
        st.download_button(
            "Download comparison as CSV", comp_df.to_csv(index=False),
            file_name="sector_comparison.csv", mime="text/csv",
        )

if section == "Query / Tables":
    st.subheader("Tables")
    con = get_connection()
    table_choice = st.selectbox("Preview table", ["instruments", "daily_prices"])
    if table_choice == "daily_prices":
        preview_symbol = st.selectbox("Filter by symbol (optional)", ["(all)"] + symbols)
        if preview_symbol == "(all)":
            preview_df = con.execute(
                "SELECT * FROM daily_prices ORDER BY date DESC LIMIT 500"
            ).df()
        else:
            preview_df = con.execute(
                "SELECT * FROM daily_prices WHERE tradingsymbol = ? ORDER BY date DESC LIMIT 1000",
                [preview_symbol],
            ).df()
    else:
        preview_df = con.execute("SELECT * FROM instruments ORDER BY tradingsymbol").df()
    st.dataframe(preview_df, width="stretch", height=300)

    st.subheader("Custom SQL")
    default_query = "SELECT tradingsymbol, count(*) AS rows, min(date) AS start, max(date) AS end\nFROM daily_prices\nGROUP BY tradingsymbol\nORDER BY rows DESC\nLIMIT 20"
    query = st.text_area("Query", value=default_query, height=140)
    run_query = st.button("Run query")
    if run_query:
        try:
            result_df = con.execute(query).df()
            st.success(f"{len(result_df)} rows returned")
            st.dataframe(result_df, width="stretch", height=400)
            st.download_button(
                "Download as CSV",
                result_df.to_csv(index=False),
                file_name="query_result.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Query failed: {e}")

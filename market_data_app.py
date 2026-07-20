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


st.title("NSE Market Data")

symbols = get_symbols()

# --- Sidebar: stage classifier settings, shared across Charts / Screener / Backtest ---
with st.sidebar:
    st.header(":material/tune: Stage classifier settings")
    st.caption("Shared across the Charts, Stage screener, and Strategy backtest tabs.")

    weekly_flat_pct = st.slider("30w MA flat threshold (%/week)", 0.05, 1.0, sc.DEFAULTS["weekly_flat_pct"], 0.05)
    daily_flat_pct = st.slider("200d/50d MA flat threshold (%/day)", 0.01, 0.3, sc.DEFAULTS["daily_flat_pct"], 0.01)
    whipsaw_band_pct = st.slider("Stage 1 whipsaw band (% of 30w MA)", 5.0, 25.0, sc.DEFAULTS["whipsaw_band_pct"], 1.0)
    distribution_vol_mult = st.slider("Stage 3 distribution volume (x 50d avg)", 1.0, 3.0, sc.DEFAULTS["distribution_vol_mult"], 0.1)
    breakout_vol_mult = st.slider("Stage 2 breakout volume (x 50d avg)", 1.5, 4.0, sc.DEFAULTS["breakout_vol_mult"], 0.1)
    failed_breakout_giveback_pct = st.slider("Failed-breakout giveback (%)", 1.0, 10.0, sc.DEFAULTS["failed_breakout_giveback_pct"], 0.5)
    min_run_weeks = st.slider("Min. weeks to confirm a stage", 1, 8, sc.DEFAULTS["min_run_weeks"], 1)

    st.divider()
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

    st.divider()
    st.header(":material/query_stats: Signal score settings")
    st.caption("A second, independent buy/sell system: a 0-100 OHLCV-derived score instead of Weinstein stages.")
    with st.expander("Factor weights", expanded=False):
        weight_return = st.slider("1-month return", 0.0, 40.0, se.DEFAULTS["weight_return"], 1.0)
        weight_ema_alignment = st.slider("EMA alignment", 0.0, 40.0, se.DEFAULTS["weight_ema_alignment"], 1.0)
        weight_volume_dryup = st.slider("Volume dry-up", 0.0, 40.0, se.DEFAULTS["weight_volume_dryup"], 1.0)
        weight_atr_compression = st.slider("ATR compression", 0.0, 40.0, se.DEFAULTS["weight_atr_compression"], 1.0)
        weight_near_high = st.slider("Near high", 0.0, 40.0, se.DEFAULTS["weight_near_high"], 1.0)
        weight_relative_strength = st.slider("Relative strength vs. market", 0.0, 40.0, se.DEFAULTS["weight_relative_strength"], 1.0)
        total_weight = (weight_return + weight_ema_alignment + weight_volume_dryup +
                         weight_atr_compression + weight_near_high + weight_relative_strength)
        st.caption(f"Max possible score: {total_weight:.0f}")

    buy_score_threshold = st.slider("Buy score threshold", 20.0, min(100.0, total_weight) if total_weight else 100.0,
                                     min(se.DEFAULTS["buy_score_threshold"], total_weight) if total_weight else 60.0, 1.0)
    with st.expander("Exit rules", expanded=False):
        se_stop_loss_pct = st.slider("Stop loss (%)", 1.0, 20.0, se.DEFAULTS["stop_loss_pct"], 0.5, key="se_sl")
        trailing_stop_pct = st.slider("Trailing stop (%)", 1.0, 30.0, se.DEFAULTS["trailing_stop_pct"], 0.5)
        max_holding_days = st.slider("Max holding period (days)", 5, 180, se.DEFAULTS["max_holding_days"], 5)

    score_params = {
        **se.DEFAULTS,
        "weight_return": weight_return, "weight_ema_alignment": weight_ema_alignment,
        "weight_volume_dryup": weight_volume_dryup, "weight_atr_compression": weight_atr_compression,
        "weight_near_high": weight_near_high, "weight_relative_strength": weight_relative_strength,
        "buy_score_threshold": buy_score_threshold, "stop_loss_pct": se_stop_loss_pct,
        "trailing_stop_pct": trailing_stop_pct, "max_holding_days": max_holding_days,
    }

    st.divider()
    if "show_help" not in st.session_state:
        st.session_state.show_help = False
    if st.button(":material/help: What does this mean?", width="stretch"):
        st.session_state.show_help = not st.session_state.show_help
    if st.session_state.show_help:
        with st.container(border=True):
            render_stage_help(stage_params)

tab_charts, tab_screener, tab_backtest, tab_signal_backtest, tab_query = st.tabs(
    ["Charts", "Stage screener", "Strategy backtest", "Signal backtest", "Query / Tables"]
)

with tab_charts:
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

with tab_screener:
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

with tab_backtest:
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

    backtest_sector_symbols = render_sector_filters("backtest")

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

    if run_backtest:
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
            if backtest_sector_symbols is not None:
                transitions = transitions[transitions["symbol"].isin(backtest_sector_symbols)]
                frames = {sym: df for sym, df in frames.items() if sym in backtest_sector_symbols}
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

with tab_signal_backtest:
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

    signal_sector_symbols = render_sector_filters("signal_backtest")

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

    if run_signal_backtest:
        signal_backtest_params = {
            **score_params,
            "start_date": pd.Timestamp(sig_start), "end_date": pd.Timestamp(sig_end),
            "total_capital": sig_total_capital, "position_size": sig_position_size,
            "participation_pct": sig_participation_pct, "enable_rotation": sig_enable_rotation,
        }
        with st.spinner("Running simulation (scoring the full universe day by day)..."):
            frames = get_price_frames()
            if signal_sector_symbols is not None:
                frames = {sym: df for sym, df in frames.items() if sym in signal_sector_symbols}
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

with tab_query:
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

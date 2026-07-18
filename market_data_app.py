"""
Local UI for querying nse_market_data.duckdb and viewing price charts.

Run:
    streamlit run market_data_app.py

Two tabs:
    Charts - pick a symbol (or several), see a candlestick + volume chart
    Query  - run arbitrary SQL against the DuckDB file, view/export results
"""

import duckdb
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

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
    return con.execute(
        "SELECT date, open, high, low, close, volume FROM daily_prices "
        "WHERE tradingsymbol = ? ORDER BY date", [symbol]
    ).df()


@st.cache_data
def get_stage_transitions(symbol: str, params: dict) -> pd.DataFrame:
    daily = get_prices(symbol)
    daily["date"] = pd.to_datetime(daily["date"])
    classified = sc.classify(daily, params)
    return sc.get_transitions(classified)


def render_stage_help(params: dict):
    """Plain-language explainer for the Stage 1-4 framework and what the
    current slider settings mean, shown in the right-hand help panel."""
    with st.container(border=True):
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


st.title("NSE Market Data")

symbols = get_symbols()
tab_charts, tab_query = st.tabs(["Charts", "Query / Tables"])

with tab_charts:
    if "show_help" not in st.session_state:
        st.session_state.show_help = False

    col1, col2, col3 = st.columns([1, 2, 1], vertical_alignment="bottom")
    with col1:
        symbol = st.selectbox("Symbol", symbols, index=symbols.index("KIRLOSENG") if "KIRLOSENG" in symbols else 0)
        interval = st.segmented_control("Interval", ["Daily", "Weekly"], default="Daily")
        show_stages = st.checkbox("Show stage annotations", value=True)
    with col3:
        if st.button(":material/help: Help", help="What do Stage 1-4 and these settings mean?"):
            st.session_state.show_help = not st.session_state.show_help

    with st.expander(":material/tune: Stage classifier settings"):
        e1, e2, e3 = st.columns(3)
        weekly_flat_pct = e1.slider("30w MA flat threshold (%/week)", 0.05, 1.0, sc.DEFAULTS["weekly_flat_pct"], 0.05)
        daily_flat_pct = e1.slider("200d/50d MA flat threshold (%/day)", 0.01, 0.3, sc.DEFAULTS["daily_flat_pct"], 0.01)
        whipsaw_band_pct = e2.slider("Stage 1 whipsaw band (% of 30w MA)", 5.0, 25.0, sc.DEFAULTS["whipsaw_band_pct"], 1.0)
        distribution_vol_mult = e2.slider("Stage 3 distribution volume (x 50d avg)", 1.0, 3.0, sc.DEFAULTS["distribution_vol_mult"], 0.1)
        breakout_vol_mult = e3.slider("Stage 2 breakout volume (x 50d avg)", 1.5, 4.0, sc.DEFAULTS["breakout_vol_mult"], 0.1)
        failed_breakout_giveback_pct = e3.slider("Failed-breakout giveback (%)", 1.0, 10.0, sc.DEFAULTS["failed_breakout_giveback_pct"], 0.5)
        min_run_weeks = e3.slider("Min. weeks to confirm a stage", 1, 8, sc.DEFAULTS["min_run_weeks"], 1)
        stage_params = {
            **sc.DEFAULTS,
            "weekly_flat_pct": weekly_flat_pct,
            "daily_flat_pct": daily_flat_pct,
            "whipsaw_band_pct": whipsaw_band_pct,
            "distribution_vol_mult": distribution_vol_mult,
            "breakout_vol_mult": breakout_vol_mult,
            "failed_breakout_giveback_pct": failed_breakout_giveback_pct,
            "min_run_weeks": min_run_weeks,
        }

    if st.session_state.show_help:
        main_col, help_col = st.columns([2.3, 1], gap="medium")
    else:
        main_col, help_col = st.container(), None

    with main_col:
        df = get_prices(symbol)
        if df.empty:
            st.warning(f"No data for {symbol}")
        else:
            df["date"] = pd.to_datetime(df["date"])
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

    if help_col is not None:
        with help_col:
            render_stage_help(stage_params)

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
    run = st.button("Run query")
    if run:
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

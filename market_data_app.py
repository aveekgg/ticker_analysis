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
    Sector leaders     - rank stocks by how consistently they beat their own
                          sector/industry equal-weight index across
                          non-overlapping return windows (pure price action)
    Query / Tables     - run arbitrary SQL against the DuckDB file
"""

import re
from datetime import datetime, timezone

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


def inject_mobile_css():
    """One responsive stylesheet so the whole app is usable on a phone. Streamlit
    exposes no server-side viewport width, so instead of device-branching the
    Python layout we restyle at <=640px via stable data-testid hooks. Desktop
    (>640px) is untouched. The main move: force every multi-column row to stack,
    which fixes all the 4-5 column control rows and metric rows at once."""
    st.markdown(
        """
        <style>
        @media (max-width: 640px) {
            /* Stack every st.columns row instead of squishing side-by-side */
            [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 0.5rem; }
            [data-testid="stColumn"] {
                flex: 1 1 100% !important;
                width: 100% !important;
                min-width: 100% !important;
            }
            /* Reclaim the wide desktop side padding */
            [data-testid="stMainBlockContainer"] { padding: 1rem 0.75rem 4rem; }
            /* Tame the oversized title / headers */
            [data-testid="stMainBlockContainer"] h1 { font-size: 1.6rem; line-height: 1.2; }
            [data-testid="stMainBlockContainer"] h2 { font-size: 1.3rem; }
            [data-testid="stMainBlockContainer"] h3 { font-size: 1.1rem; }
            /* Comfortable tap targets (~44px) */
            button[data-testid^="stBaseButton"] { min-height: 44px; }
            [data-testid="stSelectbox"] div[data-baseweb="select"] > div { min-height: 44px; }
            [data-testid="stNumberInput"] input,
            [data-testid="stTextInput"] input,
            [data-testid="stDateInput"] input { min-height: 40px; }
            [data-testid="stSlider"] [role="slider"] { height: 22px; width: 22px; }
            /* Full-width, self-scrolling data + charts */
            [data-testid="stDataFrame"] { width: 100% !important; }
            [data-testid="stPlotlyChart"], .js-plotly-plot { width: 100% !important; }
            /* The Plotly modebar icons are unusable on touch and overlap the chart */
            .js-plotly-plot .modebar { display: none !important; }
            /* Open the sidebar as a near-full-width drawer so settings are readable */
            [data-testid="stSidebar"] { min-width: 85vw !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_mobile_css()


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


# --- Sector leaders: consistent price-action outperformance vs peer group ---

LEADER_PERIODS = {
    "6 months": 182, "1 year": 365, "2 years": 730, "3 years": 1095, "5 years": 1825,
}
LEADER_WINDOWS = {  # label -> (approx days, offset) ; offset walks bounds back from today
    "1 week": (7, pd.DateOffset(weeks=1)),
    "1 month": (30, pd.DateOffset(months=1)),
    "3 months": (91, pd.DateOffset(months=3)),
    "6 months": (182, pd.DateOffset(months=6)),
    "1 year": (365, pd.DateOffset(years=1)),
}
LEADER_STEPS = {  # rolling-scan step -> how far each window's end date walks back
    "1 week": pd.DateOffset(weeks=1),
    "2 weeks": pd.DateOffset(weeks=2),
    "1 month": pd.DateOffset(months=1),
}


@st.cache_data(show_spinner="Building the universe close-price matrix (first time only)...")
def get_close_matrix() -> pd.DataFrame:
    """date x symbol matrix of closes for the whole universe."""
    frames = get_price_frames()
    return pd.DataFrame({s: df["close"] for s, df in frames.items()}).sort_index()


@st.cache_data
def get_price_date_range() -> tuple:
    """(earliest, latest) trading date in the DB -- cheap query straight off
    daily_prices so the End date picker can bound itself without triggering the
    full close-price matrix load."""
    con = get_connection()
    lo, hi = con.execute("SELECT min(date), max(date) FROM daily_prices").fetchone()
    return pd.Timestamp(lo), pd.Timestamp(hi)


@st.cache_data
def get_mcap_map() -> pd.DataFrame:
    """symbol -> current market cap (₹ cr). Kite has no market-cap field, so this
    comes from a scrape. Prefers the screener.in fundamentals scrape
    (fundamentals.csv, full-universe coverage via scrape_fundamentals.py); falls
    back to the older NSE quote scrape (nse_mcap_universe_raw.csv) if that's the
    only snapshot present. Re-run the relevant scraper to refresh."""
    try:
        df = pd.read_csv("fundamentals.csv")
        return df[["symbol", "mcap_cr"]].dropna()
    except FileNotFoundError:
        pass
    try:
        df = pd.read_csv("nse_mcap_universe_raw.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["symbol", "mcap_cr"])
    return df[df["status"] == "ok"][["symbol", "mcap_cr"]].dropna()


def mcap_segment(mcap_cr: float) -> str:
    """Approximate AMFI-style size buckets (thresholds in ₹ crore), with the
    small-cap band further split at ₹5,000cr -- the AMFI small/mid/large split
    alone still lumps a ₹32,000cr company in with a ₹500cr one."""
    if pd.isna(mcap_cr):
        return "Unknown"
    if mcap_cr >= 100_000:
        return "Large (>1L cr)"
    if mcap_cr >= 33_000:
        return "Mid (33k-1L cr)"
    if mcap_cr >= 5_000:
        return "Small (5k-33k cr)"
    return "Micro (<5k cr)"


MCAP_SEGMENT_ORDER = ["Large (>1L cr)", "Mid (33k-1L cr)", "Small (5k-33k cr)", "Micro (<5k cr)", "Unknown"]


def mcap_ladder(max_cr: float) -> list:
    """Non-uniform market-cap stops (₹ cr) for the range slider: fine steps at
    the low end where most of the universe sits, coarser as cap rises so the
    long right tail (a few ₹-lakh-crore names) doesn't swamp the slider. Steps:
    5k up to 50k, 10k to 100k, 20k to 200k, 50k to 500k, then 100k beyond."""
    stops, v = [0], 0
    for ceiling, step in [(50_000, 5_000), (100_000, 10_000), (200_000, 20_000),
                          (500_000, 50_000), (float("inf"), 100_000)]:
        while v < ceiling and v < max_cr:
            v += step
            stops.append(v)
        if v >= max_cr:
            break
    return stops


def format_mcap_cr(v: float) -> str:
    """Compact ₹-crore label, Indian style: 5000 -> '5k cr', 100000 -> '1L cr'."""
    if v >= 100_000:
        return f"{v / 100_000:g}L cr"
    if v >= 1_000:
        return f"{v / 1_000:g}k cr"
    return f"{v:g} cr"


# --- Fundamentals: per-company statement viewer, mirroring screener.in's own
# page layout so the scrape can be visually spot-checked against the source. ---

# Row order exactly as it appears on screener.in's company page (verified by
# fetching a live page this session) -- not alphabetical/discovery order, so
# the table reads the same top-to-bottom as the real page for easy comparison.
FUNDAMENTALS_METRIC_ORDER = {
    "profit_loss": ["Sales", "Expenses", "Operating Profit", "OPM %", "Other Income",
                     "Interest", "Depreciation", "Profit before tax", "Tax %", "Net Profit",
                     "EPS in Rs", "Dividend Payout %"],
    "quarters": ["Sales", "Expenses", "Operating Profit", "OPM %", "Other Income",
                  "Interest", "Depreciation", "Profit before tax", "Tax %", "Net Profit",
                  "EPS in Rs"],
    "balance_sheet": ["Equity Capital", "Reserves", "Borrowings", "Other Liabilities",
                       "Total Liabilities", "Fixed Assets", "CWIP", "Investments",
                       "Other Assets", "Total Assets"],
    "cash_flow": ["Cash from Operating Activity", "Cash from Investing Activity",
                   "Cash from Financing Activity", "Net Cash Flow", "Free Cash Flow", "CFO/OP"],
    "ratios": ["Debtor Days", "Inventory Days", "Days Payable", "Cash Conversion Cycle",
                "Working Capital Days", "ROCE %"],
}

FUNDAMENTALS_TABS = [
    ("quarters", "Quarterly results"), ("profit_loss", "Profit & loss"),
    ("balance_sheet", "Balance sheet"), ("cash_flow", "Cash flow"), ("ratios", "Ratios"),
]


def ordered_metrics_for_period(raw: pd.DataFrame, period_type: str) -> list:
    """Metric names available for one company at annual/quarterly granularity,
    in screener's own row order (concatenated across its annual statements, or
    just the quarters order for quarterly) so the chart's metric picker reads
    top-to-bottom the same as the tables below it -- with any metric outside
    the canonical order (e.g. ROE % for financials) appended alphabetically."""
    available = set(raw[raw["period_type"] == period_type]["metric"])
    if period_type == "quarterly":
        order = FUNDAMENTALS_METRIC_ORDER.get("quarters", [])
    else:
        order = [m for stmt in ("profit_loss", "balance_sheet", "cash_flow", "ratios")
                 for m in FUNDAMENTALS_METRIC_ORDER.get(stmt, [])]
    ordered = [m for m in order if m in available]
    ordered += sorted(available - set(ordered))
    return ordered


@st.cache_data
def get_fundamentals_raw(symbol: str) -> pd.DataFrame:
    con = get_connection()
    return con.execute(
        "SELECT statement, period_type, metric, period_end, value, unit "
        "FROM fundamentals_raw WHERE symbol = ? ORDER BY period_end",
        [symbol],
    ).df()


@st.cache_data
def get_fundamentals_scrape_info(symbol: str):
    con = get_connection()
    row = con.execute(
        "SELECT statement_basis, status, scraped_at FROM fundamentals_scrape_log WHERE symbol = ?",
        [symbol],
    ).fetchone()
    return row  # (basis, status, scraped_at) or None if never scraped


def format_fundamentals_value(value: float, unit: str) -> str:
    if pd.isna(value):
        return ""
    if unit == "pct":
        return f"{value:.0f}%"
    if unit == "rs_per_share":
        return f"{value:.2f}"
    if unit == "days":
        return f"{value:.0f}"
    return f"{value:,.0f}"  # 'cr'


def pivot_fundamentals_statement(raw: pd.DataFrame, statement: str):
    """metric-as-row / period-as-column tables for one statement, ordered to
    match screener.in's own layout. Periods run oldest -> newest left to right.
    Returns (display, values): display holds formatted strings for reading;
    values holds the raw floats (NaN where blank), same shape/index/columns --
    used for range-selection stats and to resolve growth-rate projections."""
    slice_df = raw[raw["statement"] == statement]
    if slice_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    formatted = slice_df.assign(
        display=slice_df.apply(lambda r: format_fundamentals_value(r["value"], r["unit"]), axis=1),
        period_label=pd.to_datetime(slice_df["period_end"]).dt.strftime("%b %Y"),
    )
    wide_display = formatted.pivot_table(index="metric", columns="period_label", values="display", aggfunc="first")
    wide_values = formatted.pivot_table(index="metric", columns="period_label", values="value", aggfunc="first")
    # Column order: chronological, not alphabetical (pivot_table sorts labels
    # alphabetically by default, which scrambles "Mar" before "Jun" etc.)
    period_order = (
        formatted[["period_label", "period_end"]].drop_duplicates()
        .sort_values("period_end")["period_label"]
    )
    cols = [p for p in period_order if p in wide_display.columns]
    wide_display = wide_display.reindex(columns=cols)
    wide_values = wide_values.reindex(columns=cols)
    # Row order: screener's own order first, then anything unexpected appended
    # (e.g. a company-specific line item) rather than silently dropped.
    canonical = FUNDAMENTALS_METRIC_ORDER.get(statement, [])
    ordered_rows = [m for m in canonical if m in wide_display.index] + \
                   [m for m in wide_display.index if m not in canonical]
    return wide_display.reindex(index=ordered_rows).fillna(""), wide_values.reindex(index=ordered_rows)


# --- User projections & notes: a separate local DuckDB file, deliberately not
# nse_market_data.duckdb. Keeps the scraped data read-only/untouched, avoids any
# lock contention with the app's existing read-only connection, and makes these
# personal annotations trivially "local" -- just another file the user owns,
# independent of the scrape pipeline. Only Profit & Loss and Ratios are
# projectable (that's where Sales/OPM %/Net Profit/ROCE % live); the other three
# statements stay historical-only. ---

PROJECTIONS_DB_FILE = "fundamentals_user_data.duckdb"
PROJECTION_STATEMENTS = {"profit_loss", "ratios"}
PROJECTION_HORIZONS = {"3 years": 3, "5 years": 5}
GROWTH_PCT_RE = re.compile(r"^([+-]?\d+(?:\.\d+)?)\s*%$")


@st.cache_resource
def get_user_db_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(PROJECTIONS_DB_FILE)
    # One-time migration: projections originally stored a pre-resolved `value
    # DOUBLE`; growth-rate projections need the raw typed text instead ('15%'
    # vs a pre-computed number). Safe to drop and recreate -- this table has
    # never held real user data outside this same session's own testing.
    try:
        existing_cols = {r[0] for r in con.execute("DESCRIBE projections").fetchall()}
    except duckdb.Error:
        existing_cols = set()
    if existing_cols and "input_raw" not in existing_cols:
        con.execute("DROP TABLE projections")
    con.execute("""
        CREATE TABLE IF NOT EXISTS projections (
            symbol      VARCHAR,
            statement   VARCHAR,
            metric      VARCHAR,
            period_end  DATE,
            input_raw   VARCHAR,
            updated_at  TIMESTAMP,
            PRIMARY KEY (symbol, statement, metric, period_end)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            symbol      VARCHAR PRIMARY KEY,
            note        VARCHAR,
            updated_at  TIMESTAMP
        )
    """)
    return con


@st.cache_data
def get_projections(symbol: str) -> pd.DataFrame:
    con = get_user_db_connection()
    return con.execute(
        "SELECT statement, metric, period_end, input_raw FROM projections WHERE symbol = ?",
        [symbol],
    ).df()


def save_projection_cell(symbol: str, statement: str, metric: str, period_end, input_raw: str):
    """Upserts one cell's raw assumption ('15%' or an absolute number); a blank
    input_raw deletes it instead of storing an empty string, so a cleared cell
    truly falls back to 'nothing projected' rather than an empty-but-present row."""
    con = get_user_db_connection()
    if input_raw is None or str(input_raw).strip() == "":
        con.execute(
            "DELETE FROM projections WHERE symbol=? AND statement=? AND metric=? AND period_end=?",
            [symbol, statement, metric, period_end],
        )
    else:
        con.execute(
            "INSERT OR REPLACE INTO projections VALUES (?, ?, ?, ?, ?, ?)",
            [symbol, statement, metric, period_end, input_raw.strip(), datetime.now(timezone.utc)],
        )
    get_projections.clear()


@st.cache_data
def get_note(symbol: str) -> str:
    con = get_user_db_connection()
    row = con.execute("SELECT note FROM notes WHERE symbol = ?", [symbol]).fetchone()
    return row[0] if row else ""


def save_note(symbol: str, text: str):
    con = get_user_db_connection()
    con.execute("INSERT OR REPLACE INTO notes VALUES (?, ?, ?)", [symbol, text, datetime.now(timezone.utc)])
    get_note.clear()


def future_period_ends(last_period_end, n_years: int) -> list:
    last = pd.Timestamp(last_period_end)
    return [(last + pd.DateOffset(years=i)).date() for i in range(1, n_years + 1)]


def infer_unit_from_metric_name(metric: str) -> str:
    """Fallback unit guess for a metric with zero historical rows for this
    company (so its unit can't be read off the scraped data) but the user still
    wants to project it. Mirrors scrape_fundamentals_history.py's infer_unit()."""
    if metric.endswith("%"):
        return "pct"
    if "Days" in metric:
        return "days"
    if metric == "EPS in Rs":
        return "rs_per_share"
    return "cr"


def resolve_projection_chain(last_actual, future_inputs: list) -> list:
    """future_inputs: [(period_end, input_raw), ...] in chronological order.
    Returns [(period_end, resolved_value_or_None), ...].

    input_raw is either a growth rate ('15%', '-5%' -- proportional growth on
    whatever the PREVIOUS period resolved to, whether that's the last actual or
    a prior projection, same idea as dragging '=B5*1.15' across a row in
    Excel), an absolute number ('16500', restarts the chain from here), or
    blank (breaks the chain -- resolves to None, and so does every subsequent
    growth-rate cell until an absolute number restarts it).

    One deliberate simplification: '15%' always means proportional growth on
    the raw number, even for already-percentage rows like OPM % (20 -> 23, not
    20 -> 35 percentage points) -- applied uniformly rather than having two
    different meanings depending on the row's unit."""
    prev = last_actual
    out = []
    for period_end, input_raw in future_inputs:
        text = "" if input_raw is None or (isinstance(input_raw, float) and pd.isna(input_raw)) else str(input_raw).strip()
        if not text:
            resolved = None
        else:
            pct_match = GROWTH_PCT_RE.match(text)
            if pct_match:
                resolved = prev * (1 + float(pct_match.group(1)) / 100) if prev is not None else None
            else:
                try:
                    resolved = float(text.replace(",", ""))
                except ValueError:
                    resolved = None
        out.append((period_end, resolved))
        prev = resolved
    return out


def pivot_fundamentals_with_projections(raw: pd.DataFrame, projections: pd.DataFrame,
                                         statement: str, horizon_years: int):
    """Like pivot_fundamentals_statement, but appends `horizon_years` extra
    future-period columns (labeled e.g. 'Mar 2027E') resolved via
    resolve_projection_chain from the user's saved raw assumptions. The table
    always shows the RESOLVED value (Excel-style: the cell shows the computed
    result; the raw formula is what you see/edit when you select that cell).

    Returns (display, values, future_cols, period_end_by_col, input_raw_by_cell):
    display/values mirror pivot_fundamentals_statement's shape (now including
    the future columns); input_raw_by_cell maps (metric, future_col_label) ->
    the raw text so a selected cell's assumption editor can prefill correctly."""
    historical_display, historical_values = pivot_fundamentals_statement(raw, statement)
    slice_df = raw[raw["statement"] == statement]
    if slice_df.empty and historical_display.empty:
        return pd.DataFrame(), pd.DataFrame(), [], {}, {}

    canonical = FUNDAMENTALS_METRIC_ORDER.get(statement, [])
    metrics = list(historical_display.index) if not historical_display.empty else canonical
    last_period = slice_df["period_end"].max() if not slice_df.empty else None

    future_ends = future_period_ends(last_period, horizon_years) if last_period is not None else []
    future_cols = [pd.Timestamp(d).strftime("%b %Y") + "E" for d in future_ends]
    period_end_by_col = dict(zip(future_cols, future_ends))

    proj_slice = projections[projections["statement"] == statement] if not projections.empty else projections
    input_by_metric_period = {}
    if proj_slice is not None and not proj_slice.empty:
        for _, r in proj_slice.iterrows():
            input_by_metric_period[(r["metric"], pd.Timestamp(r["period_end"]).date())] = r["input_raw"]

    units_by_metric = (
        raw[raw["statement"] == statement].drop_duplicates("metric").set_index("metric")["unit"].to_dict()
    )

    future_values = pd.DataFrame(index=metrics, columns=future_cols, dtype="float64")
    future_display = pd.DataFrame(index=metrics, columns=future_cols, dtype="object")
    input_raw_by_cell = {}

    for metric in metrics:
        non_null = historical_values.loc[metric].dropna() if metric in historical_values.index else pd.Series(dtype="float64")
        last_actual = non_null.iloc[-1] if not non_null.empty else None
        chain_inputs = [(pe, input_by_metric_period.get((metric, pe))) for pe in future_ends]
        resolved = resolve_projection_chain(last_actual, chain_inputs)
        unit = units_by_metric.get(metric) or infer_unit_from_metric_name(metric)
        for col, (pe, val) in zip(future_cols, resolved):
            future_values.loc[metric, col] = val
            future_display.loc[metric, col] = format_fundamentals_value(val, unit) if val is not None else ""
            input_raw_by_cell[(metric, col)] = input_by_metric_period.get((metric, pe), "") or ""

    display = pd.concat([historical_display.reindex(index=metrics).fillna(""), future_display], axis=1)
    values = pd.concat([historical_values.reindex(index=metrics), future_values], axis=1)
    return display, values, future_cols, period_end_by_col, input_raw_by_cell


@st.cache_data(show_spinner="Computing window returns vs peer-group index...")
def compute_sector_leaders(group_level: str, period_days: int, window_label: str,
                            end_date: pd.Timestamp = None, step_label: str = None) -> dict:
    """For every stock: measure how consistently it beat its equal-weighted
    peer-group index over `window_label`-long windows inside the lookback.

    Two scan modes, both anchored so windows end on/before `end_date`:
      - Non-overlapping (step_label=None): back-to-back windows, floor(period /
        window) of them. Independent samples, but few and phase-locked to where
        the end date happens to fall.
      - Rolling (step_label set): windows of the same length whose end dates
        walk back by `step_label`, so they overlap. Many more windows, robust to
        boundary placement -- but overlapping, so the hit rate is a stable
        estimate, not an independent-trial count.

    Consistency = share of a stock's windows that beat its group. Also reports
    the median and worst per-window excess so 'how much' and the downside tail
    show up, not just 'how often'. A stock only counts in windows it has data
    for; each window's group index is the mean over members with data then."""
    closes_full = get_close_matrix()
    # Snap the anchor to the last trading day at/before the requested end date so
    # the window bounds never reach past available data (no lookahead).
    if end_date is None:
        end = closes_full.index.max()
    else:
        on_or_before = closes_full.index[closes_full.index <= end_date]
        end = on_or_before.max() if len(on_or_before) else closes_full.index.min()
    start = end - pd.Timedelta(days=period_days)

    window_offset = LEADER_WINDOWS[window_label][1]
    step_offset = LEADER_STEPS[step_label] if step_label else window_offset

    # Windows end on/before `end`, each `window_offset` long and fully inside the
    # lookback; their end dates step back by `step_offset` (= the window itself
    # in non-overlapping mode, giving exactly back-to-back windows).
    windows = []  # (start, end) pairs
    we = end
    while True:
        ws = we - window_offset
        if ws < start - pd.Timedelta(days=5):  # tolerance so e.g. 4x 3mo fits "1 year"
            break
        windows.append((ws, we))
        we = we - step_offset
    windows.reverse()  # oldest first
    win_starts = [ws for ws, _ in windows]
    win_ends = [we for _, we in windows]
    period_start = min(win_starts)

    # Sample every bound off a forward-filled matrix (bounds are rarely trading
    # days). Keep a buffer before the first bound so ffill can reach it. Leading
    # NaNs (pre-listing) survive, so a stock has no value at bounds before it
    # listed and simply drops out of those windows.
    filled = closes_full.loc[closes_full.index >= period_start - pd.Timedelta(days=14)].ffill()
    start_vals = filled.reindex(win_starts, method="ffill").to_numpy()
    end_vals = filled.reindex(win_ends, method="ffill").to_numpy()
    R = pd.DataFrame(end_vals / start_vals - 1, columns=filled.columns)  # windows x symbols

    sym2grp = get_sector_map().set_index("symbol")[group_level]
    cols = [c for c in R.columns if c in sym2grp.index]
    R = R[cols]
    groups = sym2grp[cols]

    grp_mean = R.T.groupby(groups.values).mean().T       # windows x groups
    delta = R - grp_mean[groups.values].to_numpy()       # windows x symbols

    valid = delta.notna()
    beats = ((delta > 0) & valid).sum()
    n_windows = valid.sum()

    # Full-period return over the whole lookback, independent of windowing -- so
    # the group total is a clean equal-weight peer return that still works when
    # windows overlap (you can't compound overlapping window returns). Full-
    # history stocks anchor to the close at the period start; later listings fall
    # back to their first traded close in-period.
    raw_first = filled.reindex([period_start], method="ffill")[cols].iloc[0]
    in_period = closes_full.loc[closes_full.index >= period_start, cols]
    period_first = raw_first.fillna(in_period.bfill().iloc[0])
    period_last = filled.reindex([end], method="ffill")[cols].iloc[0]
    period_return = (period_last / period_first - 1) * 100
    grp_period = period_return.groupby(groups).mean()

    stats = pd.DataFrame({
        "group": groups,
        "peers": groups.map(groups.value_counts()),
        "beaten": beats,
        "windows": n_windows,
        "consistency_pct": beats.div(n_windows.where(n_windows > 0)) * 100,
        "avg_window_delta_pp": delta.mean() * 100,
        "median_window_delta_pp": delta.median() * 100,
        "worst_window_delta_pp": delta.min() * 100,
        "period_return_pct": period_return,
        "group_return_pct": groups.map(grp_period),
        "full_history": raw_first.notna(),
    })
    stats["delta_total_pp"] = stats["period_return_pct"] - stats["group_return_pct"]
    stats.index.name = "symbol"

    # Aggregate ranking: each group vs the broad market (equal-weighted whole
    # classified universe), reusing the same windows. This answers "which group
    # led at the selected level" -- one level up from the stock leaderboard,
    # where the benchmark is the market instead of the group.
    market_wret = R.mean(axis=1)                          # per window, all stocks
    gdelta = grp_mean.sub(market_wret, axis=0)            # windows x groups
    gvalid = gdelta.notna()
    gbeats = ((gdelta > 0) & gvalid).sum()
    gwin = gvalid.sum()
    market_return = period_return.mean()
    group_stats = pd.DataFrame({
        "members": groups.value_counts(),
        "beaten": gbeats,
        "windows": gwin,
        "consistency_pct": gbeats.div(gwin.where(gwin > 0)) * 100,
        "avg_window_delta_pp": gdelta.mean() * 100,
        "median_window_delta_pp": gdelta.median() * 100,
        "worst_window_delta_pp": gdelta.min() * 100,
        "group_return_pct": grp_period,
        "market_return_pct": market_return,
    })
    group_stats["delta_vs_market_pp"] = group_stats["group_return_pct"] - market_return
    group_stats.index.name = "group"

    return {"stats": stats.reset_index(), "group_stats": group_stats.reset_index(),
            "n_windows": len(windows), "start": period_start, "end": end,
            "market_return_pct": market_return,
            "mode": "rolling" if step_label else "non-overlapping", "step": step_label}


@st.cache_data
def get_group_leader_chart(group_level: str, group_name: str, top_symbols: tuple,
                            period_days: int, end_date: pd.Timestamp = None) -> pd.DataFrame:
    """Normalized (=100 at each line's start) cumulative performance of the top
    consistent outperformers vs their group's equal-weight index, over the same
    lookback window ending at `end_date` (defaults to the latest date)."""
    closes = get_close_matrix()
    if end_date is None:
        end = closes.index.max()
    else:
        on_or_before = closes.index[closes.index <= end_date]
        end = on_or_before.max() if len(on_or_before) else closes.index.min()
    closes = closes.loc[(closes.index >= end - pd.Timedelta(days=period_days)) & (closes.index <= end)]
    sector_map = get_sector_map()
    members = sector_map[sector_map[group_level] == group_name]["symbol"]
    member_closes = closes[[m for m in members if m in closes.columns]].ffill()
    idx_rets = member_closes.pct_change(fill_method=None).mean(axis=1).fillna(0)
    out = pd.DataFrame({f"{group_name} (eq-wt index)": (1 + idx_rets).cumprod() * 100})
    for s in top_symbols:
        series = member_closes[s].dropna()
        if not series.empty:
            out[s] = member_closes[s] / series.iloc[0] * 100
    return out


# --- Sector fundamentals: year-by-year sector aggregates (revenue, profit, ROCE,
# ROE, PE) to spot years a sector's fundamentals accelerated, not just its price. ---

@st.cache_data(show_spinner="Aggregating sector fundamentals across years...")
def get_sector_fundamentals_raw(group_level: str) -> pd.DataFrame:
    """Per-company-per-year fundamentals joined to sector classification and an
    as-of share price (last close on/before the result's `available_date`, so no
    lookahead) for PE and the market-cap-derived ratios below. Sales growth comes
    straight from the scrape (screener's own YoY calc); profit growth is computed
    here the same way."""
    con = get_connection()
    query = f"""
        WITH roe AS (
            SELECT symbol, period_end, value AS roe_pct
            FROM fundamentals_raw WHERE statement = 'ratios' AND metric = 'ROE %'
        ),
        borrow AS (
            SELECT symbol, period_end, MAX(value) AS borrowings_cr
            FROM fundamentals_raw
            WHERE statement = 'balance_sheet' AND metric IN ('Borrowing', 'Borrowings')
            GROUP BY symbol, period_end
        ),
        assets AS (
            SELECT symbol, period_end, value AS total_assets_cr
            FROM fundamentals_raw WHERE statement = 'balance_sheet' AND metric = 'Total Assets'
        ),
        cfo AS (
            SELECT symbol, period_end, value AS cfo_cr
            FROM fundamentals_raw WHERE statement = 'cash_flow' AND metric = 'Cash from Operating Activity'
        )
        SELECT fa.symbol, im.{group_level} AS group_name, fa.period_end,
               fa.sales_cr, fa.net_profit_cr, fa.eps_rs, fa.roce_pct, fa.opm_pct,
               fa.sales_growth_pct, roe.roe_pct, dp.close AS close_price,
               borrow.borrowings_cr, assets.total_assets_cr, cfo.cfo_cr,
               dp_end.close AS year_end_price, dp_start.close AS year_start_price
        FROM fundamentals_annual fa
        JOIN instruments im ON im.tradingsymbol = fa.symbol
        LEFT JOIN roe ON roe.symbol = fa.symbol AND roe.period_end = fa.period_end
        LEFT JOIN borrow ON borrow.symbol = fa.symbol AND borrow.period_end = fa.period_end
        LEFT JOIN assets ON assets.symbol = fa.symbol AND assets.period_end = fa.period_end
        LEFT JOIN cfo ON cfo.symbol = fa.symbol AND cfo.period_end = fa.period_end
        ASOF LEFT JOIN daily_prices dp
            ON dp.tradingsymbol = fa.symbol AND dp.date <= fa.available_date
        -- Indian FY = April-March: period_end is always ~March 31 (FY close), so
        -- the FY's first trading day is the nearest date on/after April 1 of the
        -- PRIOR calendar year -- nearest on/before period_end itself is the close.
        ASOF LEFT JOIN daily_prices dp_end
            ON dp_end.tradingsymbol = fa.symbol AND dp_end.date <= fa.period_end
        ASOF LEFT JOIN daily_prices dp_start
            ON dp_start.tradingsymbol = fa.symbol
            AND dp_start.date >= make_date(CAST(extract(year FROM fa.period_end) AS INT) - 1, 4, 1)
        WHERE im.{group_level} IS NOT NULL AND fa.sales_cr IS NOT NULL
    """
    df = con.execute(query).df()
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["fiscal_year"] = df["period_end"].dt.year
    df = df.sort_values(["symbol", "period_end"])
    df["profit_growth_pct"] = df.groupby("symbol")["net_profit_cr"].pct_change() * 100
    # A near-zero prior-year base makes plain pct-change blow up to +/-1000s%,
    # which would swamp any real acceleration signal -- drop those as noise.
    df.loc[df["profit_growth_pct"].abs() > 1000, "profit_growth_pct"] = pd.NA
    df["pe"] = df["close_price"] / df["eps_rs"]
    df.loc[(df["eps_rs"] <= 0) | df["eps_rs"].isna(), "pe"] = pd.NA
    df["operating_profit_cr"] = df["sales_cr"] * df["opm_pct"] / 100

    # No historical shares-outstanding table, so market cap is backed out via
    # PE algebra (shares = net profit / EPS, mcap = price x shares) instead of
    # scraping a separate series -- equivalent to mcap = PE x net profit, and
    # inherits PE's loss-making-year guard for free.
    df["shares_cr"] = df["net_profit_cr"] / df["eps_rs"]
    df.loc[(df["eps_rs"] <= 0) | df["eps_rs"].isna(), "shares_cr"] = pd.NA
    df["market_cap_cr"] = df["close_price"] * df["shares_cr"]
    df.loc[df["market_cap_cr"] <= 0, "market_cap_cr"] = pd.NA

    df["ps_ratio"] = df["market_cap_cr"] / df["sales_cr"]
    df.loc[df["sales_cr"] <= 0, "ps_ratio"] = pd.NA

    # EV = market cap + debt; no clean cash line in the scrape, so cash isn't
    # netted out -- EV is therefore a slight overstatement for cash-rich firms.
    df["ev_cr"] = df["market_cap_cr"] + df["borrowings_cr"].fillna(0)
    df["ev_to_ebitda"] = df["ev_cr"] / df["operating_profit_cr"]
    df.loc[df["market_cap_cr"].isna() | (df["operating_profit_cr"] <= 0), "ev_to_ebitda"] = pd.NA

    df["op_profit_to_cfo"] = df["operating_profit_cr"] / df["cfo_cr"]
    df.loc[df["cfo_cr"].isna() | (df["cfo_cr"] == 0), "op_profit_to_cfo"] = pd.NA

    df["op_profit_to_assets_pct"] = df["operating_profit_cr"] / df["total_assets_cr"] * 100
    df.loc[df["total_assets_cr"].isna() | (df["total_assets_cr"] == 0), "op_profit_to_assets_pct"] = pd.NA
    return df


def aggregate_sector_fundamentals(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (group, fiscal year): sector-wide size (summed revenue/profit,
    a 'the whole sector re-rated' view) alongside the typical company's profile
    (medians, a 'the average constituent did better' view, robust to a couple of
    giants or new listings distorting the sum)."""
    g = raw.groupby(["group_name", "fiscal_year"])
    out = g.agg(
        companies=("symbol", "nunique"),
        total_revenue_cr=("sales_cr", "sum"),
        total_profit_cr=("net_profit_cr", "sum"),
        median_revenue_growth_pct=("sales_growth_pct", "median"),
        median_profit_growth_pct=("profit_growth_pct", "median"),
        median_roce_pct=("roce_pct", "median"),
        median_roe_pct=("roe_pct", "median"),
        median_pe=("pe", "median"),
        median_opm_pct=("opm_pct", "median"),
        median_ps_ratio=("ps_ratio", "median"),
        median_ev_to_ebitda=("ev_to_ebitda", "median"),
        median_op_profit_to_cfo=("op_profit_to_cfo", "median"),
        median_op_profit_to_assets_pct=("op_profit_to_assets_pct", "median"),
        median_year_start_price=("year_start_price", "median"),
        median_year_end_price=("year_end_price", "median"),
    ).reset_index()
    return out.sort_values(["group_name", "fiscal_year"])


def yoy_layer(level_pivot: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Row-wise year-over-year change of a metric x year pivot. Rate-like
    metrics (kind == 'pct': ROCE/ROE/OPM/PE-as-%-ish/growth-rates themselves)
    show the change in **percentage points** (18% -> 22% reads as +4pp), since
    '% growth of a %' is a confusing way to read a margin or ratio move.
    Size/multiple metrics (kind in {'num', 'ratio'}) show standard YoY % change.
    Same near-zero-base guard as profit growth elsewhere in this section, so one
    noisy prior-year value near zero can't blow out the whole color scale."""
    years = sorted(level_pivot.columns)
    lvl = level_pivot.reindex(columns=years)
    if kind == "pct":
        return lvl.diff(axis=1)
    growth = lvl.pct_change(axis=1) * 100
    return growth.where(growth.abs() <= 1000)


def relative_layer(growth_pivot: pd.DataFrame, peer_growth_pivot: pd.DataFrame) -> pd.DataFrame:
    """Each row's YoY change minus the peer group's median YoY change that same
    year -- positive means beating peers that year, not just moving with them.
    `peer_growth_pivot` sets the peer group (may differ from `growth_pivot`'s
    own rows, e.g. peer = the sector's full membership even when the displayed
    rows are filtered down)."""
    peer_median = peer_growth_pivot.median(axis=0, skipna=True)
    return growth_pivot.sub(peer_median, axis=1)


def resolve_view_layer(level_pivot: pd.DataFrame, full_pivot: pd.DataFrame, kind: str,
                        base_diverging: bool, label: str, view_mode: str):
    """Apply the shared Level / YoY growth / vs. peer median view-mode toggle to
    one metric's company x year pivot. `full_pivot` (the sector's full, unfiltered
    membership) sets the peer group for 'vs. peer median' regardless of which
    rows `level_pivot` itself contains. Returns (display_pivot, diverging, title)."""
    if view_mode == "Level":
        return level_pivot, base_diverging, label
    if view_mode == "YoY growth":
        return yoy_layer(level_pivot, kind), True, f"{label} — YoY Δ"
    shown_growth = yoy_layer(level_pivot, kind)
    full_growth = yoy_layer(full_pivot, kind)
    return relative_layer(shown_growth, full_growth), True, f"{label} — vs. sector median YoY Δ"


def build_company_heatmap(display_pivot: pd.DataFrame, diverging: bool, cbar_title: str,
                           fmt: str, row_segments: list, primary_label: str = None,
                           secondary_pivot: pd.DataFrame = None, secondary_fmt: str = None,
                           secondary_label: str = None) -> go.Figure:
    """A company x year heatmap split into stacked panels by market-cap tier
    (MCAP_SEGMENT_ORDER) so a micro-cap's growth isn't visually stacked against
    a behemoth's on the same footing. One shared color scale/colorbar across
    every tier keeps colors comparable across panels.

    When `secondary_pivot` is given (same index/columns as `display_pivot`),
    color still encodes the primary metric, but each cell's text and hover
    also show the secondary metric's value in parentheses -- one grid, both
    numbers, so a correlation between them can be eyeballed directly instead
    of scanning two separate heatmaps."""
    present_segments = [seg for seg in MCAP_SEGMENT_ORDER if seg in row_segments]
    seg_counts = {seg: row_segments.count(seg) for seg in present_segments}
    zmax = display_pivot.abs().max().max() if diverging and display_pivot.notna().any().any() else None
    has_secondary = secondary_pivot is not None
    # Heatmap's texttemplate doesn't resolve %{customdata} (only hovertemplate
    # does) -- so the combined cell text is pre-rendered into a "text" matrix
    # and shown via %{text} instead; hover keeps using customdata directly.
    if has_secondary:
        hover_tmpl = (f"%{{y}} · %{{x}}<br>{primary_label}: %{{z:{fmt}}}"
                      f"<br>{secondary_label}: %{{customdata:{secondary_fmt}}}<extra></extra>")
    else:
        hover_tmpl = "%{y} · %{x}: %{z:" + fmt + "}<extra></extra>"
    fig = make_subplots(
        rows=len(present_segments), cols=1, shared_xaxes=True,
        row_heights=[seg_counts[seg] for seg in present_segments],
        vertical_spacing=min(0.15, 1.2 / max(1, len(display_pivot))),
        subplot_titles=[f"{seg} — {seg_counts[seg]} co." for seg in present_segments],
    )
    for i, seg in enumerate(present_segments, start=1):
        seg_rows = [s for s, sg in zip(display_pivot.index, row_segments) if sg == seg]
        sub = display_pivot.loc[seg_rows]
        heat_kwargs = dict(
            z=sub.values, x=[str(y) for y in sub.columns], y=sub.index,
            colorscale="RdYlGn" if diverging else "Blues",
            zmid=0 if diverging else None,
            zmin=-zmax if zmax else None, zmax=zmax if zmax else None,
            hovertemplate=hover_tmpl,
            showscale=(i == 1), colorbar=dict(title=cbar_title) if i == 1 else None,
        )
        if has_secondary:
            sub2 = secondary_pivot.loc[seg_rows, sub.columns]
            heat_kwargs["customdata"] = sub2.values
            heat_kwargs["text"] = [
                [
                    "" if pd.isna(zv) else (format(zv, fmt) if pd.isna(cv) else f"{format(zv, fmt)}<br>({format(cv, secondary_fmt)})")
                    for zv, cv in zip(z_row, c_row)
                ]
                for z_row, c_row in zip(sub.values, sub2.values)
            ]
            heat_kwargs["texttemplate"] = "%{text}"
        else:
            heat_kwargs["texttemplate"] = "%{z:" + fmt + "}"
        fig.add_trace(go.Heatmap(**heat_kwargs), row=i, col=1)
    fig.update_layout(
        height=max(280, 30 * len(display_pivot) + 50 * len(present_segments) + 40),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


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

SECTIONS = ["Charts", "Stage screener", "Strategy backtest", "Signal backtest", "Sector leaders",
            "Sector fundamentals", "Fundamentals", "Query / Tables"]

# Section nav lives at the top of the main area (not the sidebar) so it's one
# tap on mobile without opening the settings drawer, and it works identically on
# desktop. Defined before the sidebar block so the settings accordions below can
# still key their auto-open off the selected section.
section = st.selectbox("Section", SECTIONS, key="nav_section", label_visibility="collapsed")

with st.sidebar:
    # Which strategy's settings are relevant to the current section. The
    # relevant accordion opens, the other stays collapsed.
    stage_relevant = section in {"Charts", "Stage screener", "Strategy backtest"}
    signal_relevant = section in {"Charts", "Signal backtest"}

    def settings_group(title: str, relevant: bool):
        """A collapsible accordion that auto-opens when relevant. An expander's
        `expanded=` is sticky once a user toggles it, so we append an invisible
        zero-width char to the label when NOT relevant -- that makes Streamlit
        treat it as a fresh element on each relevance flip, so the open/closed
        default actually applies on navigation. Widgets always render, keeping
        the derived params valid on every section."""
        label = title if relevant else title + "​"  # zero-width space
        return st.expander(label, expanded=relevant)

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

if section == "Sector leaders":
    st.caption(
        "Pure price action: the lookback period is split into windows (counting back from the "
        "end date), each stock's window return is compared to its **equal-weighted peer-group "
        "index**, and stocks are ranked by how *consistently* they beat their group — not just by "
        "total return. A stock that beat its sector in most windows is a leadership candidate "
        "worth studying; one big spike isn't."
    )

    leader_view = st.segmented_control(
        "Rank", ["Aggregate groups", "Individual stocks"], default="Aggregate groups",
        key="leader_view",
        help="**Aggregate groups** ranks each group at the chosen level (which sector / industry "
        "led) against the broad market. **Individual stocks** ranks stocks against their own "
        "group. Either way, the drill-down below lets you open one group and see its members.",
    )

    date_lo, date_hi = get_price_date_range()
    mcap_df = get_mcap_map()
    has_mcap = not mcap_df.empty

    # Group 1 — what we measure over.
    with st.container(border=True):
        st.caption(":material/date_range: Measurement window")
        w1, w2, w3, w4 = st.columns(4, vertical_alignment="bottom")
        with w1:
            leader_level = st.selectbox(
                "Group by", ["macro_sector", "sector", "industry", "basic_industry"],
                index=1, format_func=lambda s: s.replace("_", " ").title(), key="leader_level",
            )
        with w2:
            leader_end = st.date_input(
                "End date", value=date_hi.date(),
                min_value=date_lo.date(), max_value=date_hi.date(), key="leader_end_date",
                help="The lookback period counts back from this date. Defaults to the latest "
                "trading day; set it earlier to study leadership as of a past date.",
            )
        with w3:
            leader_period = st.selectbox("Lookback period", list(LEADER_PERIODS), index=2, key="leader_period")
        period_days = LEADER_PERIODS[leader_period]
        window_opts = [w for w, (d, _) in LEADER_WINDOWS.items() if d * 4 <= period_days]
        with w4:
            leader_window = st.selectbox(
                "Comparison window", window_opts,
                index=window_opts.index("3 months") if "3 months" in window_opts else len(window_opts) - 1,
                key="leader_window",
                help="Only windows that give at least 4 full comparisons within the lookback are offered.",
            )
    leader_end_ts = pd.Timestamp(leader_end)

    # Group 2 — how we scan the window and the consistency bar.
    with st.container(border=True):
        st.caption(":material/travel_explore: Scan & consistency")
        s1, s2, s3 = st.columns(3, vertical_alignment="bottom")
        with s1:
            scan_mode = st.segmented_control(
                "Scan mode", ["Rolling", "Non-overlapping"], default="Rolling", key="leader_scan_mode",
                help="**Rolling** slides the window forward in small steps so it overlaps — many "
                "windows, robust to where boundaries fall, best for *identifying* durable leaders "
                "(but overlap means the hit rate is a stable estimate, not independent trials). "
                "**Non-overlapping** uses back-to-back windows — few but statistically independent, "
                "a good cross-check.",
            )
        with s2:
            if scan_mode == "Rolling":
                step_label = st.selectbox(
                    "Rolling step", list(LEADER_STEPS), index=2, key="leader_step",
                    help="How far each window's end date slides. Smaller steps = more windows and a "
                    "smoother, more phase-robust consistency estimate.",
                )
            else:
                step_label = None
                st.selectbox(
                    "Rolling step", ["— not applicable"], disabled=True, key="leader_step_off",
                    help="Non-overlapping uses back-to-back windows: independent samples, but few "
                    "and sensitive to the exact end date.",
                )
        with s3:
            min_consistency = st.slider(
                "Min. windows beaten (%)", 0, 100, 60, 5, key="leader_min_consistency",
                help="Only show names that beat their group in at least this share of windows. "
                "(Not applied to the Aggregate groups view, which ranks every group.)",
            )

    # Group 3 — narrow / display the results.
    with st.container(border=True):
        st.caption(":material/filter_alt: Filters & display")
        f1, f2, f3, f4 = st.columns([2.2, 1, 1, 1], vertical_alignment="bottom")
        with f1:
            if has_mcap:
                mcap_stops = mcap_ladder(mcap_df["mcap_cr"].max())
                mcap_range = st.select_slider(
                    "Market-cap range (₹ cr)", options=mcap_stops,
                    value=(mcap_stops[0], mcap_stops[-1]), format_func=format_mcap_cr,
                    key="leader_mcap_range",
                    help="Filter to a market-cap band. Steps coarsen as cap rises (5k → 10k → 20k "
                    "→ 50k → 100k). Controls for base effect; full range = no filter.",
                )
            else:
                mcap_range = None
                st.caption(":material/scale: No market-cap snapshot — size filter off. "
                           "Run `python scrape_fundamentals.py` to enable it.")
        with f2:
            top_n_choice = st.selectbox(
                "Show", ["All stocks", "Top 3 per group", "Top 5 per group"],
                key="leader_top_n",
                help="Individual-stocks view only: cap how many names show per group.",
            )
        with f3:
            min_peers = st.number_input(
                "Min. stocks in group", 2, 50, 3, 1, key="leader_min_peers",
                help="Groups with very few members make 'beating the group' close to meaningless.",
            )
        with f4:
            full_history_only = st.checkbox(
                "Full-period history only", value=True, key="leader_full_history",
                help="Exclude stocks that listed after the period started (measured on fewer windows).",
            )

    with st.expander(":material/warning: Caveats before you read too much into this"):
        st.markdown(
            "- **Survivorship bias**: the universe is *today's* >₹2000cr list, so long "
            "lookbacks are tilted toward winners by construction. Good for finding candidates "
            "to study, not for backtest-grade conclusions.\n"
            "- Sector/industry classification is **as of today**, applied to the whole history.\n"
            "- The peer-group index is **equal-weighted** and includes the stock itself, so in "
            "small groups a big mover drags its own benchmark up.\n"
            "- Stocks that listed mid-period are compared only on the windows they traded in "
            "(see the *windows* column) unless you require full-period history."
        )

    if st.button(":material/leaderboard: Build leaderboard", type="primary", key="leader_build"):
        st.session_state["leaders_active"] = True

    if not st.session_state.get("leaders_active"):
        st.info("Click **Build leaderboard** to load the price universe and rank the stocks (first load takes a minute; instant after that).")
    else:
        res = compute_sector_leaders(leader_level, period_days, leader_window, leader_end_ts, step_label)
        stats = res["stats"]
        group_stats = res["group_stats"]
        mode_txt = (f"rolling ({res['step']} step, overlapping)" if res["mode"] == "rolling"
                    else "non-overlapping")
        subj = ("each group beat the broad market" if leader_view == "Aggregate groups"
                else "each stock beat its group")
        st.caption(
            f"{res['n_windows']} {mode_txt} {leader_window} windows from "
            f"{res['start'].date()} to {res['end'].date()}. Consistency = share of these "
            f"windows {subj}."
            + (" Overlapping windows share days, so read the % as a robust estimate, not "
               "independent trials." if res["mode"] == "rolling" else "")
        )

        if has_mcap:
            stats = stats.merge(mcap_df, on="symbol", how="left")
            stats["segment"] = stats["mcap_cr"].map(mcap_segment)

        level_label = leader_level.replace("_", " ")
        if leader_view == "Aggregate groups":
            best_member = (stats[stats["full_history"]]
                           .sort_values("consistency_pct", ascending=False)
                           .groupby("group")["symbol"].first())
            # Show every group ranked (min. members still drops trivially small
            # ones). The consistency threshold is a stock-screening tool for the
            # 1000+ name universe; with only a couple dozen groups the point is
            # to see the whole ranking, so it isn't applied here.
            gdf = group_stats[group_stats["members"] >= min_peers].sort_values(
                ["consistency_pct", "avg_window_delta_pp"], ascending=False).copy()
            gdf["top_member"] = gdf["group"].map(best_member)
            gdf["record"] = gdf["beaten"].astype(str) + "/" + gdf["windows"].astype(str)
            beat_bar = int((gdf["consistency_pct"] > 50).sum())
            st.write(f"**{len(gdf)} {level_label}s ranked** by consistency of beating the broad "
                     f"market ({beat_bar} beat it in over half the windows; market return over "
                     f"the period {res['market_return_pct']:+.1f}%)")
            if gdf.empty:
                st.caption("No groups have enough members — lower **Min. stocks in group**.")
            else:
                gcols = ["group", "members", "record", "consistency_pct", "avg_window_delta_pp",
                         "median_window_delta_pp", "worst_window_delta_pp", "group_return_pct",
                         "delta_vs_market_pp", "top_member"]
                st.dataframe(
                    gdf[gcols], width="stretch", height=460, hide_index=True,
                    column_config={
                        "group": st.column_config.TextColumn(level_label),
                        "members": st.column_config.NumberColumn("members", format="%d"),
                        "record": st.column_config.TextColumn(
                            "beat market", help="Windows the group's index beat the broad-market index / windows measured"),
                        "consistency_pct": st.column_config.ProgressColumn(
                            "consistency", format="%.0f%%", min_value=0, max_value=100),
                        "avg_window_delta_pp": st.column_config.NumberColumn(
                            "avg delta / window", format="%+.1f pp",
                            help="Average (group − market) return per window, in percentage points"),
                        "median_window_delta_pp": st.column_config.NumberColumn(
                            "median delta / window", format="%+.1f pp"),
                        "worst_window_delta_pp": st.column_config.NumberColumn(
                            "worst window", format="%+.1f pp",
                            help="Largest single-window shortfall vs the market — the group's downside tail"),
                        "group_return_pct": st.column_config.NumberColumn(
                            f"return {leader_period}", format="%+.1f%%"),
                        "delta_vs_market_pp": st.column_config.NumberColumn(
                            "delta vs market", format="%+.1f pp",
                            help="Full-period group return minus the broad-market return"),
                        "top_member": st.column_config.TextColumn(
                            "top member", help="Most-consistent full-history stock in the group — its best individual leader"),
                    },
                )
                st.download_button(
                    "Download group ranking as CSV", gdf[gcols].to_csv(index=False),
                    file_name="sector_leaders_groups.csv", mime="text/csv",
                )
                st.caption(f"Switch **Rank** to *Individual stocks* to rank stocks within groups, "
                           f"or use the drill-down below to open one {level_label}.")

        if leader_view == "Individual stocks":
            filtered = stats[stats["peers"] >= min_peers]
            if full_history_only:
                filtered = filtered[filtered["full_history"]]
            if has_mcap and mcap_range is not None and (
                    mcap_range[0] > mcap_stops[0] or mcap_range[1] < mcap_stops[-1]):
                # A narrowed band excludes stocks with no mcap (NaN fails between);
                # the full range leaves everything in, unknowns included.
                filtered = filtered[filtered["mcap_cr"].between(*mcap_range)]
            filtered = filtered[filtered["consistency_pct"] >= min_consistency]
            filtered = filtered.sort_values(
                ["consistency_pct", "avg_window_delta_pp"], ascending=False
            )
            if top_n_choice != "All stocks":
                n = 3 if "3" in top_n_choice else 5
                filtered = filtered.groupby("group", sort=False).head(n)

            st.write(f"**{len(filtered)} consistent outperformers** (of {len(stats)} stocks scored)")
            if filtered.empty:
                st.caption("Nothing matches — lower the consistency bar or relax the filters.")
            else:
                display = filtered.copy()
                display["record"] = display["beaten"].astype(str) + "/" + display["windows"].astype(str)
                cols = ["symbol", "group", "record", "consistency_pct", "avg_window_delta_pp",
                        "median_window_delta_pp", "worst_window_delta_pp",
                        "period_return_pct", "group_return_pct", "delta_total_pp", "peers"]
                if has_mcap:
                    cols[2:2] = ["segment", "mcap_cr"]
                st.dataframe(
                    display[cols],
                    width="stretch", height=420, hide_index=True,
                    column_config={
                        "group": st.column_config.TextColumn(leader_level.replace("_", " ")),
                        "record": st.column_config.TextColumn(
                            "beat group", help="Windows where the stock's return beat its group index / windows it traded in"),
                        "consistency_pct": st.column_config.ProgressColumn(
                            "consistency", format="%.0f%%", min_value=0, max_value=100),
                        "avg_window_delta_pp": st.column_config.NumberColumn(
                            "avg delta / window", format="%+.1f pp",
                            help="Average (stock − group) return per window, in percentage points"),
                        "median_window_delta_pp": st.column_config.NumberColumn(
                            "median delta / window", format="%+.1f pp",
                            help="Median (stock − group) return per window — less swayed by one outlier window than the average"),
                        "worst_window_delta_pp": st.column_config.NumberColumn(
                            "worst window", format="%+.1f pp",
                            help="Largest single-window shortfall vs the group — the downside tail. A steady leader rarely lags badly in any window"),
                        "period_return_pct": st.column_config.NumberColumn(
                            f"return {leader_period}", format="%+.1f%%"),
                        "group_return_pct": st.column_config.NumberColumn(
                            "group return", format="%+.1f%%"),
                        "delta_total_pp": st.column_config.NumberColumn(
                            "delta vs group", format="%+.1f pp",
                            help="Full-period stock return minus full-period group return"),
                        "mcap_cr": st.column_config.NumberColumn("mcap (₹ cr)", format="%.0f"),
                    },
                )
                st.download_button(
                    "Download leaderboard as CSV", display[cols].to_csv(index=False),
                    file_name="sector_leaders.csv", mime="text/csv",
                )

        st.divider()
        st.subheader(":material/travel_explore: Group drill-down")
        group_names = sorted(stats["group"].dropna().unique())
        drill_group = st.selectbox(leader_level.replace("_", " ").title(), group_names, key="leader_drill_group")
        members = stats[stats["group"] == drill_group].sort_values(
            ["consistency_pct", "avg_window_delta_pp"], ascending=False
        )
        top_syms = tuple(members[members["full_history"]]["symbol"].head(5))
        if top_syms:
            chart_df = get_group_leader_chart(leader_level, drill_group, top_syms, period_days, leader_end_ts)
            st.caption(
                "Top 5 most-consistent members (full history) vs the group's equal-weight "
                "index, all normalized to 100 at the period start."
            )
            st.line_chart(chart_df, height=380)

        display_m = members.copy()
        display_m["record"] = display_m["beaten"].astype(str) + "/" + display_m["windows"].astype(str)
        mcols = ["symbol", "record", "consistency_pct", "avg_window_delta_pp",
                 "period_return_pct", "delta_total_pp"]
        if has_mcap:
            mcols[1:1] = ["segment", "mcap_cr"]
        st.dataframe(
            display_m[mcols],
            width="stretch", height=330, hide_index=True,
            column_config={
                "record": st.column_config.TextColumn("beat group"),
                "consistency_pct": st.column_config.ProgressColumn(
                    "consistency", format="%.0f%%", min_value=0, max_value=100),
                "avg_window_delta_pp": st.column_config.NumberColumn(
                    "avg delta / window", format="%+.1f pp"),
                "period_return_pct": st.column_config.NumberColumn(
                    f"return {leader_period}", format="%+.1f%%"),
                "delta_total_pp": st.column_config.NumberColumn(
                    "delta vs group", format="%+.1f pp"),
                "mcap_cr": st.column_config.NumberColumn("mcap (₹ cr)", format="%.0f"),
            },
        )
        st.caption("Every member of the group, unfiltered — so laggards are visible too.")

if section == "Sector fundamentals":
    st.caption(
        "Sector-year aggregates built from every company's annual results: sector-wide size "
        "(summed revenue/profit) alongside the typical constituent's profile (medians). Use "
        "this to spot years a sector's *fundamentals* accelerated, not just its share price."
    )

    # (aggregate column, kind, diverging colorscale). kind drives number
    # formatting: "pct" -> "%.1f%%", "num" -> whole-number ₹cr/PE, "ratio" ->
    # "%.2fx" for the multiples (EV/EBITDA, Price/Sales, op. profit/CFO).
    SF_METRICS = {
        "Median revenue growth %": ("median_revenue_growth_pct", "pct", True),
        "Median profit growth %": ("median_profit_growth_pct", "pct", True),
        "Total revenue (₹ cr)": ("total_revenue_cr", "num", False),
        "Total profit (₹ cr)": ("total_profit_cr", "num", False),
        "Median ROCE %": ("median_roce_pct", "pct", False),
        "Median ROE %": ("median_roe_pct", "pct", False),
        "Median PE": ("median_pe", "num", False),
        "Median operating margin %": ("median_opm_pct", "pct", False),
        "Median Price/Sales": ("median_ps_ratio", "ratio", False),
        "Median EV/EBITDA": ("median_ev_to_ebitda", "ratio", False),
        "Median op. profit / CFO": ("median_op_profit_to_cfo", "ratio", False),
        "Median op. profit / assets %": ("median_op_profit_to_assets_pct", "pct", False),
        "Year start price (₹)": ("median_year_start_price", "num", False),
        "Year end price (₹)": ("median_year_end_price", "num", False),
    }
    # aggregate column name -> the per-company raw column it's built from, for the
    # constituent drill-down chart (sf_raw has no total_revenue_cr/total_profit_cr,
    # those only exist post-aggregation; per-company they're just sales/profit).
    SF_RAW_COL = {
        "median_revenue_growth_pct": "sales_growth_pct", "median_profit_growth_pct": "profit_growth_pct",
        "total_revenue_cr": "sales_cr", "total_profit_cr": "net_profit_cr",
        "median_roce_pct": "roce_pct", "median_roe_pct": "roe_pct", "median_pe": "pe",
        "median_opm_pct": "opm_pct", "median_ps_ratio": "ps_ratio",
        "median_ev_to_ebitda": "ev_to_ebitda", "median_op_profit_to_cfo": "op_profit_to_cfo",
        "median_op_profit_to_assets_pct": "op_profit_to_assets_pct",
        "median_year_start_price": "year_start_price", "median_year_end_price": "year_end_price",
    }

    c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
    with c1:
        sf_level = st.selectbox(
            "Group by", ["macro_sector", "sector", "industry", "basic_industry"],
            index=1, format_func=lambda s: s.replace("_", " ").title(), key="sf_level",
        )
    with c2:
        sf_metric_label = st.selectbox(
            "Primary metric", list(SF_METRICS), key="sf_metric",
            help="Drives the sector heatmap above and the primary company heatmap below.",
        )
    sf_view_mode = st.segmented_control(
        "View", ["Level", "YoY growth", "vs. peer median"], default="Level", key="sf_view_mode",
        help="**Level** — the raw metric. **YoY growth** — year-over-year change (percentage points "
        "for rate metrics like ROCE/PE/margins, % change for size metrics like revenue/profit). "
        "**vs. peer median** — that same YoY change minus the peer group's median YoY change the same "
        "year, so a cell lights up only when it's beating its peers, not just moving with them. Peer "
        "group = the sectors shown below on the sector heatmap, or the sector's full membership on the "
        "company heatmap (even if you've filtered which companies are displayed).",
    )
    sf_view_mode = sf_view_mode or "Level"  # segmented_control returns None if clicked off
    sf_col, sf_kind, sf_diverging = SF_METRICS[sf_metric_label]
    sf_raw_col = SF_RAW_COL[sf_col]

    sf_raw = get_sector_fundamentals_raw(sf_level)
    if sf_raw.empty:
        st.info("No annual fundamentals joined to sector classification yet.")
    else:
        sf_agg = aggregate_sector_fundamentals(sf_raw)

        f1, f2 = st.columns([2, 1], vertical_alignment="bottom")
        with f2:
            sf_min_companies = st.number_input(
                "Min. companies in group-year", 1, 50, 3, 1, key="sf_min_companies",
                help="Group-years with very few reporting companies make medians noisy.",
            )
        sf_agg = sf_agg[sf_agg["companies"] >= sf_min_companies]
        latest_year = sf_agg["fiscal_year"].max() if not sf_agg.empty else None
        default_groups = (
            sf_agg[sf_agg["fiscal_year"] == latest_year]
            .sort_values("total_revenue_cr", ascending=False)["group_name"].head(8).tolist()
            if latest_year is not None else []
        )
        with f1:
            sf_groups = st.multiselect(
                sf_level.replace("_", " ").title(), sorted(sf_agg["group_name"].dropna().unique()),
                default=default_groups, key="sf_groups",
            )

        if not sf_groups:
            st.info("Pick at least one group to chart.")
        else:
            plot_df = sf_agg[sf_agg["group_name"].isin(sf_groups)]
            pivot = plot_df.pivot(index="group_name", columns="fiscal_year", values=sf_col)
            pivot = pivot.reindex(sf_groups)  # keep the picked order, not alphabetical

            st.subheader(":material/grid_on: Heatmap")
            fmt = ".1f" if sf_kind == "pct" else (".2f" if sf_kind == "ratio" else ",.0f")
            if sf_view_mode == "Level":
                display_pivot, heat_diverging, cbar_title = pivot, sf_diverging, sf_metric_label
            elif sf_view_mode == "YoY growth":
                display_pivot = yoy_layer(pivot, sf_kind)
                heat_diverging, cbar_title = True, f"{sf_metric_label} — YoY Δ"
            else:
                growth = yoy_layer(pivot, sf_kind)
                display_pivot = relative_layer(growth, growth)  # peer = the displayed sectors themselves
                heat_diverging, cbar_title = True, f"{sf_metric_label} — vs. peer median YoY Δ"
            zmax = display_pivot.abs().max().max() if heat_diverging and display_pivot.notna().any().any() else None
            fig = go.Figure(data=go.Heatmap(
                z=display_pivot.values, x=[str(y) for y in display_pivot.columns], y=display_pivot.index,
                colorscale="RdYlGn" if heat_diverging else "Blues",
                zmid=0 if heat_diverging else None,
                zmin=-zmax if zmax else None, zmax=zmax if zmax else None,
                texttemplate="%{z:" + fmt + "}", hovertemplate="%{y} · %{x}: %{z:" + fmt + "}<extra></extra>",
                colorbar_title=cbar_title,
            ))
            fig.update_layout(height=max(220, 40 * len(display_pivot) + 80), margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")

            # Click-to-drill on the chart itself turned out unreliable across
            # browsers/environments (Plotly's selection model doesn't fire
            # consistently for Heatmap traces) -- explicit dropdowns instead,
            # which always work.
            st.markdown(":material/zoom_in: **Drill into a sector-year**")
            d1, d2 = st.columns(2)
            with d1:
                drill_group = st.selectbox("Sector", sf_groups, key="sf_drill_group")
            years_available = sorted(pivot.columns[pivot.loc[drill_group].notna()].tolist(), reverse=True) \
                if drill_group in pivot.index else []
            with d2:
                drill_year = st.selectbox("Year", years_available, key="sf_drill_year") if years_available else None

            if drill_year is not None:
                cell_df = sf_raw[(sf_raw["group_name"] == drill_group) & (sf_raw["fiscal_year"] == drill_year)].copy()
                mcap_df = get_mcap_map()
                cell_df = cell_df.merge(mcap_df, on="symbol", how="left") if not mcap_df.empty \
                    else cell_df.assign(mcap_cr=pd.NA)
                cell_df = cell_df.sort_values("sales_cr", ascending=False)

                st.caption(":material/filter_alt: Filter constituents — leave at the minimum to include everyone.")
                ff1, ff2, ff3, ff4 = st.columns(4)
                with ff1:
                    min_mcap = st.number_input(
                        "Min. market cap (₹cr)", value=0, step=500, key="sf_min_mcap",
                        help="Current market cap (not historical). Companies with no market-cap "
                        "snapshot are excluded once this is raised above 0.",
                    )
                with ff2:
                    min_rev = st.number_input("Min. revenue (₹cr)", value=0, step=100, key="sf_min_rev")
                with ff3:
                    min_profit = st.number_input(
                        "Min. profit (₹cr)", value=-100_000, step=50, key="sf_min_profit",
                        help="Net profit for the selected year. Defaults very low so loss-making "
                        "companies aren't excluded until you raise it.",
                    )
                with ff4:
                    min_opm = st.number_input(
                        "Min. operating profit (₹cr)", value=-100_000, step=50, key="sf_min_opm",
                        help="Sales × OPM% for the selected year.",
                    )
                mask = (
                    cell_df["sales_cr"].fillna(-1e15).ge(min_rev)
                    & cell_df["net_profit_cr"].fillna(-1e15).ge(min_profit)
                    & cell_df["operating_profit_cr"].fillna(-1e15).ge(min_opm)
                )
                if min_mcap > 0:
                    mask &= cell_df["mcap_cr"].fillna(-1).ge(min_mcap)
                filtered_df = cell_df[mask]

                st.caption(f"{len(filtered_df)} of {len(cell_df)} companies match, for "
                           f"{drill_group} in {drill_year}, sorted by revenue.")
                st.dataframe(
                    filtered_df[["symbol", "mcap_cr", "sales_cr", "sales_growth_pct", "net_profit_cr",
                                  "profit_growth_pct", "operating_profit_cr", "roce_pct", "roe_pct", "pe"]],
                    width="stretch", hide_index=True,
                    column_config={
                        "symbol": st.column_config.TextColumn("company"),
                        "mcap_cr": st.column_config.NumberColumn("mkt cap (₹cr)", format="%,.0f"),
                        "sales_cr": st.column_config.NumberColumn("revenue (₹cr)", format="%,.0f"),
                        "sales_growth_pct": st.column_config.NumberColumn("revenue growth %", format="%+.1f"),
                        "net_profit_cr": st.column_config.NumberColumn("profit (₹cr)", format="%,.0f"),
                        "profit_growth_pct": st.column_config.NumberColumn("profit growth %", format="%+.1f"),
                        "operating_profit_cr": st.column_config.NumberColumn("op. profit (₹cr)", format="%,.0f"),
                        "roce_pct": st.column_config.NumberColumn("ROCE %", format="%.1f"),
                        "roe_pct": st.column_config.NumberColumn("ROE %", format="%.1f"),
                        "pe": st.column_config.NumberColumn("PE", format="%.1f"),
                    },
                )

                st.caption(f"Same data as a heatmap — every filtered constituent's **{sf_metric_label}** "
                           f"by year, grouped by current market-cap tier (so a small-cap's growth isn't "
                           f"visually stacked against a behemoth's), ranked by revenue within each tier.")
                sf_secondary_label = st.selectbox(
                    "Secondary metric (optional)", ["(None)"] + list(SF_METRICS), key="sf_secondary_metric",
                    help="Overlay a second heatmap for the same companies/years, plus a per-company "
                    "correlation vs. the primary metric — e.g. does revenue growth track with price "
                    "moves, or ROCE with EV/EBITDA re-rating.",
                )
                filtered_syms = filtered_df["symbol"].tolist()
                if not filtered_syms:
                    st.caption("No companies match the current filters — loosen them above.")
                else:
                    # Peer group for "vs. peer median" is the sector's FULL membership,
                    # not the filtered/top-N subset -- filtering which companies are
                    # displayed shouldn't quietly change what "the sector" means.
                    full_sector_pivot = sf_raw[sf_raw["group_name"] == drill_group].pivot_table(
                        index="symbol", columns="fiscal_year", values=sf_raw_col, aggfunc="first"
                    )
                    # Group by market-cap tier (stable sort keeps the existing
                    # revenue-descending order within each tier).
                    sym_to_seg = filtered_df.set_index("symbol")["mcap_cr"].map(mcap_segment).to_dict()
                    seg_rank = {seg: i for i, seg in enumerate(MCAP_SEGMENT_ORDER)}
                    filtered_syms_grouped = sorted(
                        filtered_syms, key=lambda s: seg_rank.get(sym_to_seg.get(s, "Unknown"), len(MCAP_SEGMENT_ORDER))
                    )
                    comp_pivot = full_sector_pivot.reindex(filtered_syms_grouped)
                    max_n = len(comp_pivot)
                    top_n = st.slider(
                        "Companies to show", 1, max_n, min(20, max_n), key="sf_drill_top_n",
                    ) if max_n > 20 else max_n
                    comp_pivot_shown = comp_pivot.head(top_n)
                    row_segments = [sym_to_seg.get(s, "Unknown") for s in comp_pivot_shown.index]

                    comp_display, comp_diverging, comp_cbar_title = resolve_view_layer(
                        comp_pivot_shown, full_sector_pivot, sf_kind, sf_diverging, sf_metric_label, sf_view_mode,
                    )

                    comp_display2 = None
                    if sf_secondary_label != "(None)":
                        sf_col2, sf_kind2, sf_diverging2 = SF_METRICS[sf_secondary_label]
                        sf_raw_col2 = SF_RAW_COL[sf_col2]
                        fmt2 = ".1f" if sf_kind2 == "pct" else (".2f" if sf_kind2 == "ratio" else ",.0f")
                        full_sector_pivot2 = sf_raw[sf_raw["group_name"] == drill_group].pivot_table(
                            index="symbol", columns="fiscal_year", values=sf_raw_col2, aggfunc="first"
                        )
                        comp_pivot2_shown = full_sector_pivot2.reindex(comp_pivot_shown.index)
                        comp_display2, comp_diverging2, comp_cbar_title2 = resolve_view_layer(
                            comp_pivot2_shown, full_sector_pivot2, sf_kind2, sf_diverging2,
                            sf_secondary_label, sf_view_mode,
                        )
                        st.caption(
                            f":material/palette: Color = **{sf_metric_label}**; the number in "
                            f"parentheses in each cell (and in the hover) = **{sf_secondary_label}** "
                            f"— scan for cells where both run high/low together, or diverge, to eyeball "
                            f"correlation directly in the grid."
                        )
                        st.plotly_chart(
                            build_company_heatmap(
                                comp_display, comp_diverging, comp_cbar_title, fmt, row_segments,
                                primary_label=sf_metric_label,
                                secondary_pivot=comp_display2.reindex(columns=comp_display.columns),
                                secondary_fmt=fmt2, secondary_label=sf_secondary_label,
                            ),
                            width="stretch",
                        )
                    else:
                        st.plotly_chart(
                            build_company_heatmap(comp_display, comp_diverging, comp_cbar_title, fmt, row_segments),
                            width="stretch",
                        )

                    if comp_display2 is not None:
                        st.markdown(
                            f":material/scatter_plot: **Correlation — {sf_metric_label} vs. {sf_secondary_label}** "
                            f"(Pearson r, per company across its own years shown above, using the "
                            f"**{sf_view_mode}** view; needs at least 3 overlapping years)."
                        )
                        corr_rows = []
                        for sym in comp_display.index:
                            paired = pd.DataFrame({"a": comp_display.loc[sym], "b": comp_display2.loc[sym]}).dropna()
                            if len(paired) >= 3:
                                corr_rows.append({
                                    "symbol": sym, "segment": sym_to_seg.get(sym, "Unknown"),
                                    "years": len(paired), "correlation": paired["a"].corr(paired["b"]),
                                })
                        if not corr_rows:
                            st.caption("No shown company has ≥3 overlapping years for both metrics.")
                        else:
                            corr_df = pd.DataFrame(corr_rows)
                            corr_df = corr_df.reindex(corr_df["correlation"].abs().sort_values(ascending=False).index)
                            st.dataframe(
                                corr_df, width="stretch", hide_index=True,
                                column_config={
                                    "symbol": st.column_config.TextColumn("company"),
                                    "segment": st.column_config.TextColumn("mkt-cap tier"),
                                    "years": st.column_config.NumberColumn("years used", format="%d"),
                                    "correlation": st.column_config.NumberColumn("correlation (r)", format="%.2f"),
                                },
                            )
            else:
                st.caption("No years with data for this sector.")

            st.subheader(":material/bolt: Sharpest year-over-year moves")
            st.caption(
                f"Biggest single-year jump in **{sf_metric_label}** vs the prior year, across all "
                f"{sf_level.replace('_', ' ')}s (not just the ones charted above) — this is where a "
                f"sector visibly took off (or fell off a cliff)."
            )
            moves = sf_agg.sort_values(["group_name", "fiscal_year"]).copy()
            moves["prior_value"] = moves.groupby("group_name")[sf_col].shift(1)
            moves["value"] = moves[sf_col]
            moves["delta"] = moves["value"] - moves["prior_value"]
            moves = moves.dropna(subset=["delta"]).reindex(
                moves["delta"].abs().sort_values(ascending=False).index
            ).head(15)
            if moves.empty:
                st.caption("Not enough consecutive years of data to compute year-over-year moves.")
            else:
                mcols = ["group_name", "fiscal_year", "companies", "prior_value", "value", "delta"]
                num_fmt = "%.1f" if sf_kind == "pct" else ("%.2fx" if sf_kind == "ratio" else "%,.0f")
                delta_fmt = "%+.1f" if sf_kind == "pct" else ("%+.2fx" if sf_kind == "ratio" else "%+,.0f")
                st.dataframe(
                    moves[mcols], width="stretch", hide_index=True,
                    column_config={
                        "group_name": st.column_config.TextColumn(sf_level.replace("_", " ")),
                        "fiscal_year": st.column_config.NumberColumn("year", format="%d"),
                        "companies": st.column_config.NumberColumn("companies", format="%d"),
                        "prior_value": st.column_config.NumberColumn("prior year", format=num_fmt),
                        "value": st.column_config.NumberColumn("this year", format=num_fmt),
                        "delta": st.column_config.NumberColumn("change", format=delta_fmt),
                    },
                )

            with st.expander(":material/table_view: Full data table"):
                detail = sf_agg[sf_agg["group_name"].isin(sf_groups)].sort_values(["group_name", "fiscal_year"])
                st.dataframe(detail, width="stretch", height=360, hide_index=True)
                st.download_button(
                    "Download as CSV", detail.to_csv(index=False),
                    file_name="sector_fundamentals.csv", mime="text/csv",
                )

            with st.expander(":material/warning: Caveats"):
                st.markdown(
                    "- **Sector classification is as-of-today**, applied to the whole history — a "
                    "company that changed sector shows its full history in its current sector.\n"
                    "- **Median ROE %** is only populated for Financial Services — screener.in "
                    "reports ROE for financials and ROCE for everything else, so every other "
                    "sector's median ROE will show blank; use median ROCE % there instead.\n"
                    "- **Totals** (revenue/profit) reflect however many companies had reported data "
                    "for that year, which grows over time as more history gets scraped/listed — a "
                    "rising total can partly be more companies present, not organic growth. Prefer "
                    "the **median growth %** metrics for a like-for-like read.\n"
                    "- **Median PE** uses the share price as of the result's public release date "
                    "(no lookahead) divided by that year's EPS — it can be distorted by companies "
                    "with near-zero or negative EPS, which are excluded.\n"
                    "- Extreme profit-growth outliers from a near-zero prior-year base (>1000%) are "
                    "dropped as noise, not capped/winsorized.\n"
                    "- **YoY growth / vs. peer median views**: for rate metrics (ROCE, ROE, OPM, PE, "
                    "the growth-rate metrics themselves) the change shown is in **percentage points**, "
                    "not a percentage of a percentage; for size/multiple metrics (revenue, profit, "
                    "Price/Sales, EV/EBITDA, op. profit/CFO) it's a standard **% change**. The first "
                    "year of any range has no prior year to compare, so it's blank in both modes.\n"
                    "- **Every year here is an Indian financial year (April-March)** — 'fiscal year "
                    "2024' means the 12 months to 31 March 2024. **Year start price** is the nearest "
                    "trading day on/after 1 April of the prior calendar year; **Year end price** is the "
                    "nearest trading day on/before that 31 March — so 'YoY growth' on either one reads "
                    "as a full-FY price return.\n"
                    "- **Secondary metric / correlation**: the correlation table is a per-company "
                    "Pearson r computed on whichever layer is on screen (Level, YoY growth, or vs. peer "
                    "median) across that company's own years — a small sample (as few as 3 points), so "
                    "treat it as a directional hint, not a statistically robust result."
                )

if section == "Fundamentals":
    st.caption(
        "Historical fundamentals scraped from screener.in's free company page (Quarterly "
        "results, Profit & Loss, Balance Sheet, Cash Flow, Ratios) — laid out the same way "
        "screener shows them, so this first pass can be spot-checked directly against the "
        "real page. Annual history typically runs 2015-2026; quarterly is the last ~3 years, "
        "since that's all screener's free page exposes."
    )

    fc1, fc2 = st.columns([2, 1])
    with fc1:
        fund_symbol = st.selectbox(
            "Company", symbols, key="fund_symbol",
            index=symbols.index("PIDILITIND") if "PIDILITIND" in symbols else 0,
        )
    with fc2:
        horizon_label = st.selectbox(
            "Projection horizon", list(PROJECTION_HORIZONS), key="fund_horizon",
            help="How many future years to add to the Profit & loss and Ratios sections below "
            "(Sales, OPM %, ROCE %, ...). Double-click a future cell and type a number, or "
            "e.g. '15%' for growth on the prior period, then press Enter.",
        )
    horizon_years = PROJECTION_HORIZONS[horizon_label]

    with st.expander(":material/edit_note: Notes", expanded=False):
        note_text = st.text_area(
            "Notes", value=get_note(fund_symbol), key=f"note_input_{fund_symbol}",
            label_visibility="collapsed", height=120,
            placeholder=f"Your notes on {fund_symbol} — thesis, risks, anything worth remembering.",
        )
        if st.button(":material/save: Save note", key="save_note_btn"):
            save_note(fund_symbol, note_text)
            st.toast(f"Note saved for {fund_symbol}", icon=":material/check:")

    scrape_info = get_fundamentals_scrape_info(fund_symbol)
    if scrape_info is None or scrape_info[1] != "ok":
        status = scrape_info[1] if scrape_info else "never scraped"
        st.warning(
            f"No fundamentals for **{fund_symbol}** ({status}). Run "
            "`python scrape_fundamentals_history.py` to (re)scrape — it's resumable and "
            "skips symbols already done.",
            icon=":material/database_off:",
        )
    else:
        basis, _, scraped_at = scrape_info
        screener_url = f"https://www.screener.in/company/{fund_symbol}/" + (
            "consolidated/" if basis == "consolidated" else ""
        )
        st.caption(
            f":material/link: [Open {fund_symbol} on screener.in]({screener_url}) to compare "
            f"side by side — **{basis}** figures, scraped {scraped_at:%Y-%m-%d %H:%M} UTC."
        )

        raw = get_fundamentals_raw(fund_symbol)
        projections = get_projections(fund_symbol)

        st.subheader(":material/show_chart: Chart view")
        if raw.empty:
            st.caption("No data to chart.")
        else:
            cv0, cv1, cv2, cv3 = st.columns([1.4, 1, 3, 1.3], vertical_alignment="bottom")
            with cv0:
                chart_mode = st.radio(
                    "Chart type", ["Overlay (line)", "Small multiples (bar)"], key="fund_chart_mode",
                    help="**Overlay** plots every selected metric as lines on one shared chart. "
                    "**Small multiples** gives each metric its own bar chart, tiled two per row — "
                    "easier to read many metrics at once since each keeps its own scale.",
                )
            with cv1:
                chart_period_label = st.segmented_control(
                    "Period", ["Annual", "Quarterly"], default="Annual", key="fund_chart_period",
                )
                chart_period_label = chart_period_label or "Annual"
            chart_period = "annual" if chart_period_label == "Annual" else "quarterly"
            available_metrics = ordered_metrics_for_period(raw, chart_period)
            default_metrics = [m for m in ["Sales", "Net Profit"] if m in available_metrics] \
                or available_metrics[:1]
            with cv2:
                chart_metrics = st.multiselect(
                    "Metrics", available_metrics, default=default_metrics, key=f"fund_chart_metrics_{chart_period}",
                )
            with cv3:
                if chart_mode == "Overlay (line)":
                    normalize = st.checkbox(
                        "Normalize to 100", key="fund_chart_normalize",
                        help="Index every selected metric to 100 at its first shown period, so metrics "
                        "with different units (₹cr vs % vs days) can be compared on one axis by growth "
                        "trajectory instead of absolute scale.",
                    )
                else:
                    normalize = False

            if not chart_metrics:
                st.info("Pick at least one metric to chart.")
            elif not available_metrics:
                st.caption(f"No {chart_period_label.lower()} data for {fund_symbol}.")
            else:
                chart_raw = raw[(raw["period_type"] == chart_period) & (raw["metric"].isin(chart_metrics))].copy()
                chart_raw["period_end"] = pd.to_datetime(chart_raw["period_end"])

                if chart_mode == "Overlay (line)":
                    figc = go.Figure()
                    unit_axis, units_present = {}, []
                    for metric in chart_metrics:
                        series = chart_raw[chart_raw["metric"] == metric].dropna(subset=["value"]).sort_values("period_end")
                        if series.empty:
                            continue
                        unit = series["unit"].iloc[0]
                        y = series["value"]
                        hover_suffix = ""
                        if normalize:
                            first = y.iloc[0]
                            if first:
                                y = y / first * 100
                                hover_suffix = " (indexed)"
                            axis = "y"
                        else:
                            if unit not in unit_axis:
                                unit_axis[unit] = "y" if not unit_axis else "y2"
                                units_present.append(unit)
                            axis = unit_axis[unit]
                        figc.add_trace(go.Scatter(
                            x=series["period_end"], y=y, mode="lines+markers", name=metric, yaxis=axis,
                            hovertemplate=f"{metric}" + "<br>%{x|%b %Y}: %{y:,.1f}" + hover_suffix + "<extra></extra>",
                        ))
                    layout_kwargs = dict(
                        height=420, margin=dict(l=10, r=10, t=30, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    if normalize:
                        layout_kwargs["yaxis"] = dict(title="Indexed to 100")
                    else:
                        if units_present:
                            layout_kwargs["yaxis"] = dict(title=units_present[0])
                        if len(units_present) > 1:
                            layout_kwargs["yaxis2"] = dict(title=units_present[1], overlaying="y", side="right")
                        if len(units_present) > 2:
                            st.caption(
                                f":material/warning: **{', '.join(units_present[2:])}** share the right "
                                f"axis with **{units_present[1]}** — scales may not line up. Check "
                                "**Normalize to 100** or pick metrics with matching units."
                            )
                    figc.update_layout(**layout_kwargs)
                    st.plotly_chart(figc, width="stretch")
                else:
                    st.caption(
                        f"One bar chart per metric ({chart_period_label.lower()} figures for "
                        f"**{fund_symbol}**), each on its own scale — tiled two per row."
                    )
                    tile_cols = st.columns(2)
                    for idx, metric in enumerate(chart_metrics):
                        series = chart_raw[chart_raw["metric"] == metric].dropna(subset=["value"]).sort_values("period_end")
                        with tile_cols[idx % 2]:
                            if series.empty:
                                st.caption(f"**{metric}** — no data.")
                                continue
                            unit = series["unit"].iloc[0]
                            bar_colors = ["#e57373" if v < 0 else "#66bb6a" for v in series["value"]] \
                                if series["value"].lt(0).any() else "#42a5f5"
                            figm = go.Figure(go.Bar(
                                x=series["period_end"], y=series["value"], marker_color=bar_colors,
                                hovertemplate="%{x|%b %Y}: %{y:,.1f}<extra></extra>",
                            ))
                            figm.update_layout(
                                title=dict(text=f"{metric} ({unit})", font=dict(size=13)),
                                height=280, margin=dict(l=10, r=10, t=36, b=10),
                                showlegend=False,
                            )
                            st.plotly_chart(figm, width="stretch")

        # Single scrolling page instead of tabs -- the jump links get you to a
        # section quickly, but plain scrolling lets you see two sections at
        # once (e.g. OPM % next to ROCE %), which switching tabs didn't allow.
        st.markdown(" · ".join(f"[{label}](#{statement})" for statement, label in FUNDAMENTALS_TABS))

        for statement, label in FUNDAMENTALS_TABS:
            st.subheader(label, anchor=statement)
            is_projectable = statement in PROJECTION_STATEMENTS
            if is_projectable:
                display, values, future_cols, period_end_by_col, input_raw_by_cell = (
                    pivot_fundamentals_with_projections(raw, projections, statement, horizon_years)
                )
            else:
                display, values = pivot_fundamentals_statement(raw, statement)
                future_cols, period_end_by_col, input_raw_by_cell = [], {}, {}

            if display.empty:
                st.caption(
                    f"No {label.lower()} data for {fund_symbol} — common for banks/NBFCs/"
                    "insurers, which screener reports on a different template."
                )
                continue

            display_indexed = display.rename_axis("Metric")

            if is_projectable:
                st.caption(
                    "Columns ending in **E** are projections — double-click a future cell, "
                    "type a plain number or e.g. **15%** for growth on the prior period "
                    "(chains forward like a spreadsheet formula), then press **Enter**."
                )
                # Keyed on a per-(statement, symbol) revision counter: after any
                # edit is saved, bumping this forces Streamlit to mount a *fresh*
                # data_editor instance instead of reusing the one still holding
                # the user's raw typed text -- so the cell redraws showing the
                # newly RESOLVED value, not what they typed (Excel-style: you
                # type the formula, Enter commits it, the cell shows the result).
                rev_key = f"fund_rev_{statement}_{fund_symbol}"
                revision = st.session_state.get(rev_key, 0)
                edited = st.data_editor(
                    display_indexed, width="stretch", height=38 * (len(display_indexed) + 1),
                    disabled=[c for c in display_indexed.columns if c not in future_cols],
                    key=f"editor_{statement}_{fund_symbol}_{revision}",
                )
                changed = False
                for metric in display_indexed.index:
                    for col in future_cols:
                        old_text, new_text = display_indexed.loc[metric, col], edited.loc[metric, col]
                        if new_text != old_text:
                            save_projection_cell(fund_symbol, statement, metric, period_end_by_col[col], new_text)
                            changed = True
                if changed:
                    st.session_state[rev_key] = revision + 1
                    st.rerun()

                saved = {(m, c): v for (m, c), v in input_raw_by_cell.items() if v}
                if saved:
                    with st.expander(f":material/functions: Current assumptions ({len(saved)})", expanded=False):
                        st.caption(" · ".join(f"**{m} / {c}**: {v}" for (m, c), v in saved.items()))
            else:
                st.caption("Select any range of cells, a row, or a column to see sum/average/etc.")
                state = st.dataframe(
                    display_indexed, width="stretch", height=38 * (len(display_indexed) + 1),
                    on_select="rerun", selection_mode=["multi-cell", "multi-row", "multi-column"],
                    key=f"grid_{statement}_{fund_symbol}",
                )

                selection = state.selection if hasattr(state, "selection") else {}
                sel_cells = list(selection.get("cells", []) or [])
                sel_rows = list(selection.get("rows", []) or [])
                sel_cols = list(selection.get("columns", []) or [])

                # Resolve every selected cell -- plus every cell implied by a
                # selected row or column -- back to its raw float via the
                # parallel `values` frame.
                metrics_list = list(display_indexed.index)
                data_cols = list(display_indexed.columns)
                resolved_cells = set(sel_cells)
                for r_idx in sel_rows:
                    if 0 <= r_idx < len(metrics_list):
                        resolved_cells.update((r_idx, c) for c in data_cols)
                for c_name in sel_cols:
                    resolved_cells.update((r_idx, c_name) for r_idx in range(len(metrics_list)))

                selected_values = [
                    values.iloc[r_idx][c_name]
                    for r_idx, c_name in resolved_cells
                    if 0 <= r_idx < len(metrics_list) and c_name in values.columns
                    and pd.notna(values.iloc[r_idx][c_name])
                ]

                if selected_values:
                    s1, s2, s3, s4, s5 = st.columns(5)
                    s1.metric("Sum", f"{sum(selected_values):,.1f}")
                    s2.metric("Average", f"{sum(selected_values) / len(selected_values):,.1f}")
                    s3.metric("Count", f"{len(selected_values)}")
                    s4.metric("Min", f"{min(selected_values):,.1f}")
                    s5.metric("Max", f"{max(selected_values):,.1f}")

    with st.expander(":material/info: How to read this / known limitations"):
        st.markdown(
            "- **Consolidated preferred, standalone fallback**: if a company reports both, "
            "consolidated (parent + subsidiaries) is used; caption above shows which applies.\n"
            "- **Quarterly depth is shallow (~3 years)** — screener's free page simply doesn't "
            "expose more; annual is the source for long-run history.\n"
            "- **ROCE % is annual-only** — screener doesn't report it per quarter.\n"
            "- **Banks/NBFCs/insurers** (Financial Services) report P&L on a different template "
            "with no Sales/OPM% rows — their Balance Sheet/Cash Flow/Ratios may still populate.\n"
            "- Figures are as scraped at the timestamp above; re-run the scraper to refresh.\n"
            "- **Projections**: double-click a future (**E**) cell, type a plain number or "
            "e.g. **15%** for growth vs. the prior period, press Enter — it saves immediately "
            "and the cell redraws showing the resolved value. Growth chains forward like a "
            "spreadsheet formula (a blank cell breaks the chain for later % entries in that "
            "row; an absolute number restarts it). Applies proportionally even to "
            "already-percentage rows like OPM % (20 → 23, not +15 percentage points). Previously "
            "entered assumptions are listed under **Current assumptions** below each projectable "
            "table, since the cell itself only shows the resolved result, not the formula.\n"
            "- **Range stats**: on Quarterly results, Balance sheet, and Cash flow, select "
            "cells, a row, or a column to see sum/average/count/min/max above it. Not available "
            "on Profit & loss / Ratios — editable tables can't also support range selection in "
            "Streamlit, so those two trade stats for direct cell entry.\n"
            "- **Projections & notes are yours** — saved locally in `fundamentals_user_data.duckdb`, "
            "separate from the scraped data, and not touched by re-scraping.\n"
            "- **Chart view**: pick any metrics (annual or quarterly) to plot over time, in either of "
            "two chart types. **Overlay (line)** puts every metric on one shared chart — metrics "
            "sharing a unit share an axis; a second unit gets its own right-hand axis; a third+ unit "
            "shares the right axis (scales won't line up — use **Normalize to 100** instead, which "
            "indexes every series to its first shown period so growth trajectories compare cleanly "
            "regardless of units). **Small multiples (bar)** instead gives each metric its own bar "
            "chart on its own scale, tiled two per row — better when comparing several metrics whose "
            "units don't mix well on one axis."
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

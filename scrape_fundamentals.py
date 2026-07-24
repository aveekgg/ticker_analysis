"""
Scrape per-company fundamentals (market cap, P/E, ROCE, quarterly sales/profit
growth, dividend yield) from the same screener.in /market/ industry pages that
`scrape_sector_classification.py` already visits.

Those industry pages render a table whose default columns are exactly the
fundamentals we want, so this reuses that scraper's INDUSTRIES list and fetch()
helper and just parses more out of each row instead of only the ticker. One
pass over ~188 basic-industry pages gives fundamentals for the whole universe,
sector-aligned.

Kite Connect exposes no fundamentals and NSE's quote endpoint only gives market
cap, so screener.in is the single source that covers all of these at once.

SETUP (one-time):
    pip install pandas requests --break-system-packages

Like the sector scraper this is resumable: progress is checkpointed to CSV
after every industry, so an interrupted or rate-limited run just re-runs and
skips already-done industries.

Output: fundamentals.csv with columns
    symbol, company_name, cmp, pe, mcap_cr, div_yield_pct, np_qtr_cr,
    profit_growth_qtr_pct, sales_qtr_cr, sales_growth_qtr_pct, roce_pct,
    basic_industry
The market pages sort by market cap descending, so page 1 is the largest names
and pagination walks down to the smaller ones -- important for segmenting the
sector leaderboard by company size.
"""

import html as html_module
import re
import time

import pandas as pd
import requests

from scrape_sector_classification import INDUSTRIES, PAGE_COUNT_RE, BASE, HEADERS


def fetch(url: str, max_retries: int = 5) -> str:
    """Like the shared scraper's fetch but also retries on network timeouts /
    connection drops, not just HTTP 429. screener.in intermittently stalls on a
    single request during a long run, and the shared fetch lets a ReadTimeout
    kill the whole (otherwise resumable) scrape -- here we back off and retry."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", 0)) or (5 * (attempt + 1))
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.text
    if last_exc:
        raise last_exc
    r.raise_for_status()

# screener.in's market-page table header text -> our output column name. The
# default column set is fixed across industry pages; we map by header label
# (not position) so a page with a reordered/missing column still lands values
# in the right place rather than silently shifting every field over.
COLUMN_MAP = {
    "CMP Rs.": "cmp",
    "P/E": "pe",
    "Mar Cap Rs.Cr.": "mcap_cr",
    "Div Yld %": "div_yield_pct",
    "NP Qtr Rs.Cr.": "np_qtr_cr",
    "Qtr Profit Var %": "profit_growth_qtr_pct",
    "Sales Qtr Rs.Cr.": "sales_qtr_cr",
    "Qtr Sales Var %": "sales_growth_qtr_pct",
    "ROCE %": "roce_pct",
}

TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.S)
ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
TH_RE = re.compile(r"<th[^>]*>(.*?)</th>", re.S)
TD_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
TICKER_RE = re.compile(r'href="/company/([^/"]+)/')
TAG_RE = re.compile(r"<[^>]+>")


def _text(cell_html: str) -> str:
    """Strip tags/entities/whitespace from a table cell's inner HTML."""
    return html_module.unescape(re.sub(r"\s+", " ", TAG_RE.sub(" ", cell_html))).strip()


def _num(text: str):
    """Screener renders blanks and unavailable values variously ('', '-'); turn
    anything non-numeric into None so a missing ROCE doesn't poison the column."""
    text = text.replace(",", "").strip()
    if not text or text in {"-", "--"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_fundamentals(page_html: str) -> list[dict]:
    """Parse the market-page table into one dict per company. Keys the numeric
    columns by header label, and pulls the NSE tradingsymbol from the /company/
    href in the Name cell (same symbol format used across the rest of the app)."""
    table_match = TABLE_RE.search(page_html)
    if not table_match:
        return []
    rows = ROW_RE.findall(table_match.group(1))
    if not rows:
        return []

    headers = [_text(h) for h in TH_RE.findall(rows[0])]
    # position of each mapped column within this page's header row
    col_index = {COLUMN_MAP[h]: i for i, h in enumerate(headers) if h in COLUMN_MAP}
    name_index = headers.index("Name") if "Name" in headers else 1

    out = []
    for row in rows[1:]:
        cells = TD_RE.findall(row)
        if len(cells) <= name_index:
            continue
        ticker_match = TICKER_RE.search(cells[name_index])
        if not ticker_match:
            continue  # BSE-only numeric scrip rows have no clean NSE symbol
        ticker = html_module.unescape(ticker_match.group(1))
        if ticker.isdigit():
            continue
        record = {"symbol": ticker, "company_name": _text(cells[name_index])}
        for key, idx in col_index.items():
            record[key] = _num(_text(cells[idx])) if idx < len(cells) else None
        out.append(record)
    return out


CHECKPOINT_FILE = "fundamentals_raw.csv"
OUTPUT_FILE = "fundamentals.csv"
REQUEST_DELAY = 1.5  # match the sector scraper: 0.4s was enough to get 429'd

if __name__ == "__main__":
    try:
        checkpoint = pd.read_csv(CHECKPOINT_FILE)
        done_industries = set(checkpoint["basic_industry"])
        rows = checkpoint.to_dict("records")
    except FileNotFoundError:
        done_industries = set()
        rows = []

    print(f"Already done: {len(done_industries)}/{len(INDUSTRIES)} industries")

    for i, (basic_industry, href) in enumerate(INDUSTRIES, 1):
        if basic_industry in done_industries:
            continue

        url = f"{BASE}{href}"
        page_html = fetch(url)
        companies = parse_fundamentals(page_html)

        page_count_match = PAGE_COUNT_RE.search(page_html)
        total_pages = int(page_count_match.group(1)) if page_count_match else 1
        for page in range(2, total_pages + 1):
            time.sleep(REQUEST_DELAY)
            companies.extend(parse_fundamentals(fetch(f"{url}?page={page}")))

        for company in companies:
            rows.append({**company, "basic_industry": basic_industry})

        print(f"[{i}/{len(INDUSTRIES)}] {basic_industry}: {len(companies)} companies"
              + (f" ({total_pages} pages)" if total_pages > 1 else ""))

        pd.DataFrame(rows).to_csv(CHECKPOINT_FILE, index=False)
        time.sleep(REQUEST_DELAY)

    # One row per symbol. A company can list under multiple industry pages; keep
    # the first (its primary basic industry, matching how sector_classification
    # dedups) so fundamentals join 1:1 onto the rest of the app's symbol keys.
    df = pd.DataFrame(rows).drop_duplicates(subset="symbol", keep="first")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved fundamentals for {len(df)} symbols to {OUTPUT_FILE}")
    if "mcap_cr" in df.columns:
        have_mcap = df["mcap_cr"].notna().sum()
        print(f"{have_mcap} have a market-cap value ({have_mcap / len(df):.0%} coverage)")

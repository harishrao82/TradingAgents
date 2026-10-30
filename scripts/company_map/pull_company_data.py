"""Pull company data for the S&P 500 company map. Standard library only.

Run from the tradingagents folder (it reads .env there):

    python3 pull_company_data.py

It is resumable: each answer is cached under ./company_map_data/raw/, so a
second run skips what it already has. If Unusual Whales' daily quota runs low
it stops the UW part cleanly; run it again after 8 PM ET to finish.

At the end it writes one combined file per source to ./company_map_data/,
which is all the analysis needs:
    universe.json  tradier.jsonl  prices.jsonl  uw_info.jsonl
    uw_exposure.jsonl  uw_ownership.jsonl  sec.jsonl  gics.jsonl

Read-only market data and public filings; no order or account endpoints.
Roughly 1,500 UW calls, 520 Tradier calls and 500 SEC calls.
"""

import datetime as dt
import html.parser
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("company_map_data")
RAW = OUT / "raw"
UW = "https://api.unusualwhales.com/api"
TRADIER = "https://api.tradier.com"
SEC_UA = "company-map research m.harish.rao@gmail.com"
UW_RESERVE = 300  # leave this many UW calls for the desk's own use today
HISTORY_YEARS = 3
CLASS_SHARES = {"BRKB": "BRK.B", "BRKA": "BRK.A", "BFB": "BF.B", "BFA": "BF.A"}


def load_env(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def fetch(url, headers, as_json=True):
    """GET with two retries. Returns (status, body, response headers)."""
    for attempt in range(3):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                text = response.read().decode("utf-8", "replace")
                return response.status, (json.loads(text) if as_json else text), dict(response.headers)
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            return exc.code, None, dict(exc.headers or {})
        except Exception as exc:
            if attempt < 2:
                time.sleep(3)
                continue
            print(f"  network error on {url.split('?')[0]}: {exc}")
            return None, None, {}


def cached(kind, key, getter):
    """Return the cached JSON for (kind, key), or call getter() and cache a non-None result."""
    path = RAW / kind / f"{key.replace('/', '_')}.json"
    if path.exists():
        return json.loads(path.read_text())
    body = getter()
    if body is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body))
    return body


# ---------------------------------------------------------------- universe

def uw_headers():
    key = os.getenv("UNUSUAL_WHALES_API_KEY") or os.getenv("UW_API_KEY")
    if not key:
        sys.exit("UNUSUAL_WHALES_API_KEY is not set")
    return {"Authorization": f"Bearer {key}", "Accept": "application/json"}


def universe():
    path = OUT / "universe.json"
    if path.exists():
        return json.loads(path.read_text())
    status, body, _ = fetch(f"{UW}/etfs/SPY/holdings?limit=600", uw_headers())
    if status != 200:
        sys.exit(f"Could not load SPY holdings from UW (HTTP {status})")
    rows = []
    for row in body.get("data") or []:
        ticker = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
        ticker = CLASS_SHARES.get(ticker, ticker).replace("-", ".")
        if ticker and ticker.replace(".", "").isalpha():
            rows.append({"ticker": ticker, "weight": row.get("weight"), "uw_sector": row.get("sector")})
    OUT.mkdir(exist_ok=True)
    path.write_text(json.dumps(rows, indent=1))
    print(f"Universe: {len(rows)} SPY holdings")
    return rows


# ---------------------------------------------------------------- Tradier

def tradier_headers():
    token = os.getenv("TRADIER_ACCESS_TOKEN")
    if not token:
        sys.exit("TRADIER_ACCESS_TOKEN is not set")
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def tradier_symbol(ticker):
    return ticker  # tickers are kept as BRK.B, which is Tradier's spelling too


def tradier_fundamentals(tickers):
    headers = tradier_headers()
    for start in range(0, len(tickers), 25):
        batch = tickers[start:start + 25]
        key = f"batch_{start:04d}_{batch[0]}"

        def get():
            url = f"{TRADIER}/beta/markets/fundamentals/company?" + urllib.parse.urlencode(
                {"symbols": ",".join(tradier_symbol(t) for t in batch)})
            status, body, _ = fetch(url, headers)
            if status != 200:
                print(f"  Tradier fundamentals HTTP {status} for {batch[0]}..")
            time.sleep(0.6)
            return body if status == 200 else None

        cached("tradier_fundamentals", key, get)
    print("Tradier fundamentals: done")


def tradier_history(tickers):
    headers = tradier_headers()
    end = dt.date.today()
    start = end - dt.timedelta(days=365 * HISTORY_YEARS + 5)
    for i, ticker in enumerate(tickers):
        def get():
            url = f"{TRADIER}/v1/markets/history?" + urllib.parse.urlencode(
                {"symbol": tradier_symbol(ticker), "interval": "daily",
                 "start": start.isoformat(), "end": end.isoformat()})
            status, body, _ = fetch(url, headers)
            time.sleep(0.55)  # Tradier market data allows about 120 calls a minute
            return body if status == 200 else None

        cached("tradier_history", ticker, get)
        if i % 100 == 99:
            print(f"  Tradier history: {i + 1}/{len(tickers)}")
    print("Tradier history: done")


# ---------------------------------------------------------------- Unusual Whales

def unusual_whales(tickers):
    headers = uw_headers()
    jobs = [("uw_info", "stock/{t}/info"), ("uw_exposure", "etfs/{t}/exposure"),
            ("uw_ownership", "institution/{t}/ownership?limit=100")]
    for kind, template in jobs:
        for i, ticker in enumerate(tickers):
            stop = {}

            def get():
                status, body, resp_headers = fetch(f"{UW}/{template.format(t=ticker)}", headers)
                used = resp_headers.get("x-uw-daily-req-count")
                limit = resp_headers.get("x-uw-token-req-limit")
                if used and limit and limit.isdigit() and int(limit) - int(used) < UW_RESERVE:
                    stop["why"] = f"UW quota low ({used} of {limit} used today)"
                if status == 429:
                    stop["why"] = "UW returned 429 (quota or rate limit)"
                time.sleep(0.25)
                return body if status == 200 else ({"data": None, "status": status} if status in (404, 422) else None)

            cached(kind, ticker, get)
            if stop:
                print(f"Stopping UW pulls: {stop['why']}. Run again after 8 PM ET to finish.")
                return False
        print(f"UW {kind}: done")
    return True


# ---------------------------------------------------------------- SEC and GICS

def sec(tickers):
    headers = {"User-Agent": SEC_UA, "Accept": "application/json"}
    cik_map = cached("sec", "company_tickers",
                     lambda: fetch("https://www.sec.gov/files/company_tickers.json", headers)[1])
    if not cik_map:
        print("SEC: could not load the ticker map, skipped")
        return
    cik = {row["ticker"].upper(): row["cik_str"] for row in cik_map.values()}
    for ticker in tickers:
        number = cik.get(ticker.replace(".", "-")) or cik.get(ticker)
        if not number:
            continue

        def get():
            status, body, _ = fetch(f"https://data.sec.gov/submissions/CIK{int(number):010d}.json", headers)
            time.sleep(0.15)  # SEC asks for at most 10 requests a second
            if status != 200:
                return None
            return {"ticker": ticker, "cik": number, **{k: body.get(k) for k in
                    ("name", "sic", "sicDescription", "ownerOrg", "category", "stateOfIncorporation", "tickers")}}

        cached("sec_submissions", ticker, get)
    print("SEC: done")


class _TableParser(html.parser.HTMLParser):
    """Collects the rows of the first table with id="constituents"."""

    def __init__(self):
        super().__init__()
        self.inside, self.rows, self.row, self.cell = False, [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "table" and dict(attrs).get("id") == "constituents":
            self.inside = True
        elif self.inside and tag == "tr":
            self.row = []
        elif self.inside and tag in ("td", "th") and self.row is not None:
            self.cell = []

    def handle_endtag(self, tag):
        if tag == "table" and self.inside:
            self.inside = False
        elif self.inside and tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif self.inside and tag == "tr" and self.row:
            self.rows.append(self.row)
            self.row = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def gics():
    def get():
        status, text, _ = fetch("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                                {"User-Agent": SEC_UA}, as_json=False)
        if status != 200:
            return None
        parser = _TableParser()
        parser.feed(text)
        header, *rows = parser.rows
        return [dict(zip(header, row)) for row in rows]

    rows = cached("gics", "wikipedia_sp500", get)
    print(f"GICS (Wikipedia): {len(rows or [])} rows")


# ---------------------------------------------------------------- combine

def combine(tickers):
    def write(name, records):
        with open(OUT / name, "w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        print(f"  {name}: {len(records)} rows")

    def load(kind, ticker):
        path = RAW / kind / f"{ticker}.json"
        return json.loads(path.read_text()) if path.exists() else None

    tradier = []
    for path in sorted((RAW / "tradier_fundamentals").glob("*.json")):
        for item in json.loads(path.read_text()) or []:
            record = {"ticker": str(item.get("request", "")).upper()}
            for result in item.get("results") or []:
                tables = result.get("tables") or {}
                for name in ("asset_classification", "historical_asset_classification"):
                    value = tables.get(name)
                    value = value[-1] if isinstance(value, list) and value else value
                    for field, v in (value or {}).items():
                        if v not in (None, "", 0):
                            record.setdefault(field, v)
                if tables.get("long_descriptions"):
                    record.setdefault("description", tables["long_descriptions"])
                profile = tables.get("company_profile") or {}
                if profile.get("headquarter"):
                    record.setdefault("hq", {k: profile["headquarter"].get(k) for k in ("city", "province", "country")})
                    record.setdefault("employees", profile.get("total_employee_number"))
            tradier.append(record)
    write("tradier.jsonl", tradier)

    prices = []
    for t in tickers:
        body = load("tradier_history", t) or {}
        days = (body.get("history") or {}).get("day") or []
        days = days if isinstance(days, list) else [days]
        prices.append({"ticker": t, "date": [d["date"] for d in days], "close": [d["close"] for d in days]})
    write("prices.jsonl", prices)

    for kind in ("uw_info", "uw_exposure", "uw_ownership"):
        records = []
        for t in tickers:
            body = load(kind, t)
            if body is not None:
                records.append({"ticker": t, "data": body.get("data")})
        write(f"{kind}.jsonl", records)

    write("sec.jsonl", [r for r in (load("sec_submissions", t) for t in tickers) if r])
    write("gics.jsonl", json.loads((RAW / "gics" / "wikipedia_sp500.json").read_text())
          if (RAW / "gics" / "wikipedia_sp500.json").exists() else [])


if __name__ == "__main__":
    load_env()
    tickers = [row["ticker"] for row in universe()]
    tradier_fundamentals(tickers)
    unusual_whales(tickers)
    sec(tickers)
    gics()
    tradier_history(tickers)
    print("\nCombining:")
    combine(tickers)
    print(f"\nDone. Files are in {OUT.resolve()}")

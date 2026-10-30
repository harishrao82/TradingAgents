"""Probe what Tradier, Unusual Whales and SEC EDGAR return about a company.

Standard library only. Run from the tradingagents folder (so it can read .env):

    python3 probe_company_data.py            # 20 default tickers
    python3 probe_company_data.py AAPL KO    # your own list

Writes raw JSON to ./company_probe/ and prints a one-screen summary. Reads only
market-data endpoints; it never touches orders or accounts. Uses about 3 UW
calls per ticker for the first 5 tickers (15 calls of the daily quota).
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT = ["AAPL", "MSFT", "NVDA", "AMD", "INTC", "JPM", "BAC", "XOM", "CVX", "KO",
           "PEP", "PFE", "MRNA", "VRTX", "TSLA", "F", "AMZN", "WMT", "NEE", "PLD"]
OUT = Path("company_probe")
SEC_UA = "company-map research m.harish.rao@gmail.com"


def load_env(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get(url, headers):
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode()[:500]}
    except Exception as exc:  # network failure, bad JSON
        return None, {"error": str(exc)}


def save(name, body):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(body, indent=1))


def tradier(tickers):
    token = os.getenv("TRADIER_ACCESS_TOKEN")
    if not token:
        print("Tradier: TRADIER_ACCESS_TOKEN not set, skipped")
        return
    url = ("https://api.tradier.com/beta/markets/fundamentals/company?"
           + urllib.parse.urlencode({"symbols": ",".join(tickers)}))
    status, body = get(url, {"Authorization": f"Bearer {token}", "Accept": "application/json"})
    save("tradier_fundamentals_company.json", body)
    print(f"Tradier fundamentals/company: HTTP {status}")
    if status != 200:
        print("  ", str(body)[:300])
        return
    for item in body if isinstance(body, list) else [body]:
        symbol = item.get("request")
        tables = {}
        for result in item.get("results") or []:
            tables.update(result.get("tables") or {})
        ac = tables.get("asset_classification") or {}
        desc = tables.get("long_descriptions") or ""
        print(f"  {symbol:5} sector {ac.get('morningstar_sector_code')} group "
              f"{ac.get('morningstar_industry_group_code')} industry {ac.get('morningstar_industry_code')} "
              f"SIC {ac.get('s_i_c')} NAICS {ac.get('n_a_i_c_s')} desc {len(str(desc))} chars")


def unusual_whales(tickers):
    key = os.getenv("UNUSUAL_WHALES_API_KEY") or os.getenv("UW_API_KEY")
    if not key:
        print("UW: UNUSUAL_WHALES_API_KEY not set, skipped")
        return
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    base = "https://api.unusualwhales.com/api"
    status, body = get(f"{base}/stock/Technology/tickers", headers)
    save("uw_sector_Technology_tickers.json", body)
    print(f"UW stock/Technology/tickers: HTTP {status}, {len(body.get('data') or []) if status == 200 else '-'} tickers")
    for ticker in tickers[:5]:
        for path in (f"stock/{ticker}/info", f"etfs/{ticker}/exposure", f"institution/{ticker}/ownership"):
            status, body = get(f"{base}/{path}?limit=50" if "ownership" in path else f"{base}/{path}", headers)
            save("uw_" + path.replace("/", "_") + ".json", body)
            data = body.get("data") if isinstance(body, dict) else None
            if path.endswith("info") and isinstance(data, dict):
                print(f"  {ticker:5} info: sector {data.get('sector')!r}, tags {data.get('uw_tags')}, "
                      f"desc {len(data.get('short_description') or '')} chars")
            else:
                print(f"  {ticker:5} {path.split('/')[-1]}: HTTP {status}, {len(data) if isinstance(data, list) else '-'} rows")
            time.sleep(0.3)


def sec(tickers):
    headers = {"User-Agent": SEC_UA, "Accept": "application/json"}
    status, body = get("https://www.sec.gov/files/company_tickers.json", headers)
    if status != 200:
        print(f"SEC company_tickers: HTTP {status}")
        return
    cik = {row["ticker"]: row["cik_str"] for row in body.values()}
    print(f"SEC: {len(cik)} tickers mapped to CIK")
    for ticker in tickers[:5]:
        if ticker not in cik:
            continue
        status, sub = get(f"https://data.sec.gov/submissions/CIK{cik[ticker]:010d}.json", headers)
        if status == 200:
            save(f"sec_submissions_{ticker}.json", {k: sub.get(k) for k in ("name", "sic", "sicDescription", "tickers", "exchanges")})
            print(f"  {ticker:5} SIC {sub.get('sic')} {sub.get('sicDescription')}")
        time.sleep(0.2)


if __name__ == "__main__":
    load_env()
    symbols = [s.upper() for s in sys.argv[1:]] or DEFAULT
    tradier(symbols)
    unusual_whales(symbols)
    sec(symbols)
    print(f"\nRaw JSON saved in {OUT.resolve()}")

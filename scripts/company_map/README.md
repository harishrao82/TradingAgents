# Company concept map

Places companies so that similar businesses sit close together, like word vectors,
and scores the result against GICS and SIC industry labels.

1. `python3 scripts/company_map/probe_company_data.py` checks what Tradier, Unusual
   Whales and SEC EDGAR return for 20 tickers (standard library only).
2. `python3 scripts/company_map/pull_company_data.py` pulls the S&P 500 (SPY holdings
   from Unusual Whales) into `./company_map_data/`. It is resumable and stops the
   Unusual Whales part early if the daily quota runs low. About 1,500 UW calls.
3. `python3 scripts/company_map/build_company_map.py company_map_data out/` builds
   the vectors and scores them (needs numpy, scipy, scikit-learn). It writes
   `peers.json` (8-10 nearest companies per ticker), `embedding.npy`,
   `map_points.json` and `summary.json`.

Views: Morningstar business description (Tradier, TF-IDF + SVD), ETF co-holding
(Unusual Whales), residual daily returns (Tradier, 3 years) and 13F ownership
(Unusual Whales). The map combines text, ETF and returns at weights 2 / 1.5 / 1.5.

First run (S&P 500, 30 Oct 2026): nearest neighbour shares the GICS sub-industry
for 73% of companies (chance 1%); 37% of the 10 nearest do. Ownership adds nothing
once the other three are in. On this account Tradier returns only the 11
Morningstar sectors, not industry codes, so SIC and GICS are the subsector labels.

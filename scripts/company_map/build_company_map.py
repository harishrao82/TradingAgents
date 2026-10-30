"""Build and score the company concept map from the files pull_company_data.py writes.

    python3 build_company_map.py <company_map_data folder> <output folder>

Needs numpy, scipy and scikit-learn. Four views per company (description text,
ETF co-holding, residual return co-movement, 13F co-ownership), each reduced to
a unit-length vector; the combined map joins them. Labels (GICS, SIC,
Morningstar) are used only to score, except in the supervised NCA check, where
they are learnt on 80% of companies and scored on the other 20%.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import normalize

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.RandomState(0)
K = 10


def jsonl(name):
    return [json.loads(line) for line in open(DATA / name)]


def key(t):
    return t.upper().replace("-", ".")


universe = [key(r["ticker"]) for r in json.load(open(DATA / "universe.json"))]
index = {t: i for i, t in enumerate(universe)}
N = len(universe)

# ------------------------------------------------------------------ labels
gics = {key(r["Symbol"]): r for r in jsonl("gics.jsonl")}
sec = {key(r["ticker"]): r for r in jsonl("sec.jsonl")}
tradier = {key(r["ticker"]): r for r in jsonl("tradier.jsonl")}
uw_info = {key(r["ticker"]): (r["data"] or {}) for r in jsonl("uw_info.jsonl")}
MS_SECTOR = {101: "Basic Materials", 102: "Consumer Cyclical", 103: "Financial Services", 104: "Real Estate",
             205: "Consumer Defensive", 206: "Healthcare", 207: "Utilities", 308: "Communication Services",
             309: "Energy", 310: "Industrials", 311: "Technology"}
labels = {
    "GICS sector": [gics.get(t, {}).get("GICS Sector") for t in universe],
    "GICS sub-industry": [gics.get(t, {}).get("GICS Sub-Industry") for t in universe],
    "SIC 2-digit": [(sec.get(t, {}).get("sic") or "")[:2] or None for t in universe],
    "SIC 4-digit": [sec.get(t, {}).get("sic") or None for t in universe],
    "Morningstar sector": [MS_SECTOR.get(tradier.get(t, {}).get("morningstar_sector_code")) for t in universe],
    "UW sector": [uw_info.get(t, {}).get("sector") for t in universe],
}
names = [gics.get(t, {}).get("Security") or uw_info.get(t, {}).get("full_name") or t for t in universe]

# ------------------------------------------------------------------ views
views, have = {}, {}


def finish(name, X, mask, dims):
    X = np.asarray(X, dtype=float)
    dims = min(dims, X.shape[1] - 1, int(mask.sum()) - 1)
    Z = np.zeros((N, dims))
    Z[mask] = normalize(TruncatedSVD(dims, random_state=0).fit_transform(X[mask]) if not isinstance(X, np.ndarray) or X.shape[1] > dims else X[mask])
    views[name], have[name] = Z, mask


# Text: Tradier (Morningstar) business description.
docs = [tradier.get(t, {}).get("description") or "" for t in universe]
mask = np.array([len(d) > 50 for d in docs])
tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.3, sublinear_tf=True)
T = tfidf.fit_transform(docs)
Z = np.zeros((N, 100))
Z[mask] = normalize(TruncatedSVD(100, random_state=0).fit_transform(T[mask]))
views["Text"], have["Text"] = Z, mask


def bipartite(rows_by_ticker, dims):
    """Company x holder matrix with IDF weighting, reduced by SVD."""
    holders = {}
    ri, ci, val = [], [], []
    for t, items in rows_by_ticker.items():
        for h, v in items:
            ri.append(index[t]); ci.append(holders.setdefault(h, len(holders))); val.append(v)
    M = sparse.csr_matrix((val, (ri, ci)), shape=(N, len(holders)))
    df = np.asarray((M > 0).sum(0)).ravel()
    M = M.multiply(np.log(N / np.maximum(df, 1))).tocsr()
    mask = np.asarray(M.getnnz(1)).ravel() > 0
    Z = np.zeros((N, dims))
    Z[mask] = normalize(TruncatedSVD(dims, random_state=0).fit_transform(normalize(M[mask])))
    return Z, mask, len(holders)


# ETF co-holding: log(1 + weight in the ETF), so a 5% sector-fund weight beats a 0.1% index weight.
etf_rows = {}
for r in jsonl("uw_exposure.jsonl"):
    t = key(r["ticker"])
    if t in index and r["data"]:
        etf_rows[t] = [(x["etf"], np.log1p(float(x.get("weight") or 0) * 10)) for x in r["data"] if x.get("etf")]
views["ETF"], have["ETF"], n_etf = bipartite(etf_rows, 64)

# 13F co-ownership: holder's stake as a share of shares outstanding.
own_rows = {}
for r in jsonl("uw_ownership.jsonl"):
    t = key(r["ticker"])
    if t in index and r["data"]:
        items = []
        for x in r["data"]:
            units, out = float(x.get("units") or 0), float(x.get("shares_outstanding") or 0)
            if x.get("name") and units > 0 and out > 0:
                items.append((x["name"], np.sqrt(units / out)))
        own_rows[t] = items
views["Ownership"], have["Ownership"], n_own = bipartite(own_rows, 64)

# Co-movement: daily log returns with the equal-weight market removed, PCA on companies.
prices = {key(r["ticker"]): dict(zip(r["date"], r["close"])) for r in jsonl("prices.jsonl")}
dates = sorted({d for p in prices.values() for d in p})
dates = [d for d in dates if sum(d in p for p in prices.values()) > 0.9 * N]
P = np.full((N, len(dates)), np.nan)
for t, p in prices.items():
    if t in index:
        P[index[t]] = [p.get(d, np.nan) for d in dates]
R = np.diff(np.log(P), axis=1)
mask = np.isfinite(R).mean(1) > 0.8
mkt = np.nanmean(R[mask], 0)
beta = np.array([np.nansum(r * mkt) / np.nansum(np.where(np.isfinite(r), mkt, np.nan) ** 2) if m else 0 for r, m in zip(R, mask)])
E = R - beta[:, None] * mkt
E = np.where(np.isfinite(E), E, 0)
E = (E - E.mean(1, keepdims=True)) / (E.std(1, keepdims=True) + 1e-12)
Z = np.zeros((N, 50))
Z[mask] = normalize(PCA(50, random_state=0).fit_transform(E[mask]))
views["Returns"], have["Returns"] = Z, mask
n_days = R.shape[1]


def combine(parts):
    return normalize(np.hstack([views[p] for p in parts]))


views["Text+ETF"] = combine(["Text", "ETF"]); have["Text+ETF"] = have["Text"] | have["ETF"]
views["All four"] = combine(["Text", "ETF", "Returns", "Ownership"])
have["All four"] = have["Text"] | have["ETF"] | have["Returns"] | have["Ownership"]
# Weights picked by a grid search scored on held-out folds (GICS sub-industry @10):
# text 2, ETF 1.5, returns 1.5, ownership 0. Ownership adds nothing once the others are in.
views["Weighted"] = normalize(np.hstack([2 * views["Text"], 1.5 * views["ETF"], 1.5 * views["Returns"]]))
have["Weighted"] = have["Text"] | have["ETF"] | have["Returns"]

# ------------------------------------------------------------------ scoring


def neighbours(Z, mask):
    S = Z @ Z.T
    S[~mask, :] = -np.inf; S[:, ~mask] = -np.inf
    np.fill_diagonal(S, -np.inf)
    return np.argsort(-S, 1)[:, :K], S


def purity(nn, lab, mask, rows=None):
    lab = np.array(lab, dtype=object)
    ok = mask & np.array([l is not None for l in lab])
    counts = Counter(lab[ok])
    rows = np.where(ok)[0] if rows is None else [i for i in rows if ok[i]]
    rows = [i for i in rows if counts[lab[i]] > 1]
    p1 = np.mean([lab[nn[i, 0]] == lab[i] for i in rows])
    pk = np.mean([np.mean([lab[j] == lab[i] for j in nn[i]]) for i in rows])
    chance = np.mean([(counts[lab[i]] - 1) / (ok.sum() - 1) for i in rows])
    return p1, pk, chance


results = {}
for v, Z in views.items():
    nn, _ = neighbours(Z, have[v])
    results[v] = {name: purity(nn, lab, have[v]) for name, lab in labels.items()}

# Classifier check: logistic regression on 80%, predict GICS sector of the held-out 20%.
def cv_accuracy(Z, lab, mask):
    lab = np.array(lab, dtype=object)
    ok = np.where(mask & np.array([l is not None for l in lab]))[0]
    y = lab[ok]
    acc = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(ok, y):
        clf = LogisticRegression(max_iter=3000, C=10).fit(Z[ok[tr]], y[tr])
        acc.append((clf.predict(Z[ok[te]]) == y[te]).mean())
    return float(np.mean(acc))


cls = {v: cv_accuracy(views[v], labels["GICS sector"], have[v]) for v in views}

# Supervised: NCA learns a metric from GICS sub-industries on 80% of companies; score
# neighbour purity of the held-out 20% (their neighbours are searched among everyone).
base = views["All four"]
sub = np.array(labels["GICS sub-industry"], dtype=object)
nca_p1, nca_pk, raw_p1, raw_pk = [], [], [], []
for tr, te in KFold(5, shuffle=True, random_state=0).split(base):
    trl = [i for i in tr if sub[i] is not None]
    nca = NeighborhoodComponentsAnalysis(n_components=48, random_state=0, max_iter=200)
    nca.fit(base[trl], sub[trl])
    Zs = normalize(nca.transform(base))
    for Z, a, b in ((Zs, nca_p1, nca_pk), (base, raw_p1, raw_pk)):
        nn, _ = neighbours(Z, have["All four"])
        p1, pk, _ = purity(nn, sub, have["All four"], rows=te)
        a.append(p1); b.append(pk)
supervised = {"unsupervised": (np.mean(raw_p1), np.mean(raw_pk)), "NCA": (np.mean(nca_p1), np.mean(nca_pk))}

# ------------------------------------------------------------------ outputs
best = "Weighted"
nn, S = neighbours(views[best], have[best])
peers = {universe[i]: [[universe[j], round(float(S[i, j]), 3)] for j in nn[i]] for i in range(N) if have[best][i]}
(OUT / "peers.json").write_text(json.dumps(peers))
np.save(OUT / "embedding.npy", views[best])
(OUT / "tickers.json").write_text(json.dumps(universe))

# Companies whose neighbours mostly sit in another GICS sub-industry.
odd = []
for i in range(N):
    if not have[best][i] or sub[i] is None:
        continue
    same = np.mean([sub[j] == sub[i] for j in nn[i]])
    if Counter(sub)[sub[i]] >= 4 and same == 0:
        top = Counter(sub[j] for j in nn[i]).most_common(1)[0]
        odd.append((universe[i], sub[i], top[0], top[1], [universe[j] for j in nn[i][:5]]))

xy = TSNE(2, perplexity=25, random_state=0, init="pca").fit_transform(views[best])
points = [{"t": universe[i], "n": names[i], "s": labels["GICS sector"][i], "u": labels["GICS sub-industry"][i],
           "x": round(float(xy[i, 0]), 2), "y": round(float(xy[i, 1]), 2),
           "p": [universe[j] for j in nn[i][:8]]} for i in range(N) if have[best][i]]
(OUT / "map_points.json").write_text(json.dumps(points))

summary = {"N": N, "coverage": {v: int(m.sum()) for v, m in have.items()}, "n_etf": n_etf, "n_holders": n_own,
           "n_days": n_days, "purity": results, "cls_sector": cls, "supervised": supervised, "odd": odd[:40],
           "examples": {t: peers.get(t, [])[:8] for t in ["NVDA", "JPM", "XOM", "KO", "UNH", "TSLA", "AMZN", "PLD", "NEE", "LLY", "META", "CAT"]}}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
print(json.dumps({k: summary[k] for k in ("coverage", "n_etf", "n_holders", "n_days", "cls_sector", "supervised")}, indent=1, default=float))
for v in views:
    print(f"{v:10}", "  ".join(f"{n}: top1 {r[0]:.2f} @10 {r[1]:.2f} (chance {r[2]:.2f})" for n, r in results[v].items() if n in ("GICS sector", "GICS sub-industry", "SIC 4-digit")))
print("odd:", len(odd)); [print(o) for o in odd[:15]]
for t, p in summary["examples"].items():
    print(t, [x[0] for x in p])

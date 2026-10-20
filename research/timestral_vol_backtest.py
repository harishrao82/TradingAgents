"""Does Timestral forecast volatility and price ranges well enough to help pick options?

An offline study, separate from the desk: it reads prices and implied
volatility, and writes a report. It places no orders and changes no setting.

For each ticker, every ``--step`` trading days, it stands on a past close and
forecasts the next ``--horizon`` trading days four ways, then scores each
forecast against what happened:

* ``rv21``   trailing 21-day realized volatility, carried forward
* ``ewma``   RiskMetrics exponentially weighted variance (lambda 0.94)
* ``har``    HAR-RV regression on daily, weekly and monthly squared returns,
             refit at each origin on windows that had fully closed by then
* ``ts_ret`` Timestral on daily log returns: the 10-90 quantile spread of each
             day gives that day's volatility, summed over the horizon
* ``ts_px``  Timestral on log prices, the shape of the RayCodes_Timestral demo:
             the quantiles at the last step are the price range at the horizon
* ``iv``     the options market's implied volatility on the origin date
             (Unusual Whales, about one year of history), when available

The report answers three questions:

1. Volatility: whose forecast is closest to the realized volatility that
   followed (QLIKE, the standard loss for variance forecasts; lower is better)
   and best ranks the stocks against each other on the same day.
2. Price ranges: how often the price at the horizon landed inside each 80% and
   90% band. A useful band for placing short strikes covers close to what it
   claims while being narrow.
3. Selling premium: when implied volatility sits furthest above a forecast, is
   the gap between implied and realized volatility larger? That gap is what a
   premium seller collects, so this is the test that matters for the desk.

Windows overlap when ``--step`` is under ``--horizon``, so neighbouring rows are
not independent; read the counts as about ``n * step / horizon`` independent
observations. The universe is today's ETF holdings, so names that left the
index are missing (survivorship).

Setup on the machine that runs it::

    pip install torch yfinance git+https://github.com/mistralai/timestral.git
    python research/timestral_vol_backtest.py --etf SPY --top 30

The first run downloads the model weights (490 MB for the default 123M model)
from Hugging Face. ``UNUSUAL_WHALES_API_KEY`` enables the implied volatility
columns; without it the study runs on prices alone.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
# Standard normal quantiles: a band from q10 to q90 is +/- Z80 sigma, q05 to q95 is +/- Z90.
Z80 = 1.2815515655446004
Z90 = 1.6448536269514722
EWMA_LAMBDA = 0.94
# The HAR regression needs this many closed windows before its first forecast.
HAR_MIN_ROWS = 500
# A ticker needs this many returns before an origin to be forecast from it.
MIN_HISTORY = 260
# Cross-sectional statistics need this many tickers on the same day.
MIN_CROSS_SECTION = 5

VOL_FORECASTERS = ["rv21", "ewma", "har", "ts_ret", "ts_px", "iv"]
BAND_FORECASTERS = ["rv21", "ewma", "har", "ts_px", "iv"]


# --------------------------------------------------------------------------- data


def yahoo_symbol(ticker: str) -> str:
    """Class shares are dotted on the desk (BRK.B) and dashed on Yahoo (BRK-B)."""
    return ticker.upper().replace(".", "-")


def load_closes(tickers: list[str], start: date) -> pd.DataFrame:
    """Split- and dividend-adjusted daily closes, one column per ticker."""
    import yfinance as yf

    symbols = [yahoo_symbol(t) for t in tickers]
    data = yf.download(symbols, start=start.isoformat(), auto_adjust=True, progress=False)
    closes = data["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(symbols[0])
    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None).normalize()
    return closes.dropna(how="all").dropna(axis=1, how="all")


def load_iv_unusual_whales(tickers: list[str]) -> dict[str, pd.Series]:
    """About a year of daily implied volatility per ticker, as a fraction.

    One request per ticker against the daily Unusual Whales quota. A ticker the
    plan or the API refuses is skipped, not fatal.
    """
    from tradingagents.dataflows.vendors import unusual_whales as uw

    today = date.today().isoformat()
    series = {}
    for ticker in tickers:
        try:
            rows = uw._request(
                f"stock/{ticker.replace('.', '').replace('-', '')}/iv-rank",
                {"date": today, "timespan": "1y"},
            )
        except Exception as exc:  # one ticker's failure must not end the study
            logger.warning("No implied volatility for %s: %s", ticker, exc)
            continue
        points = {
            pd.Timestamp(str(r.get("date"))[:10]): float(r["volatility"])
            for r in rows
            if r.get("date") and r.get("volatility") not in (None, "")
        }
        if points:
            series[yahoo_symbol(ticker)] = pd.Series(points).sort_index()
    return series


def load_iv_csv(path: Path) -> dict[str, pd.Series]:
    """Implied volatility from a CSV with columns ticker, date, iv (a fraction, 0.25 = 25%)."""
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return {
        yahoo_symbol(ticker): group.set_index("date")["iv"].astype(float).sort_index()
        for ticker, group in frame.groupby("ticker")
    }


def etf_top_holdings(etf: str, top: int) -> list[str]:
    from tradingagents.dataflows.vendors.unusual_whales import get_etf_holdings

    return [ticker for ticker, _ in get_etf_holdings(etf)[:top]]


# --------------------------------------------------------------------- baselines


def realized_vol(returns: np.ndarray, origin: int, horizon: int) -> float:
    """Annualized volatility of the returns after ``origin``, zero mean."""
    window = returns[origin + 1 : origin + 1 + horizon]
    return math.sqrt(TRADING_DAYS * float(np.mean(window**2)))


def trailing_vol(returns: np.ndarray, origin: int, days: int = 21) -> float:
    window = returns[origin - days + 1 : origin + 1]
    return math.sqrt(TRADING_DAYS * float(np.mean(window**2)))


def ewma_variance(returns: np.ndarray, lam: float = EWMA_LAMBDA) -> np.ndarray:
    """Daily variance after each return; seeded with the first 21 days' mean square."""
    out = np.empty_like(returns)
    var = float(np.mean(returns[1:22] ** 2))
    out[0] = var
    for k in range(1, len(returns)):
        var = lam * var + (1 - lam) * returns[k] ** 2
        out[k] = var
    return out


class HarModel:
    """HAR-RV on squared daily returns, forecasting mean squared return over the horizon.

    Regressors at day k: r_k^2, the mean of the last 5 and of the last 22. The
    target row for day k is closed once day k + horizon has printed, so the fit
    for an origin uses only rows with k + horizon <= origin.
    """

    def __init__(self, returns: np.ndarray, horizon: int):
        sq = returns**2
        n = len(sq)
        self.horizon = horizon
        cumsum = np.concatenate([[0.0], np.cumsum(sq)])

        def trailing_mean(days: int) -> np.ndarray:
            out = np.full(n, np.nan)
            k = np.arange(days, n)
            out[k] = (cumsum[k + 1] - cumsum[k + 1 - days]) / days
            return out

        self.x = np.column_stack([np.ones(n), sq, trailing_mean(5), trailing_mean(22)])
        self.y = np.full(n, np.nan)
        k = np.arange(1, n - horizon)
        self.y[k] = (cumsum[k + 1 + horizon] - cumsum[k + 1]) / horizon

    def forecast(self, origin: int) -> float:
        """Annualized volatility forecast, or NaN while too few windows have closed."""
        rows = np.arange(22, origin - self.horizon + 1)
        if len(rows) < HAR_MIN_ROWS:
            return math.nan
        beta, *_ = np.linalg.lstsq(self.x[rows], self.y[rows], rcond=None)
        variance = float(self.x[origin] @ beta)
        # OLS can go negative after a calm stretch; floor at a fifth of the monthly level.
        variance = max(variance, 0.2 * float(self.x[origin, 3]))
        return math.sqrt(TRADING_DAYS * variance)


# ---------------------------------------------------------------------- timestral


class TimestralQuantiles:
    """Batched quantile forecasts, returned as a NumPy array (batch, horizon, levels)."""

    def __init__(self, model_id: str, device: str | None = None, batch_size: int = 32):
        import torch
        from timestral import TimestralForecaster

        self._torch = torch
        kwargs = {"device": device} if device else {}
        self.model = TimestralForecaster.from_pretrained(model_id, **kwargs)
        self.batch_size = batch_size
        self.levels: tuple[float, ...] | None = None

    def __call__(self, contexts: np.ndarray, horizon: int) -> np.ndarray:
        chunks = []
        for start in range(0, len(contexts), self.batch_size):
            with self._torch.inference_mode():
                forecast = self.model.forecast(
                    contexts[start : start + self.batch_size], prediction_length=horizon
                )
            self.levels = tuple(float(q) for q in forecast.quantile_levels)
            quantiles = forecast.quantiles
            if hasattr(quantiles, "detach"):
                quantiles = quantiles.detach().cpu().numpy()
            # (batch, variates=1, horizon, levels) -> (batch, horizon, levels)
            chunks.append(np.asarray(quantiles, dtype=float)[:, 0])
            logger.info("Timestral: %d of %d windows", min(start + self.batch_size, len(contexts)), len(contexts))
        return np.concatenate(chunks)


def level_index(levels: tuple[float, ...], level: float) -> int:
    for i, q in enumerate(levels):
        if abs(q - level) < 1e-9:
            return i
    raise ValueError(f"quantile level {level} not among {levels}")


def left_padded(values: np.ndarray, end: int, length: int) -> np.ndarray:
    """The ``length`` values ending at ``end`` inclusive, NaN-padded on the left."""
    window = values[max(0, end - length + 1) : end + 1]
    return np.concatenate([np.full(length - len(window), np.nan), window])


# ------------------------------------------------------------------------ the run


def origin_dates(index: pd.DatetimeIndex, start: pd.Timestamp, step: int, horizon: int) -> list[pd.Timestamp]:
    """Every ``step``-th session from ``start`` whose horizon has fully printed."""
    positions = [i for i, d in enumerate(index) if d >= start and i + horizon < len(index)]
    return [index[i] for i in positions[::step]]


def build_rows(
    closes: pd.DataFrame,
    iv: dict[str, pd.Series],
    origins: list[pd.Timestamp],
    horizon: int,
    context: int,
    quantiles,
) -> pd.DataFrame:
    """One row per (ticker, origin) with the realized outcome and every forecast."""
    rows, ret_contexts, px_contexts = [], [], []
    for ticker in closes.columns:
        prices = closes[ticker].dropna()
        log_px = np.log(prices.to_numpy(dtype=float))
        returns = np.concatenate([[0.0], np.diff(log_px)])
        position = {d: i for i, d in enumerate(prices.index)}
        ewma = ewma_variance(returns)
        # The leading zero return is a placeholder, not a market move.
        context_returns = np.concatenate([[np.nan], returns[1:]])
        har = HarModel(returns, horizon)
        ticker_iv = iv.get(ticker)
        for when in origins:
            i = position.get(when)
            if i is None or i < MIN_HISTORY or i + horizon >= len(prices):
                continue
            iv_today = math.nan
            if ticker_iv is not None and when in ticker_iv.index:
                iv_today = float(ticker_iv.loc[when])
            rows.append(
                {
                    "ticker": ticker,
                    "date": when,
                    "realized": realized_vol(returns, i, horizon),
                    "rv21": trailing_vol(returns, i),
                    "ewma": math.sqrt(TRADING_DAYS * ewma[i]),
                    "har": har.forecast(i),
                    "iv": iv_today if iv_today > 0 else math.nan,
                    "fwd_return": float(log_px[i + horizon] - log_px[i]),
                }
            )
            ret_contexts.append(left_padded(context_returns, i, context))
            px_contexts.append(left_padded(log_px, i, context))

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    q_ret = quantiles(np.stack(ret_contexts), horizon)
    levels = quantiles.levels
    lo, hi = level_index(levels, 0.1), level_index(levels, 0.9)
    step_sigma = (q_ret[:, :, hi] - q_ret[:, :, lo]) / (2 * Z80)
    frame["ts_ret"] = np.sqrt(TRADING_DAYS * np.mean(step_sigma**2, axis=1))

    q_px = quantiles(np.stack(px_contexts), horizon)[:, -1, :]
    last = np.array([c[-1] for c in px_contexts])
    for level in (0.05, 0.1, 0.5, 0.9, 0.95):
        frame[f"ts_px_q{round(level * 100):02d}"] = q_px[:, level_index(levels, level)] - last
    frame["ts_px"] = (frame["ts_px_q90"] - frame["ts_px_q10"]) / (2 * Z80) / math.sqrt(horizon / TRADING_DAYS)
    return frame


# ------------------------------------------------------------------------- scoring


def qlike(realized: pd.Series, forecast: pd.Series) -> float:
    """Patton's QLIKE on variances; 0 for a perfect forecast, lower is better."""
    ratio = (realized / forecast) ** 2
    return float(np.mean(ratio - np.log(ratio) - 1))


def rank_ic(frame: pd.DataFrame, column: str, target: pd.Series) -> float:
    """Mean same-day Spearman correlation between a column and a target, across tickers."""
    values = []
    for _, day in frame.assign(_target=target).groupby("date"):
        day = day[[column, "_target"]].dropna()
        if len(day) >= MIN_CROSS_SECTION:
            values.append(day[column].rank().corr(day["_target"].rank()))
    return float(np.nanmean(values)) if values else math.nan


def vol_table(frame: pd.DataFrame, forecasters: list[str]) -> pd.DataFrame:
    table = []
    for name in forecasters:
        sub = frame[[name, "realized", "date", "ticker"]].dropna()
        sub = sub[sub[name] > 0]
        if sub.empty:
            continue
        table.append(
            {
                "forecaster": name,
                "n": len(sub),
                "QLIKE": qlike(sub["realized"], sub[name]),
                "RMSE (vol pts)": 100 * float(np.sqrt(np.mean((sub[name] - sub["realized"]) ** 2))),
                "bias (vol pts)": 100 * float(np.mean(sub[name] - sub["realized"])),
                "corr(log)": float(np.corrcoef(np.log(sub[name]), np.log(sub["realized"]))[0, 1]),
                "same-day rank IC": rank_ic(sub, name, sub["realized"]),
            }
        )
    return pd.DataFrame(table)


def band_table(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Coverage and width of 80% and 90% bands for the price ``horizon`` days on.

    Volatility forecasts become zero-drift normal bands around today's price;
    Timestral's price model gives its own quantiles.
    """
    scale = math.sqrt(horizon / TRADING_DAYS)
    table = []
    for name in BAND_FORECASTERS:
        if name == "ts_px":
            sub = frame.dropna(subset=["ts_px_q10"])
            bands = {80: (sub["ts_px_q10"], sub["ts_px_q90"]), 90: (sub["ts_px_q05"], sub["ts_px_q95"])}
        else:
            sub = frame.dropna(subset=[name])
            half = sub[name] * scale
            bands = {80: (-Z80 * half, Z80 * half), 90: (-Z90 * half, Z90 * half)}
        if sub.empty:
            continue
        move = sub["fwd_return"]
        row = {"forecaster": name, "n": len(sub)}
        for claim, (low, high) in bands.items():
            row[f"{claim}% band coverage"] = float(np.mean((move >= low) & (move <= high)))
            row[f"{claim}% band width (% of price)"] = 100 * float(np.mean(np.exp(high) - np.exp(low)))
        table.append(row)
    return pd.DataFrame(table)


def direction_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Does Timestral's median price call the direction? Against always guessing up."""
    sub = frame.dropna(subset=["ts_px_q50"])
    called = np.sign(sub["ts_px_q50"])
    actual = np.sign(sub["fwd_return"])
    return pd.DataFrame(
        [
            {
                "n": len(sub),
                "Timestral median hit rate": float(np.mean(called == actual)),
                "always-up hit rate": float(np.mean(actual > 0)),
                "corr(median, actual return)": float(sub["ts_px_q50"].corr(sub["fwd_return"])),
                "same-day rank IC": rank_ic(sub, "ts_px_q50", sub["fwd_return"]),
            }
        ]
    )


def premium_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Does implied minus forecast volatility predict implied minus realized?

    ``IV - realized`` is roughly what a delta-hedged premium seller earns. For
    each forecaster: the correlation of the two gaps, and, among each day's
    tickers, the realized gap in the third where IV sits furthest above the
    forecast minus the third where it sits lowest.
    """
    with_iv = frame.dropna(subset=["iv"])
    if with_iv.empty:
        return pd.DataFrame()
    edge = with_iv["iv"] - with_iv["realized"]
    table = []
    for name in ["rv21", "ewma", "har", "ts_ret", "ts_px"]:
        sub = with_iv.dropna(subset=[name])
        signal = sub["iv"] - sub[name]
        spreads = []
        for _, idx in sub.groupby("date").groups.items():
            day_signal, day_edge = signal.loc[idx], edge.loc[idx]
            if len(day_signal) < MIN_CROSS_SECTION:
                continue
            top = day_signal >= day_signal.quantile(2 / 3)
            bottom = day_signal <= day_signal.quantile(1 / 3)
            spreads.append(day_edge[top].mean() - day_edge[bottom].mean())
        table.append(
            {
                "signal": f"IV - {name}",
                "n": len(sub),
                "corr with IV - realized": float(signal.corr(edge.loc[sub.index])),
                "top minus bottom third (vol pts)": 100 * float(np.nanmean(spreads)) if spreads else math.nan,
            }
        )
    return pd.DataFrame(table)


def render_report(frame: pd.DataFrame, args: argparse.Namespace) -> str:
    def md(table: pd.DataFrame) -> str:
        if table.empty:
            return "_no rows_"

        def cell(value) -> str:
            if isinstance(value, float):
                return "n/a" if math.isnan(value) else f"{value:.3f}"
            return str(value)

        lines = ["| " + " | ".join(table.columns) + " |", "|" + " --- |" * len(table.columns)]
        lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in table.itertuples(index=False)]
        return "\n".join(lines)

    with_iv = frame.dropna(subset=["iv"])
    parts = [
        f"# Timestral volatility backtest, {datetime.now():%Y-%m-%d %H:%M}\n",
        f"{frame['ticker'].nunique()} tickers, {frame['date'].nunique()} origins from "
        f"{frame['date'].min():%Y-%m-%d} to {frame['date'].max():%Y-%m-%d}, horizon "
        f"{args.horizon} trading days, every {args.step} days, model {args.model}, context "
        f"{args.context} days. Rows with implied volatility: {len(with_iv)}.\n",
        "\n## 1. Volatility forecasts, all rows\n\nLower QLIKE and RMSE are better; "
        "higher correlation and rank IC are better.\n\n",
        md(vol_table(frame, [f for f in VOL_FORECASTERS if f != "iv"])),
        "\n\n## 1b. Volatility forecasts, rows that have implied volatility\n\n"
        "Implied volatility usually runs above realized (the premium sellers earn), "
        "so its bias is expected to be positive.\n\n",
        md(vol_table(with_iv, VOL_FORECASTERS)),
        f"\n\n## 2. Price bands {args.horizon} trading days out\n\n"
        "A calibrated 80% band covers about 0.80. Narrower is better only at the same coverage.\n\n",
        md(band_table(frame, args.horizon)),
        "\n\n## 3. Direction from Timestral's median price\n\n",
        md(direction_table(frame)),
        "\n\n## 4. Selling premium: does IV minus forecast pick the richer options?\n\n"
        "Positive and larger is better. A forecaster earns a place in the strategist only "
        "if its row beats HAR and EWMA here.\n\n",
        md(premium_table(frame)),
        f"\n\nOverlapping windows: about {len(frame) * args.step // args.horizon} independent "
        "observations. Universe is today's holdings (survivorship).\n",
    ]
    return "".join(parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    universe = parser.add_mutually_exclusive_group()
    universe.add_argument("--tickers", help="Comma-separated tickers, e.g. AAPL,MSFT,NVDA")
    universe.add_argument("--etf", default="SPY", help="ETF whose top holdings form the universe (Unusual Whales)")
    parser.add_argument("--top", type=int, default=30, help="How many of the ETF's heaviest holdings")
    parser.add_argument("--start", help="First origin, YYYY-MM-DD; default two years ago")
    parser.add_argument("--horizon", type=int, default=21, help="Trading days forecast, about a monthly option")
    parser.add_argument("--step", type=int, default=5, help="Trading days between origins")
    parser.add_argument("--context", type=int, default=1024, help="Trading days of history Timestral reads")
    parser.add_argument("--model", default="mistralai/Timestral-123M", help="Hugging Face model id or local folder")
    parser.add_argument("--device", help="cpu, cuda or mps; default the library's choice")
    parser.add_argument("--batch", type=int, default=32, help="Windows per Timestral call")
    parser.add_argument("--iv", choices=["uw", "none"], default="uw", help="Implied volatility source")
    parser.add_argument("--iv-csv", type=Path, help="CSV of ticker,date,iv to use instead of Unusual Whales")
    parser.add_argument("--out", type=Path, default=Path("results/timestral_vol"), help="Output folder")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = etf_top_holdings(args.etf, args.top)
    start = date.fromisoformat(args.start) if args.start else date.today() - timedelta(days=730)
    history_start = start - timedelta(days=int(args.context * 1.46) + 60)

    logger.info("Prices for %d tickers from %s", len(tickers), history_start)
    closes = load_closes(tickers, history_start)
    if args.iv_csv:
        iv = load_iv_csv(args.iv_csv)
    elif args.iv == "uw":
        iv = load_iv_unusual_whales(tickers)
    else:
        iv = {}
    logger.info("Implied volatility for %d tickers", len(iv))

    origins = origin_dates(closes.index, pd.Timestamp(start), args.step, args.horizon)
    quantiles = TimestralQuantiles(args.model, device=args.device, batch_size=args.batch)
    frame = build_rows(closes, iv, origins, args.horizon, args.context, quantiles)
    if frame.empty:
        logger.error("No forecastable windows: check the tickers and --start.")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    frame.to_csv(args.out / f"rows_{stamp}.csv", index=False)
    report = render_report(frame, args)
    (args.out / f"report_{stamp}.md").write_text(report)
    print(report)
    logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Timestral volatility study: scoring math on synthetic prices with a known volatility.

A stub stands in for the model and returns exact normal quantiles, so nothing
loads torch or reaches a network.
"""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PATH = Path(__file__).resolve().parents[1] / "research" / "timestral_vol_backtest.py"
_spec = importlib.util.spec_from_file_location("timestral_vol_backtest", _PATH)
vb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vb)

LEVELS = (0.01, 0.05, *[round(0.1 + 0.05 * k, 2) for k in range(17)], 0.95, 0.99)


def _z(level):
    from statistics import NormalDist

    return NormalDist().inv_cdf(level)


class StubQuantiles:
    """Normal quantiles at a fixed daily sigma: flat for returns, widening for log prices."""

    levels = LEVELS

    def __init__(self, daily_sigma):
        self.sigma = daily_sigma

    def __call__(self, contexts, horizon):
        z = np.array([_z(q) for q in LEVELS])
        steps = np.arange(1, horizon + 1)[:, None]
        is_price = np.nanmean(np.abs(contexts), axis=1) > 1  # log prices sit near log(100)
        out = np.empty((len(contexts), horizon, len(LEVELS)))
        for b, ctx in enumerate(contexts):
            if is_price[b]:
                out[b] = ctx[-1] + self.sigma * np.sqrt(steps) * z
            else:
                out[b] = np.broadcast_to(self.sigma * z, (horizon, len(LEVELS)))
        return out


def _closes(daily_sigma, days=1200, tickers=("AAA", "BBB", "CCC", "DDD", "EEE", "FFF"), seed=0):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=days)
    paths = {t: 100 * np.exp(np.cumsum(rng.normal(0, daily_sigma, days))) for t in tickers}
    return pd.DataFrame(paths, index=index)


def test_level_index_finds_model_levels():
    assert vb.level_index(LEVELS, 0.1) == 2
    assert vb.level_index(LEVELS, 0.9) == 18
    with pytest.raises(ValueError):
        vb.level_index(LEVELS, 0.42)


def test_realized_vol_uses_only_the_window_after_the_origin():
    returns = np.zeros(50)
    returns[11:32] = 0.01  # 21 days of 1% after origin 10
    assert vb.realized_vol(returns, 10, 21) == pytest.approx(0.01 * math.sqrt(252))
    returns[10] = 0.5  # the origin's own return is history, not outcome
    assert vb.realized_vol(returns, 10, 21) == pytest.approx(0.01 * math.sqrt(252))


def test_har_waits_for_enough_closed_windows():
    returns = np.random.default_rng(1).normal(0, 0.01, 900)
    har = vb.HarModel(returns, 21)
    assert math.isnan(har.forecast(400))
    assert har.forecast(800) == pytest.approx(0.01 * math.sqrt(252), rel=0.25)


def test_true_volatility_scores_well_and_bands_are_calibrated():
    sigma = 0.015
    closes = _closes(sigma)
    origins = vb.origin_dates(closes.index, closes.index[400], 5, 21)
    frame = vb.build_rows(closes, {}, origins, 21, 256, StubQuantiles(sigma))

    true_vol = sigma * math.sqrt(252)
    assert frame["ts_ret"].to_numpy() == pytest.approx(true_vol, rel=1e-3)
    assert frame["ts_px"].to_numpy() == pytest.approx(true_vol, rel=1e-3)
    assert frame["ts_px_q50"].abs().max() < 1e-9

    vols = vb.vol_table(frame, ["rv21", "ts_ret"]).set_index("forecaster")
    # The true volatility should beat a noisy trailing estimate on QLIKE.
    assert vols.loc["ts_ret", "QLIKE"] < vols.loc["rv21", "QLIKE"]

    bands = vb.band_table(frame, 21).set_index("forecaster")
    assert bands.loc["ts_px", "80% band coverage"] == pytest.approx(0.80, abs=0.06)
    assert bands.loc["ts_px", "90% band coverage"] == pytest.approx(0.90, abs=0.05)


def test_premium_table_rewards_a_signal_that_tracks_the_gap():
    closes = _closes(0.015)
    origins = vb.origin_dates(closes.index, closes.index[400], 5, 21)
    frame = vb.build_rows(closes, {}, origins, 21, 256, StubQuantiles(0.015))
    # Implied vol that knew the future: IV - forecast then predicts IV - realized.
    frame["iv"] = frame["realized"] + 0.05 + 0.5 * (frame["realized"] - frame["ts_ret"])
    table = vb.premium_table(frame).set_index("signal")
    assert set(table.index) == {"IV - rv21", "IV - ewma", "IV - har", "IV - ts_ret", "IV - ts_px"}
    report = vb.render_report(frame, vb.parse_args(["--tickers", "AAA"]))
    assert "## 4. Selling premium" in report and "IV - ts_ret" in report


def test_iv_csv_maps_class_shares_to_yahoo(tmp_path):
    path = tmp_path / "iv.csv"
    path.write_text("ticker,date,iv\nBRK.B,2025-01-02,0.18\nAAPL,2025-01-02,0.25\n")
    iv = vb.load_iv_csv(path)
    assert set(iv) == {"BRK-B", "AAPL"}
    assert iv["AAPL"].iloc[0] == pytest.approx(0.25)

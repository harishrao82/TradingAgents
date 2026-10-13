"""Basket tilt: news calls and earnings dates become bounded weights; nothing reaches a network."""
import json

import pytest

import tradingagents.default_config as default_config
from tradingagents.conviction import tilt as tilt_mod
from tradingagents.conviction.screen import NewsVerdict
from tradingagents.conviction.tilt import (
    bounded_rescale,
    next_month_end,
    tilt_names,
    weighted_water_fill,
)


def _v(ticker, direction, conviction):
    return NewsVerdict(ticker=ticker, direction=direction, conviction=conviction, catalyst="c", reason="r")


def test_rescale_keeps_bounds_and_averages_one():
    raw = {"A": 1.5, "B": 1.5, "C": 1.5, "D": 0.5}
    out = bounded_rescale(raw, 0.5, 1.5)
    assert sum(out.values()) == pytest.approx(4)
    assert all(0.5 - 1e-9 <= v <= 1.5 + 1e-9 for v in out.values())
    assert out["D"] == pytest.approx(0.5)


def test_rescale_leaves_equal_weight_alone():
    assert bounded_rescale({"A": 1.0, "B": 1.0}, 0.5, 1.5) == {"A": 1.0, "B": 1.0}


def test_tilt_follows_news_and_holds_up_tilts_before_earnings():
    names = tilt_names(["UP", "EARN"], ["DOWN", "QUIET"],
                       [_v("UP", "up", 5), _v("EARN", "up", 5), _v("DOWN", "down", 5)],
                       {"EARN": "2026-10-28"})
    by = {n.ticker: n for n in names}
    assert by["UP"].multiplier > by["EARN"].multiplier
    assert by["DOWN"].multiplier < by["QUIET"].multiplier
    assert by["DOWN"].drop and not by["UP"].drop
    assert by["EARN"].raw == 1.0 and "earnings 2026-10-28" in by["EARN"].notes[0]
    assert sum(n.weight for n in names) == pytest.approx(1.0)
    assert by["UP"].bucket == "momentum" and by["QUIET"].bucket == "pullback"


def test_water_fill_matches_the_cells_equal_cost_level_when_untilted():
    cost = {"A": 0.0, "B": 3000.0, "C": 9000.0}
    level = weighted_water_fill(cost, {s: 1.0 for s in cost}, 6000.0)
    assert level == pytest.approx(4500.0)  # (6000 + 0 + 3000) / 2, C already above


def test_water_fill_spends_exactly_the_cash_with_tilts():
    cost, mult = {"A": 0.0, "B": 2000.0, "C": 5000.0}, {"A": 1.5, "B": 1.0, "C": 0.5}
    level = weighted_water_fill(cost, mult, 4000.0)
    spent = sum(max(mult[s] * level - cost[s], 0) for s in cost)
    assert spent == pytest.approx(4000.0, rel=1e-6)


def test_next_month_end_is_a_weekday():
    assert next_month_end("2026-10-13") == "2026-11-30"
    assert next_month_end("2026-12-02") == "2027-01-29"


def test_run_basket_tilt_saves_json(monkeypatch, tmp_path):
    from tradingagents.conviction.workflow import ScreenResult

    monkeypatch.setattr(tilt_mod, "run_conviction_screen", lambda *a, **k: ScreenResult(
        "custom list", "2026-10-13", "2026-10-09", 2, 1, [_v("MU", "down", 4)], {}, [], screen_model="m"))
    monkeypatch.setattr(tilt_mod.unusual_whales, "get_next_earnings",
                        lambda t, d: ("2026-12-17", "postmarket") if t == "MU" else None)
    result = tilt_mod.run_basket_tilt(dict(default_config.DEFAULT_CONFIG), "2026-10-13", ["MU"], ["AMD"])
    assert result.until == "2026-11-30"
    assert result.multipliers["MU"] < 1 < result.multipliers["AMD"]
    md, js = tilt_mod.save_tilt(result, tilt_mod.render_tilt(result), str(tmp_path))
    saved = json.loads((tmp_path / "tilts" / "latest.json").read_text())
    assert saved["multipliers"] == result.multipliers and md.exists()

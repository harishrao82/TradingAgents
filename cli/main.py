import sys

import typer

from cli.display import console
from cli.models import AnalystType, AssetType
from cli.prompts import filter_analysts_for_asset_type, parse_analysts
from cli.run import run_analysis
from tradingagents.backtest import iter_grid, run_backtest, summarize
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.portfolio import load_portfolio

# prompt_toolkit's win32 output module is importable only on Windows (it asserts
# the platform at import time), so gate on the platform rather than catching the
# failure — that way a genuinely broken prompt_toolkit on Windows still surfaces
# instead of silently disabling the handler below. Off Windows this stays an
# empty tuple, which `except` accepts and never matches (#1138).
if sys.platform == "win32":  # pragma: no cover - platform dependent
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError

    _NO_CONSOLE_ERRORS: tuple[type[BaseException], ...] = (NoConsoleScreenBufferError,)
else:
    _NO_CONSOLE_ERRORS = ()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)


@app.callback(invoke_without_command=True)
def analyze(
    ctx: typer.Context,
    checkpoint: bool | None = typer.Option(
        None,
        "--checkpoint/--no-checkpoint",
        help="Enable/disable checkpoint-resume (save state after each node so a "
        "crashed run can resume). Omit to honor TRADINGAGENTS_CHECKPOINT_ENABLED.",
    ),
    clear_checkpoints: bool = typer.Option(
        False,
        "--clear-checkpoints",
        help="Delete all saved checkpoints before running (force fresh start).",
    ),
    portfolio: str = typer.Option(
        None,
        "--portfolio",
        help="JSON file with current holdings and cash, so the trader, risk and "
        "portfolio agents size against your actual position.",
    ),
    ticker: str = typer.Option(None, "--ticker", help="Ticker to analyze, e.g. NVDA or 0700.HK; skips the prompt"),
    date: str = typer.Option(None, "--date", help="Analysis date, YYYY-MM-DD; skips the prompt"),
    analysts: str = typer.Option(
        None, "--analysts", help="Comma-separated analysts, e.g. market,news; skips the prompt"
    ),
    save: bool | None = typer.Option(
        None, "--save/--no-save", help="Save the report under results_dir without asking"
    ),
    show: bool | None = typer.Option(
        None, "--show/--no-show", help="Show the full report at the end without asking"
    ),
    html: bool | None = typer.Option(
        None, "--html/--no-html",
        help="Save the report as one HTML page too, complete_report.html, without asking (default: yes)",
    ),
):
    """Run an analysis. This is what a bare `tradingagents` does.

    Flags answer their questions; with provider, models, depth and language also
    set through TRADINGAGENTS_* variables, the run asks nothing.
    """
    if ctx.invoked_subcommand is not None:
        return
    if clear_checkpoints:
        from tradingagents.graph.checkpointer import clear_all_checkpoints
        n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
        console.print(f"[yellow]Cleared {n} checkpoint(s).[/yellow]")
    portfolio_context = None
    if portfolio:
        try:
            portfolio_context = load_portfolio(portfolio)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from None

    try:
        flags = {"ticker": ticker, "date": date, "analysts": analysts, "save": save, "show": show, "html": html}
        run_analysis(checkpoint=checkpoint, portfolio=portfolio_context, flags=flags)
    except _NO_CONSOLE_ERRORS:
        # A terminal with no console buffer cannot host the interactive prompts.
        # Emit one actionable line on stderr instead of a prompt_toolkit
        # traceback; plain text, since rich may not render here either (#1138).
        typer.echo(
            "Error: no Windows console available. The interactive CLI needs a real "
            "console buffer — run it from Windows Terminal, PowerShell, or cmd.exe "
            "rather than a piped or embedded terminal.",
            err=True,
        )
        raise typer.Exit(code=1) from None


@app.command()
def backtest(
    tickers: str = typer.Argument(..., help="Comma-separated tickers, e.g. NVDA,AAPL"),
    start: str = typer.Option(..., "--start", help="First analysis date, YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="Last analysis date, YYYY-MM-DD"),
    every: int = typer.Option(7, "--every", help="Days between analysis dates"),
    analysts: str = typer.Option(
        None, "--analysts", help="Comma-separated analysts to run: market, sentiment, news, fundamentals, options, calendar; omit for all the asset type allows"
    ),
    asset_type: str = typer.Option("stock", "--asset-type", help="stock or crypto"),
    portfolio: str = typer.Option(
        None, "--portfolio", help="JSON file with holdings and cash, held constant across the grid"
    ),
    run_id: str = typer.Option(
        None, "--run-id", help="Continue an earlier sweep: its cells are skipped and its log reused"
    ),
):
    """Score past decisions over a grid of tickers and dates."""

    try:
        dates = iter_grid(start, end, every)
        book = load_portfolio(portfolio) if portfolio else None
        kind = AssetType(asset_type.strip().lower())
        # The analysts are named and checked as for an analysis; without a
        # choice, every analyst the asset type allows runs.
        chosen = (parse_analysts(analysts, kind) if analysts
                  else filter_analysts_for_asset_type(list(AnalystType), kind))
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    names = [t.strip() for t in tickers.split(",") if t.strip()]
    if not names:
        console.print("[red]No ticker to analyze; pass them comma-separated, e.g. NVDA,AAPL[/red]")
        raise typer.Exit(code=1)

    def show_progress(done, total, ticker, date):
        console.print(f"[dim][{done}/{total}] {ticker} {date}[/dim]")

    kwargs = {"asset_type": kind.value, "portfolio": book, "run_id": run_id, "progress": show_progress,
              "selected_analysts": [a.value for a in chosen]}

    try:
        result = run_backtest(names, dates, DEFAULT_CONFIG, **kwargs)
    except Exception as exc:  # a missing key or an unknown analyst is a setup error
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(summarize(result).render())
    console.print(f"\nRan {result.cells_run} cells, skipped {result.skipped}. Log: {result.log_path}")
    console.print(f"Continue or settle this sweep: --run-id {result.run_id}")
    for ticker, date, reason in result.failures:
        console.print(f"[yellow]failed:[/yellow] {ticker} {date}: {reason}")
    for ticker, reason in result.settlement_failures:
        console.print(f"[yellow]unsettled:[/yellow] {ticker}: {reason}")


@app.command()
def screen(
    etf: str = typer.Argument("SPY", help="ETF whose holdings (from Unusual Whales) make the universe"),
    date: str = typer.Option(None, "--date", help="Screen date, YYYY-MM-DD; omit for today"),
    tickers: str = typer.Option(None, "--tickers", help="Comma-separated tickers to screen instead of the ETF"),
    equity: float = typer.Option(None, "--equity", help="Account size, to size each idea to the per-trade risk"),
    screen_only: bool = typer.Option(False, "--screen-only", help="Stop after the news screen; no strategist"),
    batch: bool = typer.Option(
        False, "--batch", help="Screen through Anthropic's Batches API: half price, usually minutes, up to 24 hours"
    ),
    min_conviction: int = typer.Option(None, "--min-conviction", help="1-5; default from config (4)"),
    max_names: int = typer.Option(None, "--max-names", help="Cap on the strategist's short list"),
    direction: str = typer.Option(
        None, "--direction", help="both (default; up to half the ideas each way), up (calls) or down (puts)"
    ),
):
    """Screen an ETF's holdings for conviction news and suggest an options trade for each. Places no orders."""
    from datetime import date as _date

    from tradingagents.conviction import render_report, run_conviction_screen, save_report

    config = dict(DEFAULT_CONFIG)
    if min_conviction is not None:
        config["screen_min_conviction"] = min_conviction
    if max_names is not None:
        config["screen_max_names"] = max_names
    if batch:
        config["screen_batch"] = True
    if direction is not None:
        if direction.lower() not in ("both", "up", "down"):
            console.print("[red]--direction must be both, up or down[/red]")
            raise typer.Exit(code=1)
        config["screen_direction"] = direction.lower()
    names = [t.strip().upper() for t in tickers.split(",") if t.strip()] if tickers else None
    when = date or _date.today().isoformat()
    try:
        result = run_conviction_screen(config, when, etf=etf.upper(), tickers=names, equity=equity,
                                       screen_only=screen_only, progress=lambda m: console.print(f"[dim]{m}[/dim]"))
    except Exception as exc:  # a missing key or an unreachable vendor is a setup error
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    text = render_report(result, equity, config["risk_per_trade"])
    path = save_report(result, text, config["results_dir"])
    console.print(text)
    console.print(f"Saved: {path}")


@app.command()
def levels(
    ticker: str = typer.Argument("SPY", help="Ticker to read levels for"),
):
    """Today's levels: prior session, pivots, overnight range, expected moves and option walls. Places no orders."""
    from datetime import date as _date

    from tradingagents.conviction.levels import get_levels

    try:
        console.print(get_levels(ticker.upper(), _date.today().isoformat()))
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None


@app.command()
def tilt(
    momentum: str = typer.Option(..., "--momentum", help="Comma-separated momentum names, as the rebalance picked them"),
    pullback: str = typer.Option("", "--pullback", help="Comma-separated pullback names"),
    date: str = typer.Option(None, "--date", help="Screen date, YYYY-MM-DD; omit for today"),
    until: str = typer.Option(None, "--until", help="Earnings window end; default the next month-end rebalance"),
    batch: bool = typer.Option(False, "--batch", help="Screen through Anthropic's Batches API: half price, slower"),
    low: float = typer.Option(None, "--min", help="Lowest weight vs equal weight (default 0.5)"),
    high: float = typer.Option(None, "--max", help="Highest weight vs equal weight (default 1.5)"),
):
    """Tilt an equal-weight basket's weights from the news screen and earnings dates. Places no orders.

    Writes a report and a JSON of multipliers to <results_dir>/tilts/ (latest.json is the newest),
    which the REL-HYBRID rebalance cell reads.
    """
    from datetime import date as _date

    from tradingagents.conviction.tilt import render_tilt, run_basket_tilt, save_tilt

    config = dict(DEFAULT_CONFIG)
    if batch:
        config["screen_batch"] = True
    if low is not None:
        config["tilt_min_multiplier"] = low
    if high is not None:
        config["tilt_max_multiplier"] = high

    def _names(text):
        return [t.strip().upper() for t in (text or "").split(",") if t.strip()]

    try:
        result = run_basket_tilt(config, date or _date.today().isoformat(), _names(momentum), _names(pullback),
                                 until=until, progress=lambda m: console.print(f"[dim]{m}[/dim]"))
    except Exception as exc:  # a missing key or an unreachable vendor is a setup error
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    text = render_tilt(result)
    md, js = save_tilt(result, text, config["results_dir"])
    console.print(text)
    console.print(f"Saved: {md}\nWeights: {js}")


if __name__ == "__main__":
    app()

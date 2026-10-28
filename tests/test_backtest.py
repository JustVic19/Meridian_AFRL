"""Backtest fixtures with hand-calculated results. No model output is accepted as a metric."""

import json
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

from research.backtest import (
    BacktestRequest,
    BacktestResult,
    Bar,
    CorporateAction,
    Signal,
    run_backtest,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "backtest"


def load(name: str) -> BacktestRequest:
    payload = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    actions = tuple(
        CorporateAction(
            symbol=item["symbol"],
            session=date.fromisoformat(item["session"]),
            split_ratio=_optional(item.get("split_ratio")),
            cash_per_share=_optional(item.get("cash_per_share")),
        )
        for item in payload.get("actions", [])
    )
    factors = tuple(
        (
            item["name"],
            tuple(
                (date.fromisoformat(point["session"]), Decimal(point["value"]))
                for point in item["observations"]
            ),
        )
        for item in payload.get("factors", [])
    )
    return BacktestRequest(
        initial_cash=Decimal(payload["initial_cash"]),
        bars=tuple(
            Bar(item["symbol"], date.fromisoformat(item["session"]), Decimal(item["close"]))
            for item in payload["bars"]
        ),
        signals=tuple(
            Signal(item["symbol"], date.fromisoformat(item["session"]))
            for item in payload["signals"]
        ),
        benchmark_symbol=payload["benchmark_symbol"],
        entry_timing=payload["entry_timing"],
        holding_period_sessions=payload["holding_period_sessions"],
        sizing_fraction=Decimal(payload["sizing_fraction"]),
        transaction_cost_bps=Decimal(payload["transaction_cost_bps"]),
        slippage_bps=Decimal(payload["slippage_bps"]),
        periods_per_year=payload["periods_per_year"],
        actions=actions,
        sectors=tuple(tuple(item) for item in payload.get("sectors", [])),
        factors=factors,
    )


def run(name: str) -> BacktestResult:
    return run_backtest(load(name))


def _optional(value: str | None) -> Decimal | None:
    if value is None:
        return None
    return Decimal(value)


def test_entries_and_exits_use_the_scheduled_close() -> None:
    # Signal on Jan 2 at 80. NEXT_CLOSE buys the Jan 3 close of 100: 800 / 100 = 8 shares.
    # The hold is one session, so the exit is the Jan 6 close of 100. Cash is unchanged.
    delayed = run("entries")
    assert delayed.trades[0].entry_session == date(2020, 1, 3)
    assert delayed.trades[0].entry_price == Decimal(100)
    assert delayed.trades[0].shares == Decimal(8)
    assert delayed.trades[0].exit_session == date(2020, 1, 6)
    assert delayed.trades[0].exit_price == Decimal(100)
    assert delayed.trades[0].holding_sessions == 1
    assert delayed.portfolio.cash == Decimal(800)
    assert delayed.portfolio.positions == ()
    # CLOSE buys the signal close of 80: 800 / 80 = 10 shares, and exits the next close at 100.
    immediate = run_backtest(replace(load("entries"), entry_timing="CLOSE"))
    assert immediate.trades[0].entry_session == date(2020, 1, 2)
    assert immediate.trades[0].entry_price == Decimal(80)
    assert immediate.trades[0].shares == Decimal(10)
    assert immediate.trades[0].exit_price == Decimal(100)
    assert immediate.portfolio.cash == Decimal(1000)


def test_transaction_costs_reduce_both_sides() -> None:
    # Buy 10100 / (100 * 1.01) = 100 shares. Cost on the buy is 100.
    # Sell notional is 11000 and the 100 bps cost is 110.
    # Price gain 1000 minus costs 210 leaves pnl 790 and cash 10890.
    result = run("costs")
    assert result.trades[0].shares == Decimal(100)
    assert result.trades[0].pnl == Decimal(790)
    assert result.portfolio.cash == Decimal(10890)


def test_slippage_moves_the_fill_against_the_portfolio() -> None:
    # 100 bps raises the buy from 100 to 101 and lowers the sell from 100 to 99.
    # 10100 / 101 = 100 shares. Sale proceeds are 9900.
    result = run("slippage")
    assert result.trades[0].entry_price == Decimal(101)
    assert result.trades[0].exit_price == Decimal(99)
    assert result.trades[0].shares == Decimal(100)
    assert result.portfolio.cash == Decimal(9900)
    assert result.trades[0].pnl == Decimal(-200)


def test_split_adjusts_shares_before_the_exit() -> None:
    # 10 shares at 100 become 20 shares in a 2-for-1 split. The post-split close is 50.
    # 20 * 50 = 1000, so the split itself does not create pnl.
    result = run("split")
    assert result.trades[0].shares == Decimal(20)
    assert result.trades[0].pnl == Decimal(0)
    assert result.portfolio.cash == Decimal(1000)


def test_dividend_is_cash_on_shares_held_overnight() -> None:
    # 10 shares receive 2 per share, then sell at an unchanged 100. Cash is 1020.
    result = run("dividend")
    assert result.trades[0].pnl == Decimal(20)
    assert result.portfolio.cash == Decimal(1020)
    assert result.win_rate == Decimal(1)


def test_position_size_is_the_requested_fraction_of_equity() -> None:
    # 0.25 * 10000 / 100 = 25 shares. The other 7500 stays in cash, then the sale returns it.
    result = run("sizing")
    assert result.trades[0].shares == Decimal(25)
    assert result.portfolio.cash == Decimal(10000)


def test_drawdown_and_calmar_use_the_equity_peak() -> None:
    # Marks are 100, 150, 90, 120. The peak is 150 and the trough is 90: (150 - 90) / 150 = 0.4.
    # Three periods and periods_per_year of 3 make CAGR = 120 / 100 - 1 = 0.2.
    # Calmar = 0.2 / 0.4 = 0.5.
    result = run("drawdown")
    assert result.equity == (Decimal(100), Decimal(150), Decimal(90), Decimal(120))
    assert result.maximum_drawdown == Decimal("0.4")
    assert result.cagr == Decimal("0.2")
    assert result.calmar == Decimal("0.5")
    assert result.number_of_observations == 3


def test_sharpe_is_mean_excess_over_population_deviation() -> None:
    # Returns are 0.20 and 0.00. Mean 0.10. Population variance 0.01. Deviation 0.10.
    # With one period per year, Sharpe = 0.10 / 0.10 = 1 and volatility = 0.10.
    # There is no peak-to-trough loss, so Calmar is undefined.
    result = run("sharpe")
    assert result.sharpe == Decimal(1)
    assert result.volatility == Decimal("0.1")
    assert result.calmar is None
    assert "not indicative of future results" in result.disclaimer


def test_sortino_uses_downside_deviation() -> None:
    # The only return is -0.50. Downside deviation is 0.50, so Sortino = -1.
    # A one-point sample has population deviation 0, so Sharpe is undefined.
    result = run("sortino")
    assert result.sortino == Decimal(-1)
    assert result.sharpe is None
    assert result.profit_factor == Decimal(0)
    assert result.average_holding_period == Decimal(1)


def test_turnover_win_rate_and_profit_factor() -> None:
    # Each name receives half of 2000, buying 100 shares at 10.
    # AAA sells at 12 for pnl 200. BBB sells at 9 for pnl -100.
    # Win rate 1/2. Profit factor 200/100 = 2.
    # Traded notional 4100 over average equity 2050 is turnover 2.
    result = run("turnover")
    pnls = {trade.symbol: trade.pnl for trade in result.trades}
    assert pnls == {"AAA": Decimal(200), "BBB": Decimal(-100)}
    assert result.win_rate == Decimal("0.5")
    assert result.profit_factor == Decimal(2)
    assert result.turnover == Decimal(2)
    assert result.average_holding_period == Decimal(1)
    assert result.number_of_observations == 1


def test_benchmark_beta_sector_and_factor_exposure() -> None:
    # Strategy returns 0.10 and 0.00. Benchmark returns 0.20 and 0.00.
    # Covariance 0.005 / benchmark variance 0.01 = beta 0.5. The MKT factor matches that beta.
    # The name is invested on two of the three session marks, so TECH exposure is 2/3.
    result = run("benchmark")
    assert result.market_beta == Decimal("0.5")
    assert result.factor_exposure == (("MKT", Decimal("0.5")),)
    assert result.sector_exposure == (("TECH", Decimal("0.66666667")),)
    assert run("sharpe").sector_exposure is None
    assert run("sharpe").factor_exposure is None


def test_replay_is_identical_when_bar_order_changes() -> None:
    request = load("benchmark")
    first = run_backtest(request)
    second = run_backtest(request)
    reversed_bars = run_backtest(replace(request, bars=tuple(reversed(request.bars))))
    assert first == second == reversed_bars

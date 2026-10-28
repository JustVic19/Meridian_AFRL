"""Deterministic daily backtest. The engine, not a model, marks the portfolio."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from financial.performance import (
    DISCLAIMER,
    average_holding,
    cagr,
    calmar,
    factor_slopes,
    market_beta,
    maximum_drawdown,
    population_std,
    profit_factor,
    sharpe,
    sortino,
    turnover,
    win_rate,
)
from financial.precision import q, sqrt

_BPS = Decimal(10000)


class BacktestError(Exception):
    """The fixture cannot be simulated without inventing a price or a rule."""


@dataclass(frozen=True)
class Bar:
    symbol: str
    session: date
    close: Decimal


@dataclass(frozen=True)
class CorporateAction:
    symbol: str
    session: date
    split_ratio: Decimal | None = None
    cash_per_share: Decimal | None = None


@dataclass(frozen=True)
class Signal:
    symbol: str
    session: date


@dataclass(frozen=True)
class BacktestRequest:
    initial_cash: Decimal
    bars: tuple[Bar, ...]
    signals: tuple[Signal, ...]
    benchmark_symbol: str
    entry_timing: str
    holding_period_sessions: int
    sizing_fraction: Decimal
    transaction_cost_bps: Decimal
    slippage_bps: Decimal
    periods_per_year: int
    actions: tuple[CorporateAction, ...] = ()
    sectors: tuple[tuple[str, str], ...] = ()
    factors: tuple[tuple[str, tuple[tuple[date, Decimal], ...]], ...] = ()
    risk_free_rate_per_period: Decimal = Decimal(0)


@dataclass(frozen=True)
class Trade:
    symbol: str
    entry_session: date
    exit_session: date
    shares: Decimal
    entry_price: Decimal
    exit_price: Decimal
    pnl: Decimal
    holding_sessions: int


@dataclass(frozen=True)
class PortfolioState:
    cash: Decimal
    positions: tuple[tuple[str, Decimal], ...]


@dataclass(frozen=True)
class BacktestResult:
    equity: tuple[Decimal, ...]
    trades: tuple[Trade, ...]
    portfolio: PortfolioState
    cagr: Decimal | None
    volatility: Decimal | None
    sharpe: Decimal | None
    sortino: Decimal | None
    maximum_drawdown: Decimal | None
    calmar: Decimal | None
    win_rate: Decimal | None
    profit_factor: Decimal | None
    turnover: Decimal | None
    number_of_observations: int
    average_holding_period: Decimal | None
    market_beta: Decimal | None
    sector_exposure: tuple[tuple[str, Decimal], ...] | None
    factor_exposure: tuple[tuple[str, Decimal], ...] | None
    disclaimer: str


@dataclass
class _Position:
    symbol: str
    shares: Decimal
    entry_session: date
    entry_index: int
    entry_price: Decimal
    cash_out: Decimal
    dividends: Decimal


def run_backtest(request: BacktestRequest) -> BacktestResult:
    """Simulate fills from the bars. Equal inputs produce an equal result."""
    _validate(request)
    prices, sessions = _panel(request)
    actions = _actions(request, sessions)
    entries = _entries(request, sessions, prices)
    slip = request.slippage_bps / _BPS
    cost = request.transaction_cost_bps / _BPS
    cash = request.initial_cash
    positions: dict[str, _Position] = {}
    trades: list[Trade] = []
    equity: list[Decimal] = []
    weights: list[dict[str, Decimal]] = []
    traded = Decimal(0)
    sectors = dict(request.sectors)

    for index, session in enumerate(sessions):
        cash = _apply_actions(session, positions, actions, cash)
        cash, traded = _exit_positions(
            request, prices, positions, trades, cash, traded, slip, cost, index, session
        )
        cash, traded = _enter_positions(
            request, prices, entries, positions, cash, traded, slip, cost, index, session
        )
        if index == len(sessions) - 1 and positions:
            cash, traded = _exit_positions(
                request,
                prices,
                positions,
                trades,
                cash,
                traded,
                slip,
                cost,
                index,
                session,
                force=True,
            )
        marked = _equity(cash, positions, prices, session)
        equity.append(marked)
        weights.append(_sector_weights(positions, prices, session, sectors, marked))

    returns = _returns(equity)
    benchmark = _benchmark_returns(request, prices, sessions)
    factors = _factor_returns(request, sessions)
    growth = cagr(equity[0], equity[-1], len(returns), request.periods_per_year)
    drawdown = maximum_drawdown(equity)
    deviation = population_std(returns)
    return BacktestResult(
        equity=tuple(equity),
        trades=tuple(trades),
        portfolio=PortfolioState(
            cash,
            tuple(sorted((symbol, pos.shares) for symbol, pos in positions.items())),
        ),
        cagr=growth,
        volatility=None
        if deviation is None
        else q(deviation * _annualizer(request.periods_per_year)),
        sharpe=sharpe(returns, request.periods_per_year, request.risk_free_rate_per_period),
        sortino=sortino(returns, request.periods_per_year, request.risk_free_rate_per_period),
        maximum_drawdown=drawdown,
        calmar=calmar(growth, drawdown),
        win_rate=win_rate([trade.pnl for trade in trades]),
        profit_factor=profit_factor([trade.pnl for trade in trades]),
        turnover=turnover(traded, equity),
        number_of_observations=len(returns),
        average_holding_period=average_holding([trade.holding_sessions for trade in trades]),
        market_beta=market_beta(returns, benchmark) if benchmark is not None else None,
        sector_exposure=_average_sectors(weights) if request.sectors else None,
        factor_exposure=_factor_exposure(request, returns, factors),
        disclaimer=DISCLAIMER,
    )


def _validate(request: BacktestRequest) -> None:
    if request.initial_cash <= 0:
        raise BacktestError("initial cash must be positive")
    if request.entry_timing not in {"CLOSE", "NEXT_CLOSE"}:
        raise BacktestError("entry timing is not supported")
    if request.holding_period_sessions < 1:
        raise BacktestError("holding period must cover at least one session")
    if request.sizing_fraction <= 0 or request.sizing_fraction > 1:
        raise BacktestError("position fraction must be in (0, 1]")
    if request.transaction_cost_bps < 0 or request.slippage_bps < 0:
        raise BacktestError("costs and slippage cannot be negative")
    if request.slippage_bps >= _BPS:
        raise BacktestError("slippage consumes the sell price")
    if request.periods_per_year < 1:
        raise BacktestError("periods_per_year must be positive")
    if not request.benchmark_symbol.strip():
        raise BacktestError("a benchmark symbol is required")


def _panel(request: BacktestRequest) -> tuple[dict[tuple[str, date], Decimal], tuple[date, ...]]:
    prices: dict[tuple[str, date], Decimal] = {}
    for bar in request.bars:
        if bar.close <= 0:
            raise BacktestError("prices must be positive")
        key = (bar.symbol, bar.session)
        if key in prices:
            raise BacktestError("duplicate price")
        prices[key] = bar.close
    traded = {session for symbol, session in prices if symbol != request.benchmark_symbol}
    sessions = tuple(sorted(traded))
    if not sessions:
        raise BacktestError("the dataset has no tradable sessions")
    for session in sessions:
        if (request.benchmark_symbol, session) not in prices:
            raise BacktestError("benchmark price is missing")
    return prices, sessions


def _actions(
    request: BacktestRequest,
    sessions: tuple[date, ...],
) -> dict[tuple[str, date], CorporateAction]:
    found: dict[tuple[str, date], CorporateAction] = {}
    known = {symbol for symbol, _session in _symbols(request)}
    for action in request.actions:
        if action.symbol == request.benchmark_symbol or action.symbol not in known:
            raise BacktestError("corporate action symbol is not in the dataset")
        if action.session not in sessions:
            raise BacktestError("corporate action is not on a session")
        if (action.split_ratio is None or action.split_ratio <= 0) and (
            action.cash_per_share is None or action.cash_per_share < 0
        ):
            raise BacktestError("corporate action needs a split or a dividend")
        if (action.symbol, action.session) in found:
            raise BacktestError("duplicate corporate action")
        found[(action.symbol, action.session)] = action
    return found


def _symbols(request: BacktestRequest) -> set[tuple[str, date]]:
    return {(bar.symbol, bar.session) for bar in request.bars}


def _entries(
    request: BacktestRequest,
    sessions: tuple[date, ...],
    prices: dict[tuple[str, date], Decimal],
) -> dict[date, tuple[str, ...]]:
    index = {session: position for position, session in enumerate(sessions)}
    grouped: dict[date, list[str]] = {}
    seen: set[tuple[str, date]] = set()
    for signal in request.signals:
        if signal.symbol == request.benchmark_symbol:
            raise BacktestError("the benchmark is not traded")
        if signal.session not in index or (signal.symbol, signal.session) not in prices:
            raise BacktestError("signal has no price on its session")
        fill_index = index[signal.session]
        if request.entry_timing == "NEXT_CLOSE":
            fill_index += 1
        if fill_index >= len(sessions):
            continue
        fill = sessions[fill_index]
        if (signal.symbol, fill) not in prices:
            raise BacktestError("entry price is missing")
        if (signal.symbol, fill) in seen:
            raise BacktestError("duplicate entry")
        seen.add((signal.symbol, fill))
        grouped.setdefault(fill, []).append(signal.symbol)
    return {session: tuple(sorted(symbols)) for session, symbols in grouped.items()}


def _apply_actions(
    session: date,
    positions: dict[str, _Position],
    actions: dict[tuple[str, date], CorporateAction],
    cash: Decimal,
) -> Decimal:
    for symbol in sorted(positions):
        action = actions.get((symbol, session))
        if action is None:
            continue
        position = positions[symbol]
        if action.split_ratio is not None:
            position.shares *= action.split_ratio
        if action.cash_per_share is not None:
            dividend = position.shares * action.cash_per_share
            position.dividends += dividend
            cash += dividend
    return cash


def _exit_positions(
    request: BacktestRequest,
    prices: dict[tuple[str, date], Decimal],
    positions: dict[str, _Position],
    trades: list[Trade],
    cash: Decimal,
    traded: Decimal,
    slip: Decimal,
    cost: Decimal,
    index: int,
    session: date,
    force: bool = False,
) -> tuple[Decimal, Decimal]:
    for symbol in sorted(positions):
        position = positions[symbol]
        held = index - position.entry_index
        if held < request.holding_period_sessions and not force:
            continue
        close = prices[(symbol, session)]
        fill = close * (1 - slip)
        notional = position.shares * fill
        fee = notional * cost
        cash += notional - fee
        traded += notional
        trades.append(
            Trade(
                symbol=symbol,
                entry_session=position.entry_session,
                exit_session=session,
                shares=position.shares,
                entry_price=position.entry_price,
                exit_price=fill,
                pnl=notional - fee + position.dividends - position.cash_out,
                holding_sessions=index - position.entry_index,
            )
        )
        del positions[symbol]
    return cash, traded


def _enter_positions(
    request: BacktestRequest,
    prices: dict[tuple[str, date], Decimal],
    entries: dict[date, tuple[str, ...]],
    positions: dict[str, _Position],
    cash: Decimal,
    traded: Decimal,
    slip: Decimal,
    cost: Decimal,
    index: int,
    session: date,
) -> tuple[Decimal, Decimal]:
    symbols = [symbol for symbol in entries.get(session, ()) if symbol not in positions]
    if not symbols:
        return cash, traded
    if request.sizing_fraction * len(symbols) > 1:
        raise BacktestError("position fractions exceed the portfolio")
    equity = _equity(cash, positions, prices, session)
    for symbol in symbols:
        close = prices[(symbol, session)]
        fill = close * (1 + slip)
        shares = equity * request.sizing_fraction / (fill * (1 + cost))
        cash_out = shares * fill * (1 + cost)
        cash -= cash_out
        traded += shares * fill
        positions[symbol] = _Position(
            symbol=symbol,
            shares=shares,
            entry_session=session,
            entry_index=index,
            entry_price=fill,
            cash_out=cash_out,
            dividends=Decimal(0),
        )
    return cash, traded


def _equity(
    cash: Decimal,
    positions: dict[str, _Position],
    prices: dict[tuple[str, date], Decimal],
    session: date,
) -> Decimal:
    market = sum(
        (position.shares * prices[(position.symbol, session)] for position in positions.values()),
        Decimal(0),
    )
    return cash + market


def _sector_weights(
    positions: dict[str, _Position],
    prices: dict[tuple[str, date], Decimal],
    session: date,
    sectors: dict[str, str],
    equity: Decimal,
) -> dict[str, Decimal]:
    weights: dict[str, Decimal] = {}
    if equity == 0:
        return weights
    for position in positions.values():
        sector = sectors.get(position.symbol)
        if sector is None:
            continue
        value = position.shares * prices[(position.symbol, session)]
        weights[sector] = weights.get(sector, Decimal(0)) + value / equity
    return weights


def _returns(equity: list[Decimal]) -> list[Decimal]:
    returns: list[Decimal] = []
    for previous, current in zip(equity, equity[1:], strict=False):
        if previous == 0:
            raise BacktestError("equity reached zero")
        returns.append(current / previous - 1)
    return returns


def _benchmark_returns(
    request: BacktestRequest,
    prices: dict[tuple[str, date], Decimal],
    sessions: tuple[date, ...],
) -> list[Decimal] | None:
    closes = [prices[(request.benchmark_symbol, session)] for session in sessions]
    return _returns(closes)


def _factor_returns(
    request: BacktestRequest,
    sessions: tuple[date, ...],
) -> list[list[Decimal]] | None:
    if not request.factors:
        return None
    return_dates = sessions[1:]
    columns: list[list[Decimal]] = []
    for _name, observations in request.factors:
        by_session = dict(observations)
        if any(session not in by_session for session in return_dates):
            return None
        columns.append([by_session[session] for session in return_dates])
    return columns


def _average_sectors(weights: list[dict[str, Decimal]]) -> tuple[tuple[str, Decimal], ...]:
    totals: dict[str, Decimal] = {}
    for snapshot in weights:
        for sector, weight in snapshot.items():
            totals[sector] = totals.get(sector, Decimal(0)) + weight
    count = Decimal(len(weights))
    return tuple((sector, q(totals[sector] / count)) for sector in sorted(totals))


def _factor_exposure(
    request: BacktestRequest,
    returns: list[Decimal],
    factors: list[list[Decimal]] | None,
) -> tuple[tuple[str, Decimal], ...] | None:
    if factors is None:
        return None
    slopes = factor_slopes(returns, factors)
    if slopes is None:
        return None
    names = [name for name, _observations in request.factors]
    return tuple(zip(names, slopes, strict=True))


def _annualizer(periods_per_year: int) -> Decimal:
    return sqrt(Decimal(periods_per_year))

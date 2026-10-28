"""Deterministic performance math. Inputs are decimals; nothing here is estimated by a model."""

from __future__ import annotations

from decimal import Decimal, localcontext

from financial.precision import q, solve, sqrt

DISCLAIMER = (
    "Research output only. This simulated performance is not financial, investment, "
    "or trading advice. Past performance of any simulated strategy is not indicative "
    "of future results."
)


def cagr(start: Decimal, end: Decimal, observations: int, periods_per_year: int) -> Decimal | None:
    """Compound growth that maps the sample length onto periods_per_year."""
    if observations < 1 or start <= 0 or end <= 0:
        return None
    growth = end / start
    if periods_per_year == observations:
        return q(growth - 1)
    annual = _power(growth, periods_per_year, observations)
    return q(annual - 1)


def population_std(values: list[Decimal]) -> Decimal | None:
    """Square root of the mean squared deviation. A single point has deviation zero."""
    if not values:
        return None
    mean = sum(values, Decimal(0)) / Decimal(len(values))
    variance = sum((value - mean) ** 2 for value in values) / Decimal(len(values))
    return sqrt(variance)


def sharpe(values: list[Decimal], periods_per_year: int, risk_free: Decimal) -> Decimal | None:
    """Annualized mean excess return divided by the population standard deviation."""
    if not values:
        return None
    deviation = population_std(values)
    if deviation is None or deviation == 0:
        return None
    mean = sum(values, Decimal(0)) / Decimal(len(values))
    return q((mean - risk_free) / deviation * sqrt(Decimal(periods_per_year)))


def sortino(values: list[Decimal], periods_per_year: int, risk_free: Decimal) -> Decimal | None:
    """Annualized mean excess return divided by the downside deviation around zero excess."""
    if not values:
        return None
    excess = [value - risk_free for value in values]
    downside = sum(min(value, Decimal(0)) ** 2 for value in excess) / Decimal(len(excess))
    deviation = sqrt(downside)
    if deviation == 0:
        return None
    mean = sum(excess, Decimal(0)) / Decimal(len(excess))
    return q(mean / deviation * sqrt(Decimal(periods_per_year)))


def maximum_drawdown(equity: list[Decimal]) -> Decimal | None:
    """Largest peak-to-trough loss as a positive fraction of the peak."""
    if not equity:
        return None
    peak = equity[0]
    worst = Decimal(0)
    for value in equity:
        if value > peak:
            peak = value
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return q(worst)


def calmar(growth: Decimal | None, drawdown: Decimal | None) -> Decimal | None:
    if growth is None or drawdown is None or drawdown == 0:
        return None
    return q(growth / drawdown)


def win_rate(pnls: list[Decimal]) -> Decimal | None:
    if not pnls:
        return None
    wins = sum(1 for pnl in pnls if pnl > 0)
    return q(Decimal(wins) / Decimal(len(pnls)))


def profit_factor(pnls: list[Decimal]) -> Decimal | None:
    """Gross winning pnl divided by gross losing pnl. Undefined when there is no loss."""
    losses = sum((-pnl for pnl in pnls if pnl < 0), Decimal(0))
    if losses == 0:
        return None
    gains = sum((pnl for pnl in pnls if pnl > 0), Decimal(0))
    return q(gains / losses)


def turnover(traded_notional: Decimal, equity: list[Decimal]) -> Decimal | None:
    """Sum of buy and sell notionals divided by the average marked equity."""
    if not equity:
        return None
    average = sum(equity, Decimal(0)) / Decimal(len(equity))
    if average == 0:
        return None
    return q(traded_notional / average)


def average_holding(holding_sessions: list[int]) -> Decimal | None:
    if not holding_sessions:
        return None
    total = sum(holding_sessions, 0)
    return q(Decimal(total) / Decimal(len(holding_sessions)))


def market_beta(strategy: list[Decimal], benchmark: list[Decimal]) -> Decimal | None:
    """Population covariance divided by the benchmark variance. No intercept adjustment."""
    if len(strategy) != len(benchmark) or not strategy:
        return None
    mean_strategy = sum(strategy, Decimal(0)) / Decimal(len(strategy))
    mean_benchmark = sum(benchmark, Decimal(0)) / Decimal(len(benchmark))
    variance = sum((value - mean_benchmark) ** 2 for value in benchmark) / Decimal(len(benchmark))
    if variance == 0:
        return None
    covariance = sum(
        (left - mean_strategy) * (right - mean_benchmark)
        for left, right in zip(strategy, benchmark, strict=True)
    ) / Decimal(len(strategy))
    return q(covariance / variance)


def factor_slopes(target: list[Decimal], factors: list[list[Decimal]]) -> list[Decimal] | None:
    """Ordinary-least-squares slopes with an intercept. The intercept is not returned."""
    if not target or not factors or any(len(column) != len(target) for column in factors):
        return None
    width = len(factors) + 1
    design = [[Decimal(1)] * len(target), *factors]
    system: list[list[Decimal]] = []
    for row_index in range(width):
        coefficients = [
            sum(
                (
                    design[row_index][observation] * design[column][observation]
                    for observation in range(len(target))
                ),
                Decimal(0),
            )
            for column in range(width)
        ]
        measured = sum(
            (
                design[row_index][observation] * target[observation]
                for observation in range(len(target))
            ),
            Decimal(0),
        )
        system.append([*coefficients, measured])
    solved = solve(system)
    if solved is None:
        return None
    return [q(slope) for slope in solved[1:]]


def _power(base: Decimal, numerator: int, denominator: int) -> Decimal:
    with localcontext() as context:
        context.prec = 50
        return _nth_root(base**numerator, denominator)


def _nth_root(value: Decimal, degree: int) -> Decimal:
    if degree == 1:
        return value
    guess = value if value >= 1 else Decimal(1)
    quantum = Decimal("1e-24")
    for _ in range(80):
        previous = guess
        guess = ((degree - 1) * guess + value / (guess ** (degree - 1))) / Decimal(degree)
        if abs(guess - previous) <= quantum:
            break
    return guess

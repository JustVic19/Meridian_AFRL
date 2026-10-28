"""Fixed-precision decimal helpers used by the performance calculations."""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal, localcontext

QUANTUM = Decimal("0.00000001")


def q(value: Decimal) -> Decimal:
    return value.quantize(QUANTUM, rounding=ROUND_HALF_EVEN)


def sqrt(value: Decimal) -> Decimal:
    """Newton square root at a fixed precision. Zero stays zero."""
    if value < 0:
        raise ValueError("square root is not defined for a negative value")
    if value == 0:
        return Decimal(0)
    with localcontext() as context:
        context.prec = 50
        guess = value
        quantum = Decimal("1e-24")
        for _ in range(80):
            previous = guess
            guess = (guess + value / guess) / 2
            if abs(guess - previous) <= quantum:
                break
        return guess


def solve(system: list[list[Decimal]]) -> list[Decimal] | None:
    """Solve augmented rows. A zero pivot has no unique solution."""
    size = len(system)
    rows = [list(row) for row in system]
    with localcontext() as context:
        context.prec = 50
        for column in range(size):
            pivot = max(range(column, size), key=lambda index: abs(rows[index][column]))
            if rows[pivot][column] == 0:
                return None
            rows[column], rows[pivot] = rows[pivot], rows[column]
            scale = rows[column][column]
            for index in range(column, size + 1):
                rows[column][index] /= scale
            for other in range(size):
                if other == column:
                    continue
                factor = rows[other][column]
                for index in range(column, size + 1):
                    rows[other][index] -= factor * rows[column][index]
        return [row[size] for row in rows]

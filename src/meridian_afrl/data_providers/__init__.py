"""
Data providers layer — market data source interfaces and adapters.

Each external data source (e.g. Alpha Vantage, Polygon, Yahoo Finance) must be
implemented behind the abstract interface defined in this layer so that:
  1. Tests can use in-memory or file-based mock adapters.
  2. Data sources can be swapped without touching domain logic.

No live network calls may be made in the core test suite.
"""

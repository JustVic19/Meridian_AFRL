"""
Domain layer — pure business logic.

Rules:
- No I/O of any kind (no file reads, no HTTP, no DB).
- No imports from infrastructure, api, ai, data_providers, or ui layers.
- All dependencies flow *inward* to this layer via interfaces defined here.
"""

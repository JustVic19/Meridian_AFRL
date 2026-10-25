"""
Infrastructure layer — database, file storage, caching, logging.

This layer is the infrastructure bridge that wires external persistence
concerns to domain interfaces.
It may import from the domain layer but must NOT import from api or ui.
"""

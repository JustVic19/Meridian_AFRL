"""SEC EDGAR ingestion. Requests stay bounded and tests use local fixtures."""

from data.sec.client import SecClient, sec_client_from_env
from data.sec.ingest import IngestRequest, ingest

__all__ = ["IngestRequest", "SecClient", "ingest", "sec_client_from_env"]

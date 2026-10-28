"""SEC EDGAR client. Live calls require SEC_USER_AGENT and stay on SEC hosts."""

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

from data.sec.limits import SEC_MIN_INTERVAL_SECONDS, RateLimiter
from data.sec.parse import MalformedFiling

ALLOWED_HOSTS = frozenset({"data.sec.gov", "www.sec.gov"})
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


class SecConfigError(Exception):
    """SEC client settings are missing."""


class SecRequestError(Exception):
    """A bounded SEC request failed."""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes
    headers: Mapping[str, str]


class Transport(Protocol):
    def get(self, url: str, headers: Mapping[str, str]) -> HttpResponse:
        """Return one response without following an unbounded redirect chain."""


class SecClient:
    """Retries retryable SEC statuses and spaces requests at the fair-access rate."""

    def __init__(
        self,
        user_agent: str,
        transport: Transport,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        min_interval: float = SEC_MIN_INTERVAL_SECONDS,
        initial_backoff: float = 0.5,
        max_attempts: int = 3,
        max_backoff: float = 8.0,
    ) -> None:
        cleaned = user_agent.strip()
        if not cleaned:
            raise SecConfigError("SEC_USER_AGENT is required")
        self._user_agent = cleaned
        self._transport = transport
        self._sleep = time.sleep if sleep is None else sleep
        self._limiter = RateLimiter(
            min_interval,
            time.monotonic if clock is None else clock,
            self._sleep,
        )
        self._initial_backoff = initial_backoff
        self._max_attempts = max_attempts
        self._max_backoff = max_backoff

    def get_json(self, url: str) -> object:
        body = self.get_bytes(url)
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MalformedFiling(url) from exc

    def get_text(self, url: str) -> str:
        try:
            return self.get_bytes(url).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MalformedFiling(url) from exc

    def get_bytes(self, url: str) -> bytes:
        _require_sec_url(url)
        headers = {"User-Agent": self._user_agent, "Accept": "application/json, text/plain"}
        backoff = self._initial_backoff
        for attempt in range(self._max_attempts):
            self._limiter.acquire()
            response = self._transport.get(url, headers)
            if len(response.body) > MAX_RESPONSE_BYTES:
                raise SecRequestError("SEC response exceeds the size cap")
            if response.status == 200:
                return response.body
            retryable = response.status in {429, 500, 502, 503, 504}
            if not retryable or attempt == self._max_attempts - 1:
                raise SecRequestError(f"SEC request failed with status {response.status}")
            self._sleep(_retry_delay(response, backoff))
            backoff = min(backoff * 2, self._max_backoff)
        raise SecRequestError("SEC request failed")


def sec_client_from_env(transport: Transport | None = None) -> SecClient:
    """Build a client from SEC_USER_AGENT. No identity string is stored in code."""
    return SecClient(
        user_agent=os.environ.get("SEC_USER_AGENT", ""),
        transport=UrllibTransport() if transport is None else transport,
    )


class UrllibTransport:
    """Live transport. Tests use a fixture transport instead of this class."""

    def __init__(self, timeout: float = 30) -> None:
        self._timeout = timeout

    def get(self, url: str, headers: Mapping[str, str]) -> HttpResponse:
        request = urllib.request.Request(url, headers=dict(headers))
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return HttpResponse(
                    status=int(response.status),
                    body=response.read(MAX_RESPONSE_BYTES + 1),
                    headers=_header_map(response.headers),
                )
        except urllib.error.HTTPError as exc:
            return HttpResponse(
                status=exc.code,
                body=exc.read(MAX_RESPONSE_BYTES + 1),
                headers=_header_map(exc.headers),
            )
        except urllib.error.URLError as exc:
            raise SecRequestError("SEC request failed") from exc


def submissions_url(cik: str) -> str:
    return f"https://data.sec.gov/submissions/CIK{cik}.json"


def company_facts_url(cik: str) -> str:
    return f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


def archive_url(cik: str, accession_number: str, primary_document: str) -> str:
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession_number.replace('-', '')}/{primary_document}"
    )


def _require_sec_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise SecRequestError("refusing a request outside SEC EDGAR")


def _retry_delay(response: HttpResponse, backoff: float) -> float:
    header = _header(response.headers, "Retry-After")
    if header is not None and header.isdigit():
        return float(header)
    return backoff


def _header(headers: Mapping[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return None


def _header_map(headers: object) -> dict[str, str]:
    if not isinstance(headers, Mapping):
        return {}
    return {str(key): str(value) for key, value in headers.items()}

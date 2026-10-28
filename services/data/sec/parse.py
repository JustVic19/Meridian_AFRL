"""Parse bounded SEC JSON. Older submission archives are not requested."""

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime

REQUIRED_COLUMNS = (
    "accessionNumber",
    "filingDate",
    "reportDate",
    "acceptanceDateTime",
    "form",
    "primaryDocument",
)


class MalformedFiling(Exception):
    """A fixture or SEC payload does not have the fields this pipeline stores."""


@dataclass(frozen=True)
class CikMap:
    _by_ticker: dict[str, tuple[str, str]]

    def cik_for(self, ticker: str) -> str:
        return self._entry(ticker)[0]

    def name_for(self, ticker: str) -> str:
        return self._entry(ticker)[1]

    def _entry(self, ticker: str) -> tuple[str, str]:
        try:
            return self._by_ticker[ticker.strip().upper()]
        except KeyError:
            raise KeyError(ticker) from None


@dataclass(frozen=True)
class FilingRef:
    cik: str
    company_name: str
    exchange: str | None
    form: str
    accession_number: str
    filing_date: date
    report_date: date
    published_at: datetime
    primary_document: str


def parse_company_tickers(payload: bytes | object) -> CikMap:
    decoded = _decode(payload, "company_tickers")
    root = _dict(decoded, "company_tickers")
    by_ticker: dict[str, tuple[str, str]] = {}
    for entry in root.values():
        item = _dict(entry, "company ticker")
        ticker = item.get("ticker")
        title = item.get("title")
        if not isinstance(ticker, str) or not isinstance(title, str) or not ticker.strip():
            raise MalformedFiling("company ticker")
        by_ticker[ticker.strip().upper()] = (pad_cik(item.get("cik_str")), title)
    return CikMap(by_ticker)


def select_filings(
    payload: object,
    *,
    cik: str,
    forms: frozenset[str],
    limit: int,
    data_cutoff: datetime,
) -> tuple[FilingRef, ...]:
    """Return at most `limit` recent filings at or before the cutoff, newest first."""
    root = _dict(payload, "submissions")
    raw_cik = root.get("cik")
    if raw_cik is not None and pad_cik(raw_cik) != cik:
        raise MalformedFiling("cik")
    name = root.get("name")
    if not isinstance(name, str) or not name.strip():
        raise MalformedFiling("name")
    filings = _dict(root.get("filings"), "filings")
    recent = _dict(filings.get("recent"), "filings.recent")
    columns = {column: _list(recent.get(column), column) for column in REQUIRED_COLUMNS}
    width = {len(column) for column in columns.values()}
    if len(width) != 1:
        raise MalformedFiling("filings.recent")
    selected: list[FilingRef] = []
    count = next(iter(width))
    for index in range(count):
        form = _text(columns["form"][index], "form")
        published_at = _instant(columns["acceptanceDateTime"][index])
        if form not in forms or published_at > data_cutoff:
            continue
        selected.append(
            FilingRef(
                cik=cik,
                company_name=name,
                exchange=_exchange(root.get("exchanges")),
                form=form,
                accession_number=_text(columns["accessionNumber"][index], "accessionNumber"),
                filing_date=date.fromisoformat(_text(columns["filingDate"][index], "filingDate")),
                report_date=date.fromisoformat(_text(columns["reportDate"][index], "reportDate")),
                published_at=published_at,
                primary_document=_text(columns["primaryDocument"][index], "primaryDocument"),
            )
        )
        if len(selected) == limit:
            break
    return tuple(selected)


def period_starts(payload: object) -> dict[tuple[str, str], date]:
    """Map (accession, period end) to the XBRL period start. Instant facts are ignored."""
    root = _dict(payload, "companyfacts")
    facts = _dict(root.get("facts"), "facts")
    found: dict[tuple[str, str], date] = {}
    for taxonomy in facts.values():
        if not isinstance(taxonomy, dict):
            continue
        for concept in taxonomy.values():
            if not isinstance(concept, dict):
                continue
            units = concept.get("units")
            if not isinstance(units, dict):
                continue
            for series in units.values():
                if not isinstance(series, list):
                    continue
                for item in series:
                    if not isinstance(item, dict):
                        continue
                    start = item.get("start")
                    end = item.get("end")
                    accession = item.get("accn")
                    if not isinstance(start, str) or not isinstance(end, str):
                        continue
                    if not isinstance(accession, str) or not accession.strip():
                        continue
                    key = (accession, end)
                    start_date = date.fromisoformat(start)
                    previous = found.get(key)
                    if previous is not None and previous != start_date:
                        raise MalformedFiling("conflicting period start")
                    found[key] = start_date
    return found


def pad_cik(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise MalformedFiling("cik")
    text = str(value).strip()
    if not text.isdigit():
        raise MalformedFiling("cik")
    return f"{int(text):010d}"


def _decode(payload: bytes | object, label: str) -> object:
    if isinstance(payload, bytes):
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MalformedFiling(label) from exc
    return payload


def _dict(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise MalformedFiling(label)
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise MalformedFiling(label)
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MalformedFiling(label)
    return value


def _instant(value: object) -> datetime:
    text = _text(value, "acceptanceDateTime")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise MalformedFiling("acceptanceDateTime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MalformedFiling("acceptanceDateTime")
    return parsed.astimezone(UTC)


def _exchange(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        return None
    first = value[0]
    if not isinstance(first, str) or not first.strip():
        return None
    return first

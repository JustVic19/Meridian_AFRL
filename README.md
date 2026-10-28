# Financial Intelligence Lab

Research skeleton for US public equities. Paper research only. No live capital, no broker credentials, and no automated financial advice.

Past performance of any simulated strategy is not indicative of future results.

## Local setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
npm install --prefix apps/web
docker compose up --build
```

`docker compose up` starts:

- Frontend: http://localhost:3000
- API: http://localhost:8000
- PostgreSQL with pgvector: localhost:5433

The database container listens on 5432 inside Compose. The host port is 5433 so it can run beside the Homebrew Postgres already bound to 5432. The API still connects to `postgres:5432` on the Compose network.

The local database role is `afil` / `afil_local_dev` / database `afil`. Those defaults live in `docker-compose.yml` so Compose starts without a copied env file. Override them with a gitignored `.env` if needed. See `.env.example`.

## Tests

Run these in a second terminal after the stack is up. The stack tests do not skip when a service is down.

```bash
source .venv/bin/activate
pytest
npm test --prefix apps/web
```

Host-side database checks use `postgresql+psycopg://afil:afil_local_dev@localhost:5433/afil`. Override with `AFIL_TEST_DATABASE_URL` only when the published port is different.

## Endpoints

- `GET /health` — API liveness. `{"status":"ok"}`.
- `GET /system/status` — API, database, and pgvector. HTTP 200 when all are available. HTTP 503 when the database or pgvector is not.

## Layout

The laboratory currently includes the research schema, point-in-time checks, SEC fixtures, document retrieval, the Researcher, HypothesisScientist, Quant, and Skeptic agents, experiment-specification validation, the experiment engine, the code sandbox, the deterministic backtest engine, and the validation engine. Paper trading, strategy selection, and the Reviewer are not implemented. Nothing here deploys live capital.

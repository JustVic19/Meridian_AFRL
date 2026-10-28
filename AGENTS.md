# AGENTS.md — Autonomous Financial Research Laboratory (AFRL) V1

> **This file is the authoritative operating contract for every agent — human or
> AI — that works on this repository. Read it in full before touching any code.**

---

## 1. Mission

Build an **autonomous financial research laboratory** for US public equities.

AFRL V1 must:

1. Generate financial hypotheses.
2. Construct reproducible experiments.
3. Execute quantitative research.
4. Detect methodological flaws.
5. Perform robustness testing.
6. Maintain research memory.
7. Produce reproducible research reports.

---

## 2. V1 Boundary (Hard Limits)

| In scope | Out of scope |
|---|---|
| US-listed common equities | Non-US markets |
| Research outputs | Live capital deployment |
| Paper trading simulation | Real brokerage execution |
| Internal research use | External investor functionality |
| Hypothesis and experiment lifecycle | Automated financial advice |
| — | Unrestricted trading access |

**No live capital. No external investors. No automated financial advice. No unrestricted trading access.**

---

## 3. Core Principles

These are non-negotiable invariants. No implementation may violate them.

| # | Principle |
|---|---|
| P1 | **Reproducibility over convenience.** Every result must be reproducible from a fixed dataset version and code version. |
| P2 | **Deterministic calculations over LLM calculations.** The LLM proposes; the deterministic engine calculates. |
| P3 | **Point-in-time data integrity is mandatory.** All historical lookups must use only data available at the simulated point in time. |
| P4 | **Never permit look-ahead bias.** Any code path that allows future data to influence past calculations is a critical defect. |
| P5 | **Never overwrite completed experiments.** Experiments are append-only and immutable once finalized. |
| P6 | **Every experiment must record its dataset version and code version.** |
| P7 | **Every AI-generated artifact must be stored** and linked to the experiment that produced it. |
| P8 | **Never allow an LLM direct access to broker credentials.** |
| P9 | **Never allow generated code unrestricted production access.** Generated code runs in a sandboxed execution environment only. |
| P10 | **Never treat backtest performance as proof of future profitability.** All performance outputs must carry this disclaimer. |

---

## 4. Architecture

### 4.1 Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js · TypeScript |
| Backend / API | Python · FastAPI |
| Database | PostgreSQL · pgvector |
| Research | Python — quant engine, experiment runner |

The repository root is the Financial Intelligence Lab tree:

| Path | Responsibility |
|---|---|
| `apps/web` | Next.js · TypeScript frontend. |
| `services/api` | FastAPI HTTP boundary. |
| `services/orchestrator` | Run coordination. Not implemented in the skeleton. |
| `services/agents/*` | Observer, Researcher, HypothesisScientist, Quant, Skeptic, Reviewer. Placeholders only until an agent task is requested. |
| `services/research/*` | Datasets, backtest, statistics, validation. No financial calculations in the skeleton. |
| `services/data/*` | SEC, market, and fundamentals providers. Interfaces only until a data task is requested. |
| `services/paper_trading/*` | Portfolio, execution, and broker simulation. No live capital and no broker credentials. |
| `packages/schemas` | Pydantic request and response models. |
| `packages/prompts` | Prompt text. Not implemented in the skeleton. |
| `packages/financial` | Deterministic financial calculations. Not implemented in the skeleton. |
| `packages/evaluation` | Evaluation harness. Not implemented in the skeleton. |
| `experiments` | Immutable experiment records. |
| `migrations` | Alembic migrations. |
| `docker` | Development images and Postgres init. |

### 4.2 Dependency Rules

- `domain` never imports from `infrastructure`, `api`, `ai`, or `data_providers`.
- `api` contains no business logic. It delegates to domain services.
- All LLM calls go through an abstract AI provider interface that is mockable in tests.
- All market-data calls go through an abstract data-provider interface that is mockable in tests.
- No API key, credential, or secret may be hard-coded anywhere. Use environment variables only.

### 4.3 Financial Research Engine Rule

```
LLM proposes  →  Deterministic engine calculates  →  Validation engine challenges
                                                            ↓
                              Database remembers  ←  Risk engine controls
```

- The LLM proposes.
- The deterministic engine calculates.
- The validation engine challenges.
- The database remembers.
- The risk engine controls.

---

## 5. Agent Roster

| Agent | Role |
|---|---|
| **Observer** | Monitors markets and news; surfaces raw signals and anomalies. |
| **Researcher** | Translates signals into structured research questions and hypotheses. |
| **HypothesisScientist** | Formalises hypotheses into testable, falsifiable experiment designs. |
| **Quant** | Implements and executes deterministic quantitative experiments. |
| **Skeptic** | Challenges results: hunts for look-ahead bias, data snooping, overfitting, and other methodological flaws; drives robustness testing. |
| **Reviewer** | Audits the full experiment lifecycle; produces the final reproducible research report. |

Every agent operates under §3 (Core Principles) and §7 (Operating Protocol).

---

## 6. Development Rules

### 6.1 Feature Implementation Order

Before implementing any feature:

1. Identify the relevant **domain model**.
2. Define its **schema** (database + Pydantic).
3. Define its **API** (endpoint contract, request/response shapes).
4. Write **tests** (unit + integration).
5. **Implement**.
6. Verify **reproducibility** (same inputs → identical outputs, always).

### 6.2 Infrastructure Discipline

> Do not introduce additional infrastructure unless it solves a **demonstrated, current** problem.

Prefer simple, explicit, testable code over premature abstractions.

### 6.3 Test Requirements

- Every implementation step must include automated tests.
- Tests must be **deterministic** — no random seeds, no flaky network calls.
- The core test suite must pass with **zero external API keys**.
- External providers must be accessed through interfaces so they can be mocked.
- **Never weaken a test to make it pass.** Fix the implementation instead.
- If a test fails, fix only the current task.

---

## 7. Agent Operating Protocol

For **every** implementation task, follow this sequence exactly:

1. **Inspect the repository.** Read the relevant files before changing anything.
2. **Identify the relevant specification requirements.** Use this file and any task brief as the source of requirements.
3. **State the acceptance criteria.**
4. **Implement only the requested task.** Do not implement future roadmap items.
5. **Write tests.**
6. **Run tests.**
7. **Run lint and type checks where applicable.**
8. **Verify reproducibility.** Same inputs must produce identical outputs.
9. **Report exact results.** Files changed, functionality implemented, tests added, exact commands, pass/fail counts, and known limitations.
10. **STOP.** Do not begin the next task. Wait for an explicit instruction before proceeding.

If a test fails, fix only the current task. Never silently proceed to the next task.

---

## 8. Absolute Prohibitions

The following actions are **forbidden** under all circumstances:

- Hard-coding credentials, API keys, or secrets.
- Allowing an LLM to directly access broker or trading credentials.
- Allowing generated code to run without sandboxing.
- Using future information in historical experiment calculations (look-ahead bias).
- Overwriting or mutating a completed experiment record.
- Fabricating financial data or research results.
- Declaring success when tests fail.
- Weakening or silently changing a test to obtain a passing result.
- Proceeding to the next task without being asked.
- Treating backtest performance as proof of future profitability.

---

## 9. Disclaimer

> All financial outputs produced by AFRL are **research outputs only**.
> Nothing in this system constitutes financial, investment, or trading advice.
> Past performance of any simulated strategy is not indicative of future results.
> No live capital is deployed by this system.

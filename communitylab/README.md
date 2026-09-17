# CommunityLab

CommunityLab is an intelligent transformation engine for digital communities.
It ingests raw conversations from Discord, Slack, webhooks and flat files,
normalizes them into a single schema, and feeds them into an AI pipeline
(LangGraph + LangChain + Groq) that produces analyses, content and decisions.

## Project layout

```
communitylab/
├── src/
│   ├── ingest/         # Loaders, normalizer and validator
│   ├── analysis/       # Sentiment, themes, relevance + consolidator
│   ├── prompts/        # System prompts for analysis and copywriting
│   ├── orchestration/  # LangGraph graphs wiring the pipeline together
│   ├── generators/     # Content generators (summaries, FAQs, posts)
│   ├── decisions/      # Decision engine: member risk, recurring topics, escalation
│   ├── oci/            # Oracle Cloud Infrastructure integrations
│   ├── interface/      # Streamlit UI
│   ├── api/            # HTTP API layer
│   ├── bot/            # Discord / Slack bot adapters
│   └── utils/          # Shared helpers (Groq LLM client with fallbacks)
├── tests/              # pytest suite
├── data/sample/        # Sample datasets (messages.json)
├── scripts/            # One-off / maintenance scripts
├── notebooks/          # Exploratory notebooks
├── n8n/                # n8n workflow exports
├── docs/               # Project documentation
├── requirements.txt
└── README.md
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Ingestion layer

The ingestion layer lives in `src/ingest/` and is split into three stages:

| Module          | Responsibility                                                                   |
| --------------- | -------------------------------------------------------------------------------- |
| `loader.py`     | Read raw records from JSON/CSV files, Discord, Slack or webhook payloads.        |
| `normalizer.py` | Use Pandas to clean the raw records and map them onto the common message schema. |
| `validator.py`  | Validate the normalized rows with Pydantic before they reach the AI pipeline.    |

Typical usage:

```python
from src.ingest import load_json, normalize, validate

raw = load_json("data/sample/messages.json")
df = normalize(raw)
result = validate(df)

for message in result.valid:
    print(message.author, message.content)
```

`validate` never raises for bad rows: it returns a `ValidationResult` with the
list of valid `CommunityMessage` objects and a list of `ValidationError`
entries describing the rejected rows.

### Common schema

Every normalized message has the following columns:

| Column          | Type                | Notes                                         |
| --------------- | ------------------- | --------------------------------------------- |
| `message_id`    | `str`               | Unique per source.                            |
| `source`        | `str`               | `discord`, `slack`, `webhook`, `json`, `csv`. |
| `channel`       | `str`               | Channel / room name.                          |
| `author`        | `str`               | Display name of the author.                   |
| `author_id`     | `str \| None`       | Platform user id when available.              |
| `content`       | `str`               | Cleaned message text.                         |
| `timestamp`     | `datetime` (UTC)    | Parsed and timezone-aware.                    |
| `thread_id`     | `str \| None`       | Parent thread / reply-to id.                  |
| `reactions`     | `int`               | Total reaction count.                         |
| `category`      | `str \| None`       | Optional tag (e.g. `technical_question`).     |
| `sentiment`     | `str \| None`       | Optional pre-labelled sentiment.              |
| `metadata`      | `dict`              | Anything source-specific we want to keep.     |

## AI analysis layer

The analysis layer (`src/analysis/`) enriches every validated message with
three independent LLM analyses and merges them into one Pydantic model.

```
CommunityMessage ──► SentimentAnalyzer ──► SentimentResult ─┐
                 ├─► ThemesAnalyzer    ──► ThemesResult    ─┼─► Consolidator ──► EnrichedMessage
                 └─► RelevanceAnalyzer ──► RelevanceResult ─┘
```

| Module            | Output model      | Key fields                                                              |
| ----------------- | ----------------- | ----------------------------------------------------------------------- |
| `sentiment.py`    | `SentimentResult` | `label` (positive/negative/neutral/mixed), `score` (-1..1), `emotions`  |
| `themes.py`       | `ThemesResult`    | `primary_category`, `topics`, `technologies`, `keywords`, `summary`     |
| `relevance.py`    | `RelevanceResult` | `score` (0..1), `tier` (noise…critical), `is_actionable`, `suggested_channels` |
| `consolidator.py` | `EnrichedMessage` | `message` + the three results above, `errors`, `analyzed_at`            |

Every analyzer asks the model for **structured output** bound to its Pydantic
schema, so downstream layers always receive validated, typed data. Failures
in one analyzer are recorded in `EnrichedMessage.errors` instead of dropping
the message (pass `strict=True` to raise instead).

```python
from src.analysis import Consolidator, to_frame
from src.ingest import load_json, normalize, validate

messages = validate(normalize(load_json("data/sample/messages.json"))).valid
enriched = Consolidator().enrich_many(messages)  # needs GROQ_API_KEY
print(to_frame(enriched)[["message_id", "sentiment_label", "relevance_tier"]])
```

### LLM client

`src/utils/llm_clients.py` centralises the Groq connection:

- `LLMSettings` — model names, temperature, timeouts; the API key comes from
  `GROQ_API_KEY` (or `LLMSettings(api_key=...)`).
- `get_llm()` — primary `llama-3.3-70b-versatile` with automatic fallback to
  `llama-3.1-8b-instant` (`Runnable.with_fallbacks`).
- `get_structured_llm(schema)` — same chain, but every model is bound to a
  Pydantic schema so parsing failures on the primary also trigger the fallback.

### Prompts

`src/prompts/system_prompts.py` holds the system prompts for the three
analyzers plus channel-specific copywriting prompts (`linkedin`, `newsletter`,
`faq`, `testimonial`), available through `get_copywriting_prompt(channel)`.

### Try it

```bash
export GROQ_API_KEY=gsk_...
python scripts/analyze_sample.py                 # analyse the bundled sample
python scripts/analyze_sample.py --json out.json  # and dump the enriched records
```

The test-suite runs fully offline: LLM calls are replaced by fakes.

## Decision engine

`src/decisions/` turns a batch of `EnrichedMessage` into actions. It is pure
Python (no LLM calls) so it is deterministic and cheap to run on every batch.

| Rule | Module                | What it does                                                                                  |
| ---- | --------------------- | --------------------------------------------------------------------------------------------- |
| R1   | `member_risk.py`      | Flags members with ≥ 1 of 3 signals: negative sentiment, frustration keywords/emotions, inactivity. |
| R2   | `recurring_topics.py` | Groups doubts by normalised topic; ≥ 3 occurrences → `faq`, `mentorship` or both.               |
| R3   | `escalation.py`       | Routes every message to `INFO` / `MEDIO` / `ALTO` / `CRÍTICO` from relevance + risk.          |
|      | `notifications.py`    | Builds level-aware alerts (digest / channel / @mention / DM) with SLA deadlines.               |
|      | `engine.py`           | `DecisionEngine.run()` coordinates R1–R3 and writes the executive summary.                   |
|      | `config.py`           | Every threshold in one place (`DecisionConfig`), documented with its rationale.               |

Key defaults (all overridable through `DecisionConfig`):

- **R1** — negative if mean sentiment ≤ −0.3 or ≥ 2 negative messages; frustration if any
  keyword (EN/ES) or `frustration`/`anger`/… emotion; inactive after 14 days. One signal =
  `watch`, two = `at_risk`, three = `high`.
- **R2** — 3 occurrences is the minimum; ≥ 2 distinct authors → FAQ, a single author →
  mentorship, ≥ 5 occurrences from several people → both.
- **R3** — relevance boundaries 0.4 / 0.6 / 0.85 mirror the analyzer tiers; sentiment
  ≤ −0.3 is negative, ≤ −0.7 very negative. Two independent `ALTO` triggers escalate to
  `CRÍTICO`. SLAs: `CRÍTICO` 2h, `ALTO` 24h, `MEDIO` 72h, `INFO` none.

```python
from src.decisions import DecisionEngine

report = DecisionEngine().run(enriched)          # enriched: list[EnrichedMessage]
print(report.executive_summary)
for action in report.urgent_actions:
    print(action.level, action.owner, action.title)
for note in report.notifications:                 # ready for Slack/Discord/n8n
    print(note.target, note.title)
```

```bash
python scripts/analyze_sample.py --json /tmp/enriched.json   # Groq step (needs GROQ_API_KEY)
python scripts/decide_sample.py /tmp/enriched.json --notifications
```

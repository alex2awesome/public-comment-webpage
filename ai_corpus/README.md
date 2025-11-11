# AI Corpus Pipeline Overview

This package orchestrates connectors and shared pipeline stages to harvest AI policy comments from disparate sources. The flow is connector-agnostic: each connector implements discovery and download logic, while the pipeline layers coordinate cross-source tasks like extraction and normalization.

## Execution Flow

```
(optional) discover
      │ enumerates connector collections / dockets
      ▼
┌─────────────┐
│    crawl    │───────────────► data/comments/<connector>/<collection>.meta.jsonl
└──────┬──────┘
       │ document queue (call + response rows)
       ▼
┌────────────────┐
│ download-call  │────────────► data/comments/ai_pipeline.sqlite (call rows + payload)
└──────┬─────────┘
       │ call identifiers
       ▼
┌────────────────────┐
│ download-responses │────────► ai_pipeline.sqlite payloads (e.g., CPPA `letters`)
└──────┬─────────────┘
       │ docs without usable text
       ▼
┌────────────┐
│  extract*  │───────────────► data/comments/blobs/<sha>.txt (fallback text)
└──────┬─────┘
       │ normalized inputs
       ▼
┌──────────┐
│  export  │───────────────► data/app_data/ai_corpus.db (documents table)
└──────────┘

`discover` and `extract*` run only when needed (known IDs skip discovery; connectors that provide text skip extraction).
```

`discover` is optional; `extract*` runs only when a download lacks usable text.

- **discover**: Enumerates collections/dockets per connector. Optional when you already know the collection IDs.
- **crawl**: Produces a JSONL file describing every document to fetch (Responses and, with `--target all`, the originating call).
- **download-call**: Fetches the RFI/RFC notice and records it in `data/comments/ai_pipeline.sqlite` (payload keyed by `doc_id`).
- **download-responses**: Fetches each response document. Connector-specific logic happens here. For CPPA, this step splits bundle PDFs into individual letters via OpenAI and stores per-letter metadata in the download payload (`payload["letters"]` along with the original PDF path).
- **extract**: Generic fallback text extraction (pdfminer/HTML). Runs only when a download lacks text. CPPA skips this because the connector already provides per-letter text files.
- **export**: Normalizes documents into the main SQLite database (`data/app_data/ai_corpus.db`). Also stores text blobs in `data/comments/blobs/`. CPPA bundles are fanned out here so every letter becomes an individual `documents` row.

### Connector Specifics & Architectural Notes

- **CPPA ADMT (`cppa_admt`)** – The agency publishes massive combined PDFs. The connector’s `fetch` downloads each bundle, invokes an OpenAI-assisted page pairing routine to split the bundle into individual letters, and stores per-letter metadata in the download payload (`payload["letters"]`). No special changes were needed downstream; `export` now detects these letters and fans them out into individual normalized rows.

- **EU “Have Your Say” (`eu_have_your_say`, `eu_have_your_say_keyword`)** – There is no public API for consultation feedback, so the connectors drive Playwright sessions (or HTML scraping) in `list_documents`/`fetch` to collect HTML pages and attachments. The rest of the pipeline treats the saved HTML exactly like other raw artifacts.

- **UK GOV (`gov_uk`)** – The public API exposes consultation metadata but not individual submissions. The connector therefore discovers the consultation list via the API and scraped publication pages to retrieve any posted PDFs/HTML responses, feeding them through the standard download pipeline.

- **Regulations.gov (`regulations_gov`), NIST AI RMF (`nist_airmf`), NITRD AI RFI (`nitrd_ai_rfi`)** – These connectors leverage official APIs or static file listings, so their `fetch` implementations primarily perform authenticated API calls or deterministic file downloads.

## Typical End-to-End Run

The CLI stages form the public “API” for the harvesting pipeline. Each stage reads well-defined artifacts, writes deterministic outputs, and is safe to rerun. You can drive the whole flow through `run_ai.sh` or call individual stages for custom orchestration.

### Inputs & prerequisites

- Python environment with the `ai_corpus` package (install from the repo root with `pip install -e .`).
- Connector-specific secrets (e.g., Regulations.gov API key) exposed as environment variables or passed via flags.
- Writable paths for `data/comments/`, `data/app_data/`, and any temporary download directory.

### Fire-and-forget script

The helper script batches collections:

```bash
./run_ai.sh \
  --connector cppa_admt \
  --collection PR-02-2023 \
  --collection CPPA-GOV-2023-01
```

Internally it runs every stage with sane defaults (SQLite paths under `data/comments`, blob dir at `data/comments/blobs`, export DB at `data/app_data/ai_corpus.db`) and logs per-collection progress.

### Stage reference

| Stage | Command | Key inputs | Primary outputs |
| --- | --- | --- | --- |
| `crawl` | `python -m ai_corpus.cli.main crawl` | Connector, collection id(s), optional `--target all` | JSONL metadata under `data/comments/<connector>/<collection>.meta.jsonl` |
| `download-call` | `... download-call` | Meta file, download dir, SQLite path | Call rows inserted into `ai_pipeline.sqlite` (`downloads` table) |
| `download-responses` | `... download-responses` | Meta file, download dir, SQLite path | Response rows + connector payloads (e.g., CPPA `letters`) |
| `extract` *(optional)* | `... extract` | SQLite DB, blob dir | Fallback text files (`data/comments/blobs/<sha>.txt`) with `sha256_text` pointers |
| `export` | `... export` | Meta file, SQLite DB, blob dir, normalized DB URL | Upserts into `data/app_data/ai_corpus.db` and copies blob references |

### Running against date ranges

Both the helper script and the raw CLI forward `--start-date/--end-date` to the discovery stage so only connectors with activity in that window are enqueued.

- Batch script:

  ```bash
  ./run_ai.sh --start-date 2023-01-01 --end-date 2023-06-30
  ```

  If discovery finds nothing in the range, it falls back to the default AI docket list so you never end up with an empty run.

- Direct CLI discovery (useful for ad-hoc manifests or dry-runs):

  ```bash
  python -m ai_corpus.cli.main discover \
    --connector regulations_gov \
    --start-date 2023-11-01 \
    --end-date 2023-12-31 \
    --output discovered.json
  ```

  You can feed the resulting collection ids into subsequent `crawl`/`download-*` invocations. Passing only `--start-date` (or only `--end-date`) creates open-ended ranges.

### Manual invocation

Run each stage yourself when you need overrides (custom output paths, selective reruns, etc.):

```bash
python -m ai_corpus.cli.main crawl \
  --connector cppa_admt \
  --collection-id PR-02-2023 \
  --output data/comments/cppa_admt/PR-02-2023.meta.jsonl \
  --target all

python -m ai_corpus.cli.main download-call \
  --connector cppa_admt \
  --collection-id PR-02-2023 \
  --meta-file data/comments/cppa_admt/PR-02-2023.meta.jsonl \
  --out-dir downloads/cppa \
  --database data/comments/ai_pipeline.sqlite

python -m ai_corpus.cli.main download-responses \
  --connector cppa_admt \
  --collection-id PR-02-2023 \
  --meta-file data/comments/cppa_admt/PR-02-2023.meta.jsonl \
  --out-dir downloads/cppa \
  --database data/comments/ai_pipeline.sqlite

python -m ai_corpus.cli.main extract \
  --database data/comments/ai_pipeline.sqlite \
  --blob-dir data/comments/blobs

python -m ai_corpus.cli.main export \
  --meta-file data/comments/cppa_admt/PR-02-2023.meta.jsonl \
  --database data/comments/ai_pipeline.sqlite \
  --database-url sqlite:///data/app_data/ai_corpus.db \
  --blob-dir data/comments/blobs
```

### Failure recovery & incremental sync

- Every stage checks for prior work. `crawl` and `download-*` skip rows unless you pass `--refresh`.
- `download-responses` is idempotent; it reuses payload JSON (and CPPA letter splits) already stored in SQLite.
- `extract` records `sha256_text`, so it will not rebuild blobs once the text hash exists.
- `export` upserts into SQLite with deterministic `uid`s, allowing multiple runs without duplicates.

## Rules & Notice API

Beyond downloading comments, you can now harvest every published version of a rule/RFI across all connectors with a single CLI, complete with caching so you never re-pull a docket unnecessarily.

### Scrape Federal Register notices directly

The legacy script still works for quick FR pulls:

```bash
python scripts/scraping/get_regulations.py \
  --start 2024-01-01 --end 2024-01-15 \
  --include-types PRORULE,RULE,NOTICE \
  --include-docket-history \
  --regs-key $REGS_API_KEY \
  -o data/federal_rulemaking_jan2024.csv
```

Flags of note:

* `--include-docket-history` – after scraping the primary notices, fetch every related FR document tied to each docket (supplementals, corrections, final rules, etc.) and annotate comment-response snippets when available.
* `--history-types/--history-limit` – restrict or cap the historical documents per docket.

### Connector-aware “rules” CLI

Use the unified CLI to pull rule histories from *any* connector (Regulations.gov, CPPA, NIST, etc.), optionally filtering by discovery criteria:

```bash
# Pull every Regulations.gov docket last modified in 2022 and export their full
# Federal Register histories. Results go to rules_2022.csv, and the cache tracks what ran.
python -m ai_corpus.cli.main rules \
  --connector regulations_gov \
  --start-date 2022-01-01 --end-date 2022-12-31 \
  --output data/rules_2022.csv \
  --cache-db data/rule_cache.sqlite
```

Important flags:

* `--start-date/--end-date/--query` – pass straight to the connector’s discovery step so you can crawl every docket that changed in a time window.
* `--collection-id connector:collection` – explicitly pull one or more dockets (e.g., `--collection-id regulations_gov:EPA-HQ-OAR-2011-0028`).
* `--cache-db` – path to the shared harvest cache (default `data/harvest_cache.sqlite`); the CLI records connector/collection/row counts/output path, and skips previously harvested dockets unless…
* `--refresh-cache` – forces a re-harvest even if the cache already has that (connector, collection, artifact type).

Every row in the output CSV follows the standardized schema from `ai_corpus.rules.schema.RULE_VERSION_COLUMNS`:

* Core metadata: FR document number, title, type (Notice, Proposed Rule, etc.), publication date, agency names, docket ids, URLs, and, when available, Regulations.gov document/object IDs.
* Comment window fields: `comment_start_date`, `comment_due_date`, `comment_status`, `comment_active`.
* RFI detection: `is_rfi_rfc`, `rfi_rfc_label`, and the FR fields where the match occurred.
* Supplementary text: `abstract`, `action`, `dates`, plus extracted `<SUPLINF>` text when the XML exists.
* History metadata: `history_parent_docket`, `history_parent_fr_doc`, `history_stage`, `history_rank`, `history_relationship` (`seed`, `related`, etc.).
* Comment references: `mentions_comment_response` and a trimmed `comment_citation_snippet` whenever the notice explicitly references comments (e.g., “Response to Comment…” sections).

The cache entries can be inspected programmatically via `Database.list_harvests(artifact_type="rules")`, which returns the connector, collection id, row count, output file, metadata (such as the last history rank), and last-run timestamp.

## Data Formats & Storage

- **Metadata JSONL (`crawl`)** – Each line is a JSON object compatible with `DocMeta` (fields such as `doc_id`, `title`, `urls`, `extra`). Files live under `data/comments/<connector>/<collection>/<collection>.meta.jsonl`.
- **Download database (`data/comments/ai_pipeline.sqlite`)** – SQLite file with a `downloads` table containing one row per fetched document (or bundle). Important columns include `doc_id`, `connector`, `collection_id`, `payload` (JSON blob with connector artifacts), `text_path`, and `sha256_text` (populated when extraction occurs).
- **Blob store (`data/comments/blobs/`)** – Content-addressed hierarchy that stores text artifacts. Connectors or the `extract` stage write `<sha256>.txt` files; `export` references these paths when inserting normalized records.
- **Normalized database (`data/app_data/ai_corpus.db`)** – SQLite database backing the application. The `documents` table (see `ai_corpus.storage.db`) stores one row per comment with deterministic `uid`, metadata fields, and pointers to text/PDF artifacts. CPPA letters appear here as `doc_id` values like `<bundle>.pdf#L003`.

## Source Catalog & References

- **API-driven connectors**
  - `regulations_gov` – U.S. Regulations.gov API & downloads (`https://api.regulations.gov`, `https://downloads.regulations.gov`).
  - `nist_airmf` – NIST AI Risk Management Framework docket (`https://www.nist.gov/itl/ai-risk-management-framework/comments-2nd-draft-ai-risk-management-framework`).
  - `nitrd_ai_rfi` – OSTP/NITRD AI Action Plan RFI repository (`https://files.nitrd.gov/90-fr-9088/`).

- **HTML / Playwright connectors**
  - `cppa_admt` – California Privacy Protection Agency rulemaking portal (`https://cppa.ca.gov/regulations/ccpa_updates.html`).
  - `eu_have_your_say`, `eu_have_your_say_keyword` – European Commission “Have Your Say” feedback portal (`https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives_en`).
  - `gov_uk` – UK Government Digital Service consultation API & publication pages (`https://www.gov.uk/api`).

## Connector Responsibilities

Each connector implements:

- `discover`: list collections (dockets) offered by the source.
- `list_documents`: enumerate documents/comments for a collection.
- `fetch`: download raw assets and attach any additional artifacts needed downstream.

Connector-specific behavior (e.g., CPPA OpenAI splitting, EU Playwright scraping) belongs in `fetch`. Downstream stages expect the download record to carry everything necessary (text path, attachments, derived segments, etc.).

## Shared Components

- `ai_corpus.pipelines.download`: orchestrates connector fetches, manages caching, and stores download metadata in SQLite.
- `ai_corpus.pipelines.extract`: provides set-and-forget PDF/HTML extraction when connectors don’t supply text.
- `ai_corpus.pipelines.normalize`: maps metadata into the shared `NormalizedDocument` shape.
- `ai_corpus.storage.db` / `ai_corpus.storage.fs`: handle SQLAlchemy persistence and content-addressed blob storage for text.

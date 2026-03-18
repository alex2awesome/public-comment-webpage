# Regulations Demo

This repository houses the bulk-download harvesters, analysis notebooks, and supporting clients/services used for the regulations corpus project. See `AGENTS.md` for the full engineering guidelines that apply to every contribution.

## Agent Workflow Notes

- Always consult `AGENTS.md` when starting a session; it captures project structure, coding standards, testing expectations, and the notebook-specific preferences listed below.
- When editing notebooks, favor inline, sequential cells. Avoid large helper functions so intermediate steps remain easy to re-run.
- Keep pandas logic inside the cell that uses it—prefer method chaining, `.assign`, and `.loc[lambda df: ...]` to keep transformations visible in place.
- Never revert or overwrite ad-hoc changes the user has already made (especially in notebooks or exploratory scripts). Build on top of existing work instead of reshaping it back to a template.

## Getting Started

1. Set up Python 3.12 with the dependencies in `requirements.txt`, plus any per-directory requirements (e.g., `backend/requirements.txt`, `app/package.json`).
2. Review the directory-specific READMEs under `ai_corpus/`, `scripts/`, and `backend/` for component-level instructions.
3. Launch workflows with the commands summarized in `AGENTS.md` (e.g., bulk harvesting via `python -m ai_corpus.cli.main pipeline --connector <name>`).

For questions about crawler behavior, PDF parsing, or notebook expectations, open an issue or annotate the relevant notebook so future agents can follow along.

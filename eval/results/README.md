# Result files

- `blind_v2-single-run.*`: the only genuinely blind run (set frozen in commit f6c73b9, run once after all audit fixes).
- `dataset-*` (dev) and `heldout-*` (held-out v1): current code; both sets were used during development.
- `heldout_v1_first_run_NOT_BLIND/`: first run of held-out v1, originally published as blind. It was not: synonyms for held-out-only words had been added before it ran (see docs/AUDIT.md, C1).
- `baseline/`: first dev run before any tuning.
- `audit/`: adversarial suite before and after the audit fixes.

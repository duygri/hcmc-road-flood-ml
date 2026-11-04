# Final review fix wave

Baseline: `79fb63f150047b782de6e7400d432af446417dea`

## Findings addressed

- Parse the semantic uv version in both the Colab bootstrap and Windows replay while retaining the complete `uv --version` output for runtime provenance.
- Require the approved processed/artifact/report outputs before publishing, verify each completed stage marker and declared output digest, and verify Track B's producer digests. The final evaluation report is checked against current input paths and hashes because the evaluation CLI replaces the earlier pipeline checkpoint.
- Accept identical legacy 0-row/0-column review partition audits during semantic replay.
- Store CLI review paths as repository-relative POSIX strings so Windows pipeline and training commands share one run fingerprint.
- Pin tracked hash-critical text to LF and test actual checkouts with `core.autocrlf=true` and `false`.
- Compare normalizer semantic fields and thresholds under the existing numeric tolerance while retaining each reference-input digest in `derived_artifact_hashes`.
- Exercise successful ready publication, missing declared outputs, wrong model digests, and CI matrix `runs-on` wiring.

## RED evidence

- Before fixes, the two end-to-end integration regressions failed: no-review replay reported `missing_row_key` for the legacy audit, and the Windows review-path run failed with `Run fingerprint changed; choose a new run_id`.
- The new publisher cases showed `DID NOT RAISE` for missing required outputs, missing declared output, and a present model with an incorrect recorded digest.
- The uv parser regression failed because neither notebook path had a semantic-version parser. The autocrlf checkout regression failed because `.gitattributes` was absent. The legacy empty-audit comparison reported a missing row key.
- The normalizer regression builds two normalizers from feature inputs jittered by `1e-6`; their reference input hashes differ while their thresholds remain within `atol=1e-5`.

## GREEN evidence

- Focused unit coverage: `uv run pytest tests/unit/test_notebook_hygiene.py tests/unit/test_git_eol_policy.py tests/unit/test_ci_workflow.py tests/unit/test_run_publisher.py tests/unit/test_run_comparison.py -q --tb=short` — **74 passed in 56.52s**.
- Covering end-to-end regressions: `uv run pytest tests/integration/test_pipeline.py::test_colab_pipeline_train_evaluate_publish_share_one_run_id tests/integration/test_pipeline.py::test_present_match_review_path_keeps_pipeline_and_train_on_same_run_id -q --tb=short` — **2 passed in 63.19s**.
- Final full suite after formatting: `uv run pytest -q` — **548 passed in 360.04s**.
- LF checkout regression after narrowing `.gitattributes` to tracked text patterns: `uv run pytest tests/unit/test_git_eol_policy.py -q` — **2 passed in 2.31s**. `git check-attr` reports `text: set`, `eol: lf` for the source module, lockfile, raw manifest, and review template.
- Lint: `uv run ruff check .` — **All checks passed**.
- Format: `uv run ruff format --check .` — **66 files already formatted**.
- Lock: `uv lock --check` — **Resolved 48 packages**.
- `git diff --check` reported no whitespace errors.

## Limitations

Validation ran on the Windows workspace, including fresh Git clones under both autocrlf settings. No real Colab session, mounted Google Drive publication, or remote Linux CI job was run in this fix wave.

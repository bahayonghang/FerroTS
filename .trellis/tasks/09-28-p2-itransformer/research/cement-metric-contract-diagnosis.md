# Cement metric contract defect and frozen tooling boundary

Date: 2026-09-29, America/Chicago. Status: defect reproduced; candidate patch passed synthetic checks; frozen production tools remain unchanged.

## Finding

The P2 cement acceptance path cannot consume the metric schema produced by the Rust evaluator. `OriginalMetrics.variables` is a `BTreeMap<String, Metric>` in `crates/ferrots-core/src/metrics.rs:67`. JSON therefore stores a mapping from variable name to metrics. The accepted P1 cement entries use the same mapping. `tools/validation/recompute_predictions.py` already reads the mapping by name.

Three consumers in `tools/validation/p2_acceptance.py` incorrectly assume a list:

| Consumer | Original operation | Reproduced result |
| --- | --- | --- |
| cement_entries, line 507 | Read v[name] while iterating mapping keys | TypeError: string indices must be integers, not str |
| cement_gate, line 526 | Index the mapping with positional index 0 | KeyError: 0 |
| cement_report, line 544 | Read metric/name fields from a mapping key | TypeError: string indices must be integers, not str |

The first defect is reached after the nine cement runs complete. The second affects three-seed aggregation. The third affects the P1 comparison. The current ETT training functions do not read these cement fields. No CUDA fault was observed in this diagnosis. CPU training is not required.

The prior review statement that no additional code defect had been established is superseded for these three consumers. The original numerical and training evidence remains evidence for its recorded scope.

## Ownership and candidate correction

The metric schema belongs to core and evaluation. The correction belongs to the P2 acceptance consumer. Preserve the producer format, variable names, units, seeds, training profiles, complete-window counts, population standard deviation and numerical thresholds.

The candidate uses exact dictionary key coverage, validates each variable count and finite nonnegative original-unit metrics, aggregates each seed by variable name in configured target order, and reads P1 baselines with dictionary items. AST comparison confirms that only cement_entries, cement_gate and cement_report change. All other functions are unchanged.

Candidate files are under ignored `runs/p2-development/cement-contract-20260929-210242-974583/`:

- `p2_acceptance.original.py`: retained source copy.
- `p2_acceptance.candidate.py`: proposed implementation; not applied to tools/.
- `cement-metric-contract.patch`: reviewable source diff.
- `identity.json`: original and candidate file hashes.
- `test_cement_contract.py`: eight synthetic consumer tests.
- `original-tests/`: original failure logs and fixtures.
- `original-summary-corrected.json`: corrected test-method counts.
- `candidate-tests/`: candidate success logs and fixtures.
- `review.json`: test summary and function-level change boundary.

Original source: 8 test methods errored, 19 error events including subtests, 0 ignored. Candidate: 8 passed, 0 failed, 0 errors, 0 ignored. The first diagnostic summary incorrectly subtracted subtest events from test methods and reported a negative passed count. The raw summary is retained; the corrected summary counts distinct errored methods.

Tests cover all nine matrix entries, dictionary-order independence, missing/extra variables, complete per-variable counts, finite nonnegative metrics, missing/duplicate matrix entries, all 108 aggregate rows with ddof=0, and all 108 P1 report rows with the n=1 label. Training, prerequisites and artifact validation are mocked at their existing boundaries. These tests establish consumer behavior only. They do not count as formal CUDA training, full quality or final acceptance.

## Frozen identity impact

The current suite freezes the complete p2_acceptance.py file at SHA256 `202d4046fc429e0db75d2d4c277378fbcec48a2fd4cb861828b59787c214e7d7`. `suite_context` rejects a changed validation-file hash. Every command receipt also binds the immutable suite.json hash. Editing the existing suite.json would invalidate those receipt bindings.

Do not edit the active frozen file, replace suite.json, rewrite old receipts or weaken hash checks. The existing ETT shell and waiting continuation remain alive. No runner was restarted or terminated. The current task remains in_progress.

Applying the candidate requires an explicit evidence transition. The current P2 runner has no cross-suite import path for Rust receipts. A versioned transition must preserve the old suite, record both tool hashes and the exact diff, revalidate every reused artifact and its original source/binary/config/data/environment/command identity, and rerun affected metric gates and final quality. Unchanged training functions alone do not authorize silently accepting a mismatched tooling fingerprint.

The decision is whether to approve that explicit transition for unaffected completed training evidence, or to create a new suite and repeat the complete 105-run matrix after the correction. The implementation and approved AC3 thresholds are unchanged by this decision.

## Execution state at discovery

At 21:05 America/Chicago, official ETT had 48 entries and Rust ETT had 13 entries. ETTh2/H96/seed2022 was active. Original shell PID 43756 and continuation PID 40828 remained alive with matching creation times. D4, cement, recomputation and final acceptance were not complete.

No production source, frozen tool, model registry, configuration or suite receipt changed during diagnosis. No commit, archive or push occurred.

## Actual P1 integration: 2026-09-29 21:20 America/Chicago

The unapplied candidate consumed all nine accepted P1 baseline entries with real artifact-hash checks enabled. All 18 variable names and per-variable counts matched each horizon. The generated diagnostic contained 108 report rows; all 324 baseline metric cells matched the actual artifacts, and the n=1 / standard-deviation-not-applicable label was present. Only the report output sink was intercepted. P2 metric inputs were synthetic. No training, receipt migration or formal gate was performed. Evidence: runs/p2-development/continued-verification-20260929-211647-8010923/cement-integration.json.

## Approved evidence transition, 2026-09-29

The user explicitly selected versioned migration with full verification of unaffected ETT receipts and reruns of affected gates and final quality. See research/tooling-transition.md. The idle downstream waiter PID40828 was deliberately stopped; original ETT/D4 PID43756 remains active. The original waiting status is historical. The new suite and corrected production validator remain pending until parent ETT48 and D4 32/32 finish. No training was interrupted or duplicated.

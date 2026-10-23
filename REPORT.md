# Reproducible KION feature-search study

## Executive status

This repository contains a **methodology-validity study**, not evidence of production lift. The current corrected KION result is `results/kion_50k_corrected.json`: a 50,000-user cohort, 285,343 retained interactions, three training seeds, and two expanding temporal folds. The run completed in 2:00:38 with peak RSS 2.53 GB. The 10k JSON remains a protocol-validation artifact; the legacy 50k JSON is historical only and must not be compared with the corrected protocol.

The current conclusion is deliberately qualified: implementation-level point-in-time invariants pass; after protocol-v4 candidate correction, auto feature search shows a small AUC difference and a larger sampled-ranking difference, but popularity remains stronger and no production or causal claim follows.

## 1. Research question and estimands

**Question.** Can auditable, template-generated temporal aggregate features match or improve a manually designed LightGBM feature set?

**Primary comparison.** `auto_gain` versus `manual`; the validation-selected winner is not the primary endpoint.

**KION estimand.** Query-weighted offline discrimination of an observed high-watch pair against sampled historical-catalog unseen items. A KION row is a pair-aggregated observed watch record with day-level `last_watch_dt`; it is not a logged impression or a fully ordered event stream. We therefore do not claim CTR, causal impact, next-event prediction, or production ranking lift.

**Target.** `watched_pct >= 50`. Missing `watched_pct` is mapped to zero by the adapter and is reported as a target limitation, not as a neutral observation.

**Ranking protocol.** For each held-out positive, up to 99 negatives are sampled uniformly from the **selected-cohort historical catalog**, excluding items previously observed for that user and items first seen after the query timestamp. This is a diagnostic sampled-discrimination protocol, not an exposure evaluation.

## 2. Validity contract

- Equal timestamps are atomic: features at time `t` use only rows with timestamp `< t`.
- Test rows are excluded from feature sources and causal popularity references.
- Candidate rows are scored from the same pre-test snapshot for every method.
- Feature selection and model configuration use train/select; validation is used for early stopping; test is reported once.
- Ranking point metrics are macro means over queries. The cluster bootstrap resamples users while preserving their query counts, so its estimand matches the point metric.
- User sampling retains every row for selected users, but does not establish user/item cold-start performance.

### Known threats

KION has no impression log, no exposure propensity, and only day-level timestamps. `watched_pct` may represent final pair-level completion rather than feedback available at a decision time. The candidate catalog is cohort-restricted and synthetic. Popularity dominates the checked-in sampled ranking result, so candidate construction is a major determinant of the score. These limitations prevent causal or online claims.

## 3. Methods and baselines

- `manual`: hand-designed user/item/pair/category temporal aggregates + LightGBM.
- `auto_gain`: auditable template enumeration followed by training-only LightGBM gain top-K selection.
- `auto_forward`: validation forward selection over the same template space.
- `decay`: manual features augmented with exponential-decay aggregates.
- `linear`: regularized logistic regression over manual features.
- References: cumulative popularity, decayed popularity, category popularity, and prior user-item positive-watch rate.

The search is not unconstrained AI-generated SQL. The restricted, inspectable search space is intentional: it makes every feature traceable and leakage-auditable.

## 4. Hypotheses and status

These are declared follow-up hypotheses, not preregistered hypotheses: they were formulated after development exploration.

- **H1:** `auto_gain` improves test AUC over `manual` by at least 0.01. **Status: not supported.** Across six fold/seed observations, the mean difference is +0.00137, below the material threshold; the descriptive SD is 0.01868.
- **H2:** `auto_gain` improves sampled NDCG@10 over `manual` by at least 0.02. **Status: inconclusive pending corrected candidate construction.** The previous estimate (auto-gain 0.3164 versus manual 0.1148) was computed before excluding positive queries whose item was not historically available before the query. It must not be treated as evidence for the corrected historical-catalog estimand. Popularity baselines were also stronger (0.4463 cumulative, 0.4709 decayed).
- **H3:** decay features are more robust under temporal drift. **Status: not supported.** Decay AUC (0.6496) and NDCG@10 (0.0974) are below manual (0.6527 and 0.1148) on the six fold/seed observations.
- **H4:** future rows do not affect prefix features. **Status: supported for the implemented invariant**, based on atomic timestamp/snapshot tests. This is not exposure causality.

No method is recommended for deployment.

## 5. Checked-in evidence

| Artifact | Status | Scope |
|---|---|---|
| `results/kion_50k_corrected.json` | current corrected evidence | 50k selected users, three seeds, two folds |
| `results/kion_10k_corrected.json` | protocol validation | 10k selected users, one seed, two folds |
| `results/kion_50k_legacy_pre_correction.json` | historical only | predates catalog/semantic/uncertainty corrections |

The protocol-v4 50k JSON is `results/kion_50k_protocol_v4.json`: 50,000 selected users, 285,343 interactions, three seeds, two folds, 2,187 valid ranking queries, and 186 excluded positive queries. It contains package versions, command, source hashes, source-tree manifest/hash, thread settings, protocol metadata, split metadata, per-query metrics, and query-weighted user-cluster intervals. The prior `kion_50k_corrected.json` is retained as pre-correction provenance and must not be used for H2. When the checkout has no Git revision, `metadata.source_tree.sha256` and its file manifest provide immutable provenance for the source/config files included in the run.

## 6. Reproduction

```bash
cd recsys_corrected_artifact
.venv/bin/python -m pytest -q

.venv/bin/python compare_methods.py \
  --kion-interactions datasets/kion/interactions.csv \
  --kion-users datasets/kion/users.csv \
  --kion-items datasets/kion/items.csv \
  --kion-max-users 50000 \
  --evaluation rolling --folds 2 --seeds 42,43,44 \
  --forward-max-candidates 15 --forward-steps 3 --json \
  > results/kion_50k_reproduced.json
```

For a stronger follow-up, freeze the protocol first and run at least 50k/all users, three training seeds, at least 1,000 ranking queries per fold, and retain JSON, stderr, cost, input hashes, and environment metadata together.

## 7. Required follow-up experiment

Before claiming improvement, run a locked experiment with:

1. primary endpoint `auto_gain - manual` on temporal test AUC and query-weighted NDCG@10;
2. user-cluster CIs for AUC and ranking, plus seed dispersion;
3. candidate sensitivity: 10/50/99/499 negatives, uniform/popularity-weighted/recent sampling, cohort/full catalog;
4. target sensitivity at 25/50/75% watched completion;
5. user/item cold-start and popularity-decile slices;
6. prevalence, history-length, query-count, and test-window coverage;
7. retrieval recall, coverage, novelty, diversity, and latency;
8. an end-to-end fixed fixture with expected split and metric tolerances.

For logged-exposure claims, use MIND `official_train_dev` or another dataset with impressions. For causal claims, define exposure, treatment, outcome, time zero, propensity support, and an online/randomized bridge; KION alone cannot identify those quantities.

## 8. Evidence ledger

| Claim | Evidence | Applicability limit |
|---|---|---|
| KION contains users, items, day-level watch completion and metadata | KION repository and dataset paper cited in the project README | Public schema does not establish exposure or within-day ordering |
| Prefix features exclude future rows | `causal_features.py`, `history.py`, and causal/snapshot tests | Establishes an implementation invariant, not causal impact |
| Candidate metrics use historical unseen negatives | `candidate_eval.py` and candidate protocol metadata | Not a logged-impression ranking estimate |
| Popularity is a strong reference | Checked-in corrected development result | Local observation only; not universal |

## Final recommendation

Submit this as a careful, transparent methodology study after synchronizing the report with checked-in evidence. Do not present the unavailable 50k table as a repository result. The strongest scientific contribution is not a claimed win: it is the explicit demonstration that automated feature search must be evaluated against a strong baseline under a leakage-safe protocol, and that sampled ranking can be dominated by retrieval/popularity effects.

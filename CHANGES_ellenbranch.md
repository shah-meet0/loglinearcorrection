# Changes on `ellenbranch` (GEAM / GPCAM fixes, October 2026)

This note lists every change Ellen made to `loglinearcorrection` on `ellenbranch` while replicating
papers 184, 187 and 189 with the generalized estimators, and why each was needed. All changes are
in the generalized estimators (`GeneralizedDREAM` / GEAM, `JointGeneralizedDREAM`,
`GeneralizedPCAM` / GPCAM) and the code they share. The original DREAm model (`model.py`,
`results.py`) and the retrep repository are unchanged.

Files touched: `loglinearcorrection/generalized_dream.py`, `nonparametric.py`,
`neural_network_models.py`, `hypothesis.py`; new test `tests/test_clusters.py`.

## Summary

| Commit | What | Default behaviour changed? |
| --- | --- | --- |
| `da22953` | Fold residuals with fixed effects were computed from raw data (bug) | Yes: fixes estimates with FE |
| `6c0b244` | Standardise interest coordinates before fitting the density score | Yes (numerical) |
| `42dfb76` | Fold fixed effects estimated on training rows; FE levels spread across folds | Yes: fold construction with FE |
| `fa54ad7` | Score: keep the least Stein-violating state when none meets the threshold | Only when no state meets the threshold |
| `48f0a8c` | GPCAM uses the FE-spread folds; fixed-effect group-size check; absorbed regressor raises | Yes: GPCAM folds with FE; new warning |
| `48c5dc8` | Partial-out (location model) density score; held-out conditional Stein check | No: new option `score_method='partial_out'`; diagnostics added |
| `8a6d081` | Group-size check tolerates a small share of rows (`max_small_fe_share`) | Fewer warnings |
| `ddc0356` | Weighted networks; Stein score calibration; target weight diagnostics | Yes: with sample weights the networks are now weighted |
| `86e084c` | `n_folds=1`: no cross-fitting | No: new option |
| `b69ec66` | Cluster-robust (one-way and multiway) standard errors | No: new option `clusters=` |

The first three were pushed to GitHub earlier; the other seven are new in this push.

## Bug fixes

**Fixed-effect residuals in the folds (`da22953`).** The fold OLS coefficient was estimated on
FE-demeaned data, but the residuals passed to the phi nuisance were computed from raw `log(y)` and `X`,
so they kept the fixed-effect levels (mean about 7.9 instead of 0 in the 187 land-value panels).
`exp(phi * u)` then reached the millions and every GEAM target was off by orders of magnitude. New
`GeneralizedDREAM._fold_ols_residuals` (shared with GPCAM) demeans on the full sample with the fold
weights (held-out rows at 1e-10, as `model.py` does), fits the fold OLS on the training rows, and returns
the demeaned residuals plus each row's fixed-effect component, which `log_inner_mean` adds back so
`M_phi` is still the power mean of `y` itself. Without fixed effects nothing changes.

**Fold fixed effects (`42dfb76`).** Fixed effects are estimated on the training rows only, and folds are
built so each FE level appears in training (levels spread across folds). Previously many held-out rows
had a level never seen in training.

**GPCAM folds (`48f0a8c`).** GPCAM split on the treatment only, so with small FE groups held-out rows had
unseen levels (and the shared residual helper called a method GPCAM lacked). GPCAM now uses the same
FE-spread folds, with treatment arms spread within each level. Fold OLS now raises when a regressor of
interest is absorbed by the training-fold fixed effects, instead of silently returning `beta = 0`.

**Score state selection (`fa54ad7`).** With `stein_validation_threshold` set, a score fit where no state
met the threshold silently kept the last-epoch network, which was never validated; in 187 those folds
gave estimates far from every other seed. The state with the smallest validation Stein error is now
used, with a warning.

## New options and diagnostics

**Partial-out score (`48c5dc8`).** `density_params={'score_method': 'partial_out'}` residualises the
interest regressors on the fixed effects and controls on each training fold, `v = x - mu(FE, W)`
(same projection as the fold OLS, degrees-of-freedom corrected), and fits the score of `v`, so
`s(x | FE, W) = s_v(x - mu_hat)`. The default network score could not learn the fixed-effect structure
and stayed at the marginal Gaussian score, which makes GEAM equal OLS. Options: `score_inputs`
(`'residual'` or `'residual_and_controls'`), `dof_correction` (default True), `stein_tolerance`
(default 0.25). Default remains `score_method='network'`.

**Conditional Stein check (`48c5dc8`).** For either score method the held-out moment
`E[s (x - mu_hat)] + 1` is stored per fold in `model.score_diagnostics` (0 for the true conditional
score, about 1 for a score that ignores the fixed effects), with a warning above `stein_tolerance`.

**Fixed-effect group sizes (`48f0a8c`, `8a6d081`).** `fit()` reports group sizes in
`model.fe_group_report` (also `fixed_effect_group_report(...)`). It warns (or raises with
`small_fe_groups='raise'`; `'ignore'` to silence) when more than `max_small_fe_share` (default 1%) of
rows are in groups smaller than `min_fe_group_size` (default 5). Small groups are where GEAM struggles
most in our simulations.

**Weighted networks (`ddc0356`).** `NNModelNuisance.fit` and `NNModelDensity.fit` accept
`sample_weight` (weighted MSE; weighted Hyvarinen loss, standardisation and validation Stein moments).
GEAM and GPCAM pass the sample weights to every network by default (`fit(..., weighted_nuisance=True)`),
so the nuisances are fitted under the same weighted population the moments average over. With
weights this changes results: in our weighted simulations unweighted networks were biased.
`weighted_nuisance=False` restores the old behaviour.

**Stein score calibration (`ddc0356`).** `density_params={'score_calibration': 'stein'}` rescales the
score per fold and column so the held-out moment `E[s (x - mu_hat)] = -1` holds exactly; the moment
before calibration and the factor are kept in `model.score_diagnostics`. Off by default.

**Target weight diagnostics (`ddc0356`).** `model.target_diagnostics` reports, per `(phi, psi)`, the
Kish effective sample size of the target weights `w * M^psi` and the share of the top 1%, with a
warning below `min_target_effective_n` (default 100). With `psi = 1` a few rows can carry most of the
weight.

**No cross-fitting (`86e084c`).** `fit(n_folds=1)` fits and evaluates every nuisance and the score on
all rows (train = test). GPCAM still needs at least 2 folds. In our simulations cross-fitting was as
good or better, so this is for comparison only.

**Clustered standard errors (`b69ec66`).** New `hypothesis.influence_covariance(IF, p, clusters)`:
the covariance from observation-level influence functions, row-level as before or clustered by one
or several dimensions (Cameron, Gelbach and Miller inclusion-exclusion for multiway, negative
eigenvalues set to zero; `G/(G-1)` per clustering term unless `small_sample=False`).
- `fit(..., clusters=labels)` on GEAM, joint GEAM and GPCAM; `labels` has one row per observation and
  one column per clustering dimension (e.g. county and state-year).
- `results.with_clusters(labels)` re-clusters an existing fit without refitting; `None` restores the
  row-level covariance.
- `difference_test` / `ols_difference_test` and `InfluenceFunctionDiffTest(clusters=...)` use the
  same clusters, so GEAM-vs-OLS and GEAM-vs-PPML tests are clustered too.
- `tests/test_clusters.py` checks the covariance against statsmodels' cluster-robust OLS (one-way,
  weighted, two-way), the row-level and singleton cases, the small-sample factor and the difference
  test. On paper 187 the package's clustered OLS SEs match `pyfixest` two-way clustering exactly.

## Recommended settings (what the replications use)

```python
model = JointGeneralizedDREAM(y, X, weights=w, fixed_effects=fe_cols, interest=[0],
                              targets=[(0, 0), (0, 1), (1, 0), (1, 1)])
res = model.fit(n_folds=10, random_state=seed,
                density_params={'score_method': 'partial_out', 'score_calibration': 'stein'},
                clusters=X[['county', 'state_year']].to_numpy())
res.ols_difference_test('x', phi=1, psi=1)   # clustered like the SEs
```
Network sizes and training settings are in the replication runner
(`run_geam_fixes.py`, `network_params`, in the replication folder).

## What we learned that is not in the code

From the simulation study (known truth) and the replications:
- Without fixed effects GEAM recovers all four targets, provided the networks are weighted.
- With small fixed-effect groups (about 7 rows per group) GEAM recovers most of a nonlinearity that is
  shared across groups, but stays near OLS when the nonlinearity is group-specific.
- The row-level standard errors understate the sampling spread for some `psi = 1` and `phi = 1`
  targets (1.3 to 3.5 times); use clustered SEs and several seeds.
- `psi = 1` targets weight rows by the level of the outcome, so they are not invariant to rescaling
  the outcome period by period (e.g. nominal vs real dollars); `psi = 0` targets and log OLS are.

# Reading (b) panel estimator and estimation without cross-fitting

`loglinearcorrection.panel_readingb.PanelReadingB` estimates the paper's
$\theta_{\phi,\psi}(h)$ in a panel where the unit effect $C_i$ is part of the
unit's type ("reading (b)"). The derivation follows the user's notes of 2026-11-11.
This file records the version that is implemented, where it departs from those
notes, the conditions under which the no-cross-fitting theory of Chen and Chiang
(2026) applies, and the Monte Carlo evidence.

## Setting and target

Units $i$ are observed in rows $t$, and the panel may be unbalanced.
Assume $Y_{it}(x)=C_i\tilde Y_{it}(x)$ with
$\tilde Y_{it}(\cdot)\perp\!\!\!\perp(C_i,X_i,W_{i,-t})\mid W_{it}$. Then

$$\theta_{\phi,\psi}(h)=\frac{E[\omega\,h\,\epsilon_\phi]}{E[\omega]},\qquad
\omega=\big(\kappa_\phi(W)\,m_\phi(X,W)\big)^\psi ,$$

where $\ell_\phi=\log m_\phi$, $\epsilon_\phi=\partial\ell_\phi/\partial s$ with
$s=\log x$ (the interest column), and $\kappa_\phi(w)=M_\phi(C\mid W=w)$.
Expectations are pooled over rows and weighted by the analytic weights $w_{it}$.

## Estimation as implemented

**Within fit, one per $\phi$.** The default sieve is
$\ell(s,w)=\sum_{k=1}^{K}b_k(w)\tilde s^{k}+g(w)$, where $\tilde s$ is the
standardized interest regressor. Each $b_k(w)$ is linear in the controls $W$ and
saturated in the user `cells`, and $g(w)$ collects the main effects of $W$ and of
the cells. All fixed effects are absorbed: the unit effect and every other one, such
as time. The fit is FE-OLS of $\log Y$ for $\phi=0$ and FE-Poisson of $Y^\phi$ with
index $\phi\ell$ for $\phi\ne0$, solved by IRLS with weighted HDFE demeaning as in
ppmlhdfe. With unit effects, FE-Poisson is exactly the multinomial within
likelihood. With $K=1$ and no interactions, the $\phi=0$ fit is the package's
log-OLS first stage, and $\hat\theta_{0,\psi}=\hat\beta_{OLS}$ (tested).

A pluggable `within_model={"model_class", "arch_params", "fit_params"}` with the
`FESplineModelNuisance` interface can replace the sieve. That includes
`FENNModelNuisance` from the `nn-nuisance` branch, which should be run with
unpenalized FE scalars. The model receives
`[s, W, cell dummies, unit code, other FE codes]` and is fitted with an identity
link on $\log Y$ or a log link on $Y^\phi$. Its profile is read with the FE codes
set to 0. Its within residuals are recomputed by re-estimating the fixed effects
exactly, with the model index as an offset.

**$\kappa$, one per $\phi$.** This is a GLM of the deflated outcome
$(Ye^{-\hat\ell})^\phi$ on the $\kappa$-cell fixed effects plus the controls,
linear in $\log\kappa^\phi$: Poisson for $\phi\ne0$ and OLS of the log for
$\phi=0$. With saturated cells this is the spec's cell power mean exactly. With
continuous $W$ it is the spec's "regress on $W$ instead".

The $\kappa$ cells are the user cells crossed with every non-unit fixed effect.
This keeps two exact properties of the spec, which says time belongs in $W_{it}$
if the outcome distribution changes over time:

- $\kappa e^{\ell}$ is invariant to the unidentified additive fixed effects in $\ell$;
- the target's derivative vanishes on every nuisance direction other than the
  $s$-terms.

More than 1% of rows in cells with fewer than 20 rows triggers a warning. With `kappa_cells="user"` (or `"auto"` when crossing would leave small cells), the $\kappa$ cells are the user cells crossed only with fixed effects nested in units. The other absorbed effects stay in $\ell$, $F$ no longer vanishes on their dummies, and the automatic representer solves the joint (s-terms, absorbed dummies) system by MINRES in operator form.

**Plug-in.** $\tilde\theta=\sum w\hat\omega h\hat\epsilon/\sum w\hat\omega$, with
$\hat\omega=\exp\{\psi(\log\hat\kappa+\hat\ell)\}$.

**Debiased score.** Each row contributes

$$w_{it}\Big[\hat\omega_{it}(h\hat\epsilon_{it}-\theta)
+\psi\hat\gamma_{it}\log_\phi\!\frac{Y_{it}}{\hat\kappa_{it}e^{\hat\ell_{it}}}
+\hat\alpha_{it}\hat\rho_{it}\Big],\qquad
\hat\theta=\tilde\theta+\frac{\sum w(\psi\hat\gamma\log_\phi+\hat\alpha\hat\rho)}{\sum w\hat\omega}.$$

The pieces are as follows.

- $\rho$ is the within residual: the demeaned $\log Y-\hat\ell$ for $\phi=0$, and
  $Y^\phi-\hat\mu$ for $\phi\ne0$, with $\hat\mu=S_i\pi_{it}$ when unit effects are
  the only fixed effects.
- $\log_\phi(z)=(z^\phi-1)/\phi$, and $\log z$ at $\phi=0$.
- $\hat\gamma=k\,Z'\Pi$, where $k=\hat\kappa^\phi$ ($k=1$ at $\phi=0$), $Z$ is the
  $\kappa$ design, and $\Pi$ solves $\sum wkZZ'\,\Pi=\sum wZ\,\hat\omega(h\hat\epsilon-\tilde\theta)$.
  This is the exact delta-method term for the Poisson $\kappa$-GLM, whose
  first-order condition is $\sum wZ\{(Ye^{-\hat\ell})^\phi-\hat\kappa^\phi\}=0$.
  With saturated cells, $\gamma$ is the cell mean of $\omega(h\epsilon-\theta)$, as in
  the spec. With continuous $W$ the spec's "cell mean" generalizes to this weighted
  projection. A first version used weights $wr$ instead of $wk$, which is exact only
  for saturated cells. A test now checks that the in-sample $\kappa$ term sums to 0.
- Row-level influence functions, in the library's convention
  $\hat\theta-\theta=\sum_i p_i\mathrm{IF}_i$, are
  $\mathrm{IF}=[\hat\omega(h\hat\epsilon-\hat\theta)+\psi\hat\gamma\log_\phi+\hat\alpha\hat\rho]/\bar\omega$.
  The covariance is clustered by unit by default. Use `with_clusters` for a
  coarsening of the unit, and a warning fires if the clusters split units.

**Derivative of the target in $\ell$.** Perturbing $\ell$ by $\delta$, with $\kappa$
re-solved, gives

$$F(\delta)=\sum w\big[\omega h\,\partial_s\delta+\psi\{\omega(h\epsilon-\theta)-\gamma r\}\delta\big].$$

$F$ vanishes on every direction in the $\kappa$ design: cells, absorbed FE levels
and the controls.

### Riesz representer, `riesz="automatic"` (default)

Let $B$ be the $s$-terms of degree `riesz_degree`, which defaults to the within
degree. Let $\tilde B$ be $B$ net of all fixed effects, controls and cell dummies
(FWL) in the metric $\Lambda=w$ for OLS or $\Lambda=w\hat\mu$ for Poisson. Then

$$\hat\alpha=\tilde B c,\qquad \big(\phi_{*}\,\tilde B'\Lambda\tilde B\big)c=F(B),\qquad \phi_*=1\ (\phi=0),\ \phi\ (\phi\neq0).$$

This needs no density score, no history regression and no Neumann series.

With the sieve within fit and `riesz_degree == degree`,
$\sum w\hat\alpha\hat\rho=0$ in-sample by the within first-order conditions. The
1-fold debiased estimate therefore equals the plug-in, and its influence function
is the sieve delta-method one. That is the spec's Step 3 ("stack the within score,
the cell moments and the two sums, GMM + delta method"), which is numerically the
same thing (Ackerberg, Chen and Hahn 2012). A richer Riesz basis gives a nonzero
correction, and so does a non-sieve within model such as FESpline or FENN.

### Riesz representer, `riesz="explicit"` (spec, part 2)

$$q=-\hat\omega(h'+h\hat s+\psi\tilde\theta)-\psi\hat\gamma\hat{\bar r}$$

The terms are estimated as follows.

- $\hat s$ is the density score of $s$ given (cells, $W$), pooled over rows. It
  comes from sieve score matching (Hyvärinen),
  $\hat s=\tilde\Phi c$ with $c=-(\tilde\Phi'W\tilde\Phi)^{-1}\sum w\,\partial_s\Phi$,
  where $\Phi$ is the polynomial in $\tilde s$ with interactions and $\tilde\Phi$
  is $\Phi$ net of cell FE and controls.
- $\hat{\bar r}$ is a sieve regression of $r$ on $(s,W,\text{cells})$.
- $q$ is projected off $(\text{cells},W)$, which imposes $E[q\mid W]=0$.

$g$ solves $\sum_t w_t(g_t-\bar g^w_i)\delta_t=\sum_t w_tq_t\delta_t$ for all $\delta$
in span(basis, $\kappa$ cells). It is computed by the Neumann/backfitting
iteration

$$g\leftarrow (G'AG)^{-1}G'\big[wq+a\,Kg\big],\qquad a_{it}=w_{it}(1-w_{it}/W_i),\quad (Kg)_{it}=\frac{\sum_{s\ne t}w_{is}g_{is}}{W_i-w_{it}}.$$

For a balanced, unweighted panel this is the spec's $g=\frac{T}{T-1}q+Kg$.
Component constants, which form the kernel, are projected out at every step. If the
iteration has not converged after `neumann_max_iter` steps, CG on the equivalent
within-Gram system takes over.

Then $\alpha_t=g_t-\bar g_i$ for $\phi=0$, and
$\alpha_t=(g_t-\bar g^w_i)/(\phi\tilde\mu_t)$ for $\phi\neq0$. Here $\tilde\mu$ is a
crude Poisson "history regression" of $Y^\phi$ on unit means of $(\tilde s,\tilde
s^2,\tilde s^3,W)$, $\log T_i$ and cell effects, with the fitted index as an
offset. Its unit factor is clipped to the 1st–99th percentiles.

The diagnostics report the iterations, convergence and the empirical contraction
rate.

### Diagnostics (`result.diagnostics`)

- `connectedness`: units, singleton units, stayers, $\kappa$ cells, the smallest
  cell, and the number of components of the (unit, $\kappa$-cell) graph. For each
  slope cell it gives movers, the number of within-cell interval chains (the
  empirical within-cell components) and the share of rows in the largest chain.
- `persistence`: `within_share` is the within-unit share of the variance of $s$ net
  of cells and $W$. `rho_star` is $\max|\lambda|$ of the Neumann operator
  $P_aK=I-(G'AG)^{-1}G'WM_{unit}G$ on the Riesz basis, excluding the kernel. Values
  near 1 mean near-stayers and a large variance.
- `neumann` (explicit only), `target_effective_n`, `within_converged_phi*`.
- `reading_a_plugin`: the reading (a) plug-in where it is identified without a
  single component: $\psi=0$ (the same as (b)) and $\psi=\phi$ (weights $Y^\phi$).
  Otherwise it is NaN.

## Departures from, and remarks on, the spec

1. **The spec's representer for $\phi\neq0$ is valid but is not the plug-in's.**
   $\alpha_t=(g_t-\bar g_i)/(\phi\mu_t)$, with $g$ solving the unweighted within
   equation, satisfies the orthogonality condition. But it corresponds to a
   $1/\mu$-reweighted, history-dependent within moment. It is not the influence
   function of the Poisson within estimator. The Poisson estimator's own
   representer is an own-period function, $\alpha_t=\tilde B_t'c$ with the
   multinomial Hessian; this is the automatic one. The two coincide at $\phi=0$.

   The note that "a crude history regression is enough" is right for the bias,
   since errors enter only through products. It is not right for the variance:
   $\alpha\propto1/\tilde\mu$, and rows where the fitted level $e^{\hat\ell}$ or the
   history factor is too small dominate the correction. In simulations a single
   row's correction reached $10^3$–$10^4$ times the typical weight, at polynomial
   tails. See the Monte Carlo below.
2. **Unbalanced panels and weights.** $\frac{T}{T-1}$ and $K$ become
   $a_{it}=w_{it}(1-w_{it}/W_i)$ and the weight-adjusted $K$ above. The spec's
   per-unit $\frac1T\sum_t$ normalization becomes pooled sums over rows, which is
   identical in a balanced panel.
3. **Other fixed effects (time etc.)** are crossed into the $\kappa$ cells by
   default; see above. With `kappa_cells="user"`, $F$ does not vanish on their
   directions when $\psi\neq0$, and the automatic representer then carries the FE
   block (MINRES). The two choices are different, equally valid estimands: they
   differ in the cells over which $C$'s power mean is taken. Fixed effects nested in
   units are always cells, because the within fit cannot see them.
4. **Convergence of the Neumann series** needs the eigenvalues of the projected
   $K$ bounded away from $-1$ as well as from $+1$. For $T=2$ with negatively
   correlated periods, $-1$ is attainable. CG is the fallback.
5. **Reading (a)** is computed only as the identified plug-ins (diagnostic), with
   no influence function. The off-diagonal reading (a) estimator, which needs a
   second within fit at $\psi$ and a single component, is not implemented.

## Estimation without cross-fitting (Chen and Chiang 2026, arXiv:2602.11333)

The paper is "Cross-Fitting-Free Debiased Machine Learning with Multiway
Dependence", v5, September 2026. Its result: with an orthogonal score, nuisances
estimated on the full sample can be plugged in. The debiased GMM estimator is then
asymptotically linear and normal under (multiway) clustering, provided that:

- **Orthogonality and smoothness.** The score has Neyman orthogonality and
  mean-square continuity, and the derivative of the moment in $\eta$ is Lipschitz
  (Assumption 1(ii)).
- **Rate.** $\|\hat\eta-\eta_0\|_{2}=o_P(n^{-1/4})$, and $\hat\eta\in\mathcal H_n$
  with probability tending to one (Assumption 1(v)). The argument localizes
  $\hat\eta$ in $\{\eta\in\mathcal H_n:\|\eta-\eta_0\|\le Cn^{-1/4}\}$.
- **Complexity.** The localized score class is VC-type with index $v_n$ and
  envelope $F_n$, and eq. (3.8) holds. For one-way clustering this requires
  $v_n\log n/n^{1/2}\to0$ and $\|F_n\|_r v_n\log n/n^{1/2-1/r}\to0$. Their
  Theorem 2 gives sufficient conditions with $r>4$:
  - sparse GLM: $s\log(p/s)\vee s\log\bar N=o(n^{1/4})$;
  - trees: $L\log(Lp)\log n=o(n^{1/4})$;
  - deep ReLU networks: $LW\log(pU)\log n=o(n^{1/4})$.
- **Moments.** $\|F_n\|_{r}<\infty$ for some $r>2$, plus an equicontinuity
  condition.
- **Variance.** The plug-in cluster-robust variance of the orthogonal score at the
  full-sample nuisances is used (eq. 3.11). For one-way clustering this is the
  usual sandwich, with no extra bias or variance correction. The library applies
  $G/(G-1)$.

**How it applies here.**

- *Reading (b), sieve within fit.* The nuisances are $\ell$ (a finite sieve of
  dimension $K(1+p_W+n_{cells})$), the $\kappa$-GLM, $\gamma$ and $\alpha$, all
  finite-dimensional and smooth. The unit effects are not nuisance parameters in
  Chiang's sense. The within moments are invariant to them (OLS demeaning and the
  multinomial likelihood concentrate them out exactly), so no incidental-parameter
  complexity enters. Hence $v_n=O(Kp+n_{\kappa\text{-cells}})$, and the conditions
  hold when this is $o(n^{1/4}/\log n)$. With high-dimensional absorbed fixed
  effects crossed into the $\kappa$ cells this can fail, so the estimator warns on
  small cells. The variance is the unit-clustered sandwich of the score, which is
  what `PanelReadingBResults` reports. With `riesz_degree == degree` the 1-fold
  estimator is the sieve plug-in with sieve delta-method SEs, and the classical
  sieve theory also covers it.
- *Neural within or nuisance models (FENN).* Chiang's Theorem 2 needs tiny
  networks ($LW\log n=o(n^{1/4})$) for the guarantee, which practical MLPs exceed.
  Their simulations nonetheless show no-CF NNs comparable to CF. Treat 1-fold NN
  results as covered by the simulations, not by the theorem.
- *GEAM/GPCAM with FE scalars in the nuisances* (FESpline/FENN with
  `fe_encoding='codes'`). The per-level FE scalars make $v_n$ grow with the number
  of levels kept (levels with at least `nuisance_fe_min_rows` rows). For
  $v_n\gtrsim n^{1/4}$ Chiang's conditions fail. So does the standard
  cross-fitting theory, since held-out FE levels are estimated from few rows. That
  is an incidental-parameter problem that 1 fold does not create and does not cure.
  Reading (b) avoids it by construction.

**What changed for `n_folds=1` elsewhere** (`generalized_dream.py`):

- *GPCAM* `_make_splits` accepts `n_folds=1`, which fits and evaluates on all rows.
  The hunk is identical to the one on `nn-nuisance`, so the merge is trivial.
- *GEAM Stein check and calibration.* With $\hat\mu=Hx$ fitted on the same rows,
  Stein's identity gives
  $E[s(v)\,((I-H)v)_i]=-(1-H_{ii})$, so the target of $E[s\,(x-\hat\mu)]$ is
  $-(1-\mathrm{dof}/n)$, not $-1$. The old calibration rescaled the score by
  $1/(1-\mathrm{dof}/n)$ in the wrong direction: by 25% with 4 rows per FE level.
  `stein_target` now carries this value through `_stein_diagnostics`,
  `_calibrate_score` and `score_diagnostics`. Held-out rows keep $-1$. The
  unweighted dof ratio is used under weights, an approximation.
- *Already correct:* GEAM's `n_folds=1` split, the `1e-10` held-out weight hack
  (guarded by `n_folds > 1`), FESpline's internal penalty holdout (independent of
  folds) and the driver's per-row extraction (all rows in fold 0). These were
  verified end to end with `run_geam`/`run_gpcam` from `log_paper_replication`
  on synthetic data.
- *Not changed:* in-sample FE-OLS residuals are shrunk by about $(1-1/n_g)$. GEAM's
  $m_\phi$ at $\phi\ne0$ is fitted to them, which is an $O(1/n_g)$
  incidental-parameter bias. Cross-fitting has the mirror-image inflation, so
  neither is exact.

## Monte Carlo

The DGP is `scripts/sim_readingb.py::simulate`. There are $N=1000$ units and $T=5$
periods ($n=5000$), with $C_i=e^{c_i}$, $c_i\sim N(0,0.8^2)$, and

$$s_{it}=0.6c_i+a_i+e_{it}+0.3w_{it},$$

where $e_{it}$ is AR(1) with $\rho\in\{0.3,0.9\}$. The unit elasticity is
$\epsilon_i=b(w)+\tau(w)\xi_i$, so $\epsilon_1=b+\tau^2s$ varies with $s$ while
$\epsilon_0=b$. The outcome has multiplicative noise $v_{it}\sim N(0,0.5^2)$.

$W$ takes one of three forms: none; time-invariant
$w_i\sim\mathrm{Bern}(\mathrm{logit}^{-1}(1.5c_i))$; or time-varying
$w_{it}\sim\mathrm{Bern}(\mathrm{logit}^{-1}(0.8c_i+\zeta))$.

There are 200 replications, and the truths come from a draw of $3\times10^5$ units.
Methods:

- `plugin`: degree-3 interacted sieve, automatic Riesz, 1 fold. This is also
  `auto`.
- `auto_rich`: Riesz degree 4.
- `explicit`: Neumann representer.
- `auto5`/`explicit5`: 5-fold cross-fitting by unit.
- `lin_*`: degree-1, log-OLS-like (misspecified for $\phi=1$) within fit, corrected
  by a degree-3 Riesz basis.

Each cell is bias / RMSE / coverage of the nominal 95% interval, with
unit-clustered SEs, measured against the reading (b) truths. The summaries are in
`docs/sim/*.csv`, and `docs/sim/run_all.sh` regenerates them (about 20 minutes on
6 cores).

**Truths.** Reading (b) and reading (a) coincide at $\psi=0$ and differ at $\psi=1$.

| design | rho | (0,0) | (1,0) | (0,1) b / a | (1,1) b / a |
|---|---|---|---|---|---|
| none | 0.3 | 0.800 | 0.800 | 0.800 / 0.800 | 0.892 / 0.945 |
| none | 0.9 | 0.800 | 0.800 | 0.800 / 0.800 | 0.892 / 0.946 |
| time-inv. W | 0.3 | 1.000 | 0.997 | 1.155 / 1.161 | 1.216 / 1.237 |
| time-inv. W | 0.9 | 1.000 | 0.997 | 1.155 / 1.161 | 1.216 / 1.238 |
| time-var. W | 0.3 | 1.000 | 1.001 | 1.129 / 1.137 | 1.189 / 1.218 |
| time-var. W | 0.9 | 1.000 | 1.001 | 1.129 / 1.138 | 1.189 / 1.218 |
| time-var. W + time FE, 20% rows dropped | 0.5 | 1.000 | 1.001 | 1.129 / 1.137 | 1.189 / 1.218 |
| time-var. W, N=4000 | 0.5 | 1.000 | 1.001 | 1.129 / 1.137 | 1.189 / 1.218 |

**Main comparison.** The within fit is a degree-3 interacted sieve.

| design | rho | method | (0,0) | (1,0) | (0,1) | (1,1) |
|---|---|---|---|---|---|---|
| none | 0.3 | automatic, 1 fold (= plug-in) | -0.001 / 0.020 / 0.95 | -0.011 / 0.039 / 0.88 | +0.001 / 0.027 / 0.96 | -0.007 / 0.047 / 0.90 |
| none | 0.3 | explicit, 1 fold | -0.001 / 0.020 / 0.95 | -0.010 / 0.032 / 0.93 | +0.001 / 0.027 / 0.96 | -0.013 / 0.050 / 0.89 |
| none | 0.3 | automatic, 5 folds | -0.001 / 0.020 / 0.95 | -0.013 / 0.125 / 0.92 | -0.001 / 0.026 / 0.97 | -0.004 / 0.074 / 0.92 |
| none | 0.3 | explicit, 5 folds | -0.001 / 0.020 / 0.95 | -0.009 / 0.031 / 0.95 | -0.001 / 0.027 / 0.96 | -0.005 / 0.058 / 0.92 |
| none | 0.3 | automatic, Riesz deg 4, 1 fold | -0.001 / 0.020 / 0.95 | -0.007 / 0.028 / 0.94 | +0.001 / 0.028 / 0.96 | -0.004 / 0.049 / 0.91 |
| none | 0.9 | automatic, 1 fold (= plug-in) | -0.000 / 0.034 / 0.95 | -0.009 / 0.062 / 0.93 | +0.002 / 0.044 / 0.96 | -0.003 / 0.087 / 0.93 |
| none | 0.9 | explicit, 1 fold | -0.000 / 0.034 / 0.95 | -0.015 / 0.071 / 0.96 | +0.002 / 0.044 / 0.94 | -0.034 / 0.293 / 0.92 |
| none | 0.9 | automatic, 5 folds | -0.000 / 0.034 / 0.95 | +0.020 / 0.228 / 0.97 | -0.004 / 0.044 / 0.96 | +0.002 / 0.142 / 0.94 |
| none | 0.9 | explicit, 5 folds | -0.000 / 0.034 / 0.95 | -0.014 / 0.075 / 0.96 | -0.003 / 0.044 / 0.94 | -0.051 / 0.464 / 0.93 |
| none | 0.9 | automatic, Riesz deg 4, 1 fold | -0.001 / 0.034 / 0.95 | -0.006 / 0.048 / 0.97 | +0.002 / 0.046 / 0.95 | +0.023 / 0.163 / 0.93 |
| time-inv. W | 0.3 | automatic, 1 fold (= plug-in) | -0.000 / 0.021 / 0.94 | +0.002 / 0.050 / 0.86 | +0.005 / 0.044 / 0.93 | +0.007 / 0.078 / 0.90 |
| time-inv. W | 0.3 | explicit, 1 fold | -0.000 / 0.021 / 0.94 | -0.028 / 0.090 / 0.85 | +0.005 / 0.044 / 0.91 | -0.131 / 0.678 / 0.88 |
| time-inv. W | 0.3 | automatic, 5 folds | -0.000 / 0.021 / 0.94 | -0.002 / 0.217 / 0.92 | -0.000 / 0.045 / 0.91 | +0.004 / 0.218 / 0.92 |
| time-inv. W | 0.3 | explicit, 5 folds | -0.000 / 0.021 / 0.94 | -0.044 / 0.224 / 0.90 | +0.002 / 0.045 / 0.90 | -0.242 / 1.474 / 0.90 |
| time-inv. W | 0.3 | automatic, Riesz deg 4, 1 fold | -0.000 / 0.021 / 0.94 | -0.017 / 0.046 / 0.86 | +0.008 / 0.048 / 0.92 | +0.013 / 0.103 / 0.88 |
| time-inv. W | 0.9 | automatic, 1 fold (= plug-in) | -0.000 / 0.035 / 0.94 | -0.006 / 0.072 / 0.92 | +0.005 / 0.076 / 0.91 | +0.008 / 0.130 / 0.89 |
| time-inv. W | 0.9 | explicit, 1 fold | -0.000 / 0.035 / 0.94 | -0.079 / 0.334 / 0.85 | +0.005 / 0.076 / 0.89 | -0.434 / 2.720 / 0.84 |
| time-inv. W | 0.9 | automatic, 5 folds | -0.001 / 0.036 / 0.94 | +0.048 / 0.490 / 0.96 | -0.013 / 0.077 / 0.92 | +0.003 / 0.270 / 0.89 |
| time-inv. W | 0.9 | explicit, 5 folds | -0.001 / 0.036 / 0.94 | -0.149 / 0.750 / 0.89 | -0.007 / 0.078 / 0.88 | -0.636 / 4.332 / 0.87 |
| time-inv. W | 0.9 | automatic, Riesz deg 4, 1 fold | -0.000 / 0.035 / 0.94 | -0.009 / 0.066 / 0.90 | +0.006 / 0.080 / 0.93 | +0.031 / 0.190 / 0.89 |
| time-var. W | 0.3 | automatic, 1 fold (= plug-in) | -0.000 / 0.018 / 0.95 | -0.003 / 0.066 / 0.90 | +0.006 / 0.038 / 0.93 | +0.007 / 0.081 / 0.91 |
| time-var. W | 0.3 | explicit, 1 fold | -0.000 / 0.018 / 0.95 | -0.050 / 0.355 / 0.86 | +0.006 / 0.038 / 0.92 | -0.719 / 8.972 / 0.87 |
| time-var. W | 0.3 | automatic, 5 folds | -0.000 / 0.018 / 0.94 | +0.012 / 0.461 / 0.96 | +0.001 / 0.039 / 0.94 | +0.015 / 0.174 / 0.93 |
| time-var. W | 0.3 | explicit, 5 folds | +0.000 / 0.018 / 0.95 | -0.169 / 1.558 / 0.91 | +0.003 / 0.039 / 0.93 | -0.615 / 6.784 / 0.90 |
| time-var. W | 0.3 | automatic, Riesz deg 4, 1 fold | -0.000 / 0.018 / 0.94 | -0.030 / 0.175 / 0.88 | +0.006 / 0.039 / 0.93 | +0.008 / 0.179 / 0.89 |
| time-var. W | 0.9 | automatic, 1 fold (= plug-in) | +0.002 / 0.033 / 0.94 | +0.002 / 0.108 / 0.92 | +0.010 / 0.063 / 0.95 | +0.011 / 0.157 / 0.93 |
| time-var. W | 0.9 | explicit, 1 fold | +0.002 / 0.033 / 0.94 | -0.025 / 0.130 / 0.93 | +0.010 / 0.063 / 0.93 | -0.120 / 0.741 / 0.90 |
| time-var. W | 0.9 | automatic, 5 folds | +0.002 / 0.034 / 0.95 | +0.110 / 1.379 / 0.96 | -0.002 / 0.063 / 0.95 | +0.260 / 2.628 / 0.94 |
| time-var. W | 0.9 | explicit, 5 folds | +0.002 / 0.033 / 0.95 | -0.036 / 0.226 / 0.94 | +0.002 / 0.063 / 0.93 | -0.229 / 1.496 / 0.93 |
| time-var. W | 0.9 | automatic, Riesz deg 4, 1 fold | +0.002 / 0.033 / 0.95 | -0.021 / 0.127 / 0.93 | +0.012 / 0.071 / 0.95 | +0.220 / 2.618 / 0.92 |
| time-var. W + time FE, 20% rows dropped | 0.5 | automatic, 1 fold (= plug-in) | -0.001 / 0.022 / 0.95 | -0.005 / 0.057 / 0.89 | +0.007 / 0.045 / 0.94 | +0.003 / 0.092 / 0.91 |
| time-var. W + time FE, 20% rows dropped | 0.5 | explicit, 1 fold | -0.001 / 0.022 / 0.95 | -0.072 / 0.503 / 0.89 | +0.007 / 0.045 / 0.93 | -0.345 / 3.095 / 0.90 |
| time-var. W + time FE, 20% rows dropped | 0.5 | automatic, 5 folds | -0.000 / 0.022 / 0.95 | +0.000 / 0.313 / 0.87 | +0.000 / 0.046 / 0.94 | +0.060 / 0.587 / 0.88 |
| time-var. W, N=4000 | 0.5 | automatic, 1 fold (= plug-in) | -0.001 / 0.010 / 0.94 | -0.007 / 0.037 / 0.86 | -0.001 / 0.019 / 0.94 | -0.004 / 0.031 / 0.93 |
| time-var. W, N=4000 | 0.5 | explicit, 1 fold | -0.001 / 0.010 / 0.94 | -0.015 / 0.056 / 0.89 | -0.001 / 0.019 / 0.94 | -0.043 / 0.208 / 0.90 |

**Debiasing a misspecified within fit.** Here the within fit is degree 1 with no
interactions (the log-OLS first stage).

| design | rho | method | (0,0) | (1,0) | (0,1) | (1,1) |
|---|---|---|---|---|---|---|
| none | 0.3 | deg-1 within, plug-in | -0.001 / 0.020 / 0.95 | +0.126 / 0.145 / 0.42 | -0.001 / 0.020 / 0.95 | +0.034 / 0.079 / 0.92 |
| none | 0.3 | deg-1 within + auto Riesz deg 3 | -0.001 / 0.020 / 0.95 | -0.026 / 0.050 / 0.88 | +0.000 / 0.026 / 0.96 | -0.006 / 0.046 / 0.91 |
| none | 0.3 | deg-1 within + explicit | -0.001 / 0.020 / 0.95 | -0.006 / 0.040 / 0.94 | +0.000 / 0.026 / 0.96 | -0.002 / 0.051 / 0.89 |
| none | 0.3 | deg-1 within + auto deg 3, 5 folds | -0.001 / 0.020 / 0.95 | -0.005 / 0.090 / 0.90 | -0.000 / 0.027 / 0.96 | +0.002 / 0.059 / 0.89 |
| none | 0.9 | deg-1 within, plug-in | -0.001 / 0.034 / 0.94 | +0.128 / 0.169 / 0.71 | -0.001 / 0.034 / 0.94 | +0.036 / 0.116 / 0.93 |
| none | 0.9 | deg-1 within + auto Riesz deg 3 | -0.000 / 0.034 / 0.95 | -0.014 / 0.064 / 0.96 | +0.000 / 0.043 / 0.95 | +0.002 / 0.080 / 0.91 |
| none | 0.9 | deg-1 within + explicit | -0.000 / 0.034 / 0.95 | -0.002 / 0.055 / 0.95 | +0.000 / 0.043 / 0.94 | +0.001 / 0.076 / 0.90 |
| none | 0.9 | deg-1 within + auto deg 3, 5 folds | -0.000 / 0.034 / 0.95 | +0.010 / 0.147 / 0.97 | -0.001 / 0.044 / 0.95 | -0.001 / 0.087 / 0.93 |
| time-inv. W | 0.3 | deg-1 within, plug-in | -0.000 / 0.021 / 0.94 | +0.233 / 0.249 / 0.07 | -0.155 / 0.157 / 0.00 | +0.014 / 0.087 / 0.94 |
| time-inv. W | 0.3 | deg-1 within + auto Riesz deg 3 | -0.000 / 0.021 / 0.95 | +0.014 / 0.094 / 0.98 | -0.051 / 0.062 / 0.68 | -0.009 / 0.066 / 0.86 |
| time-inv. W | 0.3 | deg-1 within + explicit | -0.000 / 0.021 / 0.95 | -0.041 / 0.081 / 0.71 | -0.052 / 0.064 / 0.61 | -0.041 / 0.094 / 0.71 |
| time-inv. W | 0.3 | deg-1 within + auto deg 3, 5 folds | -0.000 / 0.021 / 0.95 | +0.054 / 0.334 / 0.99 | -0.051 / 0.062 / 0.68 | +0.001 / 0.107 / 0.89 |
| time-inv. W | 0.9 | deg-1 within, plug-in | -0.000 / 0.036 / 0.94 | +0.225 / 0.273 / 0.47 | -0.155 / 0.159 / 0.01 | +0.006 / 0.155 / 0.91 |
| time-inv. W | 0.9 | deg-1 within + auto Riesz deg 3 | -0.000 / 0.035 / 0.94 | +0.031 / 0.135 / 0.95 | -0.063 / 0.088 / 0.77 | -0.006 / 0.112 / 0.89 |
| time-inv. W | 0.9 | deg-1 within + explicit | -0.000 / 0.035 / 0.94 | -0.031 / 0.111 / 0.87 | -0.066 / 0.089 / 0.73 | -0.052 / 0.171 / 0.79 |
| time-inv. W | 0.9 | deg-1 within + auto deg 3, 5 folds | -0.000 / 0.035 / 0.94 | +0.061 / 0.763 / 0.98 | -0.064 / 0.089 / 0.78 | -0.028 / 0.151 / 0.89 |
| time-var. W | 0.3 | deg-1 within, plug-in | -0.000 / 0.019 / 0.94 | +0.197 / 0.217 / 0.12 | -0.129 / 0.131 / 0.00 | +0.009 / 0.091 / 0.92 |
| time-var. W | 0.3 | deg-1 within + auto Riesz deg 3 | -0.000 / 0.019 / 0.96 | -0.098 / 0.131 / 0.65 | -0.001 / 0.035 / 0.93 | -0.032 / 0.069 / 0.78 |
| time-var. W | 0.3 | deg-1 within + explicit | -0.000 / 0.019 / 0.96 | -0.047 / 0.104 / 0.64 | -0.003 / 0.038 / 0.89 | +0.003 / 0.108 / 0.80 |
| time-var. W | 0.3 | deg-1 within + auto deg 3, 5 folds | -0.001 / 0.019 / 0.96 | -0.045 / 0.584 / 0.76 | -0.002 / 0.036 / 0.93 | -0.022 / 0.115 / 0.82 |
| time-var. W | 0.9 | deg-1 within, plug-in | +0.001 / 0.035 / 0.96 | +0.216 / 0.308 / 0.57 | -0.128 / 0.132 / 0.04 | +0.028 / 0.222 / 0.93 |
| time-var. W | 0.9 | deg-1 within + auto Riesz deg 3 | +0.001 / 0.035 / 0.97 | -0.331 / 0.348 / 0.20 | +0.201 / 0.210 / 0.05 | -0.057 / 0.176 / 0.74 |
| time-var. W | 0.9 | deg-1 within + explicit | +0.001 / 0.035 / 0.97 | -0.031 / 0.206 / 0.89 | +0.189 / 0.200 / 0.06 | +0.186 / 0.292 / 0.30 |
| time-var. W | 0.9 | deg-1 within + auto deg 3, 5 folds | +0.001 / 0.035 / 0.97 | -0.178 / 1.331 / 0.41 | +0.199 / 0.208 / 0.07 | -0.066 / 0.162 / 0.81 |

### Reading the results

- **Automatic Riesz with 1 fold is the default.** In every design it is
  essentially unbiased, with $|\text{bias}|\le0.011$. Coverage is 0.91–0.96 at
  $\phi=0$.
- **$\phi=1$ coverage is 0.86–0.93** because the unit-clustered SE understates the
  SD by 15–30%. This is a heavy-tail effect of the DGP: $C=e^{c}$ with
  $c\sim N(0,0.8^2)$ is correlated with $s$, and elasticity heterogeneity puts the
  fourth moments of $Ye^{\epsilon s}$ near the boundary of finiteness. With
  $\sigma_c=0.3$, coverage at $(1,0)$ is 0.953 and SE/SD is 1.02 (150 reps). With
  $\sigma_c=0.8$ it does not clearly improve from $N=1000$ to $N=4000$. This is the
  familiar PPML small-sample SE problem, not a missing term. In-sample, the
  $\kappa$ and same-basis Riesz corrections sum to exactly zero (tested), and the IF
  is the exact sieve delta-method one. A cluster bootstrap would be the natural
  check on real data with very heavy tails.
- **The explicit representer** equals the automatic one up to noise at $\phi=0$.
  At $\phi\neq0$ it has occasional explosions: RMSE up to 9 with time-varying $W$.
  The $1/\tilde\mu$ factor and polynomial-tail levels $e^{\hat\ell}$ make a few rows
  dominate; one row's correction reached $4\times10^{3}$. Clipping the history
  factor does not cure this, because the tails of $e^{\hat\ell}$ do the damage. It
  also needs a density score, an $r$-regression and a history regression.
- **5-fold cross-fitting by unit** matches 1 fold at $\phi=0$. At $\phi=1$ its RMSE
  is 3–20 times larger. The held-out $\alpha\rho$ includes cubic-sieve
  extrapolation of $\alpha$ to held-out units, multiplied by level residuals; a
  single row moved the estimate by 0.09. This agrees with Chen and Chiang's finding
  that no-CF does as well as or better than CF for series/penalized learners at
  moderate $n$.
- **A Riesz basis richer than the within sieve** (degree 4 against 3) adds
  heavy-tailed noise at $\phi=1$ without reducing bias.
- **Debiasing a misspecified (degree-1) within fit** removes the bias when the
  Riesz basis spans the misspecification. With no $W$ at $(1,0)$, the bias falls
  from 0.126 to −0.006 (explicit) or −0.026 (automatic). The correction is only
  partial, or overshoots, when the misspecification is in the $W$ direction
  (slopes differ by $w$) and the Riesz basis lacks the $s\times W$ interactions. The
  flexible interacted within sieve is therefore the recommendation, rather than
  "log-OLS first stage + debiasing".
- **(a) against (b).** Reading (a) is $\theta_{1,1}=0.945$ against 0.892 for (b) with
  no $W$, because $C$ is correlated with $s$: (a) weights by the selected scale. The
  reading (a) diagonal plug-in in `diagnostics["reading_a_plugin"]` tracks its own
  truth to within about 0.02. Reading (b) targets are invariant to rescaling units
  (exactly at $\phi=0$; tested).
- **The persistence diagnostic** $\rho^*$ is about 0.70 at AR $\rho=0.3$. The
  Neumann series needs about 50 iterations to reach $10^{-8}$, with empirical
  contraction equal to $\rho^*$. SEs roughly double from AR $\rho=0.3$ to $0.9$.

**Runtime.** One fit, all 4 targets, 2 threads, time-varying $W$, degree 3:

| rows | fixed effects | automatic | explicit |
|---|---|---|---|
| 5,000 | unit | 0.11 s | 0.33 s |
| 5,000 | unit + year | 0.35 s | 0.60 s |
| 100,000 | unit | 2.5 s | 6.0 s |
| 100,000 | unit + year | 8.4 s | 13.5 s |

## Driver integration (`log_paper_replication`)

`detect_panel_structure(inputs.fe, inputs.regressors[col])` returns
`{unit, time, absorbed}`. The rule:

- *unit* is the FE column with the most levels among those whose median group has
  at least 2 rows and that keep at least 1% of the interest regressor's variance
  within levels;
- *time* is the other FE with the fewest levels such that (unit, level) pairs are
  unique;
- every other FE is absorbed.

If no unit qualifies, skip the coefficient: reading (b) needs within-unit variation.

```python
from loglinearcorrection.panel_readingb import PanelReadingB, detect_panel_structure

exog, fe_names = _exog(inputs)                       # regressors + FE code columns
panel = detect_panel_structure(inputs.fe, inputs.regressors[col].to_numpy())
model = PanelReadingB(inputs.y, exog, fixed_effects=fe_names, unit=panel["unit"], time=panel["time"],
                      interest=col, targets=TARGETS, weights=inputs.weights, seed=seed,
                      kappa_cells="auto")
result = model.fit(n_folds=1, riesz="automatic", degree=3)
# result.estimates[:, 0]          estimates in TARGETS order
# result.influence_functions      rows x targets, library convention: theta_hat - theta = sum p_i IF_i
# result.per_row                  every array has length n; "fold" is all 0
# result.diagnostics, result.beta[0]  (beta[0] is the log-OLS FE slope)
```

Cluster on the unit or on a coarsening of it. When the authors' clusters do not
nest units, use two-way clusters (authors' clusters, unit); `with_clusters` warns
otherwise. `kappa_cells="auto"` keeps the spec's crossed cells unless more than 1%
of rows would sit in cells under 20 rows, as happens with industry×year or gravity
exporter-year/importer-year effects. In that case the absorbed effects stay in
$\ell$, and the automatic Riesz representer carries their block (MINRES).


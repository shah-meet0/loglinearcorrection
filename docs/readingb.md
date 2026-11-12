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

More than 1% of rows in cells with fewer than 20 rows triggers a warning.

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
3. **Other fixed effects (time etc.)** are crossed into the $\kappa$ cells; see
   above. Without this, $F$ would not vanish on the time-effect directions with
   $\psi\neq0$, and the FWL construction of the automatic representer would need
   the FE block.
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

RESULTS_PLACEHOLDER

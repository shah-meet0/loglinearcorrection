"""Monte Carlo for the reading (b) panel estimator (``PanelReadingB``).

DGP (``simulate``): units i = 1..N, periods t = 1..T,

    c_i ~ N(0, sc^2),  C_i = exp(c_i)
    W:  none | tinv: w_i ~ Bern(Phi(c_i)) | tvar: w_it ~ Bern(logistic(0.8 c_i + zeta_it))
    s_it = lam c_i + a_i + e_it + 0.3 w_it,  a_i ~ N(0, sa^2),  e_it AR(1)(rho)
    log Ytilde_it(s) = (b(w) + tau(w) xi_i) s + 0.2 w + v_it,  xi_i, v_it ~ N(0, 1), N(0, sv^2)
    Y_it = C_i Ytilde_it(S_it)   [+ time effects 0.1 t when time_effects]

so ell_0 = b s + ..., ell_1 = b s + tau^2 s^2 / 2 + ..., eps_0 = b(w), eps_1 = b(w) + tau(w)^2 s.
The scale C_i is correlated with S_it (lam) and with W, so reading (a) and (b)
differ at psi = 1.  True values are computed on a large draw of the same DGP.

CLI (one design per call, results appended as CSV rows):

    python scripts/sim_readingb.py --design tinv --reps 200 --n-units 1000 --T 5 \
        --rho 0.5 --out sim.csv --methods plugin,auto,auto_rich,explicit,auto5 --workers 8
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

TARGETS = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))


def simulate(n_units=1000, T=5, design="none", rho=0.5, lam=0.6, sc=0.8, sa=0.5, se=0.6, sv=0.5,
             b=(0.8, 1.2), tau=(0.35, 0.2), time_effects=False, unbalanced=0.0, seed=0, return_truth_inputs=False):
    rng = np.random.default_rng(seed)
    n = n_units * T
    unit = np.repeat(np.arange(n_units), T)
    t = np.tile(np.arange(T), n_units)
    c = rng.normal(0, sc, n_units)
    a = rng.normal(0, sa, n_units)
    xi = rng.normal(0, 1, n_units)
    e = np.empty((n_units, T))
    e[:, 0] = rng.normal(0, se, n_units)
    for k in range(1, T):
        e[:, k] = rho * e[:, k - 1] + np.sqrt(1 - rho ** 2) * rng.normal(0, se, n_units)
    if design == "none":
        w = np.zeros(n)
    elif design == "tinv":
        w = np.repeat((rng.random(n_units) < 1 / (1 + np.exp(-1.5 * c))).astype(float), T)
    elif design == "tvar":
        z = 0.8 * c[unit] + rng.normal(0, 1, n)
        w = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    else:
        raise ValueError(design)
    wi = w.astype(int)
    s = lam * c[unit] + a[unit] + e.reshape(-1) + 0.3 * w
    bw = np.asarray(b)[wi]
    tw = np.asarray(tau)[wi]
    eps_unit = bw + tw * xi[unit]
    v = rng.normal(0, sv, n)
    tfx = 0.1 * t if time_effects else 0.0
    log_y = c[unit] + eps_unit * s + 0.2 * w + v + tfx
    frame = pd.DataFrame({"s": s, "w": w, "unit": unit, "t": t})
    out = {"y": np.exp(log_y), "frame": frame, "c": c[unit], "s": s, "w": wi, "t": t, "bw": bw, "tw": tw,
           "sv": sv, "tfx": tfx}
    if unbalanced > 0:
        keep = rng.random(n) >= unbalanced
        for key in ("y", "c", "s", "w", "t", "bw", "tw"):
            out[key] = out[key][keep]
        out["frame"] = frame[keep].reset_index(drop=True)
        if time_effects:
            out["tfx"] = 0.1 * out["t"]
    return out


def truth(design, n_big_units=400_000, **kwargs):
    """True reading (a) and (b) targets by Monte Carlo on a large draw."""
    d = simulate(n_units=n_big_units, design=design, seed=987654, **kwargs)
    s, w, c, bw, tw, sv = d["s"], d["w"], d["c"], d["bw"], d["tw"], d["sv"]
    cell = w * 1000 + d["t"] * (1 if kwargs.get("time_effects") else 0)
    out = {}
    for phi in (0.0, 1.0):
        if phi == 0.0:
            ell = bw * s + 0.2 * w + d["tfx"]
            eps = bw
            kap = pd.Series(c).groupby(cell).transform("mean").to_numpy()  # log kappa_0
        else:
            ell = bw * s + tw ** 2 * s ** 2 / 2 + 0.2 * w + sv ** 2 / 2 + d["tfx"]
            eps = bw + tw ** 2 * s
            kap = np.log(pd.Series(np.exp(c)).groupby(cell).transform("mean").to_numpy())
        for psi in (0.0, 1.0):
            if psi == 0.0:
                out[("b", phi, psi)] = out[("a", phi, psi)] = float(np.mean(eps))
                continue
            om_b = np.exp(kap + ell - np.mean(kap + ell))
            out[("b", phi, psi)] = float(np.sum(om_b * eps) / np.sum(om_b))
            om_a = np.exp(c + ell - np.mean(c + ell))
            out[("a", phi, psi)] = float(np.sum(om_a * eps) / np.sum(om_a))
    return out


METHODS = {
    # name: (fit kwargs)
    "plugin": dict(n_folds=1, riesz="automatic"),
    "auto": dict(n_folds=1, riesz="automatic"),
    "explicit": dict(n_folds=1, riesz="explicit"),
    "auto5": dict(n_folds=5, riesz="automatic"),
    # Riesz basis richer than the within sieve (degree 4 vs 3)
    "auto_rich": dict(n_folds=1, riesz="automatic", riesz_degree=4),
    "explicit5": dict(n_folds=5, riesz="explicit"),
    # misspecified (log-OLS-like, degree 1) within fit, corrected by a degree-3 Riesz basis
    "lin_plugin": dict(n_folds=1, riesz="automatic", degree=1, interact=False),
    "lin_auto3": dict(n_folds=1, riesz="automatic", degree=1, interact=False, riesz_degree=3),
    "lin_explicit": dict(n_folds=1, riesz="explicit", degree=1, interact=False, riesz_degree=3),
    "lin_auto3_5": dict(n_folds=5, riesz="automatic", degree=1, interact=False, riesz_degree=3),
}


def one_rep(args):
    rep, design, n_units, T, rho, methods, time_effects, unbalanced = args
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import warnings
    from loglinearcorrection.panel_readingb import PanelReadingB

    d = simulate(n_units=n_units, T=T, design=design, rho=rho, seed=rep, time_effects=time_effects,
                 unbalanced=unbalanced)
    frame = d["frame"]
    rows = []
    fe = ["unit", "t"] if time_effects else ["unit"]
    exog = frame[["s", "w"] + fe] if design != "none" else frame[["s"] + fe]
    for name in methods:
        kw = dict(METHODS[name])
        if design == "none" and "interact" in kw:
            pass
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = PanelReadingB(d["y"], exog, fixed_effects=fe, unit="unit", interest="s", seed=rep)
            res = model.fit(random_state=rep, **kw)
        secs = time.perf_counter() - started
        for k, (phi, psi) in enumerate(TARGETS):
            use_plugin = name in ("plugin", "lin_plugin")
            est = res.plugin_estimates[k, 0] if use_plugin else res.estimates[k, 0]
            se = res.plugin_standard_errors[k, 0] if use_plugin else res.standard_errors[k, 0]
            neu = res.diagnostics["folds"][0].get("neumann", {}).get(f"{phi:g}_{psi:g}", {})
            rows.append({"rep": rep, "design": design, "rho": rho, "n_units": n_units, "T": T, "method": name,
                         "phi": phi, "psi": psi, "estimate": est, "se": se, "seconds": secs,
                         "reading_a_plugin": res.diagnostics["reading_a_plugin"][f"{phi:g}_{psi:g}"],
                         "neumann_iter": neu.get("iterations", np.nan),
                         "rho_star": res.diagnostics["persistence"]["rho_star"]})
    return rows


def summarize(results: pd.DataFrame, truths: dict) -> pd.DataFrame:
    out = []
    for (design, rho, method, phi, psi), g in results.groupby(["design", "rho", "method", "phi", "psi"]):
        tb = truths[(design, rho)][("b", phi, psi)]
        ta = truths[(design, rho)][("a", phi, psi)]
        err = g["estimate"] - tb
        cover = np.mean(np.abs(err) <= 1.96 * g["se"])
        out.append({"design": design, "rho": rho, "method": method, "phi": phi, "psi": psi,
                    "theta_b": tb, "theta_a": ta, "mean": g["estimate"].mean(), "bias": err.mean(),
                    "sd": g["estimate"].std(), "rmse": float(np.sqrt(np.mean(err ** 2))),
                    "mean_se": g["se"].mean(), "coverage": cover, "reps": len(g),
                    "a_plugin_bias": float((g["reading_a_plugin"] - ta).mean()) if "reading_a_plugin" in g else np.nan,
                    "seconds": g["seconds"].mean()})
    return pd.DataFrame(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--design", default="none", choices=["none", "tinv", "tvar"])
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--n-units", type=int, default=1000)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--rho", type=float, nargs="+", default=[0.5])
    p.add_argument("--methods", default="plugin,explicit,auto5,lin_plugin,lin_auto3,lin_explicit")
    p.add_argument("--time-effects", action="store_true")
    p.add_argument("--unbalanced", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--out", default="sim_readingb.csv")
    p.add_argument("--truth-units", type=int, default=400_000)
    args = p.parse_args(argv)
    methods = args.methods.split(",")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    truths, all_rows = {}, []
    for rho in args.rho:
        truths[(args.design, rho)] = truth(args.design, n_big_units=args.truth_units, T=args.T, rho=rho,
                                           time_effects=args.time_effects)
        jobs = [(r, args.design, args.n_units, args.T, rho, methods, args.time_effects, args.unbalanced)
                for r in range(args.reps)]
        with ProcessPoolExecutor(args.workers) as pool:
            for rows in pool.map(one_rep, jobs):
                all_rows.extend(rows)
    results = pd.DataFrame(all_rows)
    results.to_csv(args.out.replace(".csv", "_raw.csv"), index=False)
    table = summarize(results, truths)
    table.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

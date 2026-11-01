"""TDBP replication (vanilla, barriers, discrete dividends, daily price limits).

Every parameter is defined in ``main()``. Results are printed and figures are
shown on screen; nothing is saved. Sizes are set for a fast run.
"""
import numpy as np
import pandas as pd

from tdbp import plotting
from tdbp.network import TDBPModel
from tdbp.payoffs import Payoff
from tdbp.plotting import STYLES
from tdbp.pricing import BatesFourier, BlackScholes, MonteCarlo
from tdbp.processes import GBM, MertonHeston, Simulator


# ======================================================================
# Shared helpers
# ======================================================================
def series(label, paths, step, prices, style, with_variance=False):
    """One point cloud of a checkpoint panel (the first len(prices) paths)."""
    n = len(prices)
    v = paths["V"][:n, step] if with_variance else None
    return dict(label=label, x=paths["S"][:n, step], y=np.asarray(prices), v=v, style=STYLES[style])


def print_results(title, panels, comparisons, n_print_points):
    """Mean price of every series and RMSE between pairs at each checkpoint,
    then the first ``n_print_points`` test points. MC prices exist for the
    first paths only, so pairs are compared on the common leading paths."""
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    summary = {}
    for p in panels:
        s = {x["label"]: x for x in p["series"]}
        col = {f"mean {k}": x["y"].mean() for k, x in s.items()}
        for a, b in comparisons:
            n = min(len(s[a]["y"]), len(s[b]["y"]))
            col[f"RMSE {a} vs {b}"] = np.sqrt(np.mean((s[a]["y"][:n] - s[b]["y"][:n]) ** 2))
        summary[p["title"]] = col
    with pd.option_context("display.width", 250, "display.max_columns", 50,
                           "display.float_format", "{:.4f}".format):
        print(pd.DataFrame(summary).to_string())
        for p in panels:
            n = min([n_print_points] + [len(x["y"]) for x in p["series"]])
            if n == 0:
                continue
            table = {}
            for x in p["series"]:
                table[f"{x['label']} spot"] = x["x"][:n]
                table[x["label"]] = x["y"][:n]
            print(f"\n-- {p['title']}: first {n} test points")
            print(pd.DataFrame(table).to_string(index=False))


# ======================================================================
# 1. Vanilla payoffs under GBM (Figures 6-10)
# ======================================================================
def run_vanilla(figure, caption, payoff_kind, S0, K, r, sigma, n_days, days_per_year, n_train,
                n_test, checkpoints, digital_smoothing, tdbp, mc, mc_points, n_print_points,
                seed, verbose, block):
    rng = np.random.default_rng(seed)
    n_steps, dt = n_days, 1.0 / days_per_year
    sim = Simulator(GBM(r, sigma), n_steps, dt, dividends=[], dpl_limit=None, dpl_boundary=None,
                    dpl_rho=0.0, steps_per_day=1)
    payoff = Payoff(payoff_kind, K, digital_smoothing, barrier=None, direction=None, bridge=None)
    train = sim.simulate(n_train, rng, S0=S0)
    test = sim.simulate(n_test, rng, S0=S0)

    td = TDBPModel(1, payoff, n_steps, dt, r, K, **tdbp).fit(train, verbose)
    mc_engine = MonteCarlo(sim, payoff, **mc)
    bs = BlackScholes(r, sigma)

    panels = []
    for step in checkpoints:
        tau = (n_steps - step) * dt
        panels.append(dict(step=step, title=f"t = t_{step} (day {step})", series=[
            series("Analytical", test, step, bs.price(payoff_kind, test["S"][:, step], K, tau), "ref"),
            series("MC", test, step, mc_engine.price_paths(test, step, mc_points), "mc"),
            series("TD", test, step, td.predict(test, step), "td"),
        ]))
    title = f"{figure}: European Options - {caption}"
    print_results(title, panels, [("TD", "Analytical"), ("MC", "Analytical"), ("TD", "MC")],
                  n_print_points)
    plotting.checkpoint_grid(panels, title, len(panels), 1, "Spot", "Price", block)


# ======================================================================
# 2. Barrier options (Figures 11-24)
# ======================================================================
def sample_hit_ratio(sim, payoff, n, rng, S0, target_hit_ratio, oversample, importance_weights):
    """Training set in which ``target_hit_ratio`` of the paths hit the barrier
    (slides 28-30). Importance weights p_natural / p_sample keep the weighted
    regression unbiased; without them the resampling biases the targets."""
    pool = sim.simulate(n * oversample, rng, S0=S0)
    hit = ~payoff.alive(pool["S"])[:, -1]
    p_nat = hit.mean()
    n_hit = int(round(target_hit_ratio * n))
    idx = np.concatenate([
        rng.choice(np.flatnonzero(hit), n_hit, replace=n_hit > hit.sum()),
        rng.choice(np.flatnonzero(~hit), n - n_hit, replace=n - n_hit > (~hit).sum()),
    ])
    idx = rng.permutation(idx)
    if importance_weights:
        w = np.where(hit[idx], p_nat / target_hit_ratio, (1 - p_nat) / (1 - target_hit_ratio))
    else:
        w = np.ones(n)
    print(f"  hit ratio: natural {p_nat:.3f} -> sample {n_hit / n:.3f}")
    return {k: v[idx] for k, v in pool.items()}, w


def run_barrier(figure, kind, barrier, S0, K, r, sigma, n_days, days_per_year, bar, hours_per_day,
                bridge_monitoring, continuity_correction, control_ratio, target_hit_ratio,
                oversample, importance_weights, n_train, n_test, checkpoint_days, tdbp, mc,
                mc_points, n_print_points, seed, verbose, block):
    rng = np.random.default_rng(seed)
    if bar is None:                                      # daily grid
        n_steps, dt = n_days, 1.0 / days_per_year
    else:                                                # slide 30: dt = 1 / (252 * 6.5 / bar)
        n_steps = int(round(n_days * hours_per_day / bar))
        dt = bar / (days_per_year * hours_per_day)
    sim = Simulator(GBM(r, sigma), n_steps, dt, dividends=[], dpl_limit=None, dpl_boundary=None,
                    dpl_rho=0.0, steps_per_day=1)
    option = "put" if kind == "DOP" else "call"
    direction = "down" if kind == "DOP" else "up"
    payoff = Payoff(option, K, None, barrier, direction,
                    bridge=(sigma, dt) if bridge_monitoring else None)

    if control_ratio:
        train, weights = sample_hit_ratio(sim, payoff, n_train, rng, S0, target_hit_ratio,
                                          oversample, importance_weights)
    else:
        train, weights = sim.simulate(n_train, rng, S0=S0), None
        print(f"  natural hit ratio {1 - payoff.alive(train['S'])[:, -1].mean():.3f}")
    test = sim.simulate(n_test, rng, S0=S0)
    train["alive"] = payoff.alive(train["S"]).astype(float)
    test["alive"] = payoff.alive(test["S"]).astype(float)

    def features(paths, i):          # slide 27: hit mask as an extra input
        S = paths["S"][:, i]
        return np.column_stack([np.log(S / K), np.log(S / barrier), paths["alive"][:, i]])

    td = TDBPModel(features, 3, payoff, n_steps, dt, r, K, **tdbp).fit(train, weights, verbose)
    mc_engine = MonteCarlo(sim, payoff, **mc)
    bs = BlackScholes(r, sigma)
    # Broadie-Glasserman-Kou shift of the barrier for discrete monitoring
    shift = 0.5826 * sigma * np.sqrt(dt) if continuity_correction else 0.0
    H = barrier * np.exp(-shift if direction == "down" else shift)

    panels = []
    for day in checkpoint_days:
        step = int(round(day * n_steps / n_days))
        tau = (n_steps - step) * dt
        panels.append(dict(step=step, title=f"day {day:g} (step {step})", series=[
            series("Analytical", test, step, bs.knock_out(kind, test["S"][:, step], K, H, tau), "ref"),
            series("MC", test, step, mc_engine.price_paths(test, step, mc_points), "mc"),
            series("TD", test, step, td.predict(test, step), "td"),
        ]))
    grid_name = "daily" if bar is None else f"hourly (bar = {bar:g})"
    title = (f"{figure}: {kind} price with B = {barrier:g} on {grid_name} discretisation "
             f"{'with' if control_ratio else 'without'} controlling ratio")
    print_results(title, panels, [("TD", "Analytical"), ("MC", "Analytical"), ("TD", "MC")],
                  n_print_points)
    plotting.checkpoint_grid(panels, title, len(panels), 1, "Spot", "Price", block)


def show_first_hitting_time(figure, kind, barrier, S0, r, sigma, n_days, days_per_year, n_paths,
                            seed, block):
    sim = Simulator(GBM(r, sigma), n_days, 1.0 / days_per_year, dividends=[], dpl_limit=None,
                    dpl_boundary=None, dpl_rho=0.0, steps_per_day=1)
    paths = sim.simulate(n_paths, np.random.default_rng(seed), S0=S0)
    plotting.first_hitting_time(paths["S"], np.arange(n_days + 1), barrier,
                                "down" if kind == "DOP" else "up",
                                f"{figure}: Two Sample-Paths, Barrier and Hitting Times", block)


# ======================================================================
# 3. Discrete dividends (Figures 25-36)
# ======================================================================
def run_dividends(figures, model, dividend, option, alpha, cash, div_day, S0, K, mu, sigma,
                  merton_heston, bates_u_max, bates_n_u, n_days, days_per_year, n_train, n_test,
                  checkpoints, steps_3d, tdbp, mc, mc_points, n_print_points, seed, verbose, block):
    rng = np.random.default_rng(seed)
    r, n_steps, dt = mu, n_days, 1.0 / days_per_year         # mu = r - q with q = 0
    amount = alpha if dividend == "proportional" else cash
    if model == "bs":
        process, with_var, ref_name, model_name = GBM(r, sigma), False, "BS", "GBM"
        bs = BlackScholes(r, sigma)
    else:
        process = MertonHeston(r, **merton_heston)
        with_var, ref_name, model_name = True, "Bates-CF", "Merton-Heston"
        bates = BatesFourier(process, bates_u_max, bates_n_u)
    sim = Simulator(process, n_steps, dt, dividends=[(div_day, amount, dividend)], dpl_limit=None,
                    dpl_boundary=None, dpl_rho=0.0, steps_per_day=1)
    payoff = Payoff(option, K, None, barrier=None, direction=None, bridge=None)

    train = sim.simulate(n_train, rng, S0=S0)
    test = sim.simulate(n_test, rng, S0=S0)
    train_Y, test_Y = {**train, "S": train["Y"]}, {**test, "S": test["Y"]}   # pure process Y

    def features(paths, i):
        x = [np.log(paths["S"][:, i] / K)]
        if with_var:
            x.append(np.maximum(paths["V"][:, i], 0.0))
        return np.column_stack(x)

    n_feat = 2 if with_var else 1
    if verbose:
        print("  training TDBP on S (with dividend)")
    td_S = TDBPModel(features, n_feat, payoff, n_steps, dt, r, K, **tdbp).fit(train, None, verbose)
    if verbose:
        print("  training TDBP on Y (without dividend)")
    td_Y = TDBPModel(features, n_feat, payoff, n_steps, dt, r, K, **tdbp).fit(train_Y, None, verbose)
    mc_engine = MonteCarlo(sim, payoff, **mc)

    def reference(paths, step):        # BS / Bates formula, ignoring the dividend (as on the slides)
        tau = (n_steps - step) * dt
        if with_var:
            return bates.price(option, paths["S"][:, step], paths["V"][:, step], K, tau)
        return bs.price(option, paths["S"][:, step], K, tau)

    def title_of(step):
        if step == div_day - 1:
            tag = "1 day before dividend payment"
        elif step == div_day + 1:
            tag = "1 day after dividend payment"
        else:
            tag = "before dividend" if step < div_day else "after dividend"
        return f"t={step / n_days:.2g}*T ({tag})"

    ref_with, ref_without = f"{ref_name} (with dividend)", f"{ref_name} (without dividend)"
    panels = []
    for step in checkpoints:
        panels.append(dict(step=step, title=title_of(step), series=[
            series(ref_with, test, step, reference(test, step), "ref", with_var),
            series(ref_without, test_Y, step, reference(test_Y, step), "ref_alt", with_var),
            series("MC (with dividend)", test, step, mc_engine.price_paths(test, step, mc_points),
                   "mc", with_var),
            series("TD (with dividend)", test, step, td_S.predict(test, step), "td", with_var),
            series("TD (without dividend)", test_Y, step, td_Y.predict(test_Y, step), "td_alt", with_var),
        ]))

    title = (f"Option price comparison, {option.upper()}, {model_name}, ttm={n_days} days, "
             f"strike={K:g}, {dividend} dividend")
    print_results(title, panels, [("TD (with dividend)", "MC (with dividend)"),
                                  ("TD (without dividend)", ref_without)], n_print_points)
    if "paths" in figures:
        plotting.with_without(np.arange(n_steps + 1), test["S"][0], test["Y"][0],
                              f"{figures['paths']}: Stock price in the {model_name} model "
                              f"with and without {dividend} dividend", block)
    if "side" in figures:
        plotting.checkpoint_grid(panels, f"{figures['side']}: {title}", 3, 2,
                                 "share price, S", "option price", block)
    if "3d" in figures:
        plotting.checkpoint_grid_3d(panels, f"{figures['3d']}: {title} (3D)", steps_3d,
                                    [ref_with, "MC (with dividend)", "TD (with dividend)"], block)
    if "3d_nodiv" in figures:
        plotting.checkpoint_grid_3d(panels, f"{figures['3d_nodiv']}: {title} (3D, without dividend)",
                                    steps_3d, [ref_without, "TD (without dividend)"], block)


# ======================================================================
# Main: every parameter is defined here
# ======================================================================
def main():
    # ---------------------------------------------------------------- run options
    SEED = 0
    VERBOSE = True
    BLOCK_ON_PLOTS = True             # True: close each figure window to continue
    N_PRINT_POINTS = 5                # test points printed per checkpoint (0 = summary only)

    RUN_LEARNING_RATE_FIGURE = True   # Figure 5
    RUN_VANILLA = True                # Figures 6-10
    RUN_BARRIER = False                # Figures 11-24
    RUN_DIVIDENDS = False              # Figures 25-36

    # ---------------------------------------------- TDBP networks (fast-run sizes)
    TDBP = dict(
        depth=3,                      # hidden layers (slide 21)
        hidden=32,                    # neurons per hidden layer (slide 21)
        activation="leaky_relu",      # leaky_relu | relu | tanh | softplus
        optimizer="adam",             # adam | adabelief | amsgrad | sgd
        schedule="exponential",       # exponential | cosine | exp_cosine | constant
        lr=1e-2,                      # initial learning rate (slide 19)
        batch_size=64,                # slide 21
        epochs=30,                    # first (maturity) network; slides use 120
        warm_start_epochs=8,          # networks warm-started from the next step
        td_lambda=0.0,                # 0 = TD(0) bootstrapping, 1 = Monte Carlo targets
        hard_knockout=True,           # knocked-out barrier paths are worth 0
        seed=SEED,
    )

    # ------------------------------------------- nested Monte Carlo (fast-run sizes)
    MC = dict(
        n_paths=20_000,               # simulations per MC price; slides use 200,000
        chunk=20_000,                 # paths simulated at once (memory control)
        antithetic=True,
        seed=12345,
    )
    MC_POINTS = 30                    # test points re-priced by MC at each checkpoint

    common = dict(mc=MC, mc_points=MC_POINTS, n_print_points=N_PRINT_POINTS, seed=SEED,
                  verbose=VERBOSE, block=BLOCK_ON_PLOTS)

    # ---------------------------------------------------- Figure 5: learning rates
    if RUN_LEARNING_RATE_FIGURE:
        plotting.learning_rates("Figure 5: Exponential, cosine and exponential-cosine decay",
                                1000, TDBP["lr"], BLOCK_ON_PLOTS)

    # ------------------------------------------------- 1. vanilla payoffs (slide 18)
    if RUN_VANILLA:
        VANILLA_CASES = [
            # (figure, payoff, caption)
            ("Figure 6", "forward", "Forward price"),
            ("Figure 7", "call", "European call price"),
            ("Figure 8", "put", "European put price"),
            ("Figure 9", "digital_bond", "Digital Bond S_T > K price"),
            ("Figure 10", "digital_share", "Digital Share S_T > K price"),
        ]
        for figure, payoff, caption in VANILLA_CASES:
            run_vanilla(
                figure=figure, caption=caption, payoff_kind=payoff,
                S0=50.0,
                K=50.0,
                r=0.1,
                sigma=0.3,
                n_days=120,                     # N_d = 120 daily steps
                days_per_year=252.0,            # dt = 1/252
                n_train=3000,
                n_test=120,
                checkpoints=(0, 1, 25, 115),    # I = {0, 1, 25, 115}
                digital_smoothing=None,         # e.g. 0.5 to smooth digital payoffs (slide 27)
                tdbp=TDBP,
                **common,
            )

    # ------------------------------------------ 2. barrier options (slides 30-43)
    if RUN_BARRIER:
        show_first_hitting_time(figure="Figure 11", kind="DOP", barrier=42.0, S0=50.0, r=0.1,
                                sigma=0.3, n_days=120, days_per_year=252.0, n_paths=200,
                                seed=SEED, block=BLOCK_ON_PLOTS)
        BARRIER_CASES = [
            # (figure, kind, barrier, bar (None = daily), control hit ratio)
            ("Figure 12", "DOP", 42.0, None, False),
            ("Figure 13", "DOP", 42.0, None, True),
            ("Figure 14", "DOP", 42.0, 3, False),
            ("Figure 15", "DOP", 42.0, 3, True),
            ("Figure 16", "DOP", 42.0, 1, False),
            ("Figures 17-18", "DOP", 42.0, 1, True),     # same caption on both slides
            ("Figure 19", "UOC", 58.0, None, False),
            ("Figure 20", "UOC", 58.0, None, True),
            ("Figure 21", "UOC", 58.0, 3, False),
            ("Figure 22", "UOC", 58.0, 3, True),
            ("Figures 23-24", "UOC", 58.0, 1, True),     # same caption on both slides
        ]
        for figure, kind, barrier, bar, control in BARRIER_CASES:
            run_barrier(
                figure=figure, kind=kind, barrier=barrier, bar=bar, control_ratio=control,
                S0=50.0,
                K=50.0,
                r=0.1,
                sigma=0.3,
                n_days=120,
                days_per_year=252.0,
                hours_per_day=6.5,              # hourly bars: dt = 1 / (252 * 6.5 / bar)
                bridge_monitoring=False,        # True: Brownian-bridge survival p_i (slide 15)
                continuity_correction=True,     # BGK shift of the analytical barrier
                target_hit_ratio=0.5,           # share of hitting paths when controlling
                oversample=5,                   # pool size = oversample * n_train
                importance_weights=True,        # False = naive resampling
                n_train=3000,
                n_test=120,
                checkpoint_days=(0, 1, 25, 115),
                tdbp=TDBP,
                **common,
            )

    # ---------------------------------------- 3. discrete dividends (slides 50-66)
    if RUN_DIVIDENDS:
        MERTON_HESTON = dict(           # Table 2
            kappa=2.5,
            theta=0.09,
            xi=0.2,
            rho=-0.6,
            v0=0.1,                     # V_{t0} (variance)
            lam=0.1,
            mu_j=-0.1,
            sigma_j=0.14,
        )
        DIVIDEND_CASES = [
            # (figures, model, dividend type, option)
            ({"paths": "Figure 25", "side": "Figure 26"}, "bs", "proportional", "call"),
            ({"side": "Figure 27"}, "bs", "proportional", "put"),
            ({"paths": "Figure 28", "side": "Figure 29"}, "bs", "cash", "call"),
            ({"side": "Figure 30"}, "bs", "cash", "put"),
            ({"paths": "Figure 31", "side": "Figure 32", "3d": "Figure 33"},
             "merton_heston", "proportional", "call"),
            ({"side": "Figure 34", "3d": "Figure 35", "3d_nodiv": "Figure 36"},
             "merton_heston", "cash", "call"),
        ]
        for figures, model, dividend, option in DIVIDEND_CASES:
            run_dividends(
                figures=figures, model=model, dividend=dividend, option=option,
                alpha=0.05,                     # proportional dividend alpha * S
                cash=2.5,                       # cash dividend
                div_day=60,                     # t_i = 60 delta
                S0=50.0,
                K=50.0,
                mu=0.05,                        # mu = r - q (Tables 1-2), read as r = 0.05
                sigma=0.3,                      # BS volatility (Table 1)
                merton_heston=MERTON_HESTON,
                bates_u_max=200.0,              # Fourier integration range
                bates_n_u=4000,                 # Fourier integration points
                n_days=100,                     # T = 100 days
                days_per_year=252.0,
                n_train=1000,                   # in-sample DS_I
                n_test=500,                     # out-sample DS_0
                checkpoints=(0, 20, 59, 61, 70, 90),
                steps_3d=(0, 59, 61, 90),       # panels of the 3D figures
                tdbp=TDBP,
                **common,
            )

if __name__ == "__main__":
    main()

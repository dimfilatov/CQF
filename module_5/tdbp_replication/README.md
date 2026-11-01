# TDBP replication: vanilla, barriers, discrete dividends, daily price limits

Python (TensorFlow) implementation of the frameworks in *Reinforcement Learning
Module: Applications* (D. Bloch, slides 1-114), with one run per result figure.
**The code has not been run.** Only syntax, lint and signature checks were done.

## Layout

```
run.py              main() with every parameter; one function per section
                    (run_vanilla, run_barrier, run_dividends, run_dpl), feature functions
tdbp/
  processes.py      GBM, MertonHeston, Simulator (discrete dividends, daily price limits)
  payoffs.py        Payoff (forward, call, put, digitals, knock-out barrier)
  pricing.py        BlackScholes (+ knock-out closed form), BatesFourier, MonteCarlo
  network.py        learning_rate schedules, Optimizer (Adam/AMSGrad/AdaBelief/SGD), TDBPModel
  plotting.py       slide-style figures, shown on screen
```

The package has no `__init__.py`, config file, dataclasses, abstract methods,
`__call__`, properties, class methods or static methods. Simulated paths are a
plain dict of arrays (`"S"`, `"Y"`, `"V"`, and the price-limit state). The time
grid is two numbers, `n_steps` and `dt`.

TensorFlow is used only to train the networks. Simulation, Monte Carlo and
reference pricing use NumPy/SciPy. The slides use Optax, so its Adam / AdaBelief
/ AMSGrad updates and exponential / cosine schedules are written in TensorFlow
with the same formulas.

## Running

Run from the folder containing `run.py`:

```
pip install -r requirements.txt
python run.py
```

Every parameter is in `main()`:

* run options: `SEED`, `VERBOSE`, `BLOCK_ON_PLOTS`, `N_PRINT_POINTS`, and
  `RUN_LEARNING_RATE_FIGURE` / `RUN_VANILLA` / `RUN_BARRIER` / `RUN_DIVIDENDS` /
  `RUN_DPL` to choose sections;
* network and training settings (`TDBP_VANILLA`, `TDBP_BARRIER`, `TDBP_DIVIDEND`,
  `TDBP_DPL`) and Monte Carlo settings (`MC`, `MC_POINTS`);
* the model inputs of each section and the case lists with figure numbers
  (`VANILLA_CASES`, `BARRIER_CASES`, `DIVIDEND_CASES`, `DPL_CASES`).

Sizes are set for a fast run: 30 epochs for the first network and 8 for
warm-started ones (slides: 120), 20,000 Monte Carlo paths (slides: 200,000), 30
Monte Carlo test points per checkpoint. Training-set sizes and model inputs
follow the slides.

Nothing is written to disk. For each figure the console shows a summary per
checkpoint (mean price of each series, RMSE between pairs such as TD vs
Analytical and TD vs MC) and the first `N_PRINT_POINTS` test points. For DPL it
also shows the log-return moments and the price limit effect (PLE) table.
Figures open with `plt.show()`. With `BLOCK_ON_PLOTS = True`, close each window
to continue.

## Figure map

| Slides | Figures | In `main()` |
|---|---|---|
| Learning rates | 5 | `RUN_LEARNING_RATE_FIGURE` |
| Vanilla (S0=50, K=50, r=0.1, sigma=0.3, 120 days, I={0,1,25,115}) | 6-10 | `VANILLA_CASES` |
| First hitting time | 11 | `show_first_hitting_time` |
| DOP, B=42 | 12-18 | `BARRIER_CASES` (daily, bar=3, bar=1; with/without ratio control) |
| UOC, B=58 | 19-24 | `BARRIER_CASES` |
| BS, proportional / cash dividend at day 60 | 25-30 | `DIVIDEND_CASES` ("bs", ...) |
| Merton-Heston, proportional / cash | 31-36 | `DIVIDEND_CASES` ("merton_heston", ...) |
| DPL absorbing / instant / slow, 0.5/2/5/7% | 37-72 | `DPL_CASES` |

## Assumptions where the slides are silent or ambiguous

1. **Measure / drift.** Paths are simulated under Q with drift r. The drift
   mu = 0.05 in Tables 1-14 is read as r = 0.05 with no dividend yield.
2. **Loss.** Slide 21 says "the loss function is the leaky ReLU". Leaky ReLU is
   used as the activation and the loss is MSE.
3. **TD targets.** One network per step is trained backwards on
   `e^{-r dt} * U_{i+1}` with TD(lambda) returns. lambda = 0 (pure
   bootstrapping) by default; 1 regresses on realised payoffs.
4. **Barrier levels.** B = 42 (DOP) and B = 58 (UOC) from the captions. The
   slide 30 text says ratios 0.8 and 1.2, which would give 40 and 60.
5. **Barrier references.** Reiner-Rubinstein with the Broadie-Glasserman-Kou
   shift for discrete monitoring. Like the slides, BS and MC re-price from the
   current spot while TDBP keeps knocked-out paths at 0.
6. **Ratio control.** Target hit ratio 0.5. Importance weights keep the
   regression unbiased (`importance_weights=False` for the naive version).
7. **Dividends.** The "BS (with/without dividend)" series are the plain formula
   on S and on Y, as on the slides. Y is the same shocks without the jump.
8. **Merton-Heston.** V_{t0} = 0.1 is a variance (the 3D axes are labelled V_t
   and start at 0.10). The reference is the Bates characteristic-function price
   where the slides use price expansions. Network size not given; 4 x 64.
9. **DPL time step.** One-hour steps, 8 per day, 160 in total,
   dt = 1/(252*24). This reproduces BS of about 1.65 at t=0 in Figure 39.
10. **Slow reflection.** rho = mean extra hours stuck on the boundary (1). After
    a hit, the rest of the day's increments change sign.
11. **DPL network.** Size not given; 5 x 64. Inputs: log(S/K), log(S/S_open)
    and the boundary state.
12. **Test sets for dividends and DPL** are not given (500). Nested MC prices the
    first 30 test points per checkpoint.
13. **Figures 18 and 24** repeat the captions of 17 and 23 and map to the same runs.

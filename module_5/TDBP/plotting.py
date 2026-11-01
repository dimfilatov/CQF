"""Slide-style figures, shown on screen.

A panel is {"step", "title", "series"}; a series is
{"label", "x", "y", "v", "style"} (v = variance for 3D views, or None).
block=True waits for each window to be closed.
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.stats import norm

from network import learning_rate

# slide colours: black = BS / analytical, blue or green = MC, red = TD
STYLES = {
    "ref": dict(marker="o", facecolors="none", edgecolors="black", s=28, linewidths=0.8),
    "ref_alt": dict(marker="x", color="black", s=14, linewidths=0.6),
    "mc": dict(marker="o", color="blue", s=8),
    "mc_green": dict(marker="o", color="green", s=8),
    "td": dict(marker="o", color="red", s=6),
    "td_alt": dict(marker="o", color="green", s=6),
}


def show(fig, block):
    fig.tight_layout()
    plt.show(block=block)
    if not block:
        plt.pause(0.1)


def checkpoint_grid(panels, title, rows, cols, xlabel, ylabel, block, labels=None):
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 2.6 * rows), squeeze=False)
    for ax, p in zip(axes.ravel(), panels):
        for s in p["series"]:
            if labels is None or s["label"] in labels:
                ax.scatter(s["x"], s["y"], label=s["label"], **s["style"])
        ax.set_title(p["title"], fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    show(fig, block)


def checkpoint_grid_3d(panels, title, steps, labels, block):
    fig = plt.figure(figsize=(11, 9))
    for k, p in enumerate([p for p in panels if p["step"] in steps][:4]):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        for s in p["series"]:
            if s["v"] is not None and s["label"] in labels:
                color = s["style"].get("color", s["style"].get("edgecolors"))
                ax.scatter(s["x"], s["v"], s["y"], color=color, s=6, label=s["label"])
        ax.set_xlabel("$S_t$")
        ax.set_ylabel("$V_t$")
        ax.set_zlabel("price")
        ax.set_title(p["title"], fontsize=9)
    fig.suptitle(title)
    show(fig, block)


def learning_rates(title, total_steps, lr, block):
    steps = tf.range(total_steps, dtype=tf.float32)
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for ax, schedule in zip(axes, ["exponential", "cosine", "exp_cosine"]):
        ax.plot(steps.numpy(), learning_rate(schedule, lr, steps, total_steps).numpy(), color="blue")
        ax.set_ylabel("Learning Rate")
        ax.set_title(schedule, fontsize=9)
    axes[-1].set_xlabel("Epochs/Steps")
    fig.suptitle(title)
    show(fig, block)


def first_hitting_time(S, days, barrier, direction, title, block):
    """Two sample paths (one hitting the barrier), the barrier and the hitting time."""
    inside = S > barrier if direction == "down" else S < barrier
    hit = ~inside.all(axis=1)
    i_hit = int(np.argmax(hit))
    i_miss = int(np.argmax(~hit))
    k = int(np.argmax(~inside[i_hit])) if hit.any() else S.shape[1] - 1
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(barrier, color="red", label="Barrier")
    ax.plot(days[: k + 1], S[i_hit, : k + 1], color="black", label="Path i")
    ax.plot(days, S[i_miss], color="blue", label="Path j")
    ax.axvline(days[k], color="m", lw=0.8)
    ax.annotate(f"$T_i^* = {days[k]:.0f}$ days", (days[k], S[i_hit, k]))
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.legend()
    show(fig, block)


def with_without(days, with_div, without_div, title, block):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(days, with_div)
    axes[0].set_title("with dividend", fontsize=9)
    axes[1].plot(days, without_div)
    axes[1].set_title("without dividend", fontsize=9)
    fig.suptitle(title)
    show(fig, block)


def dpl_trajectories(S, Y, steps_per_day, n_days, limit, title, block):
    m = n_days * steps_per_day
    hours = np.arange(m + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hours, Y[: m + 1], color="blue", label="no daily price limits")
    ax.plot(hours, S[: m + 1], color="red", label="with daily price limits")
    for d in range(n_days):
        start = d * steps_per_day
        ax.axvline(start, color="red", ls="--", lw=0.5)
        ax.hlines([S[start] * (1 - limit), S[start] * (1 + limit)], start, start + steps_per_day,
                  colors="red", linestyles="--", lw=0.8)
    ax.set_xlabel("hours")
    ax.set_title(title)
    ax.legend()
    show(fig, block)


def return_distributions(no_dpl, with_dpl, title, left_label, right_label, bins, block):
    horizons = sorted(no_dpl, reverse=True)
    fig, axes = plt.subplots(len(horizons), 2, figsize=(11, 2.4 * len(horizons)), squeeze=False)
    for row, h in enumerate(horizons):
        mu0, sd0 = no_dpl[h].mean(), no_dpl[h].std()
        xs = np.linspace(mu0 - 4 * sd0, mu0 + 4 * sd0, 400)
        for col, (data, label) in enumerate([(no_dpl[h], left_label), (with_dpl[h], right_label)]):
            ax = axes[row, col]
            ax.hist(data, bins=bins, density=True, alpha=0.6, label="data")
            ax.plot(xs, norm.pdf(xs, data.mean(), max(data.std(), 1e-12)), color="black", label="norm fit")
            if col == 1:
                ax.plot(xs, norm.pdf(xs, mu0, sd0), "b--", label="fit without dpl")
            ax.set_title(f"{label}, T={h}", fontsize=8)
            ax.set_ylabel("density", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].legend(fontsize=7)
    fig.suptitle(title)
    show(fig, block)

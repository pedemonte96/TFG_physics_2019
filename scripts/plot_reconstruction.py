"""Plot archived 4 x 4 Ising coupling reconstructions at three temperatures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def main():
    """Recreate the portfolio figure and report relative reconstruction errors."""
    root = Path(__file__).resolve().parents[1]
    data = root / "harmonic_temp_annealing/square_bonds_python/Plots"
    output = root / "output/coupling-reconstruction.png"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharex=True, sharey=True)
    for index, (ax, temperature) in enumerate(zip(axes, (0.5, 1.08304286, 2.0)), 1):
        original, inferred = np.loadtxt(data / f"M25000_L4_T{index}_S1_j.txt").T
        error = np.linalg.norm(inferred - original) / np.linalg.norm(original)
        print(f"T{index}: relative error = {error:.6f}")
        ax.plot([-1, 1], [-1, 1], "--", color="#64748b", linewidth=1, label="Exact recovery")
        ax.scatter(
            original, inferred, s=40, color="#0369a1", edgecolors="white", linewidths=0.5, zorder=3
        )
        ax.set(
            title=f"T{index}: T = {temperature:.3g}\nRelative error: {error:.2%}",
            xlabel="Original coupling",
            xlim=(-1.05, 1.05),
            ylim=(-1.05, 1.05),
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Inferred coupling")
    axes[-1].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("Recovering Ising interactions from spin observations", fontsize=15)
    fig.text(
        0.5,
        0.02,
        "4 x 4 periodic lattice · 25,000 configurations · bond sample 1 · archived simulated annealing results",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"Saved {output.relative_to(root)}")


if __name__ == "__main__":
    main()

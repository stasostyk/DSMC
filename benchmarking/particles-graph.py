import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def ms_formatter(x, _pos):
    return f"{x:,.0f}"


def main():
    particles_per_cell = [10, 50, 100, 200, 300]

    sim_times_ms = [
        3_877.4610,
        10_691.5235,
        33_103.6095,
        40_275.7910,
        66_375.3129,
    ]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        particles_per_cell,
        sim_times_ms,
        marker="o",
        linewidth=2.5,
        markersize=8,
    )

    ax.set_title(
        "Simulation Scaling with Problem Size",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Particles per cell (125,000 cells)", fontsize=12)
    ax.set_ylabel("Simulation time (ms)", fontsize=12)

    ax.yaxis.set_major_formatter(FuncFormatter(ms_formatter))

    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for x, y in zip(particles_per_cell, sim_times_ms):
        ax.annotate(
            f"{y:,.1f} ms",
            xy=(x, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()

    output_file = "particle_scaling.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")


if __name__ == "__main__":
    main()
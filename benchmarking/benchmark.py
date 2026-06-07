import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt


CSV_FILE = "historical-data.csv"
OUTPUT_FILE = "benchmark_speedups.png"

TIME_COL = "simulation_loop_elapsed_time_ms"

BENCHMARK_ORDER = ["B1", "B2", "B3"]

BENCHMARK_LABELS = {
    "B1": "Initial CUDA Implementation",
    "B2": "Optimized Collisions",
    "B3": "Optimized Collisions + Improved Memory Management",
}

BENCHMARK_COLORS = {
    "B1": "#4C78A8",
    "B2": "#F58518",
    "B3": "#54A24B",
}

SIZE_ORDER = ["SMALL", "LARGE"]

SIZE_LABELS = {
    "SMALL": "1,000 cells",
    "LARGE": "125,000 cells",
}


def compute_speedups(df, size):
    subset = df[df["size"] == size].copy()

    mean_times = (
        subset.groupby("benchmark")[TIME_COL]
        .mean()
        .reindex(BENCHMARK_ORDER)
    )

    b1_time = mean_times.loc["B1"]
    speedups = b1_time / mean_times

    return mean_times, speedups


def main():
    df = pd.read_csv(CSV_FILE)

    df["benchmark"] = df["benchmark"].astype(str).str.strip().str.upper()
    df["size"] = df["size"].astype(str).str.strip().str.upper()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        sharey=True,
    )

    max_speedup = 0

    for ax, size in zip(axes, SIZE_ORDER):
        mean_times, speedups = compute_speedups(df, size)
        max_speedup = max(max_speedup, speedups.max())

        x_positions = range(len(BENCHMARK_ORDER))

        bars = ax.bar(
            x_positions,
            speedups,
            color=[BENCHMARK_COLORS[b] for b in BENCHMARK_ORDER],
            width=0.65,
        )

        ax.set_title(SIZE_LABELS[size], fontsize=14, fontweight="bold")
        ax.set_xlabel("Implementation Iteration (acceleration milestones)", fontsize=12)

        ax.set_xticks([])

        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, benchmark in zip(bars, BENCHMARK_ORDER):
            speedup = speedups.loc[benchmark]
            time_ms = mean_times.loc[benchmark]

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.04,
                f"{speedup:.2f}x\n{time_ms:,.1f} ms",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    axes[0].set_ylabel("Speedup vs initial CUDA implementation", fontsize=12)
    axes[0].set_ylim(0.8, max_speedup * 1.10)

    legend_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=BENCHMARK_COLORS[b],
            label=BENCHMARK_LABELS[b],
        )
        for b in BENCHMARK_ORDER
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10,
    )

    fig.suptitle(
        "Simulation Loop Speedup by Problem Size",
        fontsize=17,
        fontweight="bold",
    )

    plt.tight_layout(rect=(0, 0.10, 1, 0.93))

    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
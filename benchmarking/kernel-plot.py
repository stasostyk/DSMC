import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt


CSV_FILE = "historical-data.csv"
OUTPUT_FILE = "kernel_speedups_large.png"

BENCHMARK_ORDER = ["B1", "B2", "B3"]

BENCHMARK_LABELS = {
    "B1": "Initial CUDA Implementation",
    "B2": "Optimized Collisions",
    "B3": "Optimized Memory Management",
}

BENCHMARK_COLORS = {
    "B1": "#4C78A8",
    "B2": "#F58518",
    "B3": "#54A24B",
}

SIZE_TO_PLOT = "LARGE"

KERNELS = [
    {
        "title": "Collision scheme kernel",
        "b1_b2_column_contains": "no_time_counter_scheme_kernel",
        "b3_column_contains": "ntcs_work_queue_kernel",
    },
    {
        "title": "Particle movement kernel",
        "column_contains": "move_particles_kernel",
    },
    {
        "title": "Sampling accumulation kernel",
        "column_contains": "accumulate_sampling_kernel",
    },
]


def find_column(df, text):
    matches = [col for col in df.columns if text in col]

    if not matches:
        raise ValueError(f"Could not find a column containing: {text}")

    return matches[0]


def get_kernel_time_column(df, kernel, benchmark):
    if "column_contains" in kernel:
        return find_column(df, kernel["column_contains"])

    if benchmark in ["B1", "B2"]:
        return find_column(df, kernel["b1_b2_column_contains"])

    if benchmark == "B3":
        return find_column(df, kernel["b3_column_contains"])

    raise ValueError(f"Unknown benchmark: {benchmark}")


def compute_kernel_speedups(df, kernel):
    mean_times = {}

    for benchmark in BENCHMARK_ORDER:
        col = get_kernel_time_column(df, kernel, benchmark)

        subset = df[df["benchmark"] == benchmark]
        mean_times[benchmark] = subset[col].mean()

    baseline_time = mean_times["B1"]

    speedups = {
        benchmark: baseline_time / mean_times[benchmark]
        for benchmark in BENCHMARK_ORDER
    }

    return mean_times, speedups


def main():
    df = pd.read_csv(CSV_FILE)

    df["benchmark"] = df["benchmark"].astype(str).str.strip().str.upper()
    df["size"] = df["size"].astype(str).str.strip().str.upper()

    df = df[df["size"] == SIZE_TO_PLOT].copy()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 6),
        sharey=True,
    )

    max_speedup = 0

    for ax, kernel in zip(axes, KERNELS):
        mean_times, speedups = compute_kernel_speedups(df, kernel)
        max_speedup = max(max_speedup, max(speedups.values()))

        x_positions = range(len(BENCHMARK_ORDER))

        bars = ax.bar(
            x_positions,
            [speedups[b] for b in BENCHMARK_ORDER],
            color=[BENCHMARK_COLORS[b] for b in BENCHMARK_ORDER],
            width=0.65,
        )

        ax.set_title(kernel["title"], fontsize=13, fontweight="bold")
        ax.set_xlabel("Implementation", fontsize=12)

        ax.set_xticks([])

        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, benchmark in zip(bars, BENCHMARK_ORDER):
            speedup = speedups[benchmark]
            time_ns = mean_times[benchmark]

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.04,
                f"{speedup:.2f}x\n{time_ns:,.3f} ns",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    axes[0].set_ylabel("Speedup vs initial CUDA implementation", fontsize=12)

    axes[0].set_ylim(0.0, max_speedup * 1.1)

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
        "Large Problem Size Kernel Speedups",
        fontsize=17,
        fontweight="bold",
    )

    fig.text(
        0.5,
        0.89,
        "125,000 cells; y-axis starts at 0.8x to make speedup differences easier to compare",
        ha="center",
        fontsize=10,
    )

    plt.tight_layout(rect=(0, 0.10, 1, 0.88))

    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
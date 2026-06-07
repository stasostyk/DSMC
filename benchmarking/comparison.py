import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def ms_formatter(x, _pos):
    """Format milliseconds with thousands separators."""
    return f"{x:,.0f}"


def main():
    devices = [
        "AMD Ryzen 5000 CPU\n(serial)",
        "V100 PCIe3 32 GB\n(CUDA)",
        "Tesla T4\n(CUDA)",
    ]

    sim_times_ms = [
        271_675.9391,
        911.5404,
        2_413.5296,
    ]

    baseline = sim_times_ms[0]
    speedups = [baseline / t for t in sim_times_ms]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(devices, sim_times_ms, width=0.62)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(ms_formatter))

    ax.set_title("Simulation Performance Comparison", fontsize=16, fontweight="bold", pad=16)
    ax.set_ylabel("Simulation time (ms, log scale)", fontsize=12)
    ax.set_xlabel("Hardware / execution mode", fontsize=12)

    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, time_ms, speedup in zip(bars, sim_times_ms, speedups):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()

        if speedup == 1:
            label = f"{time_ms:,.1f} ms\nbaseline"
        else:
            label = f"{time_ms:,.1f} ms\n{speedup:,.1f}× faster"

        ax.text(
            x,
            y * 1.15,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()

    output_file = "performance_histogram.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")


if __name__ == "__main__":
    main()
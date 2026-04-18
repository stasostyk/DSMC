import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

filename = "cmake-build-debug/fields_avg.dat"

# columns for file format: x y n ux uy uz T
scalar_col = 6   # 2=density, 6=temperature
ux_col = 3
uy_col = 4

plot_title = "Number Density"
colorbar_label = "n"

wing_x1 = -0.5
wing_x2 = 0.5
wing_y = 0.0
wing_thickness = 0.02

levels = 8
stream_density = 1.0
cmap = "jet"


def load_structured_field(filename, scalar_col, ux_col, uy_col):
    data = np.loadtxt(filename, comments="#")

    x = data[:, 0]
    y = data[:, 1]
    s = data[:, scalar_col]
    ux = data[:, ux_col]
    uy = data[:, uy_col]

    x_unique = np.sort(np.unique(x))
    y_unique = np.sort(np.unique(y))

    nx = len(x_unique)
    ny = len(y_unique)

    x_to_i = {val: i for i, val in enumerate(x_unique)}
    y_to_j = {val: j for j, val in enumerate(y_unique)}

    S = np.full((ny, nx), np.nan)
    U = np.full((ny, nx), np.nan)
    V = np.full((ny, nx), np.nan)

    for row in data:
        xv = row[0]
        yv = row[1]
        sv = row[scalar_col]
        uv = row[ux_col]
        vv = row[uy_col]

        i = x_to_i[xv]
        j = y_to_j[yv]

        S[j, i] = sv
        U[j, i] = uv
        V[j, i] = vv

    return x_unique, y_unique, S, U, V


def main():
    x, y, S, U, V = load_structured_field(filename, scalar_col, ux_col, uy_col)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

    cf = ax.contourf(x, y, S, levels=levels, cmap=cmap)
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label(colorbar_label)

    ax.streamplot(
        x, y, U, V,
        color="k",
        density=stream_density,
        linewidth=0.5,
        arrowsize=0.8,
        arrowstyle="->"
    )

    wing = Rectangle(
        (wing_x1, wing_y - wing_thickness / 2.0),
        wing_x2 - wing_x1,
        wing_thickness,
        facecolor="black",
        edgecolor="black"
    )
    ax.add_patch(wing)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(plot_title)
    ax.set_aspect("equal")
    ax.minorticks_on()
    ax.tick_params(direction="in", which="both")

    plt.tight_layout()
    plt.savefig("advanced_plot.png", dpi=300)


if __name__ == "__main__":
    main()
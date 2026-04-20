import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

filename = "cmake-build-debug/fields_avg.dat"

# columns for file format: x y z? n ux uy uz T avg_count
z_present = True

density_col = 2
temp_col = 6
ux_col = 3
uy_col = 4

plot_title = "Number Density"
colorbar_label = "n"

wing_x1 = 0.3
wing_x2 = 0.5
wing_y = 0.5
wing_thickness = 0.02

levels = 20
stream_density = 1.0
cmap = "viridis"


def load_structured_field(filename, density_col, temp_col, ux_col, uy_col):
    data = np.loadtxt(filename, comments="#")

    # extract only a single z-slice
    if (z_present):
        z_val = 0.55;
        data = data[data[:, 2] == z_val]
        # remove z column
        data = np.delete(data, 2, axis=1)

    x = data[:, 0]
    y = data[:, 1]
    n = data[:, density_col]
    T = data[:, temp_col]
    ux = data[:, ux_col]
    uy = data[:, uy_col]

    x_unique = np.sort(np.unique(x))
    y_unique = np.sort(np.unique(y))

    nx = len(x_unique)
    ny = len(y_unique)

    x_to_i = {val: i for i, val in enumerate(x_unique)}
    y_to_j = {val: j for j, val in enumerate(y_unique)}

    N = np.full((ny, nx), np.nan)
    Tg = np.full((ny, nx), np.nan)
    U = np.full((ny, nx), np.nan)
    V = np.full((ny, nx), np.nan)


    for row in data:
        xv = row[0]
        yv = row[1]
        nv = row[density_col]
        Tv = row[temp_col]

        uv = row[ux_col]
        vv = row[uy_col]

        i = x_to_i[xv]
        j = y_to_j[yv]

        N[j, i] = nv
        Tg[j, i] = Tv
        U[j, i] = uv
        V[j, i] = vv

    return x_unique, y_unique, N, Tg, U, V

def plot_field(ax, x, y, S, U, V, title, cbar_label):
    cf = ax.contourf(x, y, S, levels=levels, cmap=cmap)
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label(cbar_label)

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
        facecolor="white",
        edgecolor="white"
    )
    ax.add_patch(wing)

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.minorticks_on()
    ax.tick_params(direction="in", which="both")

def main():
    x, y, N, T, U, V = load_structured_field(filename, density_col, temp_col, ux_col, uy_col)

    # velocity magnitude
    Vmag = np.sqrt(U**2 + V**2)

    # pressure (ideal gas)
    KB = 1.380649e-23
    P = N * KB * T


    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

    plot_field(axes[0], x, y, P, U, V, "Pressure", "p")
    plot_field(axes[1], x, y, T, U, V, "Temperature", "T")
    plot_field(axes[2], x, y, Vmag, U, V, "Velocity Magnitude", "|u|")

    for ax in axes:
        ax.set_xlabel("X")
    axes[0].set_ylabel("Y")

    plt.tight_layout()
    plt.savefig("advanced_plot.png", dpi=300)


if __name__ == "__main__":
    main()
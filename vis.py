import numpy as np
import matplotlib.pyplot as plt
import sys

def load_data(filename):
    data = np.loadtxt(filename, comments="#")

    x = data[:, 0]
    y = data[:, 1]
    n = data[:, 2]
    ux = data[:, 3]
    uy = data[:, 4]
    uz = data[:, 5]
    T = data[:, 6]

    return x, y, n, ux, uy, uz, T

def reshape_field(x, y, field):
    x_unique = np.unique(x)
    y_unique = np.unique(y)

    NX = len(x_unique)
    NY = len(y_unique)

    field_2D = field.reshape((NX, NY))

    return x_unique, y_unique, field_2D

def plot_field(x, y, field, title, cmap="viridis"):
    X, Y = np.meshgrid(y, x)

    plt.figure()
    plt.pcolormesh(X, Y, field, shading='auto', cmap=cmap)
    plt.colorbar(label=title)
    plt.xlabel("y")
    plt.ylabel("x")
    plt.title(title)
    plt.tight_layout()

def main():
    if len(sys.argv) < 2:
        print("Usage: python vis.py output.dat")
        return

    filename = sys.argv[1]

    x, y, n, ux, uy, uz, T = load_data(filename)

    U = np.sqrt(ux**2 + uy**2 + uz**2) # velocity magnitude

    xg, yg, n2D = reshape_field(x, y, n)
    _, _, T2D = reshape_field(x, y, T)
    _, _, U2D = reshape_field(x, y, U)

    # plot_field(xg, yg, n2D, "Number Density n")
    plot_field(xg, yg, T2D, "Temperature T")
    # plot_field(xg, yg, U2D, "Velocity Magnitude |u|")

    plt.savefig("fields.png")

if __name__ == "__main__":
    main()
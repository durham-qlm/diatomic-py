import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from scipy.special import sph_harm
from diatomic.operators import uncoupled_basis_iter


def _make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format for
    LineCollection. Has the form:  numlines x (points per line) x 2 (x and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


def colorline(ax, x, y, colors, linewidth=3):
    """
    Adds a line with colour (and therefore transparency) varying per (x,y) point to ax.
    """
    segments = _make_segments(x, y)
    lc = LineCollection(segments, colors=colors, linewidth=linewidth)
    ax.add_collection(lc)

    return lc


def _surface_plot(ax, fxs, fys, fzs):
    # Add axis lines
    ax_len = 0.5
    ax.plot([-ax_len, ax_len], [0, 0], [0, 0], c="0.5", lw=1, alpha=0.3)
    ax.plot([0, 0], [-ax_len, ax_len], [0, 0], c="0.5", lw=1, alpha=0.3)
    ax.plot([0, 0], [0, 0], [-ax_len, ax_len], c="0.5", lw=1, alpha=0.3)
    # Set axes limits
    ax_lim = 0.5
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)
    # Set camera position
    ax.view_init(elev=15, azim=45)  # Reproduce view
    ax.set_xlim3d(-0.2, 0.2)  # Reproduce magnification
    ax.set_ylim3d(-0.2, 0.2)  # ...
    ax.set_zlim3d(-0.2, 0.2)  # ...
    # Turn off Axes
    ax.axis("off")
    # Draw
    ax.patch.set_alpha(0.0)
    ax.plot_surface(
        fxs,
        fys,
        fzs,
        rstride=1,
        cstride=1,
        cmap=plt.get_cmap("viridis"),
        linewidth=0,
        antialiased=False,
        alpha=0.3,
        shade=False,
    )


def plot_rotational_3d(ax, mol, eigenvector, plot_res=50):
    """
    Plots eigenvectors rotational distribution in 3D, ax's projection type must be '3d'
    """
    if ax.name != "3d":
        raise TypeError("Axis must have projection type of '3d'.")

    # Polar and Azimuthal angles to Sample
    theta = np.linspace(0, np.pi, plot_res)
    phi = np.linspace(0, 2 * np.pi, plot_res)
    # Create a 2-D meshgrid of (theta, phi) angles.
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    # Calculate the unit sphere Cartesian coordinates of each (theta, phi).
    xyz = np.array(
        [
            np.sin(theta_grid) * np.sin(phi_grid),
            np.sin(theta_grid) * np.cos(phi_grid),
            np.cos(theta_grid),
        ]
    )

    f_grid = np.zeros((plot_res, plot_res), dtype=np.cdouble)

    for i, (N, MN, M1, M2) in enumerate(uncoupled_basis_iter(mol.Nmax, *mol.Ii)):
        f_grid += eigenvector[i] * sph_harm(MN, N, phi_grid, theta_grid)

    Yx, Yy, Yz = np.abs(f_grid) ** 2 * xyz  # get final output cartesian coords
    _surface_plot(ax, Yx, Yy, Yz)


def plot_rotational_2d(ax, mol, eigenvector, plot_res=200, format_axes=True):
    """
    Plots eigenvectors rotational distribution in 2d at phi=0,
    ax's projection type must be 'polar'
    """

    # Polar and Azimuthal angles to Sample
    thetas = np.linspace(-np.pi, np.pi, plot_res)
    f_grid = np.zeros((plot_res), dtype=np.cdouble)

    for i, (N, MN, M1, M2) in enumerate(uncoupled_basis_iter(mol.Nmax, *mol.Ii)):
        f_grid += eigenvector[i] * sph_harm(MN, N, 0, np.abs(thetas))

    probs = np.abs(f_grid) ** 2
    xs = np.sin(thetas) * probs
    ys = np.cos(thetas) * probs
    ax.plot(xs, ys)
    ax.axes.set_aspect("equal")
    if format_axes:
        ax.plot([-0.15, 0.15], [0, 0], c="k", alpha=0.2)
        ax.plot([0, 0], [-0.15, 0.15], c="k", alpha=0.2)
        ax.set_axis_off()

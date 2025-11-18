import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from scipy.special import sph_harm
from diatomic.operators import uncoupled_basis_iter

# from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots?


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


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs2d, ys2d, zs2d = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.get_proj())
        self.set_positions((xs2d[0], ys2d[0]), (xs2d[1], ys2d[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs2d, ys2d, zs2d = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.get_proj())
        self.set_positions((xs2d[0], ys2d[0]), (xs2d[1], ys2d[1]))
        return np.min(zs2d)


def plot_polarization_ellipse(omega, gamma, delta, n_points=400, num_arrows=6):
    """
    Plot the 3D polarization ellipse for

        epsilon = (cos(omega) cos(gamma),
                   exp(i delta) sin(gamma),
                  -sin(omega) cos(gamma))

    and draw:
      - the k vector, at angle omega to the z-axis (in the x–z plane)
      - minimalist x, y, z axes as arrows
      - a lightly shaded x–y plane (z=0)
      - the polarization plane (normal to k) in light green
      - several arrows on the ellipse showing direction as t increases

    Angles in radians.
    """

    # Complex polarization vector ε
    epsilon = np.array(
        [
            np.cos(omega) * np.cos(gamma),
            np.exp(1j * delta) * np.sin(gamma),
            -np.sin(omega) * np.cos(gamma),
        ],
        dtype=complex,
    )

    # Parameter along the ellipse (plays role of ωt)
    t = np.linspace(0, 1.8 * np.pi, n_points)

    # Electric field tip in time: E(t) = Re[ε e^{-i t}]
    phase_factor = np.exp(-1j * t)
    E = np.real(np.outer(epsilon, phase_factor))  # shape (3, n_points)
    x, y, z = E

    # Set up figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")  # orthographic projection
    ax.set_box_aspect([1, 1, 1])  # equal scaling in x, y, z

    # Plot the polarization ellipse
    ax.plot(x, y, z, linewidth=2, color="C0")

    # Symmetric range
    max_extent = max(np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z)), 1e-6)
    lim = max_extent * 1.4

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    # Lightly shaded x–y plane at z = 0
    xx, yy = np.meshgrid(np.linspace(-lim, lim, 2), np.linspace(-lim, lim, 2))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.08, color="gray", linewidth=0, shade=False)

    # k vector: angle omega w.r.t. z-axis in the x–z plane
    k_hat = np.array([np.sin(omega), 0.0, np.cos(omega)])
    k_hat /= np.linalg.norm(k_hat)
    k_len = lim * 0.9
    k_vec = k_len * k_hat

    ax.quiver(
        0,
        0,
        0,
        k_vec[0],
        k_vec[1],
        k_vec[2],
        arrow_length_ratio=0.1,
        linewidth=2,
        color="C3",
    )
    ax.text(
        k_vec[0] * 1.05, k_vec[1] * 1.05, k_vec[2] * 1.05, r"$\mathbf{k}$", fontsize=12
    )

    # ---- Polarization plane (normal to k) in light green ----
    # Build an orthonormal basis {u, v, k_hat}
    if abs(k_hat[2]) < 0.9:
        tmp = np.array([0.0, 0.0, 1.0])
    else:
        tmp = np.array([0.0, 1.0, 0.0])

    u = np.cross(k_hat, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(k_hat, u)

    s_vals = np.linspace(-lim, lim, 2)
    t_vals = np.linspace(-lim, lim, 2)
    S, T = np.meshgrid(s_vals, t_vals)

    Xp = S * u[0] + T * v[0]
    Yp = S * u[1] + T * v[1]
    Zp = S * u[2] + T * v[2]

    ax.plot_surface(
        Xp, Yp, Zp, alpha=0.12, color="lightgreen", linewidth=0, shade=False
    )

    # ---- Multiple arrows decorating the ellipse (direction of increasing t) ----
    step = n_points // num_arrows
    arrow_len = 0.2 * lim  # a bit shorter so multiple arrows don't clutter

    for i in range(0, n_points - 1 - step, step):
        p0 = np.array([x[i], y[i], z[i]])
        p1 = np.array([x[i + 1], y[i + 1], z[i + 1]])  # slightly further along

        tangent = p1 - p0
        if np.linalg.norm(tangent) == 0:
            continue
        tangent /= np.linalg.norm(tangent)

        p_arrow = p0 + arrow_len * tangent

        arrow = Arrow3D(
            [p0[0], p_arrow[0]],
            [p0[1], p_arrow[1]],
            [p0[2], p_arrow[2]],
            mutation_scale=12,
            lw=2,
            arrowstyle="-|>",
            color="C1",
        )
        ax.add_artist(arrow)

    # Minimalist coordinate axes as arrows
    axis_len = lim * 0.8

    ax.quiver(
        0, 0, 0, axis_len, 0, 0, arrow_length_ratio=0.08, linewidth=1, color="black"
    )
    ax.quiver(
        0, 0, 0, 0, axis_len, 0, arrow_length_ratio=0.08, linewidth=1, color="black"
    )
    ax.quiver(
        0, 0, 0, 0, 0, axis_len, arrow_length_ratio=0.08, linewidth=1, color="black"
    )

    ax.text(axis_len * 1.05, 0, 0, "x", fontsize=10)
    ax.text(0, axis_len * 1.05, 0, "y", fontsize=10)
    ax.text(0, 0, axis_len * 1.05, "z", fontsize=10)

    # Clean up: no grid, no ticks, no box
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

    ax.set_title("3D Polarization Ellipse with k-vector", pad=20)
    plt.tight_layout()
    plt.show()

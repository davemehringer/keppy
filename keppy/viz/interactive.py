"""
keppy.viz.interactive — Interactive animated orbit viewer.

Provides ``OrbitViewer``, a matplotlib figure with live controls:

* **Animated playback** — bodies move along their recorded trajectories
  via ``matplotlib.animation.FuncAnimation``.
* **Label toggles** — ``CheckButtons`` panel to show / hide each body's
  name annotation independently.
* **Center-body selector** — ``RadioButtons`` panel to switch the
  reference-frame origin between any body or the original (absolute)
  frame.  All trails and positions are shown relative to the chosen
  center body, mimicking keplerpp's dynamic re-centering.
* **Trail control** — optional ``trail_length`` to limit how many past
  steps are drawn as a fading path.

Usage
-----
    from keppy.viz.interactive import OrbitViewer
    import matplotlib.pyplot as plt

    viewer = OrbitViewer(records, body_names=["Sun", "Earth", "Jupiter"])
    anim   = viewer.animate(interval=40)   # returns FuncAnimation
    plt.show()

The returned ``FuncAnimation`` must be kept alive (assign to a variable)
to prevent garbage collection stopping the animation.

Requirements
------------
matplotlib >= 3.7 (``pip install "keppy[viz]"``).
"""

from __future__ import annotations

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import CheckButtons, RadioButtons
except ModuleNotFoundError as _exc:
    raise ModuleNotFoundError(
        "keppy.viz requires matplotlib.  "
        "Install it with: pip install \"keppy[viz]\""
    ) from _exc

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from keppy.timestep import StepRecord

from keppy.viz.plot import _spatial_scale, _DEFAULT_COLORS


# ---------------------------------------------------------------------------
# OrbitViewer
# ---------------------------------------------------------------------------

class OrbitViewer:
    """
    Interactive animated orbit viewer.

    Parameters
    ----------
    records      : Trajectory snapshots from :func:`~keppy.timestep.run`.
    body_names   : Display names for each body.  Defaults to
                   ``"Body 0"``, ``"Body 1"``, etc.
    unit         : Spatial unit for axis labels: ``"m"``, ``"km"``, ``"au"``.
    plane        : Projection plane: ``"xy"``, ``"xz"``, or ``"yz"``.
    trail_length : Number of past frames shown as a fading trail.
                   ``None`` (default) draws the complete history up to the
                   current frame.
    figsize      : ``(width, height)`` in inches.

    Attributes
    ----------
    fig     : The :class:`~matplotlib.figure.Figure`.
    ax      : The main orbit :class:`~matplotlib.axes.Axes`.

    Notes
    -----
    Call :meth:`animate` to start playback, then ``plt.show()``.
    Keep the returned :class:`~matplotlib.animation.FuncAnimation` object
    in scope to prevent Python from garbage-collecting the animation.
    """

    _PANEL_TITLE_FS = 8   # font size for widget panel titles
    _LABEL_FS       = 8   # font size for body name annotations

    def __init__(
        self,
        records: Sequence["StepRecord"],
        *,
        body_names:   list[str] | None = None,
        unit:         str = "au",
        plane:        str = "xy",
        trail_length: int | None = None,
        figsize:      tuple[float, float] = (13, 7),
    ) -> None:
        if not records:
            raise ValueError("records is empty.")

        self.records      = list(records)
        self.n_steps      = len(self.records)
        self.n_bodies     = self.records[0].positions.shape[0]
        self.trail_length = trail_length

        if body_names is None:
            body_names = [f"Body {i}" for i in range(self.n_bodies)]
        if len(body_names) != self.n_bodies:
            raise ValueError(
                f"body_names has {len(body_names)} entries but records have "
                f"{self.n_bodies} bodies."
            )
        self.body_names = list(body_names)

        scale, unit_label      = _spatial_scale(unit)
        self._scale            = scale
        self._unit_label       = unit_label

        plane_axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
        if plane.lower() not in plane_axes:
            raise ValueError(
                f"Unknown plane {plane!r}. Choose from: {list(plane_axes)}."
            )
        self._ix, self._iy = plane_axes[plane.lower()]
        ax_names            = ["x", "y", "z"]
        self._xlabel        = f"{ax_names[self._ix]} ({unit_label})"
        self._ylabel        = f"{ax_names[self._iy]} ({unit_label})"

        # State
        self._center_idx   = -1               # -1 = absolute frame
        self._show_labels  = [True] * self.n_bodies
        self._anim: FuncAnimation | None = None

        # Pre-compute all scaled positions: shape (n_steps, n_bodies, 3)
        self._pos_all = (
            np.stack([r.positions for r in self.records]) * scale
        )

        self._build_figure(figsize)
        self._init_artists()
        self._setup_widgets()

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------

    def _build_figure(self, figsize: tuple[float, float]) -> None:
        self.fig = plt.figure(figsize=figsize)

        # Main orbit axes (left 65 %)
        self.ax = self.fig.add_axes([0.05, 0.08, 0.60, 0.87])
        self.ax.set_xlabel(self._xlabel)
        self.ax.set_ylabel(self._ylabel)
        self.ax.set_title("Orbit Viewer — use controls →", fontsize=10)
        self.ax.set_aspect("equal")

        # Widget axes (right 30 %)
        # Upper panel: label toggles
        n = self.n_bodies
        label_h = min(0.42, 0.05 + n * 0.055)
        self._ax_labels = self.fig.add_axes(
            [0.70, 0.95 - label_h, 0.28, label_h]
        )
        # Lower panel: center selector (one extra option: "absolute")
        center_h = min(0.42, 0.05 + (n + 1) * 0.055)
        self._ax_center = self.fig.add_axes(
            [0.70, 0.95 - label_h - center_h - 0.04, 0.28, center_h]
        )

        self._update_axis_limits()

    def _update_axis_limits(self) -> None:
        """Set axis limits for the current center-body frame."""
        rel = self._relative_positions()          # (n_steps, n_bodies, 3)

        # Exclude center body from limit computation (it's always at origin)
        mask = np.ones(self.n_bodies, dtype=bool)
        if self._center_idx >= 0:
            mask[self._center_idx] = False

        pts_x = rel[:, mask, self._ix]
        pts_y = rel[:, mask, self._iy]

        margin = 0.08
        xmin, xmax = float(pts_x.min()), float(pts_x.max())
        ymin, ymax = float(pts_y.min()), float(pts_y.max())
        xr = max(xmax - xmin, 1e-30)
        yr = max(ymax - ymin, 1e-30)
        self.ax.set_xlim(xmin - margin * xr, xmax + margin * xr)
        self.ax.set_ylim(ymin - margin * yr, ymax + margin * yr)

    # ------------------------------------------------------------------
    # Artists
    # ------------------------------------------------------------------

    def _init_artists(self) -> None:
        """Create matplotlib artists for bodies (scatter + text + trail)."""
        self._scatters: list = []
        self._labels:   list = []
        self._trails:   list = []

        for bi in range(self.n_bodies):
            color = _DEFAULT_COLORS[bi % len(_DEFAULT_COLORS)]

            # Current-position dot
            sc = self.ax.scatter([], [], color=color, s=40, zorder=5)
            self._scatters.append(sc)

            # Name annotation
            txt = self.ax.text(
                0, 0, self.body_names[bi],
                fontsize=self._LABEL_FS, color=color,
                ha="left", va="bottom",
                visible=self._show_labels[bi],
            )
            self._labels.append(txt)

            # Trail line
            line, = self.ax.plot([], [], color=color, alpha=0.45,
                                 linewidth=0.9, zorder=2)
            self._trails.append(line)

        self._frame_text = self.ax.text(
            0.02, 0.97, "", transform=self.ax.transAxes,
            fontsize=8, va="top", color="gray",
        )

    # ------------------------------------------------------------------
    # Widget setup
    # ------------------------------------------------------------------

    def _setup_widgets(self) -> None:
        # ---- Label toggles ----
        self._ax_labels.set_title("Show labels", fontsize=self._PANEL_TITLE_FS)
        self._chk = CheckButtons(
            self._ax_labels,
            self.body_names,
            [True] * self.n_bodies,
        )
        self._chk.on_clicked(self._on_label_toggle)

        # ---- Center-body selector ----
        self._ax_center.set_title("Center on", fontsize=self._PANEL_TITLE_FS)
        center_opts = ["(absolute)"] + self.body_names
        self._radio = RadioButtons(self._ax_center, center_opts)
        self._radio.on_clicked(self._on_center_change)

    # ------------------------------------------------------------------
    # Widget callbacks
    # ------------------------------------------------------------------

    def _on_label_toggle(self, label: str) -> None:
        """Toggle visibility of a body name annotation."""
        idx = self.body_names.index(label)
        self._show_labels[idx] = not self._show_labels[idx]
        self._labels[idx].set_visible(self._show_labels[idx])
        self.fig.canvas.draw_idle()

    def _on_center_change(self, label: str) -> None:
        """Switch the reference-frame center body."""
        if label == "(absolute)":
            self._center_idx = -1
        else:
            self._center_idx = self.body_names.index(label)
        self._update_axis_limits()
        # Redraw trails for the new frame immediately
        if self._anim is not None:
            # Trigger a frame update at the last displayed frame
            pass
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def _relative_positions(self) -> np.ndarray:
        """
        Return all positions in the current reference frame.

        Returns
        -------
        ndarray, shape (n_steps, n_bodies, 3)
            Positions relative to the center body (or absolute if
            ``_center_idx == -1``).
        """
        pos = self._pos_all
        if self._center_idx >= 0:
            center = pos[:, self._center_idx : self._center_idx + 1, :]
            pos = pos - center
        return pos

    def _trail_slice(self, frame_idx: int) -> slice:
        """Return the slice of steps to show as a trail ending at frame_idx."""
        if self.trail_length is None:
            return slice(0, frame_idx + 1)
        start = max(0, frame_idx - self.trail_length)
        return slice(start, frame_idx + 1)

    # ------------------------------------------------------------------
    # Animation update
    # ------------------------------------------------------------------

    def _update_frame(self, frame_idx: int) -> list:
        """
        FuncAnimation callback — update all artists for *frame_idx*.

        Returns a list of changed artists (used when ``blit=True``).
        """
        rel = self._relative_positions()   # (n_steps, n_bodies, 3)
        sl  = self._trail_slice(frame_idx)
        pos = rel[frame_idx]               # (n_bodies, 3), current frame

        artists = []
        for bi in range(self.n_bodies):
            x = float(pos[bi, self._ix])
            y = float(pos[bi, self._iy])

            # Update scatter
            self._scatters[bi].set_offsets([[x, y]])

            # Update label position
            self._labels[bi].set_position((x, y))

            # Update trail
            trail = rel[sl, bi, :]
            self._trails[bi].set_data(trail[:, self._ix], trail[:, self._iy])

            artists.extend([self._scatters[bi], self._labels[bi], self._trails[bi]])

        # Frame counter
        t = self.records[frame_idx].time
        self._frame_text.set_text(
            f"step {frame_idx + 1}/{self.n_steps}  "
            f"t = {t / 86400:.1f} days"
        )
        artists.append(self._frame_text)

        return artists

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def animate(
        self,
        interval: int = 50,
        repeat:   bool = True,
    ) -> FuncAnimation:
        """
        Start the animation.

        Parameters
        ----------
        interval : Milliseconds between frames (default 50 ms → ~20 fps).
        repeat   : Loop the animation (default ``True``).

        Returns
        -------
        FuncAnimation
            **Assign the return value to a variable** to prevent Python's
            garbage collector from stopping the animation::

                anim = viewer.animate()
                plt.show()
        """
        self._anim = FuncAnimation(
            self.fig,
            self._update_frame,
            frames=self.n_steps,
            interval=interval,
            blit=False,   # blit=False keeps widgets responsive
            repeat=repeat,
        )
        return self._anim

    def show(self, interval: int = 50, repeat: bool = True) -> None:
        """
        Convenience wrapper: call :meth:`animate` then ``plt.show()``.

        Keeps the :class:`~matplotlib.animation.FuncAnimation` alive
        internally.

        Parameters
        ----------
        interval : Milliseconds between frames.
        repeat   : Loop the animation.
        """
        self.animate(interval=interval, repeat=repeat)
        plt.show()

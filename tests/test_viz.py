"""Tests for keppy.viz — matplotlib visualization functions."""

import math

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
matplotlib.use("Agg")   # non-interactive backend; must be set before any other mpl import
import matplotlib.pyplot as plt

from keppy.viz import plot_trajectory, plot_orbit, plot_system, plot_energy
from keppy.body import Body
from keppy.nbody_system import NBodySystem
from keppy.orbital import Elements
from keppy.timestep import StepRecord, run, ConstantTimeStepManager
from keppy.integrator import LeapfrogIntegrator
from keppy.acceleration import PairwiseAccelerationCalculator
from keppy.constants import AU, MU_SUN, DAY, G


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sun_earth():
    v = math.sqrt(MU_SUN / AU)
    sun   = Body.from_mu("Sun",   MU_SUN,   np.zeros(3), np.zeros(3))
    earth = Body.from_mass("Earth", 5.972e24, np.array([AU, 0., 0.]), np.array([0., v, 0.]))
    sun.radius   = 6.957e8
    earth.radius = 6.378e6
    return NBodySystem([sun, earth])


@pytest.fixture
def records(sun_earth):
    integ = LeapfrogIntegrator(PairwiseAccelerationCalculator())
    tsm   = ConstantTimeStepManager(DAY)
    return run(sun_earth, integ, tsm, t_end=30 * DAY)


@pytest.fixture
def earth_elements():
    return Elements(a=AU, e=0.0167, i=0.0, node=0.0, peri=102.9, M=100.0)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_trajectory
# ---------------------------------------------------------------------------

class TestPlotTrajectory:

    def test_returns_axes(self, records):
        ax = plot_trajectory(records)
        assert ax is not None

    def test_default_plane_xy(self, records):
        ax = plot_trajectory(records)
        assert "x" in ax.get_xlabel()
        assert "y" in ax.get_ylabel()

    def test_plane_xz(self, records):
        ax = plot_trajectory(records, plane="xz")
        assert "x" in ax.get_xlabel()
        assert "z" in ax.get_ylabel()

    def test_plane_yz(self, records):
        ax = plot_trajectory(records, plane="yz")
        assert "y" in ax.get_xlabel()
        assert "z" in ax.get_ylabel()

    def test_plane_3d(self, records):
        from mpl_toolkits.mplot3d import Axes3D
        ax = plot_trajectory(records, plane="3d")
        assert isinstance(ax, Axes3D)

    def test_unit_au(self, records):
        ax = plot_trajectory(records, unit="au")
        assert "AU" in ax.get_xlabel()

    def test_unit_km(self, records):
        ax = plot_trajectory(records, unit="km")
        assert "km" in ax.get_xlabel()

    def test_unit_m(self, records):
        ax = plot_trajectory(records, unit="m")
        assert "m" in ax.get_xlabel()

    def test_custom_title(self, records):
        ax = plot_trajectory(records, title="My orbit")
        assert ax.get_title() == "My orbit"

    def test_body_names_in_legend(self, records):
        ax = plot_trajectory(records, body_names=["Sun", "Earth"])
        texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "Sun"   in texts
        assert "Earth" in texts

    def test_wrong_body_names_length_raises(self, records):
        with pytest.raises(ValueError, match="body_names"):
            plot_trajectory(records, body_names=["Sun"])   # only 1 for 2 bodies

    def test_empty_records_raises(self):
        with pytest.raises(ValueError, match="empty"):
            plot_trajectory([])

    def test_draw_on_existing_axes(self, records):
        _, ax_existing = plt.subplots()
        ax = plot_trajectory(records, ax=ax_existing)
        assert ax is ax_existing

    def test_number_of_lines(self, records):
        """One line per body."""
        ax = plot_trajectory(records)
        # lines include mark_start scatter as well as path lines
        assert len(ax.lines) == 2   # 2 bodies → 2 trajectory lines

    def test_legend_hidden(self, records):
        ax = plot_trajectory(records, legend=False)
        assert ax.get_legend() is None

    def test_mark_start_adds_scatter(self, records):
        ax = plot_trajectory(records, mark_start=True)
        # Collections hold PathCollections from scatter
        assert len(ax.collections) >= 2

    def test_mark_start_false(self, records):
        ax = plot_trajectory(records, mark_start=False)
        assert len(ax.collections) == 0

    def test_invalid_plane_raises(self, records):
        with pytest.raises(KeyError):
            plot_trajectory(records, plane="uw")

    def test_invalid_unit_raises(self, records):
        with pytest.raises(ValueError, match="unit"):
            plot_trajectory(records, unit="parsec")

    def test_trajectory_x_range_plausible(self, records):
        """Earth x-position should span approximately ±1 AU in AU units."""
        ax = plot_trajectory(records, unit="au")
        xlim = ax.get_xlim()
        # Earth starts at ~1 AU, Sun at ~0 — range should include both
        assert xlim[0] < 0.1 and xlim[1] > 0.9


# ---------------------------------------------------------------------------
# plot_orbit
# ---------------------------------------------------------------------------

class TestPlotOrbit:

    def test_returns_axes(self, earth_elements):
        ax = plot_orbit(earth_elements, MU_SUN)
        assert ax is not None

    def test_line_drawn(self, earth_elements):
        ax = plot_orbit(earth_elements, MU_SUN)
        assert len(ax.lines) == 1

    def test_orbit_closes(self, earth_elements):
        """First and last point of the plotted curve should be identical."""
        ax = plot_orbit(earth_elements, MU_SUN, n_points=100)
        line = ax.lines[0]
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        assert xdata[0] == pytest.approx(xdata[-1], rel=1e-10)
        assert ydata[0] == pytest.approx(ydata[-1], rel=1e-10)

    def test_orbit_radius_plausible(self, earth_elements):
        """For e=0.0167, all radii should be close to 1 AU."""
        ax = plot_orbit(earth_elements, MU_SUN, unit="au")
        line = ax.lines[0]
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        radii = np.sqrt(xdata**2 + ydata**2)
        assert radii.min() == pytest.approx(1.0, abs=0.02)
        assert radii.max() == pytest.approx(1.0, abs=0.02)

    def test_unit_au(self, earth_elements):
        ax = plot_orbit(earth_elements, MU_SUN, unit="au")
        assert "AU" in ax.get_xlabel()

    def test_plane_3d(self, earth_elements):
        from mpl_toolkits.mplot3d import Axes3D
        ax = plot_orbit(earth_elements, MU_SUN, plane="3d")
        assert isinstance(ax, Axes3D)

    def test_label_in_legend(self, earth_elements):
        ax = plot_orbit(earth_elements, MU_SUN, label="Earth orbit")
        ax.legend()
        texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "Earth orbit" in texts

    def test_hyperbolic_raises(self):
        el = Elements(a=-AU, e=1.5, i=0.0, node=0.0, peri=0.0, M=float("nan"))
        with pytest.raises(ValueError, match="elliptic"):
            plot_orbit(el, MU_SUN)

    def test_plot_kwargs_forwarded(self, earth_elements):
        ax = plot_orbit(earth_elements, MU_SUN, color="red", linewidth=2.5)
        line = ax.lines[0]
        assert line.get_linewidth() == pytest.approx(2.5)

    def test_draw_on_existing_axes(self, earth_elements):
        _, ax_existing = plt.subplots()
        ax = plot_orbit(earth_elements, MU_SUN, ax=ax_existing)
        assert ax is ax_existing

    def test_multiple_orbits_on_same_axes(self, earth_elements):
        """Two successive calls on the same axes should add two lines."""
        _, ax = plt.subplots()
        ax.set_aspect("equal")
        plot_orbit(earth_elements, MU_SUN, ax=ax)
        el2 = Elements(a=1.5*AU, e=0.09, i=0.0, node=0.0, peri=30.0, M=0.0)
        plot_orbit(el2, MU_SUN, ax=ax)
        assert len(ax.lines) == 2

    def test_n_points_controls_resolution(self, earth_elements):
        ax50  = plot_orbit(earth_elements, MU_SUN, n_points=50)
        ax200 = plot_orbit(earth_elements, MU_SUN, n_points=200)
        # 50+1 = 51 points (closed), 200+1 = 201 points
        assert len(ax50.lines[0].get_xdata())  == 51
        assert len(ax200.lines[0].get_xdata()) == 201


# ---------------------------------------------------------------------------
# plot_system
# ---------------------------------------------------------------------------

class TestPlotSystem:

    def test_returns_axes(self, sun_earth):
        ax = plot_system(sun_earth)
        assert ax is not None

    def test_scatter_points_drawn(self, sun_earth):
        ax = plot_system(sun_earth)
        # Each body → one PathCollection
        assert len(ax.collections) == 2

    def test_annotations_present(self, sun_earth):
        ax = plot_system(sun_earth, annotate=True)
        texts = [t.get_text() for t in ax.texts]
        assert any("Sun"   in t for t in texts)
        assert any("Earth" in t for t in texts)

    def test_annotate_false(self, sun_earth):
        ax = plot_system(sun_earth, annotate=False)
        assert len(ax.texts) == 0

    def test_unit_au(self, sun_earth):
        ax = plot_system(sun_earth, unit="au")
        assert "AU" in ax.get_xlabel()

    def test_plane_xz(self, sun_earth):
        ax = plot_system(sun_earth, plane="xz")
        assert "z" in ax.get_ylabel()

    def test_plane_3d(self, sun_earth):
        from mpl_toolkits.mplot3d import Axes3D
        ax = plot_system(sun_earth, plane="3d")
        assert isinstance(ax, Axes3D)

    def test_custom_title(self, sun_earth):
        ax = plot_system(sun_earth, title="t = 0")
        assert ax.get_title() == "t = 0"


# ---------------------------------------------------------------------------
# plot_energy
# ---------------------------------------------------------------------------

class TestPlotEnergy:

    def _mus(self, system):
        return np.array([b.mu for b in system.bodies])

    def test_returns_axes(self, records, sun_earth):
        ax = plot_energy(records, self._mus(sun_earth))
        assert ax is not None

    def test_line_drawn(self, records, sun_earth):
        ax = plot_energy(records, self._mus(sun_earth))
        assert len(ax.lines) >= 1

    def test_relative_ylabel(self, records, sun_earth):
        ax = plot_energy(records, self._mus(sun_earth), relative=True)
        assert "E" in ax.get_ylabel()

    def test_absolute_ylabel(self, records, sun_earth):
        ax = plot_energy(records, self._mus(sun_earth), relative=False)
        assert "J" in ax.get_ylabel()

    def test_time_unit_day(self, records, sun_earth):
        ax = plot_energy(records, self._mus(sun_earth), time_unit="day")
        assert "day" in ax.get_xlabel()

    def test_time_unit_year(self, records, sun_earth):
        ax = plot_energy(records, self._mus(sun_earth), time_unit="year")
        assert "year" in ax.get_xlabel()

    def test_time_unit_s(self, records, sun_earth):
        ax = plot_energy(records, self._mus(sun_earth), time_unit="s")
        assert "s" in ax.get_xlabel()

    def test_relative_error_small_leapfrog(self, records, sun_earth):
        """Leapfrog energy error should be < 0.1% over 30 days."""
        ax   = plot_energy(records, self._mus(sun_earth), relative=True)
        ydata = ax.lines[0].get_ydata()
        assert np.max(np.abs(ydata)) < 1e-3

    def test_empty_records_raises(self, sun_earth):
        with pytest.raises(ValueError, match="empty"):
            plot_energy([], self._mus(sun_earth))

    def test_draw_on_existing_axes(self, records, sun_earth):
        _, ax_existing = plt.subplots()
        ax = plot_energy(records, self._mus(sun_earth), ax=ax_existing)
        assert ax is ax_existing

    def test_plot_kwargs_forwarded(self, records, sun_earth):
        ax = plot_energy(records, self._mus(sun_earth), color="red")
        # First line (the energy curve, not the zero line) should be red
        import matplotlib.colors as mc
        color = mc.to_hex(ax.lines[0].get_color())
        assert color == mc.to_hex("red")

    def test_zero_line_present(self, records, sun_earth):
        """A dashed zero reference line should always be drawn."""
        ax    = plot_energy(records, self._mus(sun_earth))
        linestyles = [l.get_linestyle() for l in ax.lines]
        assert "--" in linestyles

    def test_invalid_time_unit_raises(self, records, sun_earth):
        with pytest.raises(ValueError, match="time unit"):
            plot_energy(records, self._mus(sun_earth), time_unit="fortnight")

    def test_custom_title(self, records, sun_earth):
        ax = plot_energy(records, self._mus(sun_earth), title="Energy check")
        assert ax.get_title() == "Energy check"


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------

class TestPackageExports:

    def test_viz_importable(self):
        import keppy.viz
        assert hasattr(keppy.viz, "plot_trajectory")
        assert hasattr(keppy.viz, "plot_orbit")
        assert hasattr(keppy.viz, "plot_system")
        assert hasattr(keppy.viz, "plot_energy")

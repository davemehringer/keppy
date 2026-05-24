"""Tests for keppy.viz.interactive — OrbitViewer."""

import math

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not installed")
matplotlib.use("Agg")   # non-interactive backend; set before any other import
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import CheckButtons, RadioButtons

from keppy.viz.interactive import OrbitViewer
from keppy.body import Body
from keppy.nbody_system import NBodySystem
from keppy.timestep import StepRecord, run, ConstantTimeStepManager
from keppy.integrator import LeapfrogIntegrator
from keppy.acceleration import PairwiseAccelerationCalculator
from keppy.constants import AU, MU_SUN, DAY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def records():
    """30-step Sun-Earth trajectory (module-scoped for speed)."""
    v     = math.sqrt(MU_SUN / AU)
    sun   = Body.from_mu("Sun",   MU_SUN,   np.zeros(3), np.zeros(3))
    earth = Body.from_mass("Earth", 5.972e24,
                           np.array([AU, 0., 0.]), np.array([0., v, 0.]))
    sys   = NBodySystem([sun, earth])
    integ = LeapfrogIntegrator(PairwiseAccelerationCalculator())
    tsm   = ConstantTimeStepManager(DAY)
    return run(sys, integ, tsm, t_end=30 * DAY)


@pytest.fixture(scope="module")
def records_3body():
    """20-step Sun-Earth-Jupiter trajectory."""
    v_e = math.sqrt(MU_SUN / AU)
    r_j = 5.2 * AU
    v_j = math.sqrt(MU_SUN / r_j)
    sun     = Body.from_mu("Sun", MU_SUN, np.zeros(3), np.zeros(3))
    earth   = Body.from_mass("Earth",   5.972e24,
                             np.array([AU, 0., 0.]), np.array([0., v_e, 0.]))
    jupiter = Body.from_mass("Jupiter", 1.898e27,
                             np.array([r_j, 0., 0.]), np.array([0., v_j, 0.]))
    sys   = NBodySystem([sun, earth, jupiter])
    integ = LeapfrogIntegrator(PairwiseAccelerationCalculator())
    tsm   = ConstantTimeStepManager(DAY)
    return run(sys, integ, tsm, t_end=20 * DAY)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestOrbitViewerConstruction:

    def test_creates_figure(self, records):
        v = OrbitViewer(records)
        assert v.fig is not None

    def test_creates_main_axes(self, records):
        v = OrbitViewer(records)
        assert v.ax is not None

    def test_default_body_names(self, records):
        v = OrbitViewer(records)
        assert v.body_names == ["Body 0", "Body 1"]

    def test_custom_body_names(self, records):
        v = OrbitViewer(records, body_names=["Sun", "Earth"])
        assert v.body_names == ["Sun", "Earth"]

    def test_wrong_body_names_length_raises(self, records):
        with pytest.raises(ValueError, match="body_names"):
            OrbitViewer(records, body_names=["Sun"])  # only 1 for 2 bodies

    def test_empty_records_raises(self):
        with pytest.raises(ValueError, match="empty"):
            OrbitViewer([])

    def test_invalid_plane_raises(self, records):
        with pytest.raises(ValueError, match="plane"):
            OrbitViewer(records, plane="xw")

    def test_three_body_system(self, records_3body):
        v = OrbitViewer(records_3body, body_names=["Sun", "Earth", "Jupiter"])
        assert v.n_bodies == 3

    def test_unit_au(self, records):
        v = OrbitViewer(records, unit="au")
        assert "AU" in v.ax.get_xlabel()

    def test_unit_km(self, records):
        v = OrbitViewer(records, unit="km")
        assert "km" in v.ax.get_xlabel()

    def test_plane_xz(self, records):
        v = OrbitViewer(records, plane="xz")
        assert "z" in v.ax.get_ylabel()

    def test_trail_length_stored(self, records):
        v = OrbitViewer(records, trail_length=10)
        assert v.trail_length == 10

    def test_default_trail_length_none(self, records):
        v = OrbitViewer(records)
        assert v.trail_length is None

    def test_axis_limits_set(self, records):
        v = OrbitViewer(records)
        xlim = v.ax.get_xlim()
        ylim = v.ax.get_ylim()
        assert xlim[1] > xlim[0]
        assert ylim[1] > ylim[0]


# ---------------------------------------------------------------------------
# Artists
# ---------------------------------------------------------------------------

class TestArtists:

    def test_scatter_count(self, records):
        v = OrbitViewer(records)
        assert len(v._scatters) == 2

    def test_label_count(self, records):
        v = OrbitViewer(records)
        assert len(v._labels) == 2

    def test_trail_count(self, records):
        v = OrbitViewer(records)
        assert len(v._trails) == 2

    def test_labels_initially_visible(self, records):
        v = OrbitViewer(records)
        for lbl in v._labels:
            assert lbl.get_visible()

    def test_label_text_matches_body_names(self, records):
        v = OrbitViewer(records, body_names=["Sun", "Earth"])
        texts = [lbl.get_text() for lbl in v._labels]
        assert "Sun"   in texts
        assert "Earth" in texts


# ---------------------------------------------------------------------------
# Label toggle (CheckButtons)
# ---------------------------------------------------------------------------

class TestLabelToggle:

    def test_toggle_hides_label(self, records):
        v = OrbitViewer(records, body_names=["Sun", "Earth"])
        assert v._labels[0].get_visible()   # Sun starts visible
        v._on_label_toggle("Sun")
        assert not v._labels[0].get_visible()

    def test_toggle_twice_restores(self, records):
        v = OrbitViewer(records, body_names=["Sun", "Earth"])
        v._on_label_toggle("Earth")
        v._on_label_toggle("Earth")
        assert v._labels[1].get_visible()

    def test_toggle_one_does_not_affect_other(self, records):
        v = OrbitViewer(records, body_names=["Sun", "Earth"])
        v._on_label_toggle("Sun")
        assert v._labels[1].get_visible()   # Earth unaffected

    def test_show_labels_state_updated(self, records):
        v = OrbitViewer(records, body_names=["Sun", "Earth"])
        v._on_label_toggle("Sun")
        assert v._show_labels[0] is False
        assert v._show_labels[1] is True

    def test_toggle_all_bodies(self, records_3body):
        v = OrbitViewer(records_3body, body_names=["Sun", "Earth", "Jupiter"])
        for name in ["Sun", "Earth", "Jupiter"]:
            v._on_label_toggle(name)
        assert all(not s for s in v._show_labels)


# ---------------------------------------------------------------------------
# Center-body selection (RadioButtons)
# ---------------------------------------------------------------------------

class TestCenterBodySelection:

    def test_default_center_absolute(self, records):
        v = OrbitViewer(records)
        assert v._center_idx == -1

    def test_center_on_first_body(self, records):
        v = OrbitViewer(records, body_names=["Sun", "Earth"])
        v._on_center_change("Sun")
        assert v._center_idx == 0

    def test_center_on_second_body(self, records):
        v = OrbitViewer(records, body_names=["Sun", "Earth"])
        v._on_center_change("Earth")
        assert v._center_idx == 1

    def test_center_absolute_resets(self, records):
        v = OrbitViewer(records, body_names=["Sun", "Earth"])
        v._on_center_change("Earth")
        v._on_center_change("(absolute)")
        assert v._center_idx == -1

    def test_center_change_updates_axis_limits(self, records):
        v = OrbitViewer(records, body_names=["Sun", "Earth"])
        xlim_before = v.ax.get_xlim()
        v._on_center_change("Earth")
        xlim_after = v.ax.get_xlim()
        # After centering on Earth, Sun appears ~1 AU away; limits must change
        assert xlim_before != xlim_after

    def test_center_earth_sun_visible(self, records):
        """Centered on Earth: Sun should be ~1 AU from origin."""
        v = OrbitViewer(records, body_names=["Sun", "Earth"])
        v._on_center_change("Earth")
        rel = v._relative_positions()   # (n_steps, n_bodies, 3)
        # Sun relative to Earth should be approximately -1 AU in x
        sun_x = rel[0, 0, 0]   # step 0, body 0 (Sun), x-component
        assert abs(sun_x) == pytest.approx(AU * v._scale, rel=1e-3)

    def test_center_body_at_origin(self, records):
        """The selected center body's relative position is always zero."""
        v = OrbitViewer(records, body_names=["Sun", "Earth"])
        v._on_center_change("Earth")
        rel = v._relative_positions()
        earth_pos = rel[:, 1, :]   # Earth (index 1) across all steps
        np.testing.assert_allclose(earth_pos, 0.0, atol=1e-30)


# ---------------------------------------------------------------------------
# Reference-frame helper
# ---------------------------------------------------------------------------

class TestRelativePositions:

    def test_absolute_unchanged(self, records):
        v = OrbitViewer(records)
        rel = v._relative_positions()
        np.testing.assert_array_equal(rel, v._pos_all)

    def test_centered_shape(self, records):
        v = OrbitViewer(records)
        v._on_center_change("Body 0")
        rel = v._relative_positions()
        assert rel.shape == v._pos_all.shape

    def test_trail_slice_full(self, records):
        v = OrbitViewer(records)
        sl = v._trail_slice(10)
        assert sl == slice(0, 11)

    def test_trail_slice_limited(self, records):
        v = OrbitViewer(records, trail_length=5)
        sl = v._trail_slice(10)
        assert sl == slice(5, 11)

    def test_trail_slice_clamped_at_zero(self, records):
        v = OrbitViewer(records, trail_length=50)
        sl = v._trail_slice(3)   # less than trail_length
        assert sl.start == 0


# ---------------------------------------------------------------------------
# Frame update
# ---------------------------------------------------------------------------

class TestFrameUpdate:

    def test_update_returns_artists(self, records):
        v = OrbitViewer(records)
        artists = v._update_frame(0)
        assert len(artists) > 0

    def test_update_first_frame(self, records):
        v = OrbitViewer(records)
        v._update_frame(0)   # should not raise

    def test_update_last_frame(self, records):
        v = OrbitViewer(records)
        v._update_frame(len(records) - 1)   # should not raise

    def test_scatter_positions_updated(self, records):
        v = OrbitViewer(records, body_names=["Sun", "Earth"])
        v._update_frame(5)
        # Earth scatter offset should be non-zero after frame 5
        offsets = v._scatters[1].get_offsets()
        assert offsets.shape == (1, 2)

    def test_frame_text_contains_step(self, records):
        v = OrbitViewer(records)
        v._update_frame(3)
        assert "step 4" in v._frame_text.get_text()

    def test_frame_text_contains_days(self, records):
        v = OrbitViewer(records)
        v._update_frame(9)
        assert "days" in v._frame_text.get_text()

    def test_trail_data_grows_with_frame(self, records):
        v = OrbitViewer(records)
        v._update_frame(0)
        trail_len_0 = len(v._trails[1].get_xdata())
        v._update_frame(15)
        trail_len_15 = len(v._trails[1].get_xdata())
        assert trail_len_15 > trail_len_0

    def test_trail_length_respected(self, records):
        v = OrbitViewer(records, trail_length=5)
        v._update_frame(29)
        trail_x = v._trails[1].get_xdata()
        assert len(trail_x) == 6   # frames 24..29 inclusive = 6 points

    def test_update_after_center_change(self, records):
        v = OrbitViewer(records, body_names=["Sun", "Earth"])
        v._on_center_change("Earth")
        v._update_frame(10)   # should not raise with relative frame active


# ---------------------------------------------------------------------------
# animate() / FuncAnimation
# ---------------------------------------------------------------------------

class TestAnimate:

    def test_animate_returns_func_animation(self, records):
        v = OrbitViewer(records)
        anim = v.animate()
        assert isinstance(anim, FuncAnimation)

    def test_animate_stored_on_viewer(self, records):
        v = OrbitViewer(records)
        anim = v.animate()
        assert v._anim is anim

    def test_animate_custom_interval(self, records):
        v = OrbitViewer(records)
        anim = v.animate(interval=100)
        assert isinstance(anim, FuncAnimation)

    def test_animate_no_repeat(self, records):
        v = OrbitViewer(records)
        anim = v.animate(repeat=False)
        assert isinstance(anim, FuncAnimation)

    def test_save_gif(self, records, tmp_path):
        """Smoke-test: save a short animation to a GIF without error."""
        pytest.importorskip("PIL", reason="Pillow not installed — skip GIF export")
        v    = OrbitViewer(records)
        anim = v.animate(interval=50)
        path = tmp_path / "orbit.gif"
        anim.save(str(path), writer="pillow", fps=5)
        assert path.exists() and path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Package export
# ---------------------------------------------------------------------------

class TestPackageExport:

    def test_orbit_viewer_exported(self):
        from keppy.viz import OrbitViewer as OV
        assert OV is OrbitViewer

"""Tests for the VTK-backed interactive orbit viewer."""

import math

import numpy as np
import pytest

vtk = pytest.importorskip("vtk", reason="VTK viewer dependency not installed")

from keppy.acceleration import PairwiseAccelerationCalculator
from keppy.body import Body
from keppy.constants import AU, DAY, MU_SUN
from keppy.integrator import LeapfrogIntegrator
from keppy.nbody_system import NBodySystem
from keppy.timestep import ConstantTimeStepManager, run
from keppy.viz.vtk_viewer import VTKOrbitViewer


@pytest.fixture
def records():
    """A short trajectory with visible motion in all three dimensions."""
    velocity = math.sqrt(MU_SUN / AU)
    sun = Body.from_mu("Sun", MU_SUN, np.zeros(3), np.zeros(3))
    earth = Body.from_mass(
        "Earth",
        5.972e24,
        np.array([AU, 0.0, 1.0e6]),
        np.array([0.0, velocity, 10.0]),
    )
    system = NBodySystem([sun, earth])
    return run(
        system,
        LeapfrogIntegrator(PairwiseAccelerationCalculator()),
        ConstantTimeStepManager(DAY),
        t_end=2 * DAY,
    )


@pytest.fixture
def viewer(records):
    result = VTKOrbitViewer(records, body_names=["Sun", "Earth"])
    result.render_window.SetOffScreenRendering(1)
    yield result
    result.render_window.Finalize()


def test_uses_black_background_and_trackball_camera(viewer):
    assert viewer.renderer.GetBackground() == (0.0, 0.0, 0.0)
    assert isinstance(viewer.interactor.GetInteractorStyle(), vtk.vtkInteractorStyleTrackballCamera)


def test_updates_all_three_coordinates(viewer):
    viewer.update_frame(1)
    expected = viewer._positions[1, 1]
    assert viewer._body_actors[1].GetPosition() == pytest.approx(expected)


def test_centering_a_body_moves_it_to_the_origin(viewer):
    viewer.center_on(1)
    viewer.update_frame(1)
    assert viewer._body_actors[1].GetPosition() == pytest.approx((0.0, 0.0, 0.0))


def test_centering_a_body_makes_it_the_camera_focal_point(viewer):
    viewer.update_frame(1)
    viewer.center_on(1)
    assert viewer.renderer.GetActiveCamera().GetFocalPoint() == pytest.approx((0.0, 0.0, 0.0))


def test_label_and_trail_toggles_change_visibility(viewer):
    viewer.toggle_labels()
    viewer.toggle_trails()
    assert all(not actor.GetVisibility() for actor in viewer._label_actors)
    assert all(not actor.GetVisibility() for actor in viewer._trail_actors)


def test_pause_stops_timer_updates_until_resumed(viewer):
    viewer.toggle_pause()
    assert viewer.paused is True
    viewer._on_timer(None, None)
    assert viewer._current_frame == 0

    viewer.toggle_pause()
    viewer._on_timer(None, None)
    assert viewer._current_frame == 1

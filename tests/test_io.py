"""Tests for keppy.io — config and trajectory I/O."""

import math
from pathlib import Path

import numpy as np
import pytest

from keppy.io import (
    BodyConfig,
    IntegratorConfig,
    TimestepConfig,
    SimulationConfig,
    load_config,
    save_config,
    config_to_system,
    build_integrator,
    build_timestep_manager,
    save_trajectory,
    load_trajectory,
    save_trajectory_csv,
)
from keppy.body import Body
from keppy.nbody_system import NBodySystem
from keppy.integrator import (
    RK4Integrator, RK45Integrator, LeapfrogIntegrator, YoshidaIntegrator,
)
from keppy.timestep import (
    ConstantTimeStepManager,
    AccelerationTimeStepManager,
    ScaledTimeStepManager,
    StepRecord,
    run,
)
from keppy.constants import AU, MU_SUN, DAY, G


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sun_earth_config() -> SimulationConfig:
    """Minimal two-body Sun-Earth config for reuse."""
    v = math.sqrt(MU_SUN / AU)
    return SimulationConfig(
        bodies=[
            BodyConfig(
                name="Sun",
                mu=MU_SUN,
                position=np.zeros(3),
                velocity=np.zeros(3),
            ),
            BodyConfig(
                name="Earth",
                mass=5.972e24,
                position=np.array([AU, 0.0, 0.0]),
                velocity=np.array([0.0, v, 0.0]),
            ),
        ],
        t_end=10 * DAY,
    )


def _sun_earth_system() -> NBodySystem:
    v = math.sqrt(MU_SUN / AU)
    sun   = Body.from_mu("Sun",   MU_SUN,   np.zeros(3), np.zeros(3))
    earth = Body.from_mass("Earth", 5.972e24, np.array([AU, 0., 0.]), np.array([0., v, 0.]))
    return NBodySystem([sun, earth])


# ---------------------------------------------------------------------------
# Config dataclass defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:

    def test_integrator_config_defaults(self):
        ic = IntegratorConfig()
        assert ic.type   == "leapfrog"
        assert ic.rtol   == pytest.approx(1e-9)
        assert ic.atol   == pytest.approx(1e-3)
        assert ic.min_dt == pytest.approx(1.0)

    def test_timestep_config_defaults(self):
        tc = TimestepConfig()
        assert tc.type            == "constant"
        assert tc.dt              == pytest.approx(86400.0)
        assert tc.a_min           == pytest.approx(0.01)
        assert tc.a_max           == pytest.approx(0.10)
        assert tc.increase_factor == pytest.approx(2.0)
        assert tc.decrease_factor == pytest.approx(2.0)
        assert tc.min_dt          == pytest.approx(1.0)
        assert tc.max_dt          is None
        assert tc.q_target        == pytest.approx(0.03)

    def test_simulation_config_defaults(self):
        cfg = _sun_earth_config()
        assert cfg.output_every == 1
        assert cfg.max_retries  == 10
        assert isinstance(cfg.integrator, IntegratorConfig)
        assert isinstance(cfg.timestep,   TimestepConfig)


# ---------------------------------------------------------------------------
# save_config / load_config round-trips
# ---------------------------------------------------------------------------

class TestSaveLoadConfig:

    def _round_trip(self, cfg: SimulationConfig, tmp_path: Path) -> SimulationConfig:
        p = tmp_path / "sim.toml"
        save_config(cfg, p)
        return load_config(p)

    def test_simulation_fields(self, tmp_path):
        cfg = _sun_earth_config()
        cfg.t_end        = 3.156e7
        cfg.output_every = 5
        cfg.max_retries  = 3
        out = self._round_trip(cfg, tmp_path)
        assert out.t_end        == pytest.approx(cfg.t_end)
        assert out.output_every == 5
        assert out.max_retries  == 3

    def test_body_state_vectors(self, tmp_path):
        cfg = _sun_earth_config()
        out = self._round_trip(cfg, tmp_path)
        assert len(out.bodies) == 2
        bc_in  = cfg.bodies[1]
        bc_out = out.bodies[1]
        np.testing.assert_allclose(bc_out.position, bc_in.position)
        np.testing.assert_allclose(bc_out.velocity, bc_in.velocity)

    def test_body_mu(self, tmp_path):
        cfg = _sun_earth_config()
        out = self._round_trip(cfg, tmp_path)
        assert out.bodies[0].mu == pytest.approx(MU_SUN, rel=1e-10)
        assert out.bodies[0].mass is None

    def test_body_mass(self, tmp_path):
        cfg = _sun_earth_config()
        out = self._round_trip(cfg, tmp_path)
        assert out.bodies[1].mass == pytest.approx(5.972e24, rel=1e-10)
        assert out.bodies[1].mu   is None

    def test_integrator_leapfrog(self, tmp_path):
        cfg = _sun_earth_config()
        out = self._round_trip(cfg, tmp_path)
        assert out.integrator.type == "leapfrog"

    def test_integrator_rk45(self, tmp_path):
        cfg = _sun_earth_config()
        cfg.integrator = IntegratorConfig(type="rk45", rtol=1e-8, atol=0.5)
        out = self._round_trip(cfg, tmp_path)
        assert out.integrator.type == "rk45"
        assert out.integrator.rtol == pytest.approx(1e-8)
        assert out.integrator.atol == pytest.approx(0.5)

    def test_timestep_constant(self, tmp_path):
        cfg = _sun_earth_config()
        cfg.timestep = TimestepConfig(type="constant", dt=3600.0)
        out = self._round_trip(cfg, tmp_path)
        assert out.timestep.type == "constant"
        assert out.timestep.dt   == pytest.approx(3600.0)

    def test_timestep_acceleration(self, tmp_path):
        cfg = _sun_earth_config()
        cfg.timestep = TimestepConfig(
            type="acceleration", dt=DAY,
            a_min=0.005, a_max=0.05, min_dt=60.0,
        )
        out = self._round_trip(cfg, tmp_path)
        assert out.timestep.type  == "acceleration"
        assert out.timestep.a_min == pytest.approx(0.005)
        assert out.timestep.a_max == pytest.approx(0.05)
        assert out.timestep.min_dt == pytest.approx(60.0)

    def test_timestep_scaled_with_max_dt(self, tmp_path):
        cfg = _sun_earth_config()
        cfg.timestep = TimestepConfig(
            type="scaled", dt=DAY, max_dt=7 * DAY,
        )
        out = self._round_trip(cfg, tmp_path)
        assert out.timestep.type   == "scaled"
        assert out.timestep.max_dt == pytest.approx(7 * DAY)

    def test_timestep_max_dt_none_omitted(self, tmp_path):
        """max_dt=None should not appear in the TOML (NaN is invalid TOML)."""
        cfg = _sun_earth_config()
        p = tmp_path / "sim.toml"
        save_config(cfg, p)
        content = p.read_text()
        assert "max_dt" not in content

    def test_body_with_radius_and_j(self, tmp_path):
        cfg = _sun_earth_config()
        cfg.bodies[0].radius          = 6.957e8
        cfg.bodies[0].j_coefficients  = [1.082e-3, -2.54e-6]
        out = self._round_trip(cfg, tmp_path)
        assert out.bodies[0].radius == pytest.approx(6.957e8)
        assert len(out.bodies[0].j_coefficients) == 2
        assert out.bodies[0].j_coefficients[0] == pytest.approx(1.082e-3, rel=1e-9)

    def test_body_id_preserved(self, tmp_path):
        cfg = _sun_earth_config()
        cfg.bodies[0].body_id = 7
        out = self._round_trip(cfg, tmp_path)
        assert out.bodies[0].body_id == 7

    def test_toml_is_human_readable(self, tmp_path):
        """Sanity-check: output should be valid text with section headers."""
        p = tmp_path / "sim.toml"
        save_config(_sun_earth_config(), p)
        content = p.read_text()
        assert "[simulation]"  in content
        assert "[integrator]"  in content
        assert "[timestep]"    in content
        assert "[[bodies]]"    in content
        assert '"Sun"'         in content


# ---------------------------------------------------------------------------
# load_config with orbital elements
# ---------------------------------------------------------------------------

class TestLoadConfigWithElements:

    def test_elements_give_correct_radius(self, tmp_path):
        """
        Body defined via elements should end up at the correct orbital radius.
        e=0, i=0 circular: r = a.
        """
        toml_text = f"""
[simulation]
t_end = 0.0

[[bodies]]
name = "Sun"
mu   = {MU_SUN!r}
position = [0.0, 0.0, 0.0]
velocity  = [0.0, 0.0, 0.0]

[[bodies]]
name        = "Earth"
mass        = 5.972e24
center_body = "Sun"
elements.a    = {AU!r}
elements.e    = 0.0
elements.i    = 0.0
elements.node = 0.0
elements.peri = 0.0
elements.M    = 0.0
"""
        p = tmp_path / "el.toml"
        p.write_text(toml_text)
        cfg = load_config(p)
        earth = cfg.bodies[1]
        r = float(np.linalg.norm(earth.position))
        assert r == pytest.approx(AU, rel=1e-9)

    def test_elements_missing_center_body_key_raises(self, tmp_path):
        toml_text = """
[simulation]
t_end = 0.0
[[bodies]]
name     = "Earth"
mass     = 5.972e24
elements.a = 1.5e11
elements.e = 0.0
elements.i = 0.0
"""
        p = tmp_path / "bad.toml"
        p.write_text(toml_text)
        with pytest.raises(ValueError, match="center_body"):
            load_config(p)

    def test_elements_unknown_center_body_raises(self, tmp_path):
        toml_text = """
[simulation]
t_end = 0.0
[[bodies]]
name        = "Earth"
mass        = 5.972e24
center_body = "NotDefined"
elements.a = 1.5e11
elements.e = 0.0
elements.i = 0.0
"""
        p = tmp_path / "bad2.toml"
        p.write_text(toml_text)
        with pytest.raises(ValueError, match="not found"):
            load_config(p)


# ---------------------------------------------------------------------------
# config_to_system
# ---------------------------------------------------------------------------

class TestConfigToSystem:

    def test_creates_nbody_system(self):
        sys = config_to_system(_sun_earth_config())
        assert isinstance(sys, NBodySystem)
        assert sys.n == 2

    def test_body_names_preserved(self):
        sys = config_to_system(_sun_earth_config())
        assert sys.get_body_by_name("Sun")   is not None
        assert sys.get_body_by_name("Earth") is not None

    def test_body_mu_correct(self):
        sys = config_to_system(_sun_earth_config(), translate_to_barycenter=False)
        sun = sys.get_body_by_name("Sun")
        assert sun.mu == pytest.approx(MU_SUN, rel=1e-10)

    def test_body_mass_converted_to_mu(self):
        sys   = config_to_system(_sun_earth_config(), translate_to_barycenter=False)
        earth = sys.get_body_by_name("Earth")
        assert earth.mu == pytest.approx(5.972e24 * G, rel=1e-9)

    def test_missing_mass_and_mu_raises(self):
        cfg = _sun_earth_config()
        cfg.bodies[0].mu   = None
        cfg.bodies[0].mass = None
        with pytest.raises(ValueError, match="mass.*mu|mu.*mass"):
            config_to_system(cfg)

    def test_radius_propagated(self):
        cfg = _sun_earth_config()
        cfg.bodies[0].radius = 6.957e8
        sys = config_to_system(cfg, translate_to_barycenter=False)
        assert sys.get_body_by_name("Sun").radius == pytest.approx(6.957e8)

    def test_j_coefficients_propagated(self):
        cfg = _sun_earth_config()
        cfg.bodies[0].j_coefficients = [1.082e-3]
        sys = config_to_system(cfg, translate_to_barycenter=False)
        assert sys.get_body_by_name("Sun").j_coefficients == pytest.approx([1.082e-3])


# ---------------------------------------------------------------------------
# build_integrator
# ---------------------------------------------------------------------------

class TestBuildIntegrator:

    @pytest.mark.parametrize("itype,cls", [
        ("rk4",      RK4Integrator),
        ("rk45",     RK45Integrator),
        ("leapfrog", LeapfrogIntegrator),
        ("yoshida",  YoshidaIntegrator),
    ])
    def test_correct_type_returned(self, itype, cls):
        cfg = _sun_earth_config()
        cfg.integrator.type = itype
        integ = build_integrator(cfg)
        assert isinstance(integ, cls)

    def test_unknown_type_raises(self):
        cfg = _sun_earth_config()
        cfg.integrator.type = "symplectic8"
        with pytest.raises(ValueError, match="symplectic8"):
            build_integrator(cfg)

    def test_rk45_params_forwarded(self):
        cfg = _sun_earth_config()
        cfg.integrator = IntegratorConfig(type="rk45", rtol=1e-7, atol=0.1, min_dt=10.0)
        integ = build_integrator(cfg)
        assert isinstance(integ, RK45Integrator)
        assert integ._rtol   == pytest.approx(1e-7)
        assert integ._atol   == pytest.approx(0.1)
        assert integ._min_dt == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# build_timestep_manager
# ---------------------------------------------------------------------------

class TestBuildTimestepManager:

    def test_constant(self):
        cfg = _sun_earth_config()
        cfg.timestep = TimestepConfig(type="constant", dt=3600.0)
        tsm = build_timestep_manager(cfg)
        assert isinstance(tsm, ConstantTimeStepManager)
        assert tsm.dt == pytest.approx(3600.0)

    def test_acceleration(self):
        cfg = _sun_earth_config()
        cfg.timestep = TimestepConfig(
            type="acceleration", dt=DAY, a_min=0.005, a_max=0.05,
        )
        tsm = build_timestep_manager(cfg)
        assert isinstance(tsm, AccelerationTimeStepManager)
        assert tsm.dt == pytest.approx(DAY)

    def test_scaled(self):
        cfg = _sun_earth_config()
        cfg.timestep = TimestepConfig(type="scaled", dt=DAY, q_target=0.02)
        tsm = build_timestep_manager(cfg)
        assert isinstance(tsm, ScaledTimeStepManager)
        assert tsm.dt == pytest.approx(DAY)

    def test_unknown_type_raises(self):
        cfg = _sun_earth_config()
        cfg.timestep.type = "magic"
        with pytest.raises(ValueError, match="magic"):
            build_timestep_manager(cfg)


# ---------------------------------------------------------------------------
# Trajectory — .npz round-trip
# ---------------------------------------------------------------------------

def _make_records(n: int = 5) -> list[StepRecord]:
    sys   = _sun_earth_system()
    from keppy.acceleration import PairwiseAccelerationCalculator
    integ = LeapfrogIntegrator(PairwiseAccelerationCalculator())
    tsm   = ConstantTimeStepManager(DAY)
    return run(sys, integ, tsm, t_end=n * DAY)


class TestSaveLoadTrajectory:

    def test_round_trip_length(self, tmp_path):
        records = _make_records(5)
        p = tmp_path / "traj.npz"
        save_trajectory(records, p)
        loaded  = load_trajectory(p)
        assert len(loaded) == 5

    def test_round_trip_times(self, tmp_path):
        records = _make_records(5)
        p = tmp_path / "traj.npz"
        save_trajectory(records, p)
        loaded = load_trajectory(p)
        for r_in, r_out in zip(records, loaded):
            assert r_out.time    == pytest.approx(r_in.time)
            assert r_out.dt_used == pytest.approx(r_in.dt_used)

    def test_round_trip_positions(self, tmp_path):
        records = _make_records(3)
        p = tmp_path / "traj.npz"
        save_trajectory(records, p)
        loaded = load_trajectory(p)
        for r_in, r_out in zip(records, loaded):
            np.testing.assert_array_equal(r_out.positions,  r_in.positions)
            np.testing.assert_array_equal(r_out.velocities, r_in.velocities)

    def test_shapes_preserved(self, tmp_path):
        records = _make_records(4)
        p = tmp_path / "traj.npz"
        save_trajectory(records, p)
        loaded = load_trajectory(p)
        assert loaded[0].positions.shape  == (2, 3)
        assert loaded[0].velocities.shape == (2, 3)

    def test_numpy_appends_npz(self, tmp_path):
        """numpy appends .npz automatically if the path omits it."""
        records = _make_records(2)
        stem = tmp_path / "traj"       # no .npz extension
        save_trajectory(records, stem)
        # numpy writes to stem.npz
        assert (tmp_path / "traj.npz").exists()
        loaded = load_trajectory(tmp_path / "traj.npz")
        assert len(loaded) == 2

    def test_save_empty_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            save_trajectory([], tmp_path / "empty.npz")


# ---------------------------------------------------------------------------
# Trajectory — CSV
# ---------------------------------------------------------------------------

class TestSaveTrajectoryCSV:

    def test_header_present(self, tmp_path):
        records = _make_records(2)
        p = tmp_path / "traj.csv"
        save_trajectory_csv(records, p)
        lines = p.read_text().splitlines()
        assert lines[0].startswith("step,time_s,dt_used_s,body_index,body_name")
        assert "x_m" in lines[0]

    def test_row_count(self, tmp_path):
        records = _make_records(3)   # 3 steps × 2 bodies = 6 data rows + 1 header
        p = tmp_path / "traj.csv"
        save_trajectory_csv(records, p)
        lines = [l for l in p.read_text().splitlines() if l]
        assert len(lines) == 1 + 3 * 2

    def test_body_names_used(self, tmp_path):
        records = _make_records(2)
        p = tmp_path / "traj.csv"
        save_trajectory_csv(records, p, body_names=["Sun", "Earth"])
        content = p.read_text()
        assert "Sun"   in content
        assert "Earth" in content

    def test_default_body_names_are_indices(self, tmp_path):
        records = _make_records(1)
        p = tmp_path / "traj.csv"
        save_trajectory_csv(records, p)
        lines = p.read_text().splitlines()
        # Should have body index "0" and "1" in rows
        assert ",0," in lines[1]
        assert ",1," in lines[2]

    def test_wrong_body_names_length_raises(self, tmp_path):
        records = _make_records(2)
        with pytest.raises(ValueError, match="body_names"):
            save_trajectory_csv(records, tmp_path / "t.csv",
                                body_names=["Sun"])  # only 1 name for 2 bodies

    def test_save_empty_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            save_trajectory_csv([], tmp_path / "empty.csv")

    def test_values_parseable(self, tmp_path):
        """Parse the CSV and check position magnitude of first body in row 1."""
        records = _make_records(1)
        p = tmp_path / "traj.csv"
        save_trajectory_csv(records, p)
        lines = p.read_text().splitlines()
        # Row 1 = first body of first step; columns: step,t,dt,idx,name,x,y,z,vx,vy,vz
        parts = lines[1].split(",")
        x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
        r = math.sqrt(x**2 + y**2 + z**2)
        # Sun should be near origin (barycenter-translated but very close)
        assert r < AU


# ---------------------------------------------------------------------------
# End-to-end: config → system → run → trajectory
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_run_and_save(self, tmp_path):
        """Build a system from config, run it, save the trajectory."""
        cfg    = _sun_earth_config()
        cfg.t_end = 5 * DAY
        sys    = config_to_system(cfg)
        integ  = build_integrator(cfg)
        tsm    = build_timestep_manager(cfg)
        records = run(sys, integ, tsm, t_end=cfg.t_end)
        assert len(records) == 5

        p = tmp_path / "traj.npz"
        save_trajectory(records, p)
        loaded = load_trajectory(p)
        assert len(loaded) == 5

    def test_config_round_trip_then_run(self, tmp_path):
        """Save a config, reload it, run the simulation."""
        cfg = _sun_earth_config()
        p   = tmp_path / "sim.toml"
        save_config(cfg, p)
        cfg2   = load_config(p)
        sys    = config_to_system(cfg2)
        integ  = build_integrator(cfg2)
        tsm    = build_timestep_manager(cfg2)
        records = run(sys, integ, tsm, t_end=3 * DAY)
        assert len(records) == 3


# ---------------------------------------------------------------------------
# keppy package exports
# ---------------------------------------------------------------------------

class TestPackageExports:

    def test_io_subpackage_importable(self):
        import keppy.io
        assert hasattr(keppy.io, "SimulationConfig")
        assert hasattr(keppy.io, "load_config")
        assert hasattr(keppy.io, "save_trajectory")

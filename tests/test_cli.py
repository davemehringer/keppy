"""Tests for keppy.cli — command-line interface."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

from keppy.cli import (
    KNOWN_BODIES,
    build_parser,
    cmd_list_bodies,
    cmd_run,
    cmd_show_config,
    cmd_solar,
    main,
    parse_duration,
)
from keppy.constants import AU, MU_SUN


# ---------------------------------------------------------------------------
# parse_duration
# ---------------------------------------------------------------------------

class TestParseDuration:

    def test_years(self):
        assert parse_duration("1y") == pytest.approx(365.25 * 86_400)

    def test_days(self):
        assert parse_duration("10d") == pytest.approx(10 * 86_400)

    def test_hours(self):
        assert parse_duration("24h") == pytest.approx(86_400)

    def test_seconds_suffix(self):
        assert parse_duration("3600s") == pytest.approx(3_600)

    def test_bare_float(self):
        assert parse_duration("86400") == pytest.approx(86_400)

    def test_fractional_days(self):
        assert parse_duration("0.5d") == pytest.approx(43_200)

    def test_fractional_years(self):
        val = parse_duration("2y")
        assert val == pytest.approx(2 * 365.25 * 86_400)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_duration("abc")


# ---------------------------------------------------------------------------
# build_parser — structure
# ---------------------------------------------------------------------------

class TestBuildParser:

    def test_returns_parser(self):
        assert build_parser() is not None

    def test_solar_subcommand(self):
        args = build_parser().parse_args(["solar"])
        assert args.command == "solar"

    def test_run_subcommand(self):
        args = build_parser().parse_args(["run", "sim.toml"])
        assert args.command == "run"

    def test_list_bodies_subcommand(self):
        args = build_parser().parse_args(["list-bodies"])
        assert args.command == "list-bodies"

    def test_solar_defaults(self):
        args = build_parser().parse_args(["solar"])
        assert args.bodies    == "sun,earth"
        assert args.duration  == "1y"
        assert args.dt        == "1d"
        assert args.integrator == "leapfrog"
        assert args.adaptive  == "none"
        assert args.horizons  is False
        assert args.epoch     == "2000-01-01"
        assert args.plot      is False
        assert args.plot_energy is False
        assert args.save      is None

    def test_solar_custom_bodies(self):
        args = build_parser().parse_args(
            ["solar", "--bodies", "sun,earth,mars"]
        )
        assert args.bodies == "sun,earth,mars"

    def test_solar_custom_integrator(self):
        args = build_parser().parse_args(["solar", "--integrator", "rk4"])
        assert args.integrator == "rk4"

    def test_solar_all_integrators_accepted(self):
        for name in ["rk4", "rk45", "leapfrog", "yoshida"]:
            args = build_parser().parse_args(["solar", "--integrator", name])
            assert args.integrator == name

    def test_solar_invalid_integrator_exits(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["solar", "--integrator", "bogus"])

    def test_solar_plot_flags(self):
        args = build_parser().parse_args(["solar", "--plot", "--plot-energy"])
        assert args.plot is True
        assert args.plot_energy is True

    def test_solar_plot_plane_is_an_initial_camera_orientation(self):
        args = build_parser().parse_args(["solar", "--plot", "--plane", "yz"])
        assert args.plane == "yz"

    def test_solar_horizons_flag(self):
        args = build_parser().parse_args(
            ["solar", "--horizons", "--epoch", "2024-06-01"]
        )
        assert args.horizons is True
        assert args.epoch == "2024-06-01"

    def test_solar_adaptive_choices(self):
        for scheme in ["none", "acceleration", "scaled"]:
            args = build_parser().parse_args(
                ["solar", "--adaptive", scheme]
            )
            assert args.adaptive == scheme

    def test_solar_trail_length(self):
        args = build_parser().parse_args(["solar", "--trail-length", "20"])
        assert args.trail_length == 20

    def test_solar_output_every(self):
        args = build_parser().parse_args(["solar", "--output-every", "5"])
        assert args.output_every == 5

    def test_run_config_stored(self):
        args = build_parser().parse_args(["run", "path/to/sim.toml"])
        assert args.config == "path/to/sim.toml"

    def test_no_command_exits(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])


# ---------------------------------------------------------------------------
# cmd_list_bodies
# ---------------------------------------------------------------------------

class TestListBodies:

    def test_returns_zero(self, capsys):
        args = build_parser().parse_args(["list-bodies"])
        assert cmd_list_bodies(args) == 0

    def test_prints_all_known_bodies(self, capsys):
        args = build_parser().parse_args(["list-bodies"])
        cmd_list_bodies(args)
        out = capsys.readouterr().out
        for name in KNOWN_BODIES:
            assert name in out

    def test_includes_sun_and_earth(self, capsys):
        args = build_parser().parse_args(["list-bodies"])
        cmd_list_bodies(args)
        out = capsys.readouterr().out
        assert "sun"   in out
        assert "earth" in out
        assert "jupiter" in out


# ---------------------------------------------------------------------------
# cmd_solar — built-in positions (no network)
# ---------------------------------------------------------------------------

def _solar_args(extra: list[str] | None = None) -> argparse.Namespace:  # type: ignore[name-defined]
    import argparse as _ap  # noqa: F401 — only used for type hint
    argv = ["solar", "--bodies", "sun,earth", "--duration", "10d", "--dt", "1d"]
    return build_parser().parse_args(argv + (extra or []))


class TestCmdSolar:

    def test_returns_zero(self, capsys):
        assert cmd_solar(_solar_args()) == 0

    def test_prints_steps(self, capsys):
        cmd_solar(_solar_args())
        assert "Steps" in capsys.readouterr().out

    def test_prints_duration(self, capsys):
        cmd_solar(_solar_args())
        assert "Duration" in capsys.readouterr().out

    def test_body_names_in_output(self, capsys):
        cmd_solar(_solar_args())
        out = capsys.readouterr().out
        assert "sun"   in out.lower()
        assert "earth" in out.lower()

    def test_rk4_integrator(self, capsys):
        assert cmd_solar(_solar_args(["--integrator", "rk4"])) == 0

    def test_rk45_integrator(self, capsys):
        assert cmd_solar(_solar_args(["--integrator", "rk45"])) == 0

    def test_yoshida_integrator(self, capsys):
        assert cmd_solar(_solar_args(["--integrator", "yoshida"])) == 0

    def test_three_bodies(self, capsys):
        args = build_parser().parse_args(
            ["solar", "--bodies", "sun,earth,mars",
             "--duration", "5d", "--dt", "1d"]
        )
        assert cmd_solar(args) == 0

    def test_five_planets(self, capsys):
        args = build_parser().parse_args(
            ["solar", "--bodies", "sun,mercury,venus,earth,mars",
             "--duration", "3d", "--dt", "1d"]
        )
        assert cmd_solar(args) == 0

    def test_unknown_body_returns_error(self, capsys):
        args = build_parser().parse_args(
            ["solar", "--bodies", "sun,krypton",
             "--duration", "1d", "--dt", "1d"]
        )
        rc = cmd_solar(args)
        assert rc != 0
        err = capsys.readouterr().err
        assert "krypton" in err

    def test_adaptive_acceleration(self, capsys):
        args = build_parser().parse_args(
            ["solar", "--bodies", "sun,earth",
             "--duration", "5d", "--dt", "1d",
             "--adaptive", "acceleration"]
        )
        assert cmd_solar(args) == 0

    def test_adaptive_scaled(self, capsys):
        args = build_parser().parse_args(
            ["solar", "--bodies", "sun,earth",
             "--duration", "5d", "--dt", "1d",
             "--adaptive", "scaled"]
        )
        assert cmd_solar(args) == 0

    def test_save_writes_file(self, tmp_path, capsys):
        out = str(tmp_path / "traj.npz")
        args = build_parser().parse_args(
            ["solar", "--bodies", "sun,earth",
             "--duration", "5d", "--dt", "1d",
             "--save", out]
        )
        assert cmd_solar(args) == 0
        assert Path(out).exists()

    def test_save_reports_path(self, tmp_path, capsys):
        out = str(tmp_path / "traj.npz")
        args = build_parser().parse_args(
            ["solar", "--bodies", "sun,earth",
             "--duration", "5d", "--dt", "1d",
             "--save", out]
        )
        cmd_solar(args)
        assert out in capsys.readouterr().out

    def test_output_every_reduces_records(self, tmp_path, capsys):
        out = str(tmp_path / "traj.npz")
        args = build_parser().parse_args(
            ["solar", "--bodies", "sun,earth",
             "--duration", "10d", "--dt", "1d",
             "--output-every", "5",
             "--save", out]
        )
        assert cmd_solar(args) == 0
        records = io_load(out)
        # 10 steps, output every 5 → 2 records
        assert len(records) == 2

    def test_moon_with_earth(self, capsys):
        """Moon can be included alongside Earth."""
        args = build_parser().parse_args(
            ["solar", "--bodies", "sun,earth,moon",
             "--duration", "3d", "--dt", "1d"]
        )
        assert cmd_solar(args) == 0

    def test_horizons_flag_calls_build(self, capsys):
        """--horizons delegates to _build_solar_system with use_horizons=True."""
        from keppy.body import Body
        from keppy.nbody_system import NBodySystem

        sun   = Body.from_mu("sun",   MU_SUN, np.zeros(3), np.zeros(3))
        v_e   = math.sqrt(MU_SUN / AU)
        earth = Body.from_mass(
            "earth", 5.972e24,
            np.array([AU, 0.0, 0.0]), np.array([0.0, v_e, 0.0]),
        )
        fake_sys = NBodySystem([sun, earth])

        with patch("keppy.cli._build_solar_system", return_value=fake_sys) as m:
            args = build_parser().parse_args(
                ["solar", "--bodies", "sun,earth",
                 "--duration", "3d", "--dt", "1d",
                 "--horizons", "--epoch", "2024-01-01"]
            )
            rc = cmd_solar(args)
            m.assert_called_once()
            _, kwargs = m.call_args
            assert kwargs["use_horizons"] is True
            assert kwargs["epoch"] == "2024-01-01"
        assert rc == 0


# ---------------------------------------------------------------------------
# cmd_run — TOML config file
# ---------------------------------------------------------------------------

def _write_config(tmp_path: Path) -> Path:
    """Write a minimal valid TOML config to tmp_path."""
    cfg = tmp_path / "sim.toml"
    cfg.write_text(
        """\
[simulation]
t_end        = 864000.0
output_every = 1

[[bodies]]
name     = "Sun"
mu       = 1.32712440018e20
position = [0.0, 0.0, 0.0]
velocity = [0.0, 0.0, 0.0]

[[bodies]]
name     = "Earth"
mass     = 5.972e24
position = [1.496e11, 0.0, 0.0]
velocity = [0.0, 29784.0, 0.0]

[integrator]
type = "leapfrog"

[timestep]
type = "constant"
dt   = 86400.0
"""
    )
    return cfg


class TestCmdRun:

    def test_runs_from_config(self, tmp_path, capsys):
        cfg  = _write_config(tmp_path)
        args = build_parser().parse_args(["run", str(cfg)])
        assert cmd_run(args) == 0

    def test_prints_steps_and_duration(self, tmp_path, capsys):
        cfg  = _write_config(tmp_path)
        args = build_parser().parse_args(["run", str(cfg)])
        cmd_run(args)
        out = capsys.readouterr().out
        assert "Steps"    in out
        assert "Duration" in out

    def test_missing_config_returns_error(self, tmp_path, capsys):
        args = build_parser().parse_args(["run", str(tmp_path / "missing.toml")])
        rc   = cmd_run(args)
        assert rc != 0

    def test_missing_config_error_message(self, tmp_path, capsys):
        path = str(tmp_path / "missing.toml")
        args = build_parser().parse_args(["run", path])
        cmd_run(args)
        err = capsys.readouterr().err
        assert "missing.toml" in err

    def test_no_config_no_show_config_returns_error(self, capsys):
        """'keppy run' without CONFIG or --show-config should error."""
        args = build_parser().parse_args(["run"])
        rc   = cmd_run(args)
        assert rc != 0

    def test_show_config_prints_template(self, capsys):
        args = build_parser().parse_args(["run", "--show-config"])
        rc   = cmd_show_config(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "[simulation]"  in out
        assert "[integrator]"  in out
        assert "[timestep]"    in out
        assert "[[bodies]]"    in out

    def test_show_config_contains_example_body(self, capsys):
        args = build_parser().parse_args(["run", "--show-config"])
        cmd_show_config(args)
        out = capsys.readouterr().out
        assert "Sun"   in out
        assert "Earth" in out

    def test_show_config_via_cmd_run(self, capsys):
        """--show-config works when routed through cmd_run."""
        args = build_parser().parse_args(["run", "--show-config"])
        rc   = cmd_run(args)
        assert rc == 0
        assert "[simulation]" in capsys.readouterr().out

    def test_saves_trajectory(self, tmp_path, capsys):
        cfg     = _write_config(tmp_path)
        out_f   = str(tmp_path / "traj.npz")
        args    = build_parser().parse_args(["run", str(cfg), "--save", out_f])
        assert cmd_run(args) == 0
        assert Path(out_f).exists()

    def test_body_names_in_output(self, tmp_path, capsys):
        cfg  = _write_config(tmp_path)
        args = build_parser().parse_args(["run", str(cfg)])
        cmd_run(args)
        out = capsys.readouterr().out
        assert "Sun"   in out
        assert "Earth" in out


# ---------------------------------------------------------------------------
# main() entry point
# ---------------------------------------------------------------------------

class TestMain:

    def test_list_bodies(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["list-bodies"])
        assert exc.value.code == 0

    def test_solar_runs(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["solar", "--bodies", "sun,earth",
                  "--duration", "3d", "--dt", "1d"])
        assert exc.value.code == 0

    def test_no_args_exits_nonzero(self):
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code != 0

    def test_unknown_command_exits_nonzero(self):
        with pytest.raises(SystemExit) as exc:
            main(["warp-speed"])
        assert exc.value.code != 0

    def test_run_config(self, tmp_path, capsys):
        cfg = _write_config(tmp_path)
        with pytest.raises(SystemExit) as exc:
            main(["run", str(cfg)])
        assert exc.value.code == 0

    def test_python_m_keppy(self, tmp_path):
        """Smoke test: python -m keppy list-bodies exits 0."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "keppy", "list-bodies"],
            capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        assert result.returncode == 0
        assert "earth" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Helper — load trajectory without importing from io at module level
# ---------------------------------------------------------------------------

def io_load(path: str):
    from keppy.io import load_trajectory
    return load_trajectory(path)

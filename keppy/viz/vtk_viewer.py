"""VTK-backed interactive 3-D orbit viewer.

``VTKOrbitViewer`` is the interactive viewer used by the ``keppy --plot``
commands.  It uses VTK's native trackball-camera interactor, so the scene can
be freely rotated by dragging the left mouse button instead of being limited
to a fixed Cartesian projection.

Controls
--------
* Left-drag: rotate the camera freely.
* Mouse wheel: zoom.
* Space: pause / resume the animation before selecting a body.
* Right-click a body: pause and centre the reference frame on that body.
* ``0``: restore the absolute reference frame.
* ``1``–``9``: centre on the corresponding body in the displayed order.
* ``n``: show / hide labels.
* ``o``: show / hide trails.

Install the optional dependency with ``pip install "keppy[vtk]"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

try:
    import vtk
except ModuleNotFoundError as _exc:
    raise ModuleNotFoundError(
        "keppy's interactive VTK viewer requires vtk. "
        "Install it with: pip install \"keppy[vtk]\""
    ) from _exc

if TYPE_CHECKING:
    from keppy.timestep import StepRecord


_SPATIAL_UNITS: dict[str, tuple[float, str]] = {
    "m": (1.0, "m"),
    "km": (1e-3, "km"),
    "au": (1.0 / 1.495_978_707e11, "AU"),
}

_COLORS: tuple[tuple[float, float, float], ...] = (
    (1.00, 0.78, 0.00),  # gold
    (0.32, 0.72, 1.00),  # light blue
    (1.00, 0.40, 0.40),  # coral
    (0.50, 1.00, 0.50),  # green
    (0.82, 0.55, 1.00),  # violet
    (1.00, 0.62, 0.30),  # orange
    (0.25, 0.95, 0.90),  # cyan
    (1.00, 0.45, 0.75),  # pink
)


class VTKOrbitViewer:
    """Interactive, animated 3-D orbit viewer with a trackball camera.

    Parameters
    ----------
    records
        Trajectory snapshots from :func:`keppy.timestep.run`.
    body_names
        Display names for each body. Defaults to ``"Body 0"``, etc.
    unit
        Spatial unit: ``"m"``, ``"km"``, or ``"au"``.
    plane
        Initial camera orientation: ``"xy"``, ``"xz"``, or ``"yz"``.
        The camera remains freely rotatable after the viewer opens.
    trail_length
        Number of past frames to show. ``None`` shows the complete history.
    background
        VTK renderer background colour. Defaults to black.
    """

    def __init__(
        self,
        records: Sequence["StepRecord"],
        *,
        body_names: list[str] | None = None,
        unit: str = "au",
        plane: str = "xy",
        trail_length: int | None = None,
        background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        if not records:
            raise ValueError("records is empty.")
        if trail_length is not None and trail_length < 0:
            raise ValueError("trail_length must be non-negative or None.")

        key = unit.lower()
        if key not in _SPATIAL_UNITS:
            raise ValueError(
                f"Unknown spatial unit {unit!r}. Choose from: {list(_SPATIAL_UNITS)}."
            )
        if plane.lower() not in {"xy", "xz", "yz"}:
            raise ValueError("plane must be one of: xy, xz, yz.")

        self.records = list(records)
        self.n_steps = len(self.records)
        self.n_bodies = self.records[0].positions.shape[0]
        if body_names is None:
            body_names = [f"Body {index}" for index in range(self.n_bodies)]
        if len(body_names) != self.n_bodies:
            raise ValueError(
                f"body_names has {len(body_names)} entries but records have "
                f"{self.n_bodies} bodies."
            )

        self.body_names = list(body_names)
        self.trail_length = trail_length
        self._scale, self._unit_label = _SPATIAL_UNITS[key]
        self._plane = plane.lower()
        self._positions = np.stack([record.positions for record in self.records]) * self._scale
        self._center_idx: int | None = None
        self._current_frame = 0
        self._labels_visible = True
        self._trails_visible = True
        self.paused = False
        self._timer_id: int | None = None

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(*background)
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetWindowName("keppy orbit viewer")
        self.render_window.SetSize(1200, 800)
        self.render_window.AddRenderer(self.renderer)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        self._body_actors: list = []
        self._label_actors: list = []
        self._trail_actors: list = []
        self._trail_points: list = []
        self._trail_lines: list = []
        self._trail_data: list = []
        self._build_scene()
        self._configure_camera()
        self._install_event_handlers()
        self.update_frame(0)

    def _build_scene(self) -> None:
        extent = float(np.ptp(self._positions, axis=(0, 1)).max())
        marker_radius = max(extent * 0.012, 1e-9)

        for index in range(self.n_bodies):
            color = _COLORS[index % len(_COLORS)]

            source = vtk.vtkSphereSource()
            source.SetRadius(marker_radius)
            source.SetThetaResolution(20)
            source.SetPhiResolution(20)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(source.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            self.renderer.AddActor(actor)
            self._body_actors.append(actor)

            label = vtk.vtkBillboardTextActor3D()
            label.SetInput(self.body_names[index])
            label.GetTextProperty().SetColor(*color)
            label.GetTextProperty().SetFontSize(16)
            self.renderer.AddActor(label)
            self._label_actors.append(label)

            points = vtk.vtkPoints()
            line = vtk.vtkPolyLine()
            lines = vtk.vtkCellArray()
            data = vtk.vtkPolyData()
            data.SetPoints(points)
            data.SetLines(lines)
            trail_mapper = vtk.vtkPolyDataMapper()
            trail_mapper.SetInputData(data)
            trail_actor = vtk.vtkActor()
            trail_actor.SetMapper(trail_mapper)
            trail_actor.GetProperty().SetColor(*color)
            trail_actor.GetProperty().SetLineWidth(2.0)
            self.renderer.AddActor(trail_actor)
            self._trail_points.append(points)
            self._trail_lines.append(line)
            self._trail_data.append(data)
            self._trail_actors.append(trail_actor)

        self._controls = vtk.vtkTextActor()
        self._controls.SetInput(
            "Left-drag: rotate  |  Wheel: zoom  |  Space: pause/resume\n"
            "Right-click body: pause + centre  |  0: absolute  |  1-9: centre body\n"
            "n: labels  |  o: trails"
        )
        self._controls.SetPosition(16, 16)
        self._controls.GetTextProperty().SetColor(0.85, 0.85, 0.85)
        self._controls.GetTextProperty().SetFontSize(16)
        self.renderer.AddActor2D(self._controls)

        self._frame_text = vtk.vtkTextActor()
        self._frame_text.SetPosition(16, 750)
        self._frame_text.GetTextProperty().SetColor(0.85, 0.85, 0.85)
        self._frame_text.GetTextProperty().SetFontSize(16)
        self.renderer.AddActor2D(self._frame_text)

    def _configure_camera(self) -> None:
        self.renderer.ResetCamera()
        camera = self.renderer.GetActiveCamera()
        focal = camera.GetFocalPoint()
        distance = camera.GetDistance()
        if self._plane == "xy":
            camera.SetPosition(focal[0], focal[1], focal[2] + distance)
            camera.SetViewUp(0.0, 1.0, 0.0)
        elif self._plane == "xz":
            camera.SetPosition(focal[0], focal[1] - distance, focal[2])
            camera.SetViewUp(0.0, 0.0, 1.0)
        else:
            camera.SetPosition(focal[0] + distance, focal[1], focal[2])
            camera.SetViewUp(0.0, 0.0, 1.0)
        self.renderer.ResetCameraClippingRange()

    def _install_event_handlers(self) -> None:
        self.interactor.AddObserver("TimerEvent", self._on_timer)
        self.interactor.AddObserver("KeyPressEvent", self._on_keypress)
        self.interactor.AddObserver("RightButtonPressEvent", self._on_right_click)

    def _relative_positions(self) -> np.ndarray:
        positions = self._positions
        if self._center_idx is not None:
            positions = positions - positions[:, self._center_idx : self._center_idx + 1]
        return positions

    def _trail_slice(self, frame_index: int) -> slice:
        if self.trail_length is None:
            return slice(0, frame_index + 1)
        return slice(max(0, frame_index - self.trail_length), frame_index + 1)

    def _update_trail(self, body_index: int, points: np.ndarray) -> None:
        vtk_points = self._trail_points[body_index]
        vtk_line = self._trail_lines[body_index]
        vtk_lines = vtk.vtkCellArray()
        vtk_points.Reset()
        vtk_line.GetPointIds().SetNumberOfIds(len(points))
        for point_index, point in enumerate(points):
            vtk_points.InsertNextPoint(*point)
            vtk_line.GetPointIds().SetId(point_index, point_index)
        if len(points) > 1:
            vtk_lines.InsertNextCell(vtk_line)
        data = self._trail_data[body_index]
        data.SetLines(vtk_lines)
        data.Modified()

    def update_frame(self, frame_index: int) -> None:
        """Update all VTK actors to display a trajectory frame."""
        if not 0 <= frame_index < self.n_steps:
            raise IndexError(f"frame_index must be in [0, {self.n_steps - 1}].")
        self._current_frame = frame_index
        positions = self._relative_positions()
        current = positions[frame_index]
        label_offset = max(float(np.ptp(self._positions, axis=(0, 1)).max()) * 0.018, 1e-9)
        for index, point in enumerate(current):
            self._body_actors[index].SetPosition(*point)
            self._label_actors[index].SetPosition(
                point[0] + label_offset, point[1] + label_offset, point[2] + label_offset
            )
            self._update_trail(index, positions[self._trail_slice(frame_index), index])

        time_days = self.records[frame_index].time / 86_400.0
        playback = "paused" if self.paused else "playing"
        self._frame_text.SetInput(
            f"step {frame_index + 1}/{self.n_steps}  t = {time_days:.1f} days  "
            f"({self._unit_label})  [{playback}]"
        )

    def center_on(self, body_index: int) -> None:
        """Make a body the origin and move it to the window centre."""
        if not 0 <= body_index < self.n_bodies:
            raise IndexError(f"body_index must be in [0, {self.n_bodies - 1}].")
        camera = self.renderer.GetActiveCamera()
        view_vector = np.asarray(camera.GetPosition()) - np.asarray(camera.GetFocalPoint())
        self._center_idx = body_index
        camera.SetFocalPoint(0.0, 0.0, 0.0)
        camera.SetPosition(*view_vector)
        self.renderer.ResetCameraClippingRange()
        self.update_frame(self._current_frame)
        self.render_window.Render()

    def reset_center(self) -> None:
        """Return to the absolute coordinate frame while preserving the view."""
        if self._center_idx is None:
            return
        offset = self._positions[self._current_frame, self._center_idx]
        self._translate_camera(offset)
        self._center_idx = None
        self.update_frame(self._current_frame)
        self.render_window.Render()

    def _translate_camera(self, delta: np.ndarray) -> None:
        camera = self.renderer.GetActiveCamera()
        position = np.asarray(camera.GetPosition()) + delta
        focal = np.asarray(camera.GetFocalPoint()) + delta
        camera.SetPosition(*position)
        camera.SetFocalPoint(*focal)
        self.renderer.ResetCameraClippingRange()

    def toggle_labels(self) -> None:
        """Show or hide all body labels."""
        self._labels_visible = not self._labels_visible
        for actor in self._label_actors:
            actor.SetVisibility(self._labels_visible)
        self.render_window.Render()

    def toggle_trails(self) -> None:
        """Show or hide all trajectory trails."""
        self._trails_visible = not self._trails_visible
        for actor in self._trail_actors:
            actor.SetVisibility(self._trails_visible)
        self.render_window.Render()

    def set_paused(self, paused: bool) -> None:
        """Set playback state without changing the displayed frame."""
        self.paused = paused
        self.update_frame(self._current_frame)
        self.render_window.Render()

    def toggle_pause(self) -> None:
        """Pause or resume VTK timer playback."""
        self.set_paused(not self.paused)

    def _on_timer(self, _obj, _event) -> None:
        if self.paused:
            return
        self.update_frame((self._current_frame + 1) % self.n_steps)
        self.render_window.Render()

    def _on_keypress(self, _obj, _event) -> None:
        key = self.interactor.GetKeySym()
        if key == "space":
            self.toggle_pause()
        elif key == "n":
            self.toggle_labels()
        elif key == "o":
            self.toggle_trails()
        elif key in {"0", "KP_0"}:
            self.reset_center()
        elif len(key) == 1 and key.isdigit():
            body_index = int(key) - 1
            if body_index < self.n_bodies:
                self.set_paused(True)
                self.center_on(body_index)

    def _on_right_click(self, _obj, _event) -> None:
        x, y = self.interactor.GetEventPosition()
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.02)
        picker.Pick(x, y, 0, self.renderer)
        picked_actor = picker.GetActor()
        if picked_actor in self._body_actors:
            self.set_paused(True)
            self.center_on(self._body_actors.index(picked_actor))

    def show(self, interval: int = 40) -> None:
        """Open the native VTK render window and start trajectory playback."""
        if interval <= 0:
            raise ValueError("interval must be positive.")
        self.render_window.Render()
        self.interactor.Initialize()
        self._timer_id = self.interactor.CreateRepeatingTimer(interval)
        self.interactor.Start()

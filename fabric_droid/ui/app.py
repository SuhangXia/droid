"""PySide6 desktop UI for Fabric-DROID multimodal sensor collection."""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from fabric_droid.io_utils import atomic_write_json
from fabric_droid.schemas import COLLECTION_EVENT_NAMES
from fabric_droid.sensors.ati import ATIStream
from fabric_droid.sensors.camera import CameraSpec, CameraStream
from fabric_droid.ui.devices import (
    CameraDevice,
    DeviceInventory,
    discover_devices,
    is_gelsight_device,
    preferred_gelsight_serial,
)
from fabric_droid.ui.episodes import EpisodeSummary, discover_episodes
from fabric_droid.ui.session import CollectionConfig, StreamedSensorEpisode

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPO_ROOT / "outputs/fabric_droid_profiles/gui_latest.json"
HOME_PATH = REPO_ROOT / "configs/robot/franka_home_pose.json"
POLYMETIS_PYTHON = Path("/home/suhang/anaconda3/envs/droid-polymetis-client/bin/python")
POLYMETIS_PREFIX = POLYMETIS_PYTHON.parent.parent
D435_MANUAL_EXPOSURE = 141.0  # 15% below the measured default of 166.
D435_MANUAL_WHITE_BALANCE = 3780.0
D435_MANUAL_TINT = -17.0
D435_RGB_WIDTH = 640
D435_RGB_HEIGHT = 480
D435_RGB_FPS = 30
WRIST_D435_WIDTH = 640
WRIST_D435_HEIGHT = 480
WRIST_D435_FPS = 30
CAMERA_LAYOUT_ID = "dual_d435_wrist_v1"
DEFAULT_OUTPUT_ROOT = "/home/suhang/datasets2/frabric_pi"


class VideoPane(QtWidgets.QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.title = QtWidgets.QLabel(title)
        self.title.setStyleSheet("font-weight: 600; color: #d7dde8;")
        self.image = QtWidgets.QLabel("Waiting for stream")
        self.image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(360, 250)
        self.image.setStyleSheet("background: #101318; color: #7f8998;")
        layout.addWidget(self.title)
        layout.addWidget(self.image, 1)

    def set_message(self, message: str) -> None:
        self.image.setPixmap(QtGui.QPixmap())
        self.image.setText(message)

    def set_frame(self, image_bgr: np.ndarray, subtitle: str) -> None:
        target_width = max(1, self.image.width())
        target_height = max(1, self.image.height())
        source_height, source_width = image_bgr.shape[:2]
        scale = min(
            target_width / source_width,
            target_height / source_height,
            1.0,
        )
        if scale < 1.0:
            import cv2

            image_bgr = cv2.resize(
                image_bgr,
                (
                    max(1, int(round(source_width * scale))),
                    max(1, int(round(source_height * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
        rgb = np.ascontiguousarray(image_bgr[..., ::-1])
        height, width = rgb.shape[:2]
        qimage = QtGui.QImage(
            rgb.data,
            width,
            height,
            int(rgb.strides[0]),
            QtGui.QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.image.setText("")
        self.image.setPixmap(pixmap)
        self.title.setText(subtitle)


class ForceScope(QtWidgets.QWidget):
    FZ_COLOR = "#ffe66d"

    def __init__(self) -> None:
        super().__init__()
        self.samples = np.empty((0, 6), dtype=np.float64)
        self.status = "ATI not connected"
        self.setMinimumSize(360, 250)

    def set_samples(self, samples: np.ndarray, status: str) -> None:
        self.samples = samples[-2500:].copy()
        self.status = status
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#101318"))
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QColor("#7f8998"))
        painter.drawText(12, 20, self.status)
        bounds = self.rect().adjusted(12, 58, -12, -24)
        painter.setPen(QtGui.QColor("#303743"))
        painter.drawRect(bounds)

        fz = self.samples[:, 2] if self.samples.ndim == 2 and self.samples.shape[1] >= 3 else np.empty(0)
        finite_fz = fz[np.isfinite(fz)]
        if finite_fz.size:
            latest_fz = float(finite_fz[-1])
            painter.setPen(QtGui.QColor(self.FZ_COLOR))
            value_font = QtGui.QFont(painter.font())
            value_font.setPointSize(16)
            value_font.setBold(True)
            painter.setFont(value_font)
            painter.drawText(
                QtCore.QRectF(12, 27, max(0, self.width() - 24), 28),
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
                f"Fz  {latest_fz:+.3f} N",
            )

        if fz.size < 2 or finite_fz.size < 2:
            painter.setPen(QtGui.QColor("#7f8998"))
            painter.drawText(bounds, QtCore.Qt.AlignmentFlag.AlignCenter, "Waiting for Fz data")
            return

        scale = max(1.0, float(np.nanpercentile(np.abs(fz), 98)))
        center = bounds.center().y()
        painter.setPen(QtGui.QPen(QtGui.QColor("#3b4553"), 1.0, QtCore.Qt.PenStyle.DashLine))
        painter.drawLine(bounds.left(), int(center), bounds.right(), int(center))

        clean_fz = np.nan_to_num(fz, nan=0.0, posinf=scale, neginf=-scale)
        x = np.linspace(bounds.left(), bounds.right(), clean_fz.shape[0])
        y = center - np.clip(clean_fz / scale, -1, 1) * bounds.height() * 0.45
        path = QtGui.QPainterPath(QtCore.QPointF(float(x[0]), float(y[0])))
        for px, py in zip(x[1:], y[1:]):
            path.lineTo(float(px), float(py))
        painter.setPen(QtGui.QPen(QtGui.QColor(self.FZ_COLOR), 1.6))
        painter.drawPath(path)

        painter.setFont(QtGui.QFont())
        painter.setPen(QtGui.QColor("#a9b2bf"))
        painter.drawText(bounds.left() + 5, bounds.top() + 16, f"+{scale:.2f} N")
        painter.drawText(bounds.left() + 5, bounds.bottom() - 5, f"-{scale:.2f} N")


class _TaskSignals(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)


class _Task(QtCore.QRunnable):
    def __init__(self, function: Callable[[], Any]) -> None:
        super().__init__()
        self.function = function
        self.signals = _TaskSignals()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            result = self.function()
        except Exception as exc:
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.signals.finished.emit(result)


class CollectionWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Fabric-DROID Data Collection")
        self.resize(1540, 920)
        self.inventory = DeviceInventory((), ())
        self.preview_streams: dict[str, CameraStream] = {}
        self.preview_ati: ATIStream | None = None
        self.recorder: StreamedSensorEpisode | None = None
        self.freedrive_process: QtCore.QProcess | None = None
        self.go_home_process: QtCore.QProcess | None = None
        self.gripper_open_process: QtCore.QProcess | None = None
        self.teleop_process: QtCore.QProcess | None = None
        self.freedrive_active = False
        self.teleop_active = False
        self._freedrive_output = ""
        self._go_home_output = ""
        self._teleop_output = ""
        self._go_home_confirmation_sent = False
        self._pending_recording_config: CollectionConfig | None = None
        self._pending_recording_stop_success: bool | None = None
        self._recording_failure_reason = ""
        self._recording_sensor_fault_handled = False
        self._last_capture_gate_ready: bool | None = None
        self.episode_summaries: list[EpisodeSummary] = []
        self.busy = False
        self.recording = False
        self._tasks: set[_Task] = set()
        self._last_counts: dict[str, tuple[int, float]] = {}
        self._last_preview_frame_indices: dict[str, int] = {}
        self.preview_restart_timer = QtCore.QTimer(self)
        self.preview_restart_timer.setSingleShot(True)
        self.preview_restart_timer.setInterval(200)
        self.preview_restart_timer.timeout.connect(self.start_preview)
        self._build_ui()
        self._load_profile()
        # ATI is mandatory for this collection workflow and cannot be bypassed
        # by a stale profile or accidental checkbox click.
        self.ati_checkbox.setChecked(True)
        self.ati_checkbox.setEnabled(False)
        self.ati_checkbox.toggled.connect(self.ati_toggled)
        self.timer = QtCore.QTimer(self)
        # 15 Hz UI refresh is smooth enough for monitoring and leaves camera
        # acquisition/encoding threads headroom. Recording retains native rate.
        self.timer.setInterval(66)
        self.timer.timeout.connect(self.poll_streams)
        self.timer.start()
        self.episode_timer = QtCore.QTimer(self)
        self.episode_timer.setInterval(2000)
        self.episode_timer.timeout.connect(self.refresh_episode_list)
        self.episode_timer.start()
        self.refresh_home_status()
        self.refresh_episode_list()
        QtCore.QTimer.singleShot(100, self.refresh_devices)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        root.addWidget(splitter)
        controls = QtWidgets.QScrollArea()
        controls.setWidgetResizable(True)
        controls.setMinimumWidth(370)
        controls.setMaximumWidth(470)
        panel = QtWidgets.QWidget()
        self.form = QtWidgets.QVBoxLayout(panel)
        controls.setWidget(panel)
        splitter.addWidget(controls)
        splitter.addWidget(self._build_views())
        splitter.setStretchFactor(1, 1)
        self._add_device_group()
        self._add_robot_group()
        self._add_dataset_group()
        self._add_episode_group()
        self._add_record_group()
        self.form.addStretch(1)
        self.statusBar().showMessage("Ready")
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #181c22; color: #d7dde8; }
            QGroupBox { border: 1px solid #3a424f; border-radius: 5px; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
                background: #101318; border: 1px solid #3a424f; padding: 5px;
            }
            QPushButton { background: #2b3440; border: 1px solid #4a5565; padding: 7px; border-radius: 4px; }
            QPushButton:hover { background: #354252; }
            QPushButton:disabled { color: #687283; background: #232831; }
            """
        )

    def _build_views(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(widget)
        self.panes = {
            "wrist_image_left": VideoPane("Wrist D435"),
            "exterior_image_1_left": VideoPane("Third-person D435"),
            "gelsight_left": VideoPane("GelSight tactile"),
        }
        self.force_scope = ForceScope()
        force_frame = QtWidgets.QFrame()
        force_layout = QtWidgets.QVBoxLayout(force_frame)
        force_title = QtWidgets.QLabel("ATI Nano17 · Fz @ 500 Hz (all 6 axes are still saved)")
        force_title.setStyleSheet("font-weight: 600;")
        force_layout.addWidget(force_title)
        force_layout.addWidget(self.force_scope)
        grid.addWidget(self.panes["wrist_image_left"], 0, 0)
        grid.addWidget(self.panes["exterior_image_1_left"], 0, 1)
        grid.addWidget(self.panes["gelsight_left"], 1, 0)
        grid.addWidget(force_frame, 1, 1)
        return widget

    def _add_device_group(self) -> None:
        group = QtWidgets.QGroupBox("Device mapping (by stable serial)")
        layout = QtWidgets.QFormLayout(group)
        self.wrist_combo = QtWidgets.QComboBox()
        self.exterior_combo = QtWidgets.QComboBox()
        self.gelsight_combo = QtWidgets.QComboBox()
        layout.addRow("Wrist D435", self.wrist_combo)
        layout.addRow("Third-person D435", self.exterior_combo)
        layout.addRow("GelSight / UVC", self.gelsight_combo)
        row = QtWidgets.QHBoxLayout()
        self.refresh_button = QtWidgets.QPushButton("Refresh devices")
        self.swap_button = QtWidgets.QPushButton("Swap D435")
        self.preview_button = QtWidgets.QPushButton("Start preview")
        row.addWidget(self.refresh_button)
        row.addWidget(self.swap_button)
        row.addWidget(self.preview_button)
        layout.addRow(row)
        self.ati_checkbox = QtWidgets.QCheckBox(
            "ATI required (no data/all zeros disables recording)"
        )
        self.ati_checkbox.setChecked(True)
        self.ati_checkbox.setEnabled(False)
        self.ati_endpoint = QtWidgets.QLineEdit("tcp://192.168.1.20:5555")
        layout.addRow(self.ati_checkbox)
        layout.addRow("ATI endpoint", self.ati_endpoint)
        self.white_balance_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        # Slider units are 10 K so every position is a value supported by D435.
        self.white_balance_slider.setRange(280, 650)
        self.white_balance_slider.setSingleStep(1)
        self.white_balance_slider.setPageStep(10)
        self.white_balance_slider.setValue(int(D435_MANUAL_WHITE_BALANCE / 10))
        self.white_balance_slider.setToolTip(
            "Controls both wrist and third D435; disables auto white balance for both"
        )
        self.white_balance_value = QtWidgets.QLabel(
            f"{int(D435_MANUAL_WHITE_BALANCE)} K"
        )
        self.white_balance_value.setMinimumWidth(62)
        self.white_balance_value.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight
            | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        white_balance_row = QtWidgets.QHBoxLayout()
        white_balance_row.addWidget(self.white_balance_slider, 1)
        white_balance_row.addWidget(self.white_balance_value)
        layout.addRow("Dual D435 Blue ↔ Amber", white_balance_row)
        self.tint_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.tint_slider.setRange(-100, 100)
        self.tint_slider.setSingleStep(1)
        self.tint_slider.setPageStep(10)
        self.tint_slider.setValue(int(D435_MANUAL_TINT))
        self.tint_slider.setToolTip(
            "Software Tint: negative shifts green, positive shifts red/magenta for both D435s"
        )
        self.tint_value = QtWidgets.QLabel(f"{int(D435_MANUAL_TINT):+d}")
        self.tint_value.setMinimumWidth(62)
        self.tint_value.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight
            | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        tint_row = QtWidgets.QHBoxLayout()
        tint_row.addWidget(self.tint_slider, 1)
        tint_row.addWidget(self.tint_value)
        layout.addRow("Dual D435 Green ↔ Red", tint_row)
        self.white_balance_apply_timer = QtCore.QTimer(self)
        self.white_balance_apply_timer.setSingleShot(True)
        self.white_balance_apply_timer.setInterval(80)
        self.white_balance_apply_timer.timeout.connect(
            self.apply_preview_color_balance
        )
        self.white_balance_slider.valueChanged.connect(
            self.white_balance_changed
        )
        self.white_balance_slider.sliderReleased.connect(self.save_profile)
        self.tint_slider.valueChanged.connect(self.tint_changed)
        self.tint_slider.sliderReleased.connect(self.save_profile)
        self.device_status = QtWidgets.QLabel("Not scanned yet")
        self.device_status.setWordWrap(True)
        layout.addRow(self.device_status)
        self.capture_gate_status = QtWidgets.QLabel("Capture gate: waiting for 3 video streams and ATI")
        self.capture_gate_status.setWordWrap(True)
        self.capture_gate_status.setStyleSheet("color: #ffb86c; font-weight: 600;")
        layout.addRow(self.capture_gate_status)
        self.form.addWidget(group)
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.swap_button.clicked.connect(self.swap_d435)
        self.preview_button.clicked.connect(self.toggle_preview)
        for combo in (self.wrist_combo, self.exterior_combo, self.gelsight_combo):
            combo.currentIndexChanged.connect(self.assignment_changed)

    def _add_dataset_group(self) -> None:
        group = QtWidgets.QGroupBox("Dataset info")
        layout = QtWidgets.QFormLayout(group)
        self.output_root = QtWidgets.QLineEdit(DEFAULT_OUTPUT_ROOT)
        self.output_browse = QtWidgets.QPushButton("Browse")
        self.output_browse.clicked.connect(self.choose_output)
        output_row = QtWidgets.QHBoxLayout()
        output_row.addWidget(self.output_root, 1)
        output_row.addWidget(self.output_browse)
        layout.addRow("Output directory", output_row)
        self.session_id = QtWidgets.QLineEdit(f"session_{time.strftime('%Y%m%d')}")
        self.episode_id = QtWidgets.QLineEdit(self.next_episode_id())
        self.swatch_uid = QtWidgets.QLineEdit("swatch_001")
        self.operator = QtWidgets.QLineEdit("suhang")
        self.destination = QtWidgets.QComboBox()
        self.destination.addItem("target_tray (target tray)", "target_tray")
        self.split = QtWidgets.QComboBox()
        for value in ("train", "validation", "heldout_test"):
            self.split.addItem(value, value)
        layout.addRow("Session ID", self.session_id)
        layout.addRow("Episode ID", self.episode_id)
        layout.addRow("Swatch UID", self.swatch_uid)
        layout.addRow("Operator", self.operator)
        layout.addRow("Target tray", self.destination)
        layout.addRow("Split", self.split)
        self.form.addWidget(group)
        self.output_root.editingFinished.connect(self.dataset_path_changed)

    def _add_robot_group(self) -> None:
        group = QtWidgets.QGroupBox("Franka Home / Manual placement")
        layout = QtWidgets.QVBoxLayout(group)
        ip_row = QtWidgets.QFormLayout()
        self.robot_ip = QtWidgets.QLineEdit("192.168.0.116")
        ip_row.addRow("Robot IP", self.robot_ip)
        layout.addLayout(ip_row)
        buttons = QtWidgets.QHBoxLayout()
        self.freedrive_button = QtWidgets.QPushButton("Enable freedrive")
        self.save_home_button = QtWidgets.QPushButton("Save Home")
        self.go_home_button = QtWidgets.QPushButton("Go Home")
        buttons.addWidget(self.freedrive_button)
        buttons.addWidget(self.save_home_button)
        buttons.addWidget(self.go_home_button)
        layout.addLayout(buttons)
        self.robot_status = QtWidgets.QLabel("Robot: idle")
        self.robot_status.setWordWrap(True)
        self.home_status = QtWidgets.QLabel("Home: checking")
        self.home_status.setWordWrap(True)
        layout.addWidget(self.robot_status)
        layout.addWidget(self.home_status)
        self.form.addWidget(group)
        self.freedrive_button.clicked.connect(self.toggle_freedrive)
        self.save_home_button.clicked.connect(self.record_home)
        self.go_home_button.clicked.connect(self.go_home)

    def _add_episode_group(self) -> None:
        group = QtWidgets.QGroupBox("Collected episodes")
        layout = QtWidgets.QVBoxLayout(group)
        header = QtWidgets.QHBoxLayout()
        self.episode_count_label = QtWidgets.QLabel("0 episodes")
        refresh = QtWidgets.QPushButton("Refresh")
        open_button = QtWidgets.QPushButton("Open folder")
        header.addWidget(self.episode_count_label)
        header.addStretch(1)
        header.addWidget(refresh)
        header.addWidget(open_button)
        layout.addLayout(header)
        self.episode_table = QtWidgets.QTableWidget(0, 7)
        self.episode_table.setHorizontalHeaderLabels(
            ("Episode", "Status", "Duration", "E/W/G frames", "ATI", "Swatch", "Split")
        )
        self.episode_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.episode_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.episode_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.episode_table.verticalHeader().setVisible(False)
        self.episode_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        for column in range(1, 7):
            self.episode_table.horizontalHeader().setSectionResizeMode(
                column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
        self.episode_table.setMinimumHeight(190)
        layout.addWidget(self.episode_table)
        self.form.addWidget(group)
        refresh.clicked.connect(self.refresh_episode_list)
        open_button.clicked.connect(self.open_selected_episode)
        self.episode_table.cellDoubleClicked.connect(lambda row, column: self.open_episode_row(row))

    def _add_record_group(self) -> None:
        group = QtWidgets.QGroupBox("Recording")
        layout = QtWidgets.QVBoxLayout(group)
        row = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("● Start recording")
        self.stop_button = QtWidgets.QPushButton("■ Stop and save")
        self.abort_button = QtWidgets.QPushButton("Abort")
        row.addWidget(self.start_button)
        row.addWidget(self.stop_button)
        row.addWidget(self.abort_button)
        layout.addLayout(row)
        self.record_label = QtWidgets.QLabel("IDLE")
        self.record_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #8bd5a7;")
        layout.addWidget(self.record_label)
        event_grid = QtWidgets.QGridLayout()
        self.event_buttons: list[QtWidgets.QPushButton] = []
        for index, name in enumerate(COLLECTION_EVENT_NAMES):
            button = QtWidgets.QPushButton(name)
            button.clicked.connect(lambda checked=False, value=name: self.mark_event(value))
            self.event_buttons.append(button)
            event_grid.addWidget(button, index // 2, index % 2)
        layout.addLayout(event_grid)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        self.log.setMinimumHeight(145)
        layout.addWidget(self.log)
        self.form.addWidget(group)
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(lambda: self.stop_recording(True))
        self.abort_button.clicked.connect(lambda: self.stop_recording(False))
        self.update_controls()

    def log_line(self, message: str) -> None:
        self.log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {message}")

    def next_episode_id(self) -> str:
        return f"episode_{time.strftime('%Y%m%d_%H%M%S')}"

    def dataset_path_changed(self) -> None:
        self.save_profile()
        self.refresh_episode_list()

    def refresh_episode_list(self) -> None:
        root_text = self.output_root.text().strip()
        self.episode_summaries = discover_episodes(Path(root_text)) if root_text else []
        self.episode_table.setSortingEnabled(False)
        self.episode_table.setRowCount(len(self.episode_summaries))
        for row, episode in enumerate(self.episode_summaries):
            duration = "—" if episode.duration_sec is None else f"{episode.duration_sec:.1f}s"
            cameras = "/".join(str(value) for value in episode.camera_counts)
            ati = (
                f"{episode.ati_count} @ {episode.ati_hz:.1f}Hz"
                if episode.ati_count
                else "unavailable"
            )
            values = (
                episode.episode_id,
                episode.status,
                duration,
                cameras,
                ati,
                episode.swatch_uid or "—",
                episode.split or "—",
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, str(episode.path))
                if column == 1:
                    color = "#8bd5a7" if episode.status == "COMPLETE" else "#ffb86c"
                    item.setForeground(QtGui.QColor(color))
                self.episode_table.setItem(row, column, item)
        self.episode_count_label.setText(f"{len(self.episode_summaries)} episodes")

    def open_episode_row(self, row: int) -> None:
        if not 0 <= row < len(self.episode_summaries):
            return
        QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(self.episode_summaries[row].path))
        )

    def open_selected_episode(self) -> None:
        row = self.episode_table.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.information(self, "Episodes", "Please select an episode first.")
            return
        self.open_episode_row(row)

    def _selected_device(self, combo: QtWidgets.QComboBox) -> CameraDevice | None:
        serial = combo.currentData()
        devices = (*self.inventory.realsense, *self.inventory.uvc)
        return next((device for device in devices if device.serial == serial), None)

    def refresh_devices(self) -> None:
        if self.recording or self.busy:
            self.log_line("Cannot scan devices while recording/saving.")
            return
        remembered = self._profile_values()
        self.stop_preview()
        self._last_preview_frame_indices.clear()
        self.inventory = discover_devices()
        self._fill_combo(
            self.wrist_combo,
            self.inventory.realsense,
            remembered.get("wrist_serial"),
        )
        self._fill_combo(self.exterior_combo, self.inventory.realsense, remembered.get("exterior_serial"))
        uvc = sorted(
            self.inventory.uvc,
            key=lambda value: (
                not is_gelsight_device(value),
                value.name,
                value.serial,
            ),
        )
        selected_gelsight = preferred_gelsight_serial(
            uvc,
            remembered.get("gelsight_serial"),
        )
        self._fill_combo(self.gelsight_combo, uvc, selected_gelsight)
        if (
            self.wrist_combo.count() >= 2
            and self.wrist_combo.currentIndex()
            == self.exterior_combo.currentIndex()
        ):
            self.exterior_combo.setCurrentIndex(1)
        warning = "\n".join(self.inventory.warnings)
        self.device_status.setText(
            f"D435: {len(self.inventory.realsense)} | "
            f"UVC: {len(self.inventory.uvc)}"
            + (f"\n{warning}" if warning else "")
        )
        self.log_line(
            "Scan result: "
            + ", ".join(f"D435 {device.serial}" for device in self.inventory.realsense)
            + "; "
            + ", ".join(f"UVC {device.serial}" for device in self.inventory.uvc)
        )
        if self.gelsight_combo.count():
            gelsight = self._selected_device(self.gelsight_combo)
            if gelsight is not None and not is_gelsight_device(gelsight):
                self.log_line(
                    "Gelsight not detected; notebook camera will not be saved as GelSight."
                    "Please check GelSight USB connection and rescan."
                )
        self.update_controls()
        selected = self._selected_device(self.gelsight_combo)
        if (
            len(self.inventory.realsense) >= 2
            and selected is not None
            and is_gelsight_device(selected)
        ):
            QtCore.QTimer.singleShot(100, self.start_preview)

    @staticmethod
    def _fill_combo(
        combo: QtWidgets.QComboBox,
        devices: tuple[CameraDevice, ...] | list[CameraDevice],
        selected_serial: str | None,
    ) -> None:
        combo.blockSignals(True)
        combo.clear()
        for device in devices:
            combo.addItem(device.label(), device.serial)
        if selected_serial:
            index = combo.findData(selected_serial)
            if index >= 0:
                combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def swap_d435(self) -> None:
        wrist = self.wrist_combo.currentIndex()
        exterior = self.exterior_combo.currentIndex()
        self.preview_restart_timer.stop()
        self.wrist_combo.blockSignals(True)
        self.exterior_combo.blockSignals(True)
        try:
            self.wrist_combo.setCurrentIndex(exterior)
            self.exterior_combo.setCurrentIndex(wrist)
        finally:
            self.wrist_combo.blockSignals(False)
            self.exterior_combo.blockSignals(False)
        self.start_preview()

    def assignment_changed(self) -> None:
        self.preview_restart_timer.stop()
        if self.preview_streams:
            self.stop_preview()
            self.log_line("Device mapping changed, restarting preview automatically.")
        if not self.recording and not self.busy:
            self.preview_restart_timer.start()
        self.update_controls()

    @staticmethod
    def _process_running(process: QtCore.QProcess | None) -> bool:
        return process is not None and process.state() != QtCore.QProcess.ProcessState.NotRunning

    def _new_robot_process(self) -> QtCore.QProcess:
        process = QtCore.QProcess(self)
        process.setProgram(str(POLYMETIS_PYTHON))
        process.setWorkingDirectory(str(REPO_ROOT))
        process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert("CONDA_PREFIX", str(POLYMETIS_PREFIX))
        environment.insert("CONDA_DEFAULT_ENV", "droid-polymetis-client")
        environment.insert(
            "PATH",
            f"{POLYMETIS_PREFIX / 'bin'}:{environment.value('PATH')}",
        )
        process.setProcessEnvironment(environment)
        return process

    def _append_robot_output(self, prefix: str, process: QtCore.QProcess) -> str:
        text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in text.rstrip().splitlines():
            self.log_line(f"{prefix} {line}")
        return text

    def toggle_freedrive(self) -> None:
        if self._process_running(self.freedrive_process):
            self.stop_freedrive()
            return
        if (
            self.recording
            or self.busy
            or self._process_running(self.go_home_process)
            or self._process_running(self.gripper_open_process)
        ):
            return
        if not POLYMETIS_PYTHON.is_file():
            QtWidgets.QMessageBox.critical(
                self,
                "Polymetis environment missing",
                f"Not found: {POLYMETIS_PYTHON}",
            )
            return
        self.freedrive_active = False
        self._freedrive_output = ""
        process = self._new_robot_process()
        process.setArguments(
            [
                str(REPO_ROOT / "tools/franka_freedrive_home.py"),
                "--robot-ip",
                self.robot_ip.text().strip(),
                "--home-pose-file",
                str(HOME_PATH),
                "--enable-robot",
                "--preflight-confirmed",
                "--non-interactive-ui",
                "--overwrite-home",
                "--max-duration-sec",
                "1800",
            ]
        )
        process.readyReadStandardOutput.connect(self._read_freedrive_output)
        process.finished.connect(self._freedrive_finished)
        process.errorOccurred.connect(
            lambda error: self.log_line(f"[freedrive] QProcess error: {process.errorString()}")
        )
        self.freedrive_process = process
        self.robot_status.setText("Robot: freedrive preflight…")
        self.log_line("Running freedrive preflight; entering low impedance after passing.")
        process.start()
        self.update_controls()

    def _read_freedrive_output(self) -> None:
        process = self.freedrive_process
        if process is None:
            return
        text = self._append_robot_output("[freedrive]", process)
        self._freedrive_output = (self._freedrive_output + text)[-30000:]
        if "LOW IMPEDANCE ACTIVE" in self._freedrive_output or "[freedrive] elapsed=" in self._freedrive_output:
            self.freedrive_active = True
            self.robot_status.setText("Robot: LOW IMPEDANCE ACTIVE · manual drag enabled")
            self.update_controls()

    def stop_freedrive(self) -> None:
        process = self.freedrive_process
        if not self._process_running(process):
            return
        self._pending_recording_config = None
        self._signal_freedrive_stop()

    def _signal_freedrive_stop(self) -> None:
        process = self.freedrive_process
        if not self._process_running(process):
            return
        self.robot_status.setText("Robot: safely stopping freedrive…")
        try:
            os.kill(int(process.processId()), signal.SIGINT)
        except (OSError, ProcessLookupError) as exc:
            self.log_line(f"Failed to send freedrive SIGINT: {exc}")

    def record_home(self) -> None:
        process = self.freedrive_process
        if not self.freedrive_active or not self._process_running(process):
            self.log_line("Ignoring Save Home: enable freedrive first.")
            return
        self.robot_status.setText("Robot: saving Home and stopping low impedance…")
        process.write(b"SAVE_HOME\n")
        self.log_line("Requested current Home save.")

    def _freedrive_finished(
        self,
        exit_code: int,
        exit_status: QtCore.QProcess.ExitStatus,
    ) -> None:
        process = self.freedrive_process
        if process is not None:
            self._append_robot_output("[freedrive]", process)
            process.deleteLater()
        self.freedrive_process = None
        pending_recording = self._pending_recording_config
        self._pending_recording_config = None
        was_active = self.freedrive_active
        self.freedrive_active = False
        if exit_code == 0:
            self.robot_status.setText("Robot: idle · Home saved")
            self.log_line("Freedrive ended, Home saved successfully.")
        elif exit_code == 130 and was_active:
            self.robot_status.setText("Robot: idle · freedrive stopped, Home not saved")
        else:
            self.robot_status.setText(f"Robot: freedrive exited ({exit_code})")
            self.log_line(f"Freedrive exit: code={exit_code}, status={exit_status.name}")
        self.refresh_home_status()
        self.update_controls()
        if pending_recording is not None:
            self.log_line("Freedrive confirmed stopped; starting recording automatically.")
            QtCore.QTimer.singleShot(
                0,
                lambda config=pending_recording: self._begin_recording(config),
            )

    def go_home(self) -> None:
        if self._process_running(self.go_home_process):
            self.stop_go_home()
            return
        if (
            self.recording
            or self.busy
            or self._process_running(self.freedrive_process)
            or self._process_running(self.gripper_open_process)
            or self._process_running(self.teleop_process)
        ):
            return
        if not HOME_PATH.is_file():
            self.log_line(f"Ignoring go home: Home file does not exist ({HOME_PATH})")
            return
        process = self._new_robot_process()
        process.setArguments(
            [
                str(REPO_ROOT / "tools/franka_gripper_control.py"),
                "--robot-ip",
                self.robot_ip.text().strip(),
                "--open",
                "--speed-mm-s",
                "20",
                "--force-n",
                "5",
                "--enable-gripper",
                "--preflight-confirmed",
                "--non-interactive-ui",
            ]
        )
        process.readyReadStandardOutput.connect(self._read_gripper_open_output)
        process.finished.connect(self._gripper_open_finished)
        process.errorOccurred.connect(
            lambda error: self._gripper_open_process_error(process, error)
        )
        self.gripper_open_process = process
        self.robot_status.setText("Robot: opening gripper fully…")
        self.log_line("Go Home step 1/2: open gripper first.")
        process.start()
        self.update_controls()

    def _read_gripper_open_output(self) -> None:
        process = self.gripper_open_process
        if process is not None:
            self._append_robot_output("[open-gripper]", process)

    def _gripper_open_process_error(
        self,
        process: QtCore.QProcess,
        error: QtCore.QProcess.ProcessError,
    ) -> None:
        self.log_line(f"[open-gripper] QProcess error: {process.errorString()}")
        if (
            error == QtCore.QProcess.ProcessError.FailedToStart
            and self.gripper_open_process is process
        ):
            self.gripper_open_process = None
            process.deleteLater()
            self.robot_status.setText("Robot: failed to start gripper-open process · go-home cancelled")
            self.update_controls()

    def _gripper_open_finished(
        self,
        exit_code: int,
        exit_status: QtCore.QProcess.ExitStatus,
    ) -> None:
        process = self.gripper_open_process
        if process is not None:
            self._append_robot_output("[open-gripper]", process)
            process.deleteLater()
        self.gripper_open_process = None
        if exit_code != 0:
            self.robot_status.setText("Robot: gripper not fully open · go-home cancelled")
            self.log_line(
                f"Gripper open failed: code={exit_code}, status={exit_status.name}; "
                "robot go-home did not start."
            )
            self.update_controls()
            return
        self.log_line("Gripper confirmed open. Go Home step 2/2: start robot trajectory.")
        self._start_go_home_motion()

    def _start_go_home_motion(self) -> None:
        if (
            self.recording
            or self.busy
            or self._process_running(self.freedrive_process)
            or self._process_running(self.gripper_open_process)
            or self._process_running(self.teleop_process)
            or self._process_running(self.go_home_process)
        ):
            self.robot_status.setText("Robot: state changed · go-home cancelled")
            self.log_line("Robot state changed after gripper open; canceling go-home for safety.")
            self.update_controls()
            return
        self._go_home_output = ""
        self._go_home_confirmation_sent = False
        process = self._new_robot_process()
        process.setArguments(
            [
                str(REPO_ROOT / "tools/franka_go_home.py"),
                "--robot-ip",
                self.robot_ip.text().strip(),
                "--home-pose-file",
                str(HOME_PATH),
                "--time-to-go-sec",
                "10",
                "--enable-robot",
                "--preflight-confirmed",
            ]
        )
        process.readyReadStandardOutput.connect(self._read_go_home_output)
        process.finished.connect(self._go_home_finished)
        process.errorOccurred.connect(
            lambda error: self.log_line(f"[go-home] QProcess error: {process.errorString()}")
        )
        self.go_home_process = process
        self.robot_status.setText("Robot: go-home preflight…")
        self.log_line("Reading current joints and computing Home offset; auto-starting 10s trajectory.")
        process.start()
        self.update_controls()

    def stop_go_home(self) -> None:
        process = self.go_home_process
        if not self._process_running(process):
            return
        self.robot_status.setText("Robot: stopping go-home…")
        try:
            os.kill(int(process.processId()), signal.SIGINT)
        except (OSError, ProcessLookupError) as exc:
            self.log_line(f"Failed to send go-home SIGINT: {exc}")

    def _read_go_home_output(self) -> None:
        process = self.go_home_process
        if process is None:
            return
        text = self._append_robot_output("[go-home]", process)
        self._go_home_output = (self._go_home_output + text)[-30000:]
        if (
            "type MOVE TO HOME exactly" in self._go_home_output
            and not self._go_home_confirmation_sent
        ):
            self._go_home_confirmation_sent = True
            process.write(b"MOVE TO HOME\n")
            self.robot_status.setText("Robot: MOVING TO HOME · 10s")
            self.log_line("go-home preflight auto-approved by UI; gate started.")

    def _go_home_finished(
        self,
        exit_code: int,
        exit_status: QtCore.QProcess.ExitStatus,
    ) -> None:
        process = self.go_home_process
        if process is not None:
            self._append_robot_output("[go-home]", process)
            process.deleteLater()
        self.go_home_process = None
        if exit_code == 0:
            self.robot_status.setText("Robot: HOME REACHED")
            self.log_line("Franka safely reached saved Home.")
        elif exit_code == 130:
            self.robot_status.setText("Robot: go-home stopped by user")
        else:
            self.robot_status.setText(f"Robot: go-home exited ({exit_code})")
            self.log_line(f"go-home exit: code={exit_code}, status={exit_status.name}")
        self.update_controls()

    def _start_teleop_for_recording(self) -> None:
        if (
            not self.recording
            or self.recorder is None
            or self._process_running(self.teleop_process)
        ):
            return
        if not POLYMETIS_PYTHON.is_file():
            reason = f"teleop failed to start: missing {POLYMETIS_PYTHON}"
            self.log_line(f"Teleop failed to start: missing {POLYMETIS_PYTHON}")
            QtCore.QTimer.singleShot(
                0,
                lambda value=reason: self.stop_recording(False, value),
            )
            return
        self.teleop_active = False
        self._teleop_output = ""
        process = self._new_robot_process()
        process.setArguments(
            [
                str(REPO_ROOT / "tools/franka_quest_teleop.py"),
                "--robot-ip",
                self.robot_ip.text().strip(),
                "--enable-robot",
                "--preflight-confirmed",
                "--non-interactive-ui",
                "--max-duration-sec",
                "0",
                "--rate",
                "15",
                "--max-linear-speed",
                "0.02",
                "--max-angular-speed-deg",
                "10",
                "--disable-workspace-limit",
                "--enable-gripper",
                "--gripper-max-closedness",
                "0.98",
                "--telemetry-output",
                str(self.recorder.robot_telemetry_path),
            ]
        )
        process.readyReadStandardOutput.connect(self._read_teleop_output)
        process.finished.connect(self._teleop_finished)
        process.errorOccurred.connect(
            lambda error: self._teleop_process_error(process, error)
        )
        self.teleop_process = process
        self.robot_status.setText("Robot: Quest/Franka preflight…")
        self.log_line(
            "Sensors started recording; auto-starting Quest teleop (release RG + index trigger)."
        )
        process.start()
        self.update_controls()

    def _teleop_process_error(
        self,
        process: QtCore.QProcess,
        error: QtCore.QProcess.ProcessError,
    ) -> None:
        self.log_line(f"[teleop] QProcess error: {process.errorString()}")
        if (
            error == QtCore.QProcess.ProcessError.FailedToStart
            and self.teleop_process is process
        ):
            self.teleop_process = None
            process.deleteLater()
            if self.recording:
                self.log_line("Teleop process failed to start; aborting this episode.")
                QtCore.QTimer.singleShot(
                    0,
                    lambda: self.stop_recording(
                        False,
                        "teleop process failed to start",
                    ),
                )
            self.update_controls()

    def _read_teleop_output(self) -> None:
        process = self.teleop_process
        if process is None:
            return
        text = self._append_robot_output("[teleop]", process)
        self._teleop_output = (self._teleop_output + text)[-30000:]
        if "[quest]" in self._teleop_output:
            self.teleop_active = True
            self.robot_status.setText(
                "Robot: QUEST TELEOP ACTIVE · hold RG to move, release to hold, B to stop"
            )
            self.update_controls()

    def _signal_teleop_stop(self) -> None:
        process = self.teleop_process
        if not self._process_running(process):
            return
        self.robot_status.setText("Robot: safely stopping teleop…")
        try:
            os.kill(int(process.processId()), signal.SIGINT)
        except (OSError, ProcessLookupError) as exc:
            self.log_line(f"Failed to send teleop SIGINT: {exc}")

    def _teleop_finished(
        self,
        exit_code: int,
        exit_status: QtCore.QProcess.ExitStatus,
    ) -> None:
        process = self.teleop_process
        if process is not None:
            self._append_robot_output("[teleop]", process)
            process.deleteLater()
        output = self._teleop_output
        self.teleop_process = None
        self.teleop_active = False
        pending_success = self._pending_recording_stop_success
        self._pending_recording_stop_success = None

        if pending_success is not None:
            self.log_line("Teleop policy confirmed stopped; continuing sensor save.")
            QtCore.QTimer.singleShot(
                0,
                lambda success=pending_success: self._finish_recording(success),
            )
        elif self.recording and exit_code == 0 and "b_button" in output:
            self.log_line("Quest B pressed; stopping and saving this episode.")
            QtCore.QTimer.singleShot(0, lambda: self.stop_recording(True))
        elif self.recording:
            self.log_line(
                f"Teleop abnormal exit: code={exit_code}, status={exit_status.name}; "
                "this episode marked as incomplete."
            )
            QtCore.QTimer.singleShot(0, lambda: self.stop_recording(False))
        elif exit_code not in (0, 130):
            self.log_line(f"Teleop exit: code={exit_code}, status={exit_status.name}")
        self.robot_status.setText("Robot: idle")
        self.update_controls()

    def refresh_home_status(self) -> None:
        if not HOME_PATH.is_file():
            self.home_status.setText(f"Home not recorded: {HOME_PATH}")
            return
        try:
            payload = json.loads(HOME_PATH.read_text(encoding="utf-8"))
            saved_at = str(payload.get("saved_at_utc", "unknown"))
            position = payload.get("position", [])
            joints = payload.get("joint_positions_rad", [])
            self.home_status.setText(
                f"Home: {saved_at} · xyz={position} · joints={len(joints)}"
            )
        except (OSError, json.JSONDecodeError) as exc:
            self.home_status.setText(f"Home invalid: {exc}")

    def validate_assignment(self) -> tuple[CameraDevice, CameraDevice, CameraDevice]:
        wrist = self._selected_device(self.wrist_combo)
        exterior = self._selected_device(self.exterior_combo)
        gelsight = self._selected_device(self.gelsight_combo)
        if wrist is None or exterior is None:
            raise RuntimeError("Please identify and select two D435 cameras.")
        if wrist.kind != "d435" or exterior.kind != "d435":
            raise RuntimeError("Wrist and third-person views must both be D435.")
        if wrist.serial == exterior.serial:
            raise RuntimeError("Wrist and third-person views cannot be the same D435.")
        if gelsight is None:
            raise RuntimeError("Please select a GelSight/UVC device.")
        if not is_gelsight_device(gelsight):
            raise RuntimeError(
                "Selected UVC device is not recognized as GelSight; laptop camera cannot replace tactile view."
            )
        return wrist, exterior, gelsight

    def toggle_preview(self) -> None:
        if self.preview_streams:
            self.stop_preview()
        else:
            self.start_preview()

    def start_preview(self) -> None:
        self.preview_restart_timer.stop()
        if self.recording or self.busy:
            return
        try:
            wrist, exterior, gelsight = self.validate_assignment()
        except Exception as exc:
            self.log_line(str(exc))
            return
        self.stop_preview()
        white_balance = self.current_d435_white_balance()
        tint = self.current_d435_tint()
        specs = [
            CameraSpec(
                "wrist_image_left",
                "d435",
                wrist.serial,
                wrist.serial,
                WRIST_D435_WIDTH,
                WRIST_D435_HEIGHT,
                WRIST_D435_FPS,
                exposure=D435_MANUAL_EXPOSURE,
                white_balance=white_balance,
                tint=tint,
                require_usb3=True,
            ),
            CameraSpec(
                "exterior_image_1_left",
                "d435",
                exterior.serial,
                exterior.serial,
                D435_RGB_WIDTH,
                D435_RGB_HEIGHT,
                D435_RGB_FPS,
                exposure=D435_MANUAL_EXPOSURE,
                white_balance=white_balance,
                tint=tint,
                require_usb3=True,
            ),
            CameraSpec(
                "gelsight_left",
                "uvc",
                gelsight.source,
                gelsight.serial,
                3280,
                2464,
                25,
                "MJPG",
            ),
        ]
        self.preview_streams = {
            spec.name: CameraStream(
                spec,
                max_buffer_frames=2,
                frame_output_size=(640, 480)
                if spec.name == "gelsight_left"
                else None,
            )
            for spec in specs
        }
        for stream in self.preview_streams.values():
            stream.start()
        if self.ati_checkbox.isChecked():
            self.preview_ati = ATIStream(
                self.ati_endpoint.text().strip(),
                recv_timeout_ms=250,
                max_buffer_samples=2500,
            )
            self.preview_ati.start()
        self.save_profile()
        self.log_line(
            "Three streams started; verify wrist D435 / third D435 / GelSight mapping."
        )
        self.update_controls()

    def current_d435_white_balance(self) -> float:
        return float(self.white_balance_slider.value() * 10)

    def current_d435_tint(self) -> float:
        return float(self.tint_slider.value())

    def white_balance_changed(self, slider_value: int) -> None:
        kelvin = int(slider_value * 10)
        self.white_balance_value.setText(f"{kelvin} K")
        if self.preview_streams and not self.recording:
            self.white_balance_apply_timer.start()

    def tint_changed(self, slider_value: int) -> None:
        self.tint_value.setText(f"{slider_value:+d}")
        if self.preview_streams and not self.recording:
            self.white_balance_apply_timer.start()

    def apply_preview_color_balance(self) -> None:
        kelvin = self.current_d435_white_balance()
        tint = self.current_d435_tint()
        applied: list[str] = []
        failures: list[str] = []
        for name in ("wrist_image_left", "exterior_image_1_left"):
            stream = self.preview_streams.get(name)
            if stream is None:
                continue
            try:
                white_balance_applied = stream.set_manual_white_balance(kelvin)
                tint_applied = stream.set_tint(tint)
                if white_balance_applied and tint_applied:
                    applied.append(name)
            except Exception as exc:
                failures.append(f"{name}: {exc}")
        if applied:
            self.statusBar().showMessage(
                f"Dual D435: {int(kelvin)} K, green-magenta tint {int(tint):+d}",
                2500,
            )
        if failures:
            self.log_line("Failed to set white balance: " + "; ".join(failures))

    def stop_preview(self) -> None:
        self.preview_restart_timer.stop()
        streams = list(self.preview_streams.values())
        self.preview_streams = {}
        # Signal all camera workers first so their normal worker-owned cleanup
        # happens concurrently. Never call OpenCV/RealSense release from Qt.
        for stream in streams:
            stream.request_stop()
        for stream in reversed(streams):
            try:
                stream.stop(timeout=2.0)
            except Exception as exc:
                self.log_line(f"Stop {stream.name}: {exc}")
        if self.preview_ati is not None:
            ati = self.preview_ati
            self.preview_ati = None
            try:
                ati.stop(timeout=2.0)
            except Exception as exc:
                self.log_line(f"Stop ATI: {exc}")
        self.force_scope.set_samples(np.empty((0, 6)), "ATI not connected/closed")
        self._last_preview_frame_indices.clear()
        self.update_controls()

    def ati_toggled(self, enabled: bool) -> None:
        """Apply ATI preview changes immediately without reopening cameras."""

        if not self.preview_streams or self.recording or self.busy:
            return
        if self.preview_ati is not None:
            ati = self.preview_ati
            self.preview_ati = None
            try:
                ati.stop(timeout=2.0)
            except Exception as exc:
                self.log_line(f"Stop ATI: {exc}")
        if enabled:
            self.preview_ati = ATIStream(
                self.ati_endpoint.text().strip(),
                recv_timeout_ms=250,
                max_buffer_samples=2500,
            )
            self.preview_ati.start()
            self.log_line(f"ATI preview enabled: {self.ati_endpoint.text().strip()}")
        else:
            self.force_scope.set_samples(np.empty((0, 6)), "ATI disabled")
            self.log_line("ATI preview disabled")
        self.save_profile()

    def _profile_values(self) -> dict[str, Any]:
        if not PROFILE_PATH.is_file():
            return {}
        try:
            return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _load_profile(self) -> None:
        profile = self._profile_values()
        for widget, key in (
            (self.session_id, "session_id"),
            (self.operator, "operator"),
            (self.ati_endpoint, "ati_endpoint"),
            (self.robot_ip, "robot_ip"),
        ):
            if profile.get(key):
                widget.setText(str(profile[key]))
        if (
            profile.get("camera_layout") == CAMERA_LAYOUT_ID
            and profile.get("output_root")
        ):
            self.output_root.setText(str(profile["output_root"]))
        self.ati_checkbox.setChecked(True)
        try:
            white_balance = int(
                round(float(profile.get("d435_white_balance", D435_MANUAL_WHITE_BALANCE)) / 10)
            )
        except (TypeError, ValueError):
            white_balance = int(D435_MANUAL_WHITE_BALANCE / 10)
        self.white_balance_slider.setValue(
            max(
                self.white_balance_slider.minimum(),
                min(self.white_balance_slider.maximum(), white_balance),
            )
        )
        try:
            tint = int(round(float(profile.get("d435_tint", D435_MANUAL_TINT))))
        except (TypeError, ValueError):
            tint = int(D435_MANUAL_TINT)
        self.tint_slider.setValue(
            max(self.tint_slider.minimum(), min(self.tint_slider.maximum(), tint))
        )

    def save_profile(self) -> None:
        wrist = self._selected_device(self.wrist_combo)
        exterior = self._selected_device(self.exterior_combo)
        gelsight = self._selected_device(self.gelsight_combo)
        atomic_write_json(
            PROFILE_PATH,
            {
                "camera_layout": CAMERA_LAYOUT_ID,
                "wrist_serial": wrist.serial if wrist else None,
                "wrist_source": wrist.serial if wrist else None,
                "wrist_kind": "d435",
                "exterior_serial": exterior.serial if exterior else None,
                "gelsight_serial": gelsight.serial if gelsight else None,
                "gelsight_source": gelsight.source if gelsight else None,
                "ati_enabled": True,
                "ati_endpoint": self.ati_endpoint.text().strip(),
                "output_root": self.output_root.text().strip(),
                "session_id": self.session_id.text().strip(),
                "operator": self.operator.text().strip(),
                "robot_ip": self.robot_ip.text().strip(),
                "d435_white_balance": int(self.current_d435_white_balance()),
                "d435_tint": int(self.current_d435_tint()),
                "saved_wall_time_ns": time.time_ns(),
            },
        )

    def choose_output(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Fabric-DROID output directory",
            self.output_root.text(),
        )
        if selected:
            self.output_root.setText(selected)
            self.dataset_path_changed()

    def _collection_config(self) -> CollectionConfig:
        wrist, exterior, gelsight = self.validate_assignment()
        required = {
            "output root": self.output_root.text().strip(),
            "episode ID": self.episode_id.text().strip(),
            "session ID": self.session_id.text().strip(),
            "swatch UID": self.swatch_uid.text().strip(),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(f"Missing fields: {', '.join(missing)}")
        return CollectionConfig(
            output_root=Path(required["output root"]).expanduser().resolve(),
            episode_id=required["episode ID"],
            session_id=required["session ID"],
            swatch_uid=required["swatch UID"],
            destination_tray=str(self.destination.currentData()),
            split=str(self.split.currentData()),
            operator=self.operator.text().strip() or "unknown",
            wrist_serial=wrist.serial,
            wrist_kind="d435",
            wrist_source=wrist.serial,
            wrist_width=WRIST_D435_WIDTH,
            wrist_height=WRIST_D435_HEIGHT,
            wrist_fps=WRIST_D435_FPS,
            wrist_pixel_format=None,
            wrist_exposure_scale=None,
            wrist_exposure_absolute=None,
            wrist_white_balance_temperature=None,
            exterior_serial=exterior.serial,
            gelsight_source=gelsight.source,
            gelsight_serial=gelsight.serial,
            robot_ip=self.robot_ip.text().strip(),
            ati_enabled=True,
            ati_endpoint=self.ati_endpoint.text().strip(),
            rgb_white_balance=self.current_d435_white_balance(),
            rgb_tint=self.current_d435_tint(),
        )

    def _run_task(
        self,
        function: Callable[[], Any],
        success: Callable[[Any], None],
        failed: Callable[[str], None],
    ) -> None:
        task = _Task(function)
        self._tasks.add(task)

        def finish(value: Any) -> None:
            self._tasks.discard(task)
            success(value)

        def fail(message: str) -> None:
            self._tasks.discard(task)
            failed(message)

        task.signals.finished.connect(finish)
        task.signals.failed.connect(fail)
        QtCore.QThreadPool.globalInstance().start(task)

    def _preview_sensor_health(self) -> dict[str, Any]:
        required_cameras = {
            "exterior_image_1_left",
            "wrist_image_left",
            "gelsight_left",
        }
        reports: dict[str, dict[str, Any]] = {}
        reasons: list[str] = []
        for name in sorted(required_cameras):
            stream = self.preview_streams.get(name)
            if stream is None:
                report = {"ready": False, "reasons": ["stream is missing"]}
            else:
                report = stream.readiness(max_age_sec=1.0)
            reports[name] = report
            reasons.extend(f"{name}: {reason}" for reason in report["reasons"])
        if self.preview_ati is None:
            ati_report: dict[str, Any] = {
                "ready": False,
                "reasons": ["stream is missing"],
            }
        else:
            ati_report = self.preview_ati.readiness(max_age_sec=1.0)
        reasons.extend(f"ati_nano17: {reason}" for reason in ati_report["reasons"])
        return {
            "ready": not reasons,
            "reasons": reasons,
            "cameras": reports,
            "ati": ati_report,
        }

    def _capture_gate_health(self) -> dict[str, Any]:
        if self.recording and self.recorder is not None:
            return self.recorder.sensor_health_report()
        return self._preview_sensor_health()

    def _render_capture_gate(self, health: dict[str, Any]) -> None:
        ready = bool(health["ready"])
        if ready:
            self.capture_gate_status.setText(
                "Capture gate: READY · 3 streams + ATI non-zero data valid"
            )
            self.capture_gate_status.setStyleSheet(
                "color: #8bd5a7; font-weight: 600;"
            )
        else:
            summary = "; ".join(health["reasons"][:3])
            remaining = len(health["reasons"]) - 3
            if remaining > 0:
                summary += f"; and {remaining} more"
            self.capture_gate_status.setText(f"Capture gate: BLOCKED · {summary}")
            self.capture_gate_status.setStyleSheet(
                "color: #ffb86c; font-weight: 600;"
            )

    def start_recording(self) -> None:
        if (
            self.recording
            or self.busy
            or self._process_running(self.go_home_process)
            or self._process_running(self.gripper_open_process)
            or self._process_running(self.teleop_process)
        ):
            return
        health = self._preview_sensor_health()
        if not health["ready"]:
            reason = "; ".join(health["reasons"])
            self._render_capture_gate(health)
            self.record_label.setText("BLOCKED · Sensors not ready")
            self.record_label.setText("BLOCKED · Sensors not ready")
            self.log_line(f"Start rejected: {reason}")
            return
        try:
            config = self._collection_config()
            self.save_profile()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Fabric-DROID", str(exc))
            return
            if self._process_running(self.freedrive_process):
                self._pending_recording_config = config
                self.record_label.setText("WAITING · Closing freedrive")
                self.log_line("Starting recording: safely closing freedrive, then starting sensor recording.")
                self._signal_freedrive_stop()
                self.update_controls()
                return
        self._begin_recording(config)

    def _begin_recording(self, config: CollectionConfig) -> None:
        if (
            self.recording
            or self.busy
            or self._process_running(self.freedrive_process)
            or self._process_running(self.go_home_process)
            or self._process_running(self.gripper_open_process)
            or self._process_running(self.teleop_process)
        ):
            return
        self.stop_preview()
        self.recorder = StreamedSensorEpisode(config)
        self._recording_failure_reason = ""
        self._recording_sensor_fault_handled = False
        self.busy = True
        self.record_label.setText("WARMUP · Verifying 3 camera streams and ATI")
        self.update_controls()

        def started(_: Any) -> None:
            self.busy = False
            self.recording = True
            self.record_label.setText("● RECORDING")
            self.record_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #ff6b6b;")
            self.log_line(f"Start recording {config.episode_id}")
            self._start_teleop_for_recording()
            self.update_controls()

        def failed(message: str) -> None:
            self.busy = False
            self.recording = False
            self.recorder = None
            self.log_line(f"Start failed: {message}")
            self.record_label.setText("START FAILED")
            self.log_line("No COMPLETE episode generated; fix sensors and retry.")
            self.episode_id.setText(self.next_episode_id())
            self.refresh_episode_list()
            self.update_controls()
            QtCore.QTimer.singleShot(200, self.start_preview)

        self._run_task(self.recorder.start, started, failed)

    def stop_recording(
        self,
        success: bool,
        failure_reason: str = "",
    ) -> None:
        if not self.recording or self.busy or self.recorder is None:
            return
        if failure_reason:
            self._recording_failure_reason = failure_reason
            self.recorder.latch_sensor_fault(failure_reason)
        if self._pending_recording_stop_success is not None:
            return
        if self._process_running(self.teleop_process):
            self._pending_recording_stop_success = success
            self.record_label.setText("WAITING · Safely stopping teleop")
            self.log_line("Stopping recording: terminate teleop policy started by this UI first.")
            self._signal_teleop_stop()
            self.update_controls()
            return
        self._finish_recording(success)

    def _finish_recording(self, success: bool) -> None:
        if not self.recording or self.busy or self.recorder is None:
            return
        recorder = self.recorder
        if success and "episode_end" not in {event.name for event in recorder.events}:
            recorder.mark_event("episode_end")
        self.recording = False
        self.busy = True
        self.record_label.setText("SAVING · Closing and atomically saving")
        self.update_controls()

        def stopped(path: Any) -> None:
            self.busy = False
            self.recorder = None
            saved_path = Path(path)
            complete = (saved_path / "COMPLETE.json").is_file()
            self.record_label.setText("SAVED" if complete else "INCOMPLETE")
            color = "#8bd5a7" if complete else "#ffb86c"
            self.record_label.setStyleSheet(
                f"font-size: 16px; font-weight: 700; color: {color};"
            )
            self.log_line(
                f"{'Saved' if complete else 'Sensor gate failed; kept as incomplete'}: {path}"
            )
            self._recording_failure_reason = ""
            self._recording_sensor_fault_handled = False
            self.episode_id.setText(self.next_episode_id())
            self.refresh_episode_list()
            self.update_controls()
            QtCore.QTimer.singleShot(200, self.start_preview)

        def failed(message: str) -> None:
            self.busy = False
            self.recorder = None
            self.record_label.setText("SAVE FAILED")
            self.log_line(f"Save failed: {message}")
            QtWidgets.QMessageBox.critical(self, "Save failed", message)
            self.update_controls()

        self._run_task(
            lambda: recorder.stop(
                success=success,
                failure_reason=(
                    self._recording_failure_reason
                    if self._recording_failure_reason
                    else ""
                    if success
                    else "operator aborted"
                ),
            ),
            stopped,
            failed,
        )

    def mark_event(self, name: str) -> None:
        if not self.recording or self.recorder is None:
            return
        try:
            self.recorder.mark_event(name)
        except Exception as exc:
            self.log_line(f"Event failed: {name}: {exc}")
        else:
            self.log_line(f"Event: {name}")

    def _active_camera_streams(self) -> dict[str, CameraStream]:
        if self.recorder is not None and (self.recording or self.busy):
            return self.recorder.camera_streams()
        return self.preview_streams

    def _stream_rate(self, name: str, count: int) -> float:
        now = time.monotonic()
        previous = self._last_counts.get(name)
        self._last_counts[name] = (count, now)
        if previous is None or now <= previous[1]:
            return 0.0
        return max(0.0, (count - previous[0]) / (now - previous[1]))

    def poll_streams(self) -> None:
        streams = self._active_camera_streams()
        for name, pane in self.panes.items():
            stream = streams.get(name)
            if stream is None:
                pane.set_message("Waiting for device")
                continue
            snapshot = stream.snapshot()
            if stream.error is not None:
                pane.set_message(f"ERROR\n{stream.error}")
                continue
            frame = stream.latest_frame()
            if frame is None:
                pane.set_message("Waiting for first frame…")
                continue
            if self._last_preview_frame_indices.get(name) == frame.frame_index:
                continue
            self._last_preview_frame_indices[name] = frame.frame_index
            rate = self._stream_rate(name, int(snapshot["count"]))
            label = {
                "wrist_image_left": "Wrist D435",
                "exterior_image_1_left": "Third-person D435",
                "gelsight_left": "GelSight tactile",
            }[name]
            frame_height, frame_width = frame.image_bgr.shape[:2]
            exposure_suffix = ""
            runtime = snapshot.get("runtime_properties", {})
            if (
                stream.spec.kind == "d435"
                and not bool(runtime.get("auto_exposure", True))
                and runtime.get("exposure") is not None
            ):
                exposure_suffix = (
                    f" · Exposure {float(runtime['exposure']):g} (locked)"
                )
            if (
                stream.spec.kind == "d435"
                and not bool(runtime.get("auto_white_balance", True))
                and runtime.get("white_balance") is not None
            ):
                exposure_suffix += (
                    f" · White balance {float(runtime['white_balance']):g} K (locked)"
                )
            pane.set_frame(
                frame.image_bgr,
                f"{label} · {frame_width}×{frame_height} · "
                f"S/N {stream.spec.serial} · {rate:.1f} fps"
                f"{exposure_suffix}",
            )
        ati = self.recorder.ati_stream() if self.recorder is not None else self.preview_ati
        if ati is None:
            self.force_scope.set_samples(np.empty((0, 6)), "ATI disabled (data marked unavailable)")
        else:
            samples = ati.latest_samples(2500)
            values = np.asarray(
                [[sample.fx, sample.fy, sample.fz, sample.tx, sample.ty, sample.tz] for sample in samples],
                dtype=np.float64,
            )
            status = (
                f"ATI · {len(samples)} buffered · {ati.endpoint}"
                if samples
                else f"ATI waiting for data · {ati.endpoint}"
            )
            self.force_scope.set_samples(values.reshape(-1, 6), status)
        health = self._capture_gate_health()
        self._render_capture_gate(health)
        gate_ready = bool(health["ready"])
        if self._last_capture_gate_ready is None or gate_ready != self._last_capture_gate_ready:
            self._last_capture_gate_ready = gate_ready
            self.update_controls()
        if (
            self.recording
            and self.recorder is not None
            and not gate_ready
            and not self._recording_sensor_fault_handled
        ):
            reason = "mandatory sensor lost during recording: " + "; ".join(
                health["reasons"]
            )
            self._recording_sensor_fault_handled = True
            self._recording_failure_reason = reason
            self.recorder.latch_sensor_fault(reason)
            self.record_label.setText("SENSOR LOST · Safely stopping")
            self.log_line(reason)
            QtCore.QTimer.singleShot(
                0,
                lambda value=reason: self.stop_recording(False, value),
            )
        if (
            self.recording
            and self.recorder is not None
            and self.recorder.started_ns is not None
            and self._pending_recording_stop_success is None
        ):
            elapsed = (time.monotonic_ns() - self.recorder.started_ns) / 1e9
            self.record_label.setText(f"● RECORDING · {elapsed:.1f} s")

    def update_controls(self) -> None:
        idle = not self.recording and not self.busy
        freedrive_running = self._process_running(self.freedrive_process)
        go_home_running = self._process_running(self.go_home_process)
        gripper_open_running = self._process_running(self.gripper_open_process)
        teleop_running = self._process_running(self.teleop_process)
        robot_idle = (
            not freedrive_running
            and not go_home_running
            and not gripper_open_running
            and not teleop_running
        )
        has_assignment = (
            self.wrist_combo.count() >= 2
            and self.exterior_combo.count() >= 2
            and self.gelsight_combo.count() >= 1
        )
        sensor_gate_ready = bool(self._preview_sensor_health()["ready"]) if idle else False
        self.start_button.setEnabled(
            idle
            and has_assignment
            and sensor_gate_ready
            and not go_home_running
            and not gripper_open_running
            and not teleop_running
            and self._pending_recording_config is None
        )
        stopping_recording = self._pending_recording_stop_success is not None
        self.stop_button.setEnabled(self.recording and not self.busy and not stopping_recording)
        self.abort_button.setEnabled(self.recording and not self.busy and not stopping_recording)
        self.refresh_button.setEnabled(idle)
        self.swap_button.setEnabled(idle and self.wrist_combo.count() >= 2)
        self.preview_button.setEnabled(idle and has_assignment)
        self.preview_button.setText("Stop preview" if self.preview_streams else "Start preview")
        self.freedrive_button.setEnabled(
            (freedrive_running and self._pending_recording_config is None)
            or (idle and robot_idle)
        )
        self.freedrive_button.setText("Stop freedrive" if freedrive_running else "Enable freedrive")
        self.save_home_button.setEnabled(
            self.freedrive_active
            and freedrive_running
            and self._pending_recording_config is None
        )
        self.go_home_button.setEnabled(
            go_home_running
            or (
                idle
                and robot_idle
                and HOME_PATH.is_file()
            )
        )
        if gripper_open_running:
            self.go_home_button.setText("Opening gripper…")
        elif go_home_running:
            self.go_home_button.setText("Stop go-home")
        else:
            self.go_home_button.setText("Open gripper and go home")
        self.robot_ip.setEnabled(idle and robot_idle)
        self.output_browse.setEnabled(idle)
        for widget in (
            self.wrist_combo,
            self.exterior_combo,
            self.gelsight_combo,
            self.ati_endpoint,
            self.white_balance_slider,
            self.tint_slider,
            self.output_root,
            self.session_id,
            self.episode_id,
            self.swatch_uid,
            self.operator,
            self.destination,
            self.split,
        ):
            widget.setEnabled(idle)
        for button in self.event_buttons:
            button.setEnabled(self.recording and not self.busy and not stopping_recording)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        running_robot_processes = [
            process
            for process in (
                self.freedrive_process,
                self.go_home_process,
                self.gripper_open_process,
                self.teleop_process,
            )
            if self._process_running(process)
        ]
        if running_robot_processes:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Robot policy running",
                "Robot policy was started by this UI must stop before exit. Stop safely and exit now?",
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._pending_recording_config = None
            for process in running_robot_processes:
                try:
                    os.kill(int(process.processId()), signal.SIGINT)
                except (OSError, ProcessLookupError):
                    pass
            stopped = [process.waitForFinished(5000) for process in running_robot_processes]
            if not all(stopped):
                QtWidgets.QMessageBox.critical(
                    self,
                    "Failed to confirm policy stopped",
                    "Robot subprocesses did not exit within 5s. UI remains open; use physical emergency stop and check Polymetis.",
                )
                event.ignore()
                return
        if self.busy:
            QtWidgets.QMessageBox.information(
                self,
                "Processing",
                "Camera is warming up or data is saving; please wait before exiting.",
            )
            event.ignore()
            return
        if self.recording:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Recording in progress",
                "Exiting now marks this episode as incomplete. Exit anyway?",
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            if self.recorder is not None:
                try:
                    self.recorder.stop(success=False, failure_reason="UI closed during recording")
                except Exception as exc:
                    self.log_line(f"Exit stop failed: {exc}")
        self.stop_preview()
        event.accept()


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Fabric-DROID Collection")
    window = CollectionWindow()
    window.show()
    return app.exec()

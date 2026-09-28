"""UVC and RealSense RGB stream readers."""

from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from fabric_droid.sensors.base import SensorStream


@dataclass(frozen=True)
class CameraSpec:
    name: str
    kind: str
    source: int | str
    serial: str
    width: int = 640
    height: int = 480
    fps: float = 30.0
    pixel_format: str | None = None
    exposure: float | None = None
    white_balance: float | None = None
    auto_white_balance: bool | None = None
    tint: float = 0.0
    uvc_exposure_scale: float | None = None
    uvc_exposure_absolute: int | None = None
    uvc_white_balance_temperature: int | None = None
    require_usb3: bool = False


@dataclass(frozen=True)
class CameraFrame:
    frame_index: int
    timestamp_monotonic_ns: int
    frame_received_timestamp_ns: int
    device_timestamp_ns: int
    image_bgr: np.ndarray


def _scaled_exposure_absolute(
    current: int,
    scale: float,
    minimum: int,
    maximum: int,
    step: int,
) -> int:
    if current < minimum or current > maximum:
        raise ValueError(
            f"current exposure {current} is outside [{minimum}, {maximum}]"
        )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("exposure scale must be finite and positive")
    if step <= 0:
        raise ValueError("exposure step must be positive")
    requested = float(current) * float(scale)
    aligned = minimum + round((requested - minimum) / step) * step
    return int(max(minimum, min(maximum, aligned)))


def _lock_uvc_exposure_v4l2(
    source: int | str,
    *,
    scale: float | None = None,
    exposure_absolute: int | None = None,
) -> dict[str, Any]:
    """Lock standard UVC exposure_absolute from a scale or an exact value."""

    if (scale is None) == (exposure_absolute is None):
        raise ValueError("provide exactly one of scale or exposure_absolute")

    device = f"/dev/video{source}" if str(source).isdigit() else str(source)

    def run(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", device, *arguments],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"unable to run v4l2-ctl for {device}: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"v4l2-ctl failed for {device}: {detail or result.returncode}"
            )
        return result.stdout

    controls = run("--list-ctrls")
    auto_match = re.search(
        r"^\s*(exposure_auto|auto_exposure)\b",
        controls,
        flags=re.MULTILINE,
    )
    if auto_match is None:
        raise RuntimeError(f"{device} does not expose a UVC auto-exposure control")
    auto_control = auto_match.group(1)
    limits = re.search(
        r"^\s*(exposure_absolute|exposure_time_absolute)\b.*?"
        r"\bmin=(-?\d+)\s+max=(-?\d+)\s+step=(-?\d+)",
        controls,
        flags=re.MULTILINE,
    )
    if limits is None:
        raise RuntimeError(f"{device} does not expose UVC absolute-exposure limits")
    exposure_control = limits.group(1)
    current_output = run(f"--get-ctrl={exposure_control}")
    current_match = re.search(
        rf"^\s*{re.escape(exposure_control)}\s*:\s*(-?\d+)",
        current_output,
        flags=re.MULTILINE,
    )
    if current_match is None:
        raise RuntimeError(f"unable to read {exposure_control} from {device}")

    minimum, maximum, step = (int(value) for value in limits.groups()[1:])
    current = int(current_match.group(1))
    if exposure_absolute is None:
        assert scale is not None
        target = _scaled_exposure_absolute(
            current,
            scale,
            minimum,
            maximum,
            step,
        )
    else:
        requested = max(minimum, min(maximum, int(exposure_absolute)))
        target = minimum + round((requested - minimum) / step) * step
        target = int(max(minimum, min(maximum, target)))
    # Both UVC menu spellings use value 1 for Manual Mode.
    run(f"--set-ctrl={auto_control}=1")
    run(f"--set-ctrl={exposure_control}={target}")
    readback_output = run(
        f"--get-ctrl={auto_control},{exposure_control}"
    )
    readback_match = re.search(
        rf"^\s*{re.escape(exposure_control)}\s*:\s*(-?\d+)",
        readback_output,
        flags=re.MULTILINE,
    )
    readback = int(readback_match.group(1)) if readback_match else target
    return {
        "auto_exposure": False,
        "exposure_initial": current,
        "exposure_target": target,
        "exposure": readback,
        "exposure_scale": float(scale) if scale is not None else None,
        "exposure_control": f"v4l2 {exposure_control}",
        "auto_exposure_control": auto_control,
        "exposure_range": [minimum, maximum],
        "exposure_step": step,
    }


def _lock_uvc_white_balance_v4l2(
    source: int | str,
    temperature: int,
) -> dict[str, Any]:
    """Disable UVC auto white balance and lock a color temperature."""

    device = f"/dev/video{source}" if str(source).isdigit() else str(source)

    def run(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", device, *arguments],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"unable to run v4l2-ctl for {device}: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"v4l2-ctl failed for {device}: {detail or result.returncode}"
            )
        return result.stdout

    controls = run("--list-ctrls")
    auto_match = re.search(
        r"^\s*(white_balance_automatic|white_balance_temperature_auto)\b",
        controls,
        flags=re.MULTILINE,
    )
    if auto_match is None:
        raise RuntimeError(
            f"{device} does not expose a UVC auto-white-balance control"
        )
    auto_control = auto_match.group(1)
    limits = re.search(
        r"^\s*(white_balance_temperature)\b.*?"
        r"\bmin=(-?\d+)\s+max=(-?\d+)\s+step=(-?\d+)",
        controls,
        flags=re.MULTILINE,
    )
    if limits is None:
        raise RuntimeError(
            f"{device} does not expose white_balance_temperature limits"
        )
    temperature_control = limits.group(1)
    minimum, maximum, step = (int(value) for value in limits.groups()[1:])
    requested = max(minimum, min(maximum, int(temperature)))
    target = minimum + round((requested - minimum) / step) * step
    target = int(max(minimum, min(maximum, target)))

    run(f"--set-ctrl={auto_control}=0")
    run(f"--set-ctrl={temperature_control}={target}")
    readback_output = run(
        f"--get-ctrl={auto_control},{temperature_control}"
    )
    readback_match = re.search(
        rf"^\s*{re.escape(temperature_control)}\s*:\s*(-?\d+)",
        readback_output,
        flags=re.MULTILINE,
    )
    readback = int(readback_match.group(1)) if readback_match else target
    return {
        "auto_white_balance": False,
        "white_balance": readback,
        "white_balance_target": target,
        "white_balance_control": f"v4l2 {temperature_control}",
        "auto_white_balance_control": auto_control,
        "white_balance_range": [minimum, maximum],
        "white_balance_step": step,
    }


class CameraStream(SensorStream):
    def __init__(
        self,
        spec: CameraSpec,
        *,
        max_buffer_frames: int | None = None,
        frame_callback: Callable[[CameraFrame], None] | None = None,
        frame_output_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(spec.name)
        if max_buffer_frames is not None and max_buffer_frames < 1:
            raise ValueError("max_buffer_frames must be at least 1")
        self.spec = spec
        self.frames: list[CameraFrame] = []
        self.max_buffer_frames = max_buffer_frames
        self.frame_callback = frame_callback
        self.frame_output_size = frame_output_size
        self._total_frame_count = 0
        self._lock = threading.Lock()
        self._control_lock = threading.Lock()
        self._tint = float(spec.tint)
        self._realsense_color_sensor: Any = None
        self._gap_monitor_start_ns: int | None = None
        self._rate_monitor_start_count: int | None = None
        self._max_inter_frame_gap_ns = 0
        self.dropped_frame_count = 0
        self.runtime_properties: dict[str, Any] = {}

    def run(self) -> None:
        if self.spec.kind == "d435":
            self._run_realsense()
        elif self.spec.kind == "uvc":
            self._run_uvc()
        else:
            raise ValueError(f"unsupported camera kind: {self.spec.kind}")

    def _append(self, image: np.ndarray, device_timestamp_ns: int = -1) -> None:
        now = time.monotonic_ns()
        image = self._apply_tint(image)
        if self.frame_output_size is not None:
            source_height, source_width = image.shape[:2]
            if (source_width, source_height) != self.frame_output_size:
                import cv2

                image = cv2.resize(
                    image,
                    self.frame_output_size,
                    interpolation=cv2.INTER_AREA,
                )
        with self._lock:
            if self.frames:
                expected = 1e9 / self.spec.fps
                gap = now - self.frames[-1].timestamp_monotonic_ns
                self.dropped_frame_count += max(0, round(gap / expected) - 1)
            if self._gap_monitor_start_ns is not None:
                previous_ns = (
                    self.frames[-1].timestamp_monotonic_ns
                    if self.frames
                    and self.frames[-1].timestamp_monotonic_ns
                    >= self._gap_monitor_start_ns
                    else self._gap_monitor_start_ns
                )
                self._max_inter_frame_gap_ns = max(
                    self._max_inter_frame_gap_ns,
                    now - previous_ns,
                )
            frame = CameraFrame(self._total_frame_count, now, now, device_timestamp_ns, image.copy())
            self.frames.append(frame)
            self._total_frame_count += 1
            if self.max_buffer_frames is not None and len(self.frames) > self.max_buffer_frames:
                del self.frames[: len(self.frames) - self.max_buffer_frames]
        if self.frame_callback is not None:
            self.frame_callback(frame)

    def reset_gap_monitor(self, start_monotonic_ns: int) -> None:
        """Start a new episode-local historical frame-gap monitor."""

        if start_monotonic_ns <= 0:
            raise ValueError("gap monitor start must be a positive timestamp")
        with self._lock:
            self._gap_monitor_start_ns = start_monotonic_ns
            self._rate_monitor_start_count = self._total_frame_count
            latest_ns = (
                self.frames[-1].timestamp_monotonic_ns
                if self.frames
                else None
            )
            self._max_inter_frame_gap_ns = (
                latest_ns - start_monotonic_ns
                if latest_ns is not None
                and latest_ns >= start_monotonic_ns
                else 0
            )

    def _apply_tint(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.spec.kind != "d435":
            return image_bgr
        with self._control_lock:
            tint = self._tint
        if abs(tint) < 0.5:
            return image_bgr
        # Positive values add red/magenta and remove green; negative values do
        # the inverse. Limit each channel correction to ±25% at slider ends.
        amount = float(np.clip(tint, -100.0, 100.0)) * 0.0025
        gains = np.asarray([1.0, 1.0 - amount, 1.0 + amount], dtype=np.float32)
        # Avoid allocating a full float32 image for every frame from both
        # D435s. OpenCV applies the same saturated per-channel transform in
        # native code.
        import cv2

        return cv2.transform(image_bgr, np.diag(gains))

    def latest_frame(self, *, copy_image: bool = False) -> CameraFrame | None:
        """Return the latest complete frame for a live UI.

        Recording streams retain every frame by default. Preview streams can set
        ``max_buffer_frames`` to keep memory bounded while this method remains
        safe to poll from the Qt thread.
        """

        with self._lock:
            if not self.frames:
                return None
            frame = self.frames[-1]
            if not copy_image:
                return frame
            return CameraFrame(
                frame.frame_index,
                frame.timestamp_monotonic_ns,
                frame.frame_received_timestamp_ns,
                frame.device_timestamp_ns,
                frame.image_bgr.copy(),
            )

    def readiness(
        self,
        *,
        minimum_frames: int = 2,
        max_age_sec: float = 1.0,
        max_gap_sec: float | None = None,
        since_monotonic_ns: int | None = None,
    ) -> dict[str, Any]:
        """Report whether this camera has a fresh, valid frame."""

        if minimum_frames < 1:
            raise ValueError("minimum_frames must be positive")
        if max_age_sec <= 0:
            raise ValueError("max_age_sec must be positive")
        if max_gap_sec is not None and max_gap_sec <= 0:
            raise ValueError("max_gap_sec must be positive")
        with self._lock:
            count = self._total_frame_count
            frames = list(self.frames)
            max_gap_ns = self._max_inter_frame_gap_ns
        if since_monotonic_ns is not None:
            frames = [
                frame
                for frame in frames
                if frame.timestamp_monotonic_ns >= since_monotonic_ns
            ]
        gate_count = count if since_monotonic_ns is None else len(frames)
        frame = frames[-1] if frames else None
        latest_ns = frame.timestamp_monotonic_ns if frame is not None else None
        age_sec = (
            None
            if latest_ns is None
            else max(0.0, (time.monotonic_ns() - latest_ns) / 1e9)
        )
        image_valid = bool(
            frame is not None
            and isinstance(frame.image_bgr, np.ndarray)
            and frame.image_bgr.ndim == 3
            and frame.image_bgr.shape[2] == 3
            and frame.image_bgr.size > 0
        )
        reasons: list[str] = []
        if self.error is not None:
            reasons.append(f"stream error: {self.error}")
        if gate_count < minimum_frames:
            qualifier = (
                ""
                if since_monotonic_ns is None
                else " since episode start"
            )
            reasons.append(
                f"only {gate_count}/{minimum_frames} frames received{qualifier}"
            )
        if age_sec is None or age_sec > max_age_sec:
            reasons.append(
                "no frames received"
                if age_sec is None
                else f"last frame is stale ({age_sec:.3f}s)"
            )
        if frame is not None and not image_valid:
            reasons.append("latest frame is invalid")
        max_gap_sec_observed = max_gap_ns / 1e9
        if (
            max_gap_sec is not None
            and max_gap_sec_observed > max_gap_sec
        ):
            reasons.append(
                "historical frame gap "
                f"{max_gap_sec_observed:.3f}s exceeds {max_gap_sec:.3f}s"
            )
        return {
            "ready": not reasons,
            "reasons": reasons,
            "count": count,
            "gate_count": gate_count,
            "latest_timestamp_monotonic_ns": latest_ns,
            "age_sec": age_sec,
            "image_valid": image_valid,
            "minimum_frames": minimum_frames,
            "max_age_sec": max_age_sec,
            "max_gap_sec": max_gap_sec,
            "max_gap_sec_observed": max_gap_sec_observed,
            "since_monotonic_ns": since_monotonic_ns,
        }

    def _run_uvc(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for UVC capture") from exc
        source: int | str = int(self.spec.source) if str(self.spec.source).isdigit() else self.spec.source
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"unable to open UVC camera {self.spec.name} at {source}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.spec.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.spec.height)
        capture.set(cv2.CAP_PROP_FPS, self.spec.fps)
        if self.spec.pixel_format:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.spec.pixel_format.upper()))
        exposure_report: dict[str, Any] = {}
        if (
            self.spec.uvc_exposure_scale is not None
            or self.spec.uvc_exposure_absolute is not None
        ):
            try:
                if self.spec.uvc_exposure_absolute is None:
                    # Allow auto exposure to settle before converting its
                    # current exposure into a repeatable manual setting.
                    warmup_frames = max(
                        5,
                        min(30, int(round(self.spec.fps * 0.5))),
                    )
                    for _ in range(warmup_frames):
                        ok, _image = capture.read()
                        if not ok:
                            raise RuntimeError(
                                f"camera {self.spec.name} failed during exposure warmup"
                            )
                exposure_report = _lock_uvc_exposure_v4l2(
                    source,
                    scale=self.spec.uvc_exposure_scale,
                    exposure_absolute=self.spec.uvc_exposure_absolute,
                )
            except Exception:
                capture.release()
                raise
        if self.spec.exposure is not None:
            capture.set(cv2.CAP_PROP_EXPOSURE, self.spec.exposure)
        white_balance_report: dict[str, Any] = {}
        if self.spec.uvc_white_balance_temperature is not None:
            try:
                white_balance_report = _lock_uvc_white_balance_v4l2(
                    source,
                    self.spec.uvc_white_balance_temperature,
                )
            except Exception:
                capture.release()
                raise
        if self.spec.white_balance is not None:
            capture.set(cv2.CAP_PROP_WB_TEMPERATURE, self.spec.white_balance)
        self.runtime_properties = {
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "pixel_format": self.spec.pixel_format,
            "exposure": float(capture.get(cv2.CAP_PROP_EXPOSURE)),
            "white_balance": float(capture.get(cv2.CAP_PROP_WB_TEMPERATURE)),
            **exposure_report,
            **white_balance_report,
        }
        try:
            while not self._stop_event.is_set():
                ok, image = capture.read()
                if not ok:
                    if self._stop_event.is_set():
                        break
                    raise RuntimeError(f"camera {self.spec.name} stopped producing frames")
                self._append(image)
        finally:
            # OpenCV backend ownership stays in this worker thread. Releasing
            # VideoCapture from the Qt thread while read() is active can
            # segfault inside cv2.abi3.so.
            capture.release()

    def _run_realsense(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("pyrealsense2 is required for D435 capture") from exc
        pipeline = rs.pipeline()
        config = rs.config()
        if self.spec.serial:
            config.enable_device(self.spec.serial)
        config.enable_stream(rs.stream.color, self.spec.width, self.spec.height, rs.format.bgr8, int(self.spec.fps))
        profile = pipeline.start(config)
        device = profile.get_device()
        usb_type = (
            device.get_info(rs.camera_info.usb_type_descriptor)
            if device.supports(rs.camera_info.usb_type_descriptor)
            else ""
        )
        if self.spec.require_usb3 and usb_type and not usb_type.startswith("3"):
            pipeline.stop()
            raise RuntimeError(
                f"RealSense {self.spec.serial} is connected as USB {usb_type}; "
                "move it to a SuperSpeed USB 3 port (lsusb -t must show 5000M)"
            )
        sensor = device.first_color_sensor()
        with self._control_lock:
            self._realsense_color_sensor = sensor
        if self.spec.exposure is not None:
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 0.0)
            if not sensor.supports(rs.option.exposure):
                raise RuntimeError(
                    f"RealSense color sensor {self.spec.serial} does not support manual exposure"
                )
            exposure_range = sensor.get_option_range(rs.option.exposure)
            requested_exposure = float(self.spec.exposure)
            if not exposure_range.min <= requested_exposure <= exposure_range.max:
                raise RuntimeError(
                    f"RealSense exposure {requested_exposure} is outside "
                    f"[{exposure_range.min}, {exposure_range.max}]"
                )
            sensor.set_option(rs.option.exposure, requested_exposure)
        if self.spec.auto_white_balance is True:
            if not sensor.supports(rs.option.enable_auto_white_balance):
                raise RuntimeError(
                    f"RealSense color sensor {self.spec.serial} does not support auto white balance"
                )
            sensor.set_option(rs.option.enable_auto_white_balance, 1.0)
        elif self.spec.white_balance is not None:
            if sensor.supports(rs.option.enable_auto_white_balance):
                sensor.set_option(rs.option.enable_auto_white_balance, 0.0)
            if not sensor.supports(rs.option.white_balance):
                raise RuntimeError(
                    f"RealSense color sensor {self.spec.serial} does not support manual white balance"
                )
            white_balance_range = sensor.get_option_range(rs.option.white_balance)
            requested_white_balance = float(self.spec.white_balance)
            if not white_balance_range.min <= requested_white_balance <= white_balance_range.max:
                raise RuntimeError(
                    f"RealSense white balance {requested_white_balance} is outside "
                    f"[{white_balance_range.min}, {white_balance_range.max}]"
                )
            sensor.set_option(rs.option.white_balance, requested_white_balance)
        self.runtime_properties = {
            "width": self.spec.width,
            "height": self.spec.height,
            "fps": self.spec.fps,
            "usb_type_descriptor": usb_type or None,
            "auto_exposure": sensor.get_option(rs.option.enable_auto_exposure)
            if sensor.supports(rs.option.enable_auto_exposure)
            else None,
            "exposure": sensor.get_option(rs.option.exposure) if sensor.supports(rs.option.exposure) else None,
            "auto_white_balance": sensor.get_option(rs.option.enable_auto_white_balance)
            if sensor.supports(rs.option.enable_auto_white_balance)
            else None,
            "white_balance": sensor.get_option(rs.option.white_balance)
            if sensor.supports(rs.option.white_balance)
            else None,
            "tint": self._tint,
        }
        try:
            while not self._stop_event.is_set():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError:
                    if self._stop_event.is_set():
                        break
                    raise
                color = frames.get_color_frame()
                if not color:
                    continue
                self._append(np.asanyarray(color.get_data()), int(color.get_timestamp() * 1e6))
        finally:
            with self._control_lock:
                self._realsense_color_sensor = None
            # RealSense pipeline ownership also stays in its worker thread.
            pipeline.stop()

    def set_manual_white_balance(self, kelvin: float) -> bool:
        """Apply one manual white-balance value to a running D435 stream."""

        if self.spec.kind != "d435":
            return False
        requested = float(kelvin)
        if not np.isfinite(requested) or not 2800.0 <= requested <= 6500.0:
            raise ValueError("D435 white balance must be within [2800, 6500] K")
        with self._control_lock:
            sensor = self._realsense_color_sensor
            if sensor is None:
                return False
            import pyrealsense2 as rs

            if not sensor.supports(rs.option.white_balance):
                raise RuntimeError(
                    f"RealSense color sensor {self.spec.serial} does not support manual white balance"
                )
            if sensor.supports(rs.option.enable_auto_white_balance):
                sensor.set_option(rs.option.enable_auto_white_balance, 0.0)
            sensor.set_option(rs.option.white_balance, requested)
            actual = float(sensor.get_option(rs.option.white_balance))
            self.runtime_properties["auto_white_balance"] = (
                float(sensor.get_option(rs.option.enable_auto_white_balance))
                if sensor.supports(rs.option.enable_auto_white_balance)
                else None
            )
            self.runtime_properties["white_balance"] = actual
            return True

    def set_tint(self, value: float) -> bool:
        """Set software green↔red tint for subsequent D435 frames."""

        requested = float(value)
        if not np.isfinite(requested) or not -100.0 <= requested <= 100.0:
            raise ValueError("D435 tint must be within [-100, 100]")
        if self.spec.kind != "d435":
            return False
        with self._control_lock:
            self._tint = requested
            self.runtime_properties["tint"] = requested
        return True

    def stop(self, timeout: float = 5.0) -> None:
        super().stop(timeout=timeout)

    def snapshot(self) -> dict[str, Any]:
        now_ns = time.monotonic_ns()
        with self._lock:
            count = self._total_frame_count
            buffered_count = len(self.frames)
            last_ns = self.frames[-1].timestamp_monotonic_ns if self.frames else None
            max_gap_ns = self._max_inter_frame_gap_ns
            rate_start_ns = self._gap_monitor_start_ns
            rate_start_count = self._rate_monitor_start_count
        rate_elapsed_sec = (
            (now_ns - rate_start_ns) / 1e9
            if rate_start_ns is not None and now_ns > rate_start_ns
            else 0.0
        )
        measured_hz = (
            max(0, count - rate_start_count) / rate_elapsed_sec
            if rate_start_count is not None and rate_elapsed_sec > 0
            else None
        )
        return {
            "name": self.name,
            "spec": asdict(self.spec),
            "runtime_properties": self.runtime_properties,
            "healthy": self.error is None and count > 0,
            "count": count,
            "buffered_count": buffered_count,
            "max_buffer_frames": self.max_buffer_frames,
            "frame_output_size": self.frame_output_size,
            "last_timestamp_monotonic_ns": last_ns,
            "episode_max_inter_frame_gap_sec": max_gap_ns / 1e9,
            "rate_monitor_elapsed_sec": rate_elapsed_sec,
            "measured_hz": measured_hz,
            "dropped_frame_count": self.dropped_frame_count,
        }

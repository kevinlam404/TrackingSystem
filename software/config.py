from __future__ import annotations
from pathlib import Path
from typing import Any
from math import isfinite
import yaml

REQUIRED_SECTIONS = (
    "camera",
    "detector",
    "behavior",
    "control",
    "gimbal",
    "interface",
)


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("The configuration must contain a YAML mapping")

    missing_sections = [
        section
        for section in REQUIRED_SECTIONS
        if section not in config
    ]

    if missing_sections:
        missing = ", ".join(missing_sections)
        raise ValueError(
            f"Missing configuration sections: {missing}"
        )

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Reject invalid settings before opening a camera or servo controller."""
    for section in REQUIRED_SECTIONS:
        if not isinstance(config.get(section), dict):
            raise ValueError(f'{section} must be a mapping')

    def number(section, key, low, high, integer=False, default=None):
        value = config[section].get(key, default)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not isfinite(value) or not low <= value <= high
                or integer and not isinstance(value, int)):
            raise ValueError(f'{section}.{key} must be {"an integer" if integer else "finite"} in {low}..{high}')
        return value

    if config['camera'].get('source') not in ('webcam', 'picamera'):
        raise ValueError('camera.source must be webcam or picamera')
    for key in ('width', 'height'):
        number('camera', key, 1, 8192, integer=True)
    number('camera', 'fps', 1, 240)
    number('camera', 'device', 0, 255, integer=True, default=0)
    number('detector', 'confidence', 0, 1)
    number('detector', 'image_size', 32, 2048, integer=True)
    for key in ('model_path', 'tracker_config'):
        if not isinstance(config['detector'].get(key), str) or not config['detector'][key]:
            raise ValueError(f'detector.{key} must be a nonempty path')
    number('behavior', 'acquisition_frames', 1, 1000, integer=True)
    number('behavior', 'lost_timeout_seconds', 0, 60)
    number('behavior', 'scan_pan_speed', -1, 1)
    if not isinstance(config['behavior'].get('target_label'), str) or not config['behavior']['target_label']:
        raise ValueError('behavior.target_label must be nonempty')
    number('control', 'update_rate_hz', 1, 100, default=40)
    number('control', 'stale_frame_timeout_seconds', 0.001, 10)
    number('control', 'dead_zone', 0, 1)
    maximum = number('control', 'maximum_pan_speed', 0, 1)
    number('control', 'minimum_pan_speed', 0, maximum)
    number('control', 'tilt_speed_dps', 0.001, 360)
    number('control', 'pan_acceleration_per_second', 0.001, 20, default=1)
    low = number('gimbal', 'tilt_min_degrees', -90, 0)
    high = number('gimbal', 'tilt_max_degrees', 0, 90)
    if low == high:
        raise ValueError('Tilt limits must span a nonzero range')
    if config['gimbal'].get('backend') not in ('simulated', 'pca9685'):
        raise ValueError('gimbal.backend must be simulated or pca9685')
    browser = config['interface'].get('browser', {})
    if not isinstance(browser, dict):
        raise ValueError('interface.browser must be a mapping')
    if browser.get('enabled', False):
        if not isinstance(browser.get('host', '127.0.0.1'), str):
            raise ValueError('Browser host must be a string')
        port = browser.get('port', 8080)
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError('Browser port must be an integer in 1..65535')

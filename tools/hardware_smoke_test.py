from __future__ import annotations

import argparse
from time import sleep

from software.config import load_config
from software.gimbal import PCA9685Gimbal, SimulatedGimbal


def create_gimbal(config: dict, physical: bool):
    gimbal_config = config['gimbal']
    if not physical:
        return SimulatedGimbal(
            gimbal_config['tilt_min_degrees'],
            gimbal_config['tilt_max_degrees'],
        )
    p = gimbal_config['pca9685']
    return PCA9685Gimbal(
        pan_channel=p['pan_channel'],
        tilt_channel=p['tilt_channel'],
        pan_min_pulse_us=p['pan_min_pulse_us'],
        pan_stop_pulse_us=p['pan_stop_pulse_us'],
        pan_max_pulse_us=p['pan_max_pulse_us'],
        tilt_min_pulse_us=p['tilt_min_pulse_us'],
        tilt_max_pulse_us=p['tilt_max_pulse_us'],
        tilt_min_degrees=gimbal_config['tilt_min_degrees'],
        tilt_max_degrees=gimbal_config['tilt_max_degrees'],
        pan_reversed=p['pan_reversed'],
        tilt_reversed=p['tilt_reversed'],
        address=p['address'],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Conservative gimbal smoke test')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--physical', action='store_true')
    parser.add_argument('--i-understand', action='store_true')
    args = parser.parse_args()
    if args.physical and not args.i_understand:
        raise SystemExit(
            'Physical mode requires --i-understand. Remove the camera load, '
            'keep the power switch accessible, and test one axis at a time.'
        )
    gimbal = create_gimbal(load_config(args.config), args.physical)
    try:
        gimbal.enable()
        gimbal.center()
        print('center:', gimbal.get_status())
        for angle in (-10.0, 0.0, 10.0, 0.0):
            gimbal.set_tilt_angle(angle)
            print('tilt:', gimbal.get_status())
            sleep(0.75)
        for speed in (0.05, 0.0, -0.05, 0.0):
            gimbal.set_pan_speed(speed)
            print('pan:', gimbal.get_status())
            sleep(0.5)
        print('PASS: commands completed without a driver exception')
    finally:
        gimbal.disable()
        print('outputs stopped:', gimbal.get_status())


if __name__ == '__main__':
    main()

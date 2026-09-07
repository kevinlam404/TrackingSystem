from __future__ import annotations
import argparse
import board
import busio
from adafruit_pca9685 import PCA9685

PWM_FREQUENCY_HZ = 50
PWM_PERIOD_US = 1_000_000 / PWM_FREQUENCY_HZ

def pulse_us_to_duty_cycle(pulse_us: int) -> int:
    """converts a pulse width in microseconds to a 16 bit duty cycle"""
    duty_cycle=round(
        (pulse_us / PWM_PERIOD_US) * 0xFFFF
    )
    return max(0,min(0xFFFF,duty_cycle))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='safely calibrate one PCA9685 servo channel'
    )

    parser.add_argument(
        '--channel',
        type=int,
        required=True,
        help='pca9685 channel 0 through 15'
    )
    parser.add_argument(
        '--address',
        type=lambda value: int(value,0),
        default=0x40,
        help='pca9685 i2c adress: default: 0x40'
    )
    parser.add_argument(
        '--start-pulse',
        type=int,
        default=1500,
        help='starting pulse width in microseconds',
    )
    parser.add_argument(
        '--step',
        type=int,
        default=5,
        help='pulse adjustment step in microseconds'
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if not 0 <= args.channel <= 15:
        raise ValueError('channel must between 0 and 15')

    pulse_us = args.start_pulse
    step_us = args.step

    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c, address=args.address)
    pca.frequency = PWM_FREQUENCY_HZ
    channel = pca.channels[args.channel]
    print()
    print('SERVO Calibration')
    print('------------------')
    print(f"Channel: {args.channel}")
    print(f"Address: {hex(args.address)}")
    print()
    print('The servo is currently OFF.')
    print()
    print('Commands:')
    print(' on          send the displayed pulse')
    print(' +           increase the pulse by one step')
    print(' -           decrease the pulse by one step')
    print(' step <us>   change the adjustment step')
    print(' pulse <us>  send an exact pulse')
    print(' off         stop sending PWM')
    print(' status      display current settings')
    print(' quit        disable PWM and exit')
    enabled = False

    try:
        while True:
            command = input(
                f"[pulse={pulse_us} us, step={step_us} us, "
                f"enabled={enabled}] > "
            ).strip().lower()

            if command == 'on':
                channel.duty_cycle = pulse_us_to_duty_cycle(
                    pulse_us
                )
                enabled=True
            elif command == '+':
                pulse_us += step_us
                if enabled:
                    channel.duty_cycle=(
                        pulse_us_to_duty_cycle(pulse_us)
                    )
            elif command == '-':
                pulse_us -= step_us
                if enabled:
                    channel.duty_cycle=(
                        pulse_us_to_duty_cycle(pulse_us)
                    )
            elif command.startswith('step '):
                _, value = command.split(maxsplit=1)
                step_us = int(value)
                if step_us <= 0:
                    raise ValueError('step must be greater than zero')
            elif command.startswith('pulse '):
                _, value = command.split(maxsplit=1)
                requested_pulse= int(value)
                #conservative calibration boundary
                if not 900 <= requested_pulse <= 2100:
                    print('rejected: pulse must remain between 900 and 2100 us')
                    continue
                pulse_us = requested_pulse
                if enabled:
                    channel.duty_cycle = (
                        pulse_us_to_duty_cycle(pulse_us)
                    )
            elif command  == 'off':
                channel.duty_cycle = 0
                enabled=False
            elif command == 'status':
                print(f"Pulse: {pulse_us} us")
                print(f"Step: {step_us} us")
                print(f"Enabled: {enabled}")
            elif command in ('quit', 'q', 'exit'):
                break
            elif not command:
                continue
            else:
                print('Unknown command')
    except KeyboardInterrupt:
        print('\nEmergency keyboard stop')

    finally:
        channel.duty_cycle = 0
        pca.deinit()
        print('PWM disabled')


if __name__ == "__main__":
    main()
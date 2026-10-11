from __future__ import annotations
import argparse

PWM_FREQUENCY_HZ = 50
PWM_PERIOD_US = 1_000_000 / PWM_FREQUENCY_HZ
MIN_PULSE_US = 900
MAX_PULSE_US = 2100

def validate_pulse(pulse_us: int) -> int:
    if not MIN_PULSE_US <= pulse_us <= MAX_PULSE_US:
        raise ValueError('pulse must remain between 900 and 2100 us')
    return pulse_us

def pulse_us_to_duty_cycle(pulse_us: int) -> int:
    """converts a pulse width in microseconds to a 16 bit duty cycle"""
    validate_pulse(pulse_us)
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

    pulse_us = validate_pulse(args.start_pulse)
    step_us = args.step
    if step_us <= 0:
        raise ValueError('step must be greater than zero')

    import board
    import busio
    from adafruit_pca9685 import PCA9685

    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c, address=args.address)
    pca.frequency = PWM_FREQUENCY_HZ
    channel = pca.channels[args.channel]
    channel.duty_cycle = 0
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
                if pulse_us + step_us > MAX_PULSE_US:
                    print('Rejected: upper pulse limit reached')
                    continue
                pulse_us = validate_pulse(pulse_us + step_us)
                if enabled:
                    channel.duty_cycle=(
                        pulse_us_to_duty_cycle(pulse_us)
                    )
            elif command == '-':
                if pulse_us - step_us < MIN_PULSE_US:
                    print('Rejected: lower pulse limit reached')
                    continue
                pulse_us = validate_pulse(pulse_us - step_us)
                if enabled:
                    channel.duty_cycle=(
                        pulse_us_to_duty_cycle(pulse_us)
                    )
            elif command.startswith('step '):
                _, value = command.split(maxsplit=1)
                try:
                    requested_step = int(value)
                    if requested_step <= 0:
                        raise ValueError('step must be greater than zero')
                except ValueError as error:
                    print(f'Rejected: {error}')
                    continue
                step_us = requested_step
            elif command.startswith('pulse '):
                _, value = command.split(maxsplit=1)
                try:
                    requested_pulse = validate_pulse(int(value))
                except ValueError as error:
                    print(f'Rejected: {error}')
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
    except (KeyboardInterrupt, EOFError):
        print('\nEmergency keyboard stop')

    finally:
        try:
            channel.duty_cycle = 0
        finally:
            pca.deinit()
        print('PWM disabled')


if __name__ == "__main__":
    main()

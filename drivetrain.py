import math
from collections import namedtuple

from scipy import constants
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

MAX_DISTANCE = 50  # ft

def plot_comparison(comparison):
    plot_drivetrains(
            [
                DefaultDrivetrainFactory.create(heavy=True, fast=True),
                DrivetrainToCompare.create(
                    heavy=True, fast=True, comparison=comparison),
                DefaultDrivetrainFactory.create(heavy=True, fast=False),
                DrivetrainToCompare.create(
                    heavy=True, fast=False, comparison=comparison),
            ],
            max_feet=MAX_DISTANCE
        )


def plot_drivetrains(drivetrains, max_feet=None):
    axes = None
    labels = []
    for dt in drivetrains:
        sim = dt.forward_sim(sim_time=10, init_velocity=0)
        axes = plot_simulation(sim, axes=axes, max_feet=max_feet)

        labels.append(dt.latex_description)

    axes[0].set_title('Drivetrain Characterization')
    axes[0].set_ylabel('Position (ft)')

    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Velocity (ft/s)')

    plt.legend(labels)
    # plt.legend(labels, loc="upper left", bbox_to_anchor=(1,1))
    # plt.tight_layout()


def plot_simulation(sim, axes=None, max_feet=None):
    if axes is None:
        fig, axes = plt.subplots(2, 1, sharex=True)

    time = sim.time
    position_ft = sim.position * 3.28084
    velocity_fps = sim.velocity * 3.28084

    if max_feet is not None:
        for index, p in enumerate(position_ft):
            if p >= max_feet:
                time = time[:index + 1]
                position_ft = position_ft[:index + 1]
                velocity_fps = velocity_fps[:index + 1]
                break

    axes[0].plot(time, position_ft)
    axes[1].plot(time, velocity_fps)

    return axes


class Drivetrain:
    def __init__(
        self,
        mass,
        motor,
        gear_ratio,
        wheel_diameter,
        voltage_bat,
        name=None,
        resistance_bat=None,
        wheel_friction_coef=None,
        current_limit=None
    ):
        if resistance_bat is None:
            resistance_bat = 0

        self._mass = mass
        self.motor = motor
        self.gear_ratio = gear_ratio
        self.wheel_radius = wheel_diameter
        self.voltage_bat = voltage_bat
        self.resistance_bat = resistance_bat
        self.current_limit = current_limit
        self._wheel_friction_coef = wheel_friction_coef

        self._update_slip_force()

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, mass):
        self._mass = mass
        self._update_slip_force()

    @property
    def wheel_friction_coef(self):
        return self._wheel_friction_coef

    @wheel_friction_coef.setter
    def wheel_friction_coef(self, wheel_friction_coef):
        self._wheel_friction_coef = wheel_friction_coef
        self._update_slip_force()

    @property
    def latex_description(self):
        des = r'{0:.1f} ft/s, {1:.0f} lbs'.format(
                self.frictionless_max_velocity * 3.28084, self.mass * 2.2)

        if self.motor.name is not None:
            des = r'{0}, {1}'.format(self.motor.name, des)

        if self.current_limit:
            des = r'{0}, {1} A limit'.format(des, self.current_limit)

        if self.resistance_bat:
            des = r'{0}, ${1}={2} \Omega$'.format(
                    des, 'R_{bat}', self.resistance_bat)

        return des

    @property
    def frictionless_max_velocity(self):
        """
        :returns: The maximum achievable robot velocity (100% efficiency)
        """

        return self.voltage_bat / self.motor.back_emf_const / self.gear_ratio \
            * self.wheel_radius

    def _update_slip_force(self):
        if self.wheel_friction_coef is not None:
            self._slip_force = self.mass * constants.g \
                * self.wheel_friction_coef
        else:
            self._slip_force = None

    def forward_sim(
            self, sim_time=None, init_velocity=None, minimum_steps_num=None):
        """
        Returns the simulated state of a forward moving drivetrain

        :param sim_time: The time to simulate for
        :param velocity_init: The initial velocity in m/s
        :returns: A matrix with the simulated velocities in the first row
            and the simulated positions in the second one
        """

        if sim_time is None:
            sim_time = 10
        if init_velocity is None:
            init_velocity = 0
        if minimum_steps_num is None:
            minimum_steps_num = 100

        solution = solve_ivp(
                fun=self.forward_ode,
                t_span=(0.0, sim_time),
                y0=np.array([init_velocity, 0.0]),
                vectorized=True,
                max_step=sim_time / minimum_steps_num
            )

        return Simulation(
                time=solution['t'],
                position=solution['y'][1],
                velocity=solution['y'][0]
            )

    def forward_ode(self, t, y):
        """
        Defines the system of ODEs for a forward moving drivetrain

        :param t: A scalar of the current time step
        :param y: Shape (n,) or (n,k) of the system variables
        :returns: Time derivatives of the system variables
            with the same shape as y
        """

        shape = y.shape
        y = np.reshape(y, (-1, 1))  # make y 2D if it is not already

        velocity = y[0, :]
        # position = y[1, :]

        omega_motor = velocity / self.wheel_radius * self.gear_ratio  # rad/s

        # TODO: include motor impedance effects
        motor_current = self._motor_current(omega_motor, 0)
        if self.current_limit:
            motor_current = min((motor_current, self.current_limit))

        force = motor_current * self.motor.torque_const * self.gear_ratio \
            / self.wheel_radius
        if self._slip_force is not None:
            force = min((force, self._slip_force))

        return np.reshape(np.vstack((force / self.mass, velocity)), shape)

    def _motor_current(self, omega_motor, motor_current_dot):
        return (self.voltage_bat
                - (self.motor.impedance * motor_current_dot)
                - (self.motor.back_emf_const * omega_motor)) \
                / (self.resistance_bat + self.motor.resistance)


class Motor:
    def __init__(
        self,
        torque_const,
        back_emf_const,
        resistance,
        impedance,
        name=None
    ):
        self.torque_const = torque_const
        self.back_emf_const = back_emf_const
        self.resistance = resistance
        self.impedance = impedance
        self.name = name

    def combine(self, num_motors):
        motor = self

        if motor.name is not None:
            name = "{0}x {1}".format(num_motors, motor.name)
            # unicode times code as an alternative
            # name = u"{0}\u00D7 {1}".format(num_motors, motor.name)
        else:
            name = None

        # only have to adjust resistance and impedance,
        # based on the equivalent values of those components
        # in parallel
        return Motor(
                motor.torque_const,
                motor.back_emf_const,
                motor.resistance / num_motors,
                motor.impedance / num_motors,
                name=name
            )


Simulation = namedtuple('Simulation', 'time, position, velocity')


class MotorFactory:
    motor_list = {
        'CIM': {
            'voltage': 12,  # volts
            'free_speed': 5330,  # RPM
            'free_current': 2.7,  # amps
            'stall_torque': 2.41,  # Nm
            'stall_current': 131,  # amps
            'impedance': 0
            },
        'neo': {           # 10.65 fast_gear, 13.84 slow_gear
            'voltage': 12,  # volts
            'free_speed': 5676,  # RPM
            'free_current': 1.8,  # amps
            'stall_torque': 2.6,  # Nm
            'stall_current': 105,   # amps
            'impedance': 0
        },
        '775pro': {          # 35.08 fast_gear, 45.61 slow_gear
            'voltage': 12,  # volts
            'free_speed': 18700,  # RPM
            'free_current': 0.7,  # amps
            'stall_torque': 0.71,  # Nm
            'stall_current': 134,   # amps
            'impedance': 0
        },
        'miniCIM': {        # 10.96 fast_gear, 14.24 slow_gear
            'voltage': 12,  # volts
            'free_speed': 5840,  # RPM
            'free_current': 3,  # amps
            'stall_torque': 1.4,  # Nm
            'stall_current': 89,   # amps
            'impedance': 0
        }
    }

    @classmethod
    def create(cls, motor_name):
        specs = cls.motor_list[motor_name]
        free_speed = specs['free_speed'] * 2 * math.pi / 60  # rad/s
        voltage = specs['voltage']
        stall_current = specs['stall_current']

        resistance = voltage / stall_current
        torque_const = specs['stall_torque'] / stall_current
        back_emf_const = (voltage - (resistance * specs['free_current'])) \
            / free_speed

        return Motor(
                torque_const,
                back_emf_const,
                resistance,
                specs['impedance'],
                name=motor_name
            )


class DrivetrainComparisonFactory:
    def __init__(
        self,
        mass=None,
        motor=None,
        num_motors=None,
        gear_ratio=None,
        wheel_diameter=None,
        voltage_bat=None,
        resistance_bat=None,
        wheel_friction_coef=None,
        current_limit=None
    ):
        if mass is None:
            mass = 130
        if motor is None:
            motor = MotorFactory.create("CIM")
        if num_motors is None:
            num_motors = 1
        if wheel_diameter is None:
            wheel_diameter = 4  # in
        if voltage_bat is None:
            voltage_bat = 12

        self.mass = mass
        self.motor = motor
        self.num_motors = num_motors
        self.wheel_diameter = wheel_diameter
        self.voltage_bat = voltage_bat
        self.resistance_bat = resistance_bat
        self.wheel_friction_coef = wheel_friction_coef
        self.current_limit = current_limit

    def create(
        self,
        mass=None,
        motor=None,
        num_motors=None,
        gear_ratio=None,
        wheel_diameter=None,
        voltage_bat=None,
        resistance_bat=None,
        wheel_friction_coef=None,
        current_limit=None
    ):
        from collections.abc import Collection
        from itertools import product

        function_args = {
                'mass': mass,
                'motor': motor,
                'num_motors': num_motors,
                'gear_ratio': gear_ratio,
                'wheel_diameter': wheel_diameter,
                'voltage_bat': voltage_bat,
                'resistance_bat': resistance_bat,
                'wheel_friction_coef': wheel_friction_coef,
                'current_limit': current_limit,
            }
        # make any argument that is not a collection into a collection
        for key, value in function_args.items():
            if (
                not isinstance(value, Collection)
                or isinstance(value, str)
            ):
                function_args[key] = (value,)

        # get a list of drivetrains for each combo
        drivetrains = []
        for combo in product(*function_args.values()):
            # if None, set parameters to the default value
            # of the factory instance
            params = {}
            for index, key in enumerate(function_args.keys()):
                combo_value = combo[index]
                if combo_value is None:
                    params[key] = getattr(self, key)
                else:
                    params[key] = combo_value

            motor = params['motor']
            if isinstance(motor, str):
                motor = MotorFactory.create(motor)

            mass_kg = params['mass'] / 2.2
            wheel_diameter_meters = params['wheel_diameter'] * 2.54 / 100

            drivetrains.append(Drivetrain(
                    mass_kg,
                    motor.combine(params['num_motors']),
                    params['gear_ratio'],
                    wheel_diameter_meters,
                    params['voltage_bat'],
                    resistance_bat=params['resistance_bat'],
                    wheel_friction_coef=params['wheel_friction_coef'],
                    current_limit=params['current_limit']
                ))

        return drivetrains


class DrivetrainToCompare:
    num_motors = 4
    wheel_diameter = 4  # in
    voltage_bat = 12
    wheel_friction_coef = 1.1

    fast_gear = 35.08
    slow_gear = 45.61
    heavy_mass = 130  # lbs
    light_mass = 80  # lbs

    @classmethod
    def create(
        cls,
        heavy,
        fast,
        comparison,
        resistance_bat=None,
        current_limit=None
    ):
        motor = MotorFactory.create(comparison)
        num_motors = cls.num_motors
        wheel_diameter = cls.wheel_diameter
        voltage_bat = cls.voltage_bat
        wheel_friction_coef = cls.wheel_friction_coef

        if heavy:
            mass = cls.heavy_mass
        else:
            mass = cls.light_mass

        if fast:
            gear_ratio = cls.fast_gear
        else:
            gear_ratio = cls.slow_gear

        mass_kg = mass / 2.2
        wheel_diameter_meters = wheel_diameter * 2.54 / 100

        return Drivetrain(
                mass_kg,
                motor.combine(num_motors),
                gear_ratio,
                wheel_diameter_meters,
                voltage_bat,
                resistance_bat=resistance_bat,
                wheel_friction_coef=wheel_friction_coef,
                current_limit=current_limit
            )


if __name__ == "__main__":
    main()

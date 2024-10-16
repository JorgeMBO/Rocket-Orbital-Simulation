import numpy as np
import matplotlib.pyplot as plt


class Planet:
    def __init__(self, mass, radius):
        self.mass = mass
        self.radius = radius
        self.position = np.array([0, 0], dtype=float)  # Planet at origin


class SpaceObject:
    def __init__(self, mass, position, velocity):
        self.initial_mass = mass  # Store initial mass
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)


class Rocket(SpaceObject):
    def __init__(self, mass, fuel_mass, position, velocity):
        super().__init__(mass + fuel_mass, position, velocity)
        self.fuel_mass = fuel_mass
        self.time = 0  # Initialize time
        self.thrust_event = None  # Current thrust event
        self.thrust_queue = []  # Queue of scheduled thrust events

    def schedule_thrust_event(self, start_time, duration, thrust, burn_rate, event_type='general'):
        event = {
            'start_time': start_time,
            'end_time': start_time + duration,
            'thrust': thrust,
            'burn_rate': burn_rate,
            'active': False,
            'type': event_type
        }
        self.thrust_queue.append(event)

    def update_mass_and_thrust(self, dt):
        thrust_acc = np.array([0, 0], dtype=float)
        for event in self.thrust_queue:
            if event['start_time'] <= self.time < event['end_time']:
                if not event['active']:
                    event['active'] = True
                    self.thrust_event = event
                # Update thrust_direction every time step
                if event['type'] == 'pitch_maneuver':
                    # First burn: Pitch maneuver
                    pitch_time = event['end_time'] - event['start_time']
                    elapsed_time = self.time - event['start_time']
                    # Angle changes from 90 degrees (vertical) to 0 degrees (horizontal)
                    angle = np.deg2rad(90 - 90 * (elapsed_time / pitch_time))
                    thrust_direction = np.array([np.cos(angle), np.sin(angle)])
                elif event['type'] == 'circularization':
                    # Second burn: Thrust in the direction of current velocity
                    velocity = self.velocity
                    speed = np.linalg.norm(velocity)
                    if speed != 0:
                        thrust_direction = velocity / speed
                    else:
                        thrust_direction = np.array([1, 0])  # Default direction
                else:
                    # Default thrust direction (if any other type)
                    thrust_direction = np.array([0, 1])  # Upward by default
                # Update mass based on burn rate
                if self.fuel_mass > 0:
                    fuel_consumed = min(event['burn_rate'] * dt, self.fuel_mass)
                    self.fuel_mass -= fuel_consumed
                    self.mass -= fuel_consumed  # Update total mass
                    # Thrust acceleration
                    thrust_acc = thrust_direction * event['thrust'] / self.mass
                else:
                    thrust_acc = np.array([0, 0], dtype=float)
                break
        else:
            self.thrust_event = None  # No active thrust event
        return thrust_acc


class Simulation:
    G = 6.67430e-11  # Gravitational constant

    def __init__(self, planet, rocket, time_step, total_time, escape_threshold_multiplier=10):
        self.planet = planet
        self.rocket = rocket
        self.dt = time_step
        self.total_time = total_time
        self.positions = []  # To store all positions for plotting
        self.energies = []   # To store total energy at each time step

        # Set escape threshold based on a multiplier of the planet's radius
        self.escape_threshold = escape_threshold_multiplier * self.planet.radius

    def calculate_acceleration(self, position, thrust_acc):
        distance_vector = position - self.planet.position
        distance = np.linalg.norm(distance_vector)

        # Gravitational acceleration
        if distance == 0:
            gravity_acc = np.array([0, 0], dtype=float)
        else:
            gravity_force_magnitude = (Simulation.G * self.planet.mass) / distance ** 2
            gravity_force_direction = -distance_vector / distance
            gravity_acc = gravity_force_direction * gravity_force_magnitude

        # Total acceleration
        total_acceleration = gravity_acc + thrust_acc
        return total_acceleration

    def calculate_total_energy(self):
        kinetic_energy = 0.5 * self.rocket.mass * np.linalg.norm(self.rocket.velocity) ** 2
        potential_energy = -Simulation.G * self.planet.mass * self.rocket.mass / np.linalg.norm(self.rocket.position)
        return kinetic_energy + potential_energy

    def run(self):
        """Run the simulation using RK4 integration."""
        num_steps = int(self.total_time / self.dt)
        previous_distance = None
        apogee_detected = False

        for step in range(num_steps):
            time = step * self.dt
            self.rocket.time = time  # Update rocket's internal time

            # Update mass and get current thrust acceleration
            thrust_acc = self.rocket.update_mass_and_thrust(self.dt)

            # Store initial values
            r0 = self.rocket.position.copy()
            v0 = self.rocket.velocity.copy()

            # Calculate k1
            a0 = self.calculate_acceleration(r0, thrust_acc)
            k1_v = a0 * self.dt
            k1_r = v0 * self.dt

            # Calculate k2
            r1 = r0 + 0.5 * k1_r
            v1 = v0 + 0.5 * k1_v
            a1 = self.calculate_acceleration(r1, thrust_acc)
            k2_v = a1 * self.dt
            k2_r = v1 * self.dt

            # Calculate k3
            r2 = r0 + 0.5 * k2_r
            v2 = v0 + 0.5 * k2_v
            a2 = self.calculate_acceleration(r2, thrust_acc)
            k3_v = a2 * self.dt
            k3_r = v2 * self.dt

            # Calculate k4
            r3 = r0 + k3_r
            v3 = v0 + k3_v
            a3 = self.calculate_acceleration(r3, thrust_acc)
            k4_v = a3 * self.dt
            k4_r = v3 * self.dt

            # Update position and velocity
            self.rocket.position += (k1_r + 2 * k2_r + 2 * k3_r + k4_r) / 6
            self.rocket.velocity += (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6

            # Store position
            self.positions.append(self.rocket.position.copy())

            # Calculate and store total energy
            total_energy = self.calculate_total_energy()
            self.energies.append(total_energy)

            # Calculate current distance from the planet
            current_distance = np.linalg.norm(self.rocket.position)

            # Apogee detection
            if not apogee_detected:
                if previous_distance is not None and current_distance < previous_distance:
                    # Apogee has been reached
                    apogee_detected = True
                    apogee_time = time
                    print(f"Apogee detected at time {apogee_time} seconds.")

                    # Schedule second burn at apogee
                    self.rocket.schedule_thrust_event(
                        start_time=apogee_time,
                        duration=60,             # Duration of second burn in seconds
                        thrust=5e6,              # Thrust force for circularization
                        burn_rate=500,           # Fuel consumption rate
                        event_type='circularization'
                    )
                else:
                    previous_distance = current_distance

            # Collision check with planet's surface
            distance_from_surface = current_distance - self.planet.radius
            if distance_from_surface <= 0:
                # Check if moving towards the planet
                radial_unit_vector = (self.rocket.position - self.planet.position) / current_distance
                radial_velocity = np.dot(self.rocket.velocity, radial_unit_vector)
                if radial_velocity < 0:
                    collision_time = time
                    print(f"Rocket has crashed into the planet at time {collision_time} seconds.")
                    break

            # Escape threshold check
            if current_distance >= self.escape_threshold:
                escape_time = time
                print(f"Rocket has escaped the planet's gravitational influence at time {escape_time} seconds.")
                break

        else:
            # If the loop completes without breaking, check if in orbit
            print("Simulation completed. Checking if the rocket is in orbit.")
            final_speed = np.linalg.norm(self.rocket.velocity)
            orbital_speed = np.sqrt(Simulation.G * self.planet.mass / current_distance)
            if abs(final_speed - orbital_speed) / orbital_speed < 0.05:
                print("Rocket is in a stable orbit.")
            else:
                print("Rocket is not in a stable orbit.")

    def plot_trajectory(self):
        """Plot the complete trajectory of the rocket."""
        if not self.positions:
            print("No trajectory data to plot.")
            return

        x_positions, y_positions = zip(*self.positions)
        plt.figure(figsize=(8, 8))
        plt.plot(x_positions, y_positions, label='Trajectory')

        # Draw the planet
        planet_circle = plt.Circle((0, 0), self.planet.radius, color='y', label='Planet', alpha=0.5)
        plt.gca().add_artist(planet_circle)

        plt.xlabel("x (meters)")
        plt.ylabel("y (meters)")
        plt.title("Rocket Launch Simulation with Automated Second Burn")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    def plot_energy(self):
        """Plot the total energy of the rocket over time."""
        if not self.energies:
            print("No energy data to plot.")
            return

        time = np.arange(len(self.energies)) * self.dt
        plt.figure()
        plt.plot(time, self.energies)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Total Energy (Joules)")
        plt.title("Total Energy Over Time")
        plt.grid(True)
        plt.show()


# Initialize planet and rocket with initial conditions
earth = Planet(mass=5.972e24, radius=6371e3)

# Rocket initial parameters
rocket_mass = 1e5       # Mass of the rocket without fuel in kg
fuel_mass = 4e5         # Initial fuel mass in kg (includes fuel for second burn)

# Initial position and velocity
initial_position = [0, earth.radius + 1]  # Start just above Earth's surface along y-axis
initial_velocity = [0, 0]                 # Starting at rest

# Create the rocket
rocket = Rocket(
    mass=rocket_mass,
    fuel_mass=fuel_mass,
    position=initial_position,
    velocity=initial_velocity
)

# Schedule the first burn (launch and ascent with pitch maneuver)
rocket.schedule_thrust_event(
    start_time=0,
    duration=150,            # Duration of first burn in seconds
    thrust=22e6,             # Thrust force in Newtons
    burn_rate=2000,          # Fuel consumption rate in kg/s
    event_type='pitch_maneuver'
)

# Run the simulation
sim = Simulation(
    planet=earth,
    rocket=rocket,
    time_step=1,        # Time step in seconds
    total_time=100000,   # Total simulation time in seconds
    escape_threshold_multiplier=30
)

sim.run()
sim.plot_trajectory()
#sim.plot_energy()




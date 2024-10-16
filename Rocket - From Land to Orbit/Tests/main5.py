import numpy as np
import matplotlib.pyplot as plt

# -------------------- Class Definitions --------------------

class Planet:
    def __init__(self, mass, radius):
        self.mass = mass
        self.radius = radius
        self.position = np.array([0, 0], dtype=float)  # Planet at origin

class SpaceObject:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

class Stage:
    def __init__(self, mass, fuel_mass):
        self.mass = mass  # Dry mass of the stage
        self.fuel_mass = fuel_mass  # Fuel mass of the stage
        self.active = False  # Whether the stage's engine is currently active
        self.attached = True  # Whether the stage is still attached to the rocket

class Rocket(SpaceObject):
    def __init__(self, stages, payload_mass, position, velocity, planet):
        super().__init__(position, velocity)
        self.stages = stages  # List of Stage objects
        self.payload_mass = payload_mass  # Payload mass
        self.time = 0  # Initialize time
        self.thrust_queue = []  # Queue of scheduled thrust events
        self.pitch_angles = []  # List to store pitch angles
        self.planet = planet
        self.stage_separation_events = []  # To record stage separation events
        self.burn_events = []  # Track burns for annotation
        self.apogee_events = []  # Track apogees

    @property
    def mass(self):
        # Total mass is the sum of payload mass plus all attached stages' mass and fuel mass
        total_mass = self.payload_mass + sum((stage.mass + stage.fuel_mass) for stage in self.stages if stage.attached)
        return total_mass

    def schedule_thrust_event(self, start_time, duration, burn_rate, Isp, event_type='general',
                              stage_index=None, pitch_exponent=1):
        event = {
            'start_time': start_time,
            'end_time': start_time + duration,
            'burn_rate': burn_rate,
            'Isp': Isp,
            'type': event_type,
            'stage_index': stage_index,
            'pitch_exponent': pitch_exponent
        }
        self.thrust_queue.append(event)
        # Record the start time and position for plotting
        self.burn_events.append({'time': start_time, 'position': self.position.copy(), 'type': event_type})

    def get_thrust_acceleration(self, time, position, velocity):
        thrust_acc = np.array([0, 0], dtype=float)
        for event in self.thrust_queue:
            if event['start_time'] <= time < event['end_time']:
                # Get the stage associated with this event
                stage_index = event.get('stage_index')
                if stage_index is None:
                    # Use the last attached stage if not specified
                    attached_stages = [i for i, s in enumerate(self.stages) if s.attached]
                    if not attached_stages:
                        continue  # No attached stages left
                    stage_index = attached_stages[-1]
                stage = self.stages[stage_index]

                # **Skip calculation if the stage is detached**
                if not stage.attached:
                    continue

                # Determine thrust direction based on event type
                if event['type'] == 'vertical_ascent':
                    thrust_direction = np.array([0, 1])  # Upward
                    self.pitch_angles.append(90)  # Pitch angle is 90 degrees
                elif event['type'] == 'pitch_maneuver':
                    pitch_time = event['end_time'] - event['start_time']
                    elapsed_time = time - event['start_time']
                    fraction = (elapsed_time / pitch_time) ** event['pitch_exponent']
                    fraction = min(fraction, 1.0)
                    angle = np.deg2rad(90 - 90 * fraction)
                    thrust_direction = np.array([np.cos(angle), np.sin(angle)])
                    self.pitch_angles.append(np.rad2deg(angle))
                elif event['type'] == 'circularization':
                    speed = np.linalg.norm(velocity)
                    thrust_direction = velocity / speed if speed != 0 else np.array([1, 0])
                    self.pitch_angles.append(None)
                else:
                    thrust_direction = np.array([0, 1])  # Default upward
                    self.pitch_angles.append(None)

                # Activate the stage's engine if it's not active
                if not stage.active:
                    stage.active = True
                    print(f"Stage {stage_index + 1} engine activated at time {time} seconds.")

                # Check if stage has fuel
                if stage.fuel_mass > 0:
                    # Calculate thrust
                    burn_rate = event['burn_rate']
                    Isp = event['Isp']
                    thrust = Isp * burn_rate * 9.80665  # Thrust = Isp * burn_rate * g0
                    mass = self.mass  # Use current mass
                    if mass <= 0:
                        raise ValueError("Rocket mass has become zero or negative.")
                    thrust_acc += thrust_direction * thrust / mass
                else:
                    # Stage is out of fuel
                    pass  # No thrust acceleration
        return thrust_acc

    def update_mass(self, dt):
        total_fuel_consumed = 0.0
        for event in self.thrust_queue:
            # Calculate overlap between [self.time, self.time + dt] and [event['start_time'], event['end_time']]
            overlap_start = max(self.time, event['start_time'])
            overlap_end = min(self.time + dt, event['end_time'])
            overlap_duration = overlap_end - overlap_start
            if overlap_duration > 0:
                # Get the stage associated with this event
                stage_index = event.get('stage_index')
                if stage_index is None:
                    # Use the last attached stage if not specified
                    attached_stages = [i for i, s in enumerate(self.stages) if s.attached]
                    if not attached_stages:
                        continue  # No attached stages left
                    stage_index = attached_stages[-1]
                stage = self.stages[stage_index]

                # **Check if the stage is attached before proceeding**
                if not stage.attached:
                    continue  # Skip this event

                # Activate the stage's engine if it's not active
                if not stage.active:
                    stage.active = True
                    print(f"Stage {stage_index + 1} engine activated at time {self.time} seconds.")

                if stage.fuel_mass > 0:
                    # Fuel that can be consumed is min(burn_rate * overlap_duration, stage.fuel_mass)
                    fuel_consumed = min(event['burn_rate'] * overlap_duration, stage.fuel_mass)
                    stage.fuel_mass -= fuel_consumed
                    total_fuel_consumed += fuel_consumed
                    # Check if stage runs out of fuel
                    if stage.fuel_mass <= 0 and stage.attached:
                        stage.active = False
                        stage.attached = False
                        speed = np.linalg.norm(self.velocity)
                        altitude = np.linalg.norm(self.position) - self.planet.radius
                        self.stage_separation_events.append({
                            'time': self.time + overlap_duration,
                            'stage_index': stage_index + 1,
                            'speed': speed,
                            'altitude': altitude,
                            'position': self.position.copy()  # Add position here
                        })
                        print(f"Stage {stage_index + 1} jettisoned at time {self.time + overlap_duration} seconds.")
                        print(f"Speed at stage separation: {speed:.2f} m/s")
                        print(f"Altitude at stage separation: {altitude:.2f} meters")
                else:
                    # Stage is out of fuel
                    if stage.attached:
                        print(f"Stage {stage_index + 1} is out of fuel and being jettisoned at time {self.time} seconds.")
                        stage.active = False
                        stage.attached = False
                        print(f"Stage {stage_index + 1} jettisoned at time {self.time} seconds.")
        return total_fuel_consumed

    def plot_pitch_angle(self):
        times = np.arange(len(self.pitch_angles))
        angles = [angle if angle is not None else np.nan for angle in self.pitch_angles]

        plt.figure()
        plt.plot(times, angles)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Pitch Angle (degrees)')
        plt.title('Pitch Angle Over Time')
        plt.grid(True)
        plt.show()

class Simulation:
    G = 6.67430e-11  # Gravitational constant

    def __init__(self, planet, rocket, time_step, total_time, escape_threshold_multiplier=10):
        self.planet = planet
        self.rocket = rocket
        self.dt = time_step
        self.total_time = total_time
        self.positions = []  # To store all positions for plotting
        self.energies = []   # To store total energy at each time step
        self.orbit_counter = 0  # To count the number of orbits
        self.previous_distance = None
        self.previous_trend = None  # 'increasing' or 'decreasing' or None
        self.passed_perigee = False  # Flag to indicate perigee has been passed
        self.ascent_phase_complete = False  # Flag to indicate ascent phase completion

        # Set escape threshold based on a multiplier of the planet's radius
        self.escape_threshold = escape_threshold_multiplier * self.planet.radius

        # Set a threshold altitude to consider ascent phase complete (e.g., 100 km)
        self.ascent_altitude_threshold = self.planet.radius + 100e3  # 100 km above the surface

        # Variables for burn scheduling
        self.apogee_detected = False
        self.third_burn_scheduled = False
        self.second_apogee_detected = False

        # Define a minimum horizontal velocity threshold for apogee detection
        self.horizontal_velocity_threshold = 1000  # in m/s (adjust as needed)

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

        # Atmospheric drag
        altitude = distance - self.planet.radius
        if altitude < 150e3:  # Apply drag below 150 km
            drag_acc = self.calculate_drag(self.rocket.velocity, altitude)
        else:
            drag_acc = np.array([0, 0], dtype=float)

        # Total acceleration
        total_acceleration = gravity_acc + thrust_acc + drag_acc
        return total_acceleration

    def calculate_drag(self, velocity, altitude):
        # In Earth-fixed frame, atmosphere is stationary relative to Earth's surface
        relative_velocity = velocity  # Rocket's velocity relative to atmosphere

        rho0 = 1.225  # Sea level atmospheric density (kg/m^3)
        h_scale = 8500  # Scale height of the atmosphere (m)
        rho = rho0 * np.exp(-altitude / h_scale)  # Exponential atmosphere model
        Cd = 0.5  # Drag coefficient (assumed)
        A = 10  # Cross-sectional area (m^2) (assumed)
        v = np.linalg.norm(relative_velocity)
        if v == 0:
            return np.array([0, 0], dtype=float)
        drag_force = -0.5 * Cd * A * rho * v * relative_velocity / v
        mass = self.rocket.mass
        if mass <= 0:
            raise ValueError("Rocket mass has become zero or negative.")
        drag_acc = drag_force / mass
        return drag_acc

    def calculate_total_energy(self):
        kinetic_energy = 0.5 * self.rocket.mass * np.linalg.norm(self.rocket.velocity) ** 2
        potential_energy = -Simulation.G * self.planet.mass * self.rocket.mass / np.linalg.norm(self.rocket.position)
        return kinetic_energy + potential_energy

    def run(self):
        """Run the simulation using RK4 integration."""
        num_steps = int(self.total_time / self.dt)
        self.previous_distance = np.linalg.norm(self.rocket.position)

        for step in range(num_steps):
            time = step * self.dt
            self.rocket.time = time  # Update rocket's internal time

            # Store initial values
            r0 = self.rocket.position.copy()
            v0 = self.rocket.velocity.copy()

            # -------------------- RK4 Integration Steps --------------------

            # Calculate k1
            a0 = self.calculate_acceleration(r0, self.rocket.get_thrust_acceleration(time, r0, v0))
            k1_v = a0 * self.dt
            k1_r = v0 * self.dt

            # Calculate k2
            t_half = time + 0.5 * self.dt
            r1 = r0 + 0.5 * k1_r
            v1 = v0 + 0.5 * k1_v
            a1 = self.calculate_acceleration(r1, self.rocket.get_thrust_acceleration(t_half, r1, v1))
            k2_v = a1 * self.dt
            k2_r = v1 * self.dt

            # Calculate k3
            r2 = r0 + 0.5 * k2_r
            v2 = v0 + 0.5 * k2_v
            a2 = self.calculate_acceleration(r2, self.rocket.get_thrust_acceleration(t_half, r2, v2))
            k3_v = a2 * self.dt
            k3_r = v2 * self.dt

            # Calculate k4
            t_full = time + self.dt
            r3 = r0 + k3_r
            v3 = v0 + k3_v
            a3 = self.calculate_acceleration(r3, self.rocket.get_thrust_acceleration(t_full, r3, v3))
            k4_v = a3 * self.dt
            k4_r = v3 * self.dt

            # Update position and velocity
            self.rocket.position += (k1_r + 2 * k2_r + 2 * k3_r + k4_r) / 6
            self.rocket.velocity += (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6

            # -------------------- Update Mass and Consume Fuel --------------------
            self.rocket.update_mass(self.dt)

            # Store position and energy
            self.positions.append(self.rocket.position.copy())
            total_energy = self.calculate_total_energy()
            self.energies.append(total_energy)

            # Calculate current distance from the planet
            current_distance = np.linalg.norm(self.rocket.position)

            # -------------------- Check Ascent Phase Completion --------------------
            if not self.ascent_phase_complete and current_distance >= self.ascent_altitude_threshold:
                self.ascent_phase_complete = True
                self.passed_perigee = False  # Reset perigee flag after ascent phase
                print(f"Ascent phase complete at time {time} seconds.")

            # -------------------- Orbit Counting and Trend Detection --------------------
            if self.ascent_phase_complete:
                # Determine the current trend based on distance change
                if self.previous_distance is not None:
                    if current_distance > self.previous_distance:
                        current_trend = 'increasing'
                    elif current_distance < self.previous_distance:
                        current_trend = 'decreasing'
                    else:
                        current_trend = self.previous_trend  # No change

                    # Detect apogee: Trend changes from increasing to decreasing
                    if self.previous_trend == 'increasing' and current_trend == 'decreasing':
                        if not self.apogee_detected:
                            # First apogee detection
                            self.apogee_detected = True
                            apogee_time = time
                            self.rocket.apogee_events.append({
                                'time': apogee_time,
                                'position': self.rocket.position.copy()
                            })
                            print(f"First apogee detected at time {apogee_time} seconds.")

                            # Schedule Stage 3 burn at first apogee
                            self.rocket.schedule_thrust_event(
                                start_time=apogee_time,
                                duration=66.5,  # Adjusted duration (in seconds)
                                burn_rate=500,  # Adjusted burn rate (kg/s)
                                Isp=450,  # Specific impulse in seconds
                                event_type='circularization',
                                stage_index=2  # Explicitly using the third stage (index 2)
                            )
                            self.third_burn_scheduled = True
                        else:
                            # Subsequent apogees after perigee passage
                            if self.passed_perigee and not self.third_burn_scheduled:
                                self.second_apogee_detected = True
                                third_burn_time = time
                                print(f"Second apogee detected at time {third_burn_time} seconds.")

                                # Calculate required delta-V at second apogee
                                orbit_radius = np.linalg.norm(self.rocket.position)
                                mu = Simulation.G * self.planet.mass
                                required_speed = np.sqrt(mu / orbit_radius)
                                current_speed = np.linalg.norm(self.rocket.velocity)
                                delta_v = required_speed - current_speed

                                print(f"Delta-V required for circularization at second apogee: {delta_v:.2f} m/s")

                                # Schedule third burn with the third stage
                                self.rocket.schedule_thrust_event(
                                    start_time=third_burn_time,
                                    duration=130,       # Adjust duration as needed (in seconds)
                                    burn_rate=300,      # Adjust burn rate as needed (kg/s)
                                    Isp=430,            # Specific impulse for third stage in seconds
                                    event_type='circularization',
                                    stage_index=2       # Explicitly using the third stage (index 2)
                                )
                                self.third_burn_scheduled = True

                    # Detect perigee: Trend changes from decreasing to increasing
                    if self.previous_trend == 'decreasing' and current_trend == 'increasing':
                        if self.apogee_detected and not self.passed_perigee:
                            self.orbit_counter += 1
                            self.passed_perigee = True
                            print(f"Perigee passage detected at time {time} seconds. Orbits completed: {self.orbit_counter}")

                    # Update trend
                    self.previous_trend = current_trend
                else:
                    # Initialize trend
                    self.previous_trend = None

            # -------------------- Apogee Detection for Scheduling Burns --------------------
            # Apogee detection is handled within trend detection above

            # -------------------- Collision Check with Planet's Surface --------------------
            distance_from_surface = current_distance - self.planet.radius
            if distance_from_surface <= 0:
                # Check if moving towards the planet
                if current_distance != 0:
                    radial_unit_vector = (self.rocket.position - self.planet.position) / current_distance
                else:
                    radial_unit_vector = np.array([0, 0], dtype=float)
                radial_velocity = np.dot(self.rocket.velocity, radial_unit_vector)
                if radial_velocity < 0:
                    collision_time = time
                    print(f"Rocket has crashed into the planet at time {collision_time} seconds.")
                    break

            # -------------------- Escape Threshold Check --------------------
            if current_distance >= self.escape_threshold:
                escape_time = time
                print(f"Rocket has escaped the planet's gravitational influence at time {escape_time} seconds.")
                # Calculate final velocity
                final_velocity = np.linalg.norm(self.rocket.velocity)
                print(f"Final velocity: {final_velocity / 1000:.2f} km/s")
                break

            # -------------------- Mass Check --------------------
            if self.rocket.mass <= 0:
                print(f"Rocket mass has become zero or negative at time {time} seconds.")
                break

            # -------------------- Detailed Logging Every 100 Seconds --------------------
            if step % max(1, int(100 / self.dt)) == 0:
                altitude = current_distance - self.planet.radius
                speed = np.linalg.norm(self.rocket.velocity)
                horizontal_speed = self.rocket.velocity[0]
                vertical_speed = self.rocket.velocity[1]
                print(f"Time: {time}s, Altitude: {altitude:.2f}m, Speed: {speed:.2f}m/s, "
                      f"Horizontal Speed: {horizontal_speed:.2f}m/s, Vertical Speed: {vertical_speed:.2f}m/s")

            # Update previous distance for next iteration
            self.previous_distance = current_distance

        else:
            # If the loop completes without breaking, check if in orbit
            print("Simulation completed. Checking if the rocket is in orbit.")
            final_speed = np.linalg.norm(self.rocket.velocity)
            if current_distance != 0:
                orbital_speed = np.sqrt(Simulation.G * self.planet.mass / current_distance)
                speed_ratio = abs(final_speed - orbital_speed) / orbital_speed
                if speed_ratio < 0.05:
                    print("Rocket is in a stable orbit.")
                else:
                    print("Rocket is not in a stable orbit.")
            else:
                print("Final distance is zero; cannot determine orbit status.")

        # -------------------- Final Summary --------------------
        final_altitude = current_distance - self.planet.radius
        print(f"Final altitude: {final_altitude / 1000:.2f} km")
        print(f"Total orbits completed: {self.orbit_counter}")

    def plot_trajectory(self):
        if not self.positions:
            print("No trajectory data to plot.")
            return

        x_positions, y_positions = zip(*self.positions)
        plt.figure(figsize=(8, 8))
        plt.plot(x_positions, y_positions, label='Trajectory')

        # Draw the planet
        planet_circle = plt.Circle((0, 0), self.planet.radius, color='y', label='Planet', alpha=0.5)
        plt.gca().add_artist(planet_circle)

        # Mark stage separations
        for event in self.rocket.stage_separation_events:
            plt.scatter(event['position'][0], event['position'][1], color='red', marker='x', s=100,
                        label=f"Stage {event['stage_index']} Separation" if event['stage_index'] == 1 else "")

        # Mark burn events
        for event in self.rocket.burn_events:
            plt.scatter(event['position'][0], event['position'][1], color='blue', marker='o', s=80,
                        label=event['type'].capitalize() + " Burn" if event['type'] == 'vertical_ascent' else "")

        # Mark apogee events
        for event in self.rocket.apogee_events:
            plt.scatter(event['position'][0], event['position'][1], color='green', marker='^', s=100, label="Apogee")

        # Label the axes and add legend
        plt.xlabel("x (meters)")
        plt.ylabel("y (meters)")
        plt.title("Rocket Launch Trajectory with Staging and Burns")
        plt.legend(loc="best")
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

    def plot_altitude(self):
        """Plot the altitude over time."""
        if not self.positions:
            print("No position data to plot.")
            return

        altitudes = [np.linalg.norm(pos) - self.planet.radius for pos in self.positions]
        time = np.arange(len(altitudes)) * self.dt

        plt.figure()
        plt.plot(time, np.array(altitudes) / 1000)  # Convert to kilometers
        plt.xlabel("Time (seconds)")
        plt.ylabel("Altitude (km)")
        plt.title("Altitude Over Time")
        plt.grid(True)
        plt.show()
# -------------------- Main Simulation Code --------------------

# Initialize the planet (Earth)
earth = Planet(
    mass=5.972e24,     # Mass of Earth in kg
    radius=6371e3      # Radius of Earth in meters
)

# Define whether to use the third stage
use_third_stage = True  # Set to False to opt-out of the third stage

# Define rocket stages
# Stage 1 parameters
stage1_mass = 40000       # Dry mass of first stage in kg
stage1_fuel_mass = 500000 # Fuel mass of first stage in kg

# Stage 2 parameters
stage2_mass = 8000        # Dry mass of second stage in kg
stage2_fuel_mass = 100000 # Fuel mass of second stage in kg

# Payload mass
payload_mass = 2000  # Mass of the payload in kg

# Create Stage objects
stage1 = Stage(mass=stage1_mass, fuel_mass=stage1_fuel_mass)
stage2 = Stage(mass=stage2_mass, fuel_mass=stage2_fuel_mass)
stages = [stage1, stage2]

# If using the third stage, define and add it
if use_third_stage:
    # Stage 3 parameters
    stage3_mass = 3000        # Dry mass of third stage in kg
    stage3_fuel_mass = 50000  # Fuel mass of third stage in kg

    # Create Stage object
    stage3 = Stage(mass=stage3_mass, fuel_mass=stage3_fuel_mass)
    stages.append(stage3)

# Initial position and velocity of the rocket
initial_position = [0, earth.radius + 1]  # Just above Earth's surface along y-axis

# In Earth-fixed frame, initial velocity is zero
initial_velocity = [0, 0]  # No initial horizontal velocity

# Create the rocket with the defined stages and payload
rocket = Rocket(
    stages=stages,
    payload_mass=payload_mass,
    position=initial_position,
    velocity=initial_velocity,
    planet=earth
)

# Calculate burn durations based on fuel mass and burn rate
stage1_burn_rate = 3400  # Burn rate for Stage 1
stage1_Isp = 300         # Specific impulse for Stage 1
stage1_burn_duration = stage1.fuel_mass / stage1_burn_rate

# Schedule the first burn: Vertical ascent with Stage 1
vertical_ascent_duration = 90  # Duration for vertical ascent
if vertical_ascent_duration > stage1_burn_duration:
    vertical_ascent_duration = stage1_burn_duration  # Adjust if necessary

rocket.schedule_thrust_event(
    start_time=0,
    duration=vertical_ascent_duration,
    burn_rate=stage1_burn_rate,
    Isp=stage1_Isp,
    event_type='vertical_ascent',
    stage_index=0,
    pitch_exponent=1
)

# Remaining fuel mass in Stage 1 after vertical ascent
stage1_remaining_fuel_mass = stage1.fuel_mass - (stage1_burn_rate * vertical_ascent_duration)
stage1_remaining_burn_duration = stage1_remaining_fuel_mass / stage1_burn_rate

# Schedule pitch maneuver with Stage 1
pitch_maneuver_duration_stage1 = stage1_remaining_burn_duration
if pitch_maneuver_duration_stage1 > 0:
    rocket.schedule_thrust_event(
        start_time=vertical_ascent_duration,
        duration=pitch_maneuver_duration_stage1,
        burn_rate=stage1_burn_rate,
        Isp=stage1_Isp,
        event_type='pitch_maneuver',
        stage_index=0,
        pitch_exponent=0.8
    )

# Schedule pitch maneuver with Stage 2
stage2_burn_rate = 600   # Burn rate for Stage 2
stage2_Isp = 400          # Specific impulse for Stage 2
stage2_burn_duration = stage2.fuel_mass / stage2_burn_rate

pitch_maneuver_start_time_stage2 = vertical_ascent_duration + pitch_maneuver_duration_stage1

rocket.schedule_thrust_event(
    start_time=pitch_maneuver_start_time_stage2,
    duration=stage2_burn_duration,
    burn_rate=stage2_burn_rate,
    Isp=stage2_Isp,
    event_type='pitch_maneuver',
    stage_index=1,
    pitch_exponent=1
)

# Create the simulation object
sim = Simulation(
    planet=earth,
    rocket=rocket,
    time_step=1,        # Time step in seconds
    total_time=20000,   # Total simulation time in seconds
    escape_threshold_multiplier=40  # Threshold to consider the rocket has escaped
)

# Run the simulation
sim.run()

# Plot the trajectory of the rocket
sim.plot_trajectory()

# Plot the pitch angle over time
#rocket.plot_pitch_angle()

# Plot the altitude over time
#sim.plot_altitude()

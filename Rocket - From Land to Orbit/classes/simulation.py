import numpy as np
import matplotlib.pyplot as plt
from utils.plotting import plot_trajectory as plot_traj_util



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
        plot_traj_util(
            positions=self.positions,
            planet_radius=self.planet.radius,
            stage_separation_events=self.rocket.stage_separation_events,
            burn_events=self.rocket.burn_events,
            apogee_events=self.rocket.apogee_events
        )

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
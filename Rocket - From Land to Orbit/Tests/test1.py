import numpy as np
import matplotlib.pyplot as plt


class Planet:
    def __init__(self, mass, radius):
        self.mass = mass
        self.radius = radius
        self.position = np.array([0, 0])  # Planet at origin


class SpaceObject:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

    def update_position(self, dt):
        self.position += self.velocity * dt

    def update_velocity(self, acceleration, dt):
        self.velocity += acceleration * dt


class Simulation:
    G = 6.67430e-11  # Gravitational constant

    def __init__(self, planet, space_object, time_step, total_time, escape_threshold_multiplier=10):
        self.planet = planet
        self.space_object = space_object
        self.dt = time_step
        self.total_time = total_time
        self.positions = []  # To store all positions for plotting

        # Set escape threshold based on a multiplier of the planet's radius
        self.escape_threshold = escape_threshold_multiplier * self.planet.radius

    def calculate_gravitational_force(self):
        distance_vector = self.space_object.position - self.planet.position
        distance = np.linalg.norm(distance_vector)

        force_magnitude = (Simulation.G * self.planet.mass * self.space_object.mass) / distance ** 2
        force_direction = -distance_vector / distance
        return force_magnitude * force_direction

    def run(self):
        """Run the simulation for a fixed total time or until escape/collision occurs."""
        for _ in range(int(self.total_time / self.dt)):
            force = self.calculate_gravitational_force()
            acceleration = force / self.space_object.mass

            self.space_object.update_velocity(acceleration, self.dt)
            self.space_object.update_position(self.dt)

            # Store the current position for plotting
            self.positions.append(self.space_object.position.copy())

            # Calculate current distance from the planet
            current_distance = np.linalg.norm(self.space_object.position)

            # Collision check with planet's surface
            if current_distance <= self.planet.radius:
                print("Object has collided with the planet.")
                break

            # Escape threshold check
            if current_distance >= self.escape_threshold:
                print("Object has escaped Earth's gravitational influence.")
                break

    def plot_trajectory(self):
        """Plot the complete trajectory of the space object."""
        x_positions, y_positions = zip(*self.positions)
        plt.figure(figsize=(8, 8))
        plt.plot(x_positions, y_positions, label='Complete Trajectory')
        plt.plot(0, 0, 'yo', label='Planet')  # Planet at origin
        plt.xlabel("x (meters)")
        plt.ylabel("y (meters)")
        plt.title("Complete Orbit Simulation with Gravitational Attraction")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()


# Initialize planet and space object with slightly higher altitude and adjusted velocity
earth = Planet(mass=5.972e24, radius=6371e3)
satellite = SpaceObject(mass=1000, position=[8e6, 0], velocity=[0, 7200])  # Adjusted for a longer orbit path

# Run the simulation for a longer total time to capture more of the orbit
sim = Simulation(planet=earth, space_object=satellite, time_step=500, total_time=2000000)
sim.run()
sim.plot_trajectory()




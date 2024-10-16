import numpy as np
import matplotlib.pyplot as plt
from classes.spaceobject import SpaceObject



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

# utils/plotting.py

import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(positions, planet_radius, stage_separation_events, burn_events, apogee_events):
    if not positions:
        print("No trajectory data to plot.")
        return

    x_positions, y_positions = zip(*positions)
    plt.figure(figsize=(8, 8))
    plt.plot(x_positions, y_positions, label='Trajectory')

    # Draw the planet
    planet_circle = plt.Circle((0, 0), planet_radius, color='y', label='Planet', alpha=0.5)
    plt.gca().add_artist(planet_circle)

    # Mark stage separations
    for event in stage_separation_events:
        plt.scatter(event['position'][0], event['position'][1], color='red', marker='x', s=100,
                    label=f"Stage {event['stage_index']} Separation" if event['stage_index'] == 1 else "")

    # Mark burn events
    for event in burn_events:
        plt.scatter(event['position'][0], event['position'][1], color='blue', marker='o', s=80,
                    label=event['type'].capitalize() + " Burn" if event['type'] == 'vertical_ascent' else "")

    # Mark apogee events
    for event in apogee_events:
        plt.scatter(event['position'][0], event['position'][1], color='green', marker='^', s=100, label="Apogee")

    # Label the axes and add legend
    plt.xlabel("x (meters)")
    plt.ylabel("y (meters)")
    plt.title("Rocket Launch Trajectory with Staging and Burns")
    plt.legend(loc="best")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

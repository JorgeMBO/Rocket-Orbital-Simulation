
from config import PLANET, PAYLOAD, STAGES, SIMULATION, THRUST_EVENTS, PHYSICS

from classes.planet import Planet
from classes.stage import Stage
from classes.rocket import Rocket
from classes.simulation import Simulation

# -------------------- Main Simulation Code --------------------
def initialize_planet():
    return Planet(
        mass=PLANET['mass'],
        radius=PLANET['radius']
    )

def initialize_stages():
    stages = []
    for stage_params in STAGES:
        stage = Stage(mass=stage_params['mass'], fuel_mass=stage_params['fuel_mass'])
        stages.append(stage)
    return stages

def initialize_rocket(stages, planet):
    initial_position = [0, planet.radius + 1]  # Just above Earth's surface along y-axis
    initial_velocity = [0, 0]  # No initial horizontal velocity
    return Rocket(
        stages=stages,
        payload_mass=PAYLOAD['mass'],
        position=initial_position,
        velocity=initial_velocity,
        planet=planet
    )

def schedule_initial_burns(rocket, stages):
    # Stage 1 Parameters
    stage1 = stages[0]
    stage1_burn_rate = stage1.burn_rate = STAGES[0]['burn_rate']
    stage1_Isp = stage1.Isp = STAGES[0]['Isp']
    stage1_burn_duration = stage1.fuel_mass / stage1_burn_rate

    # Schedule Vertical Ascent
    vertical_ascent_duration = SIMULATION['vertical_ascent_duration']
    if vertical_ascent_duration > stage1_burn_duration:
        vertical_ascent_duration = stage1_burn_duration  # Adjust if necessary

    rocket.schedule_thrust_event(
        start_time=0,
        duration=vertical_ascent_duration,
        burn_rate=stage1_burn_rate,
        Isp=stage1_Isp,
        event_type=THRUST_EVENTS['vertical_ascent']['type'],
        stage_index=0,
        pitch_exponent=1
    )

    # Remaining fuel mass in Stage 1 after vertical ascent
    stage1_remaining_fuel_mass = stage1.fuel_mass - (stage1_burn_rate * vertical_ascent_duration)
    stage1_remaining_burn_duration = stage1_remaining_fuel_mass / stage1_burn_rate

    # Schedule Pitch Maneuver with Stage 1
    if stage1_remaining_burn_duration > 0:
        rocket.schedule_thrust_event(
            start_time=vertical_ascent_duration,
            duration=stage1_remaining_burn_duration,
            burn_rate=stage1_burn_rate,
            Isp=stage1_Isp,
            event_type=THRUST_EVENTS['pitch_maneuver_stage1']['type'],
            stage_index=0,
            pitch_exponent=THRUST_EVENTS['pitch_maneuver_stage1']['pitch_exponent']
        )

    # Stage 2 Parameters
    stage2 = stages[1]
    stage2_burn_rate = stage2.burn_rate = STAGES[1]['burn_rate']
    stage2_Isp = stage2.Isp = STAGES[1]['Isp']
    stage2_burn_duration = stage2.fuel_mass / stage2_burn_rate

    # Schedule Pitch Maneuver with Stage 2
    pitch_maneuver_start_time_stage2 = vertical_ascent_duration + stage1_remaining_burn_duration

    rocket.schedule_thrust_event(
        start_time=pitch_maneuver_start_time_stage2,
        duration=stage2_burn_duration,
        burn_rate=stage2_burn_rate,
        Isp=stage2_Isp,
        event_type=THRUST_EVENTS['pitch_maneuver_stage2']['type'],
        stage_index=1,
        pitch_exponent=1
    )

    # If using the third stage, additional scheduling can be handled within the Simulation class
    # ...

def main():
    # Initialize components
    earth = initialize_planet()
    stages = initialize_stages()
    rocket = initialize_rocket(stages, earth)
    schedule_initial_burns(rocket, stages)

    # Create the simulation object
    sim = Simulation(
        planet=earth,
        rocket=rocket,
        time_step=SIMULATION['time_step'],        # Time step in seconds
        total_time=SIMULATION['total_time'],      # Total simulation time in seconds
        escape_threshold_multiplier=SIMULATION['escape_threshold_multiplier']  # Threshold to consider the rocket has escaped
    )


    # Run the simulation
    sim.run()

    # Plot the trajectory of the rocket
    sim.plot_trajectory()

    # Optionally, plot other graphs
    # rocket.plot_pitch_angle()
    # sim.plot_altitude()

if __name__ == "__main__":
    main()

# config.py

# Planet Parameters
PLANET = {
    'mass': 5.972e24,     # Mass of Earth in kg
    'radius': 6371e3      # Radius of Earth in meters
}

# Payload Parameters
PAYLOAD = {
    'mass': 2000  # Payload mass in kg
}

# Stage Parameters
STAGES = [
    {   # Stage 1
        'mass': 40000,         # Dry mass in kg
        'fuel_mass': 500000,   # Fuel mass in kg
        'burn_rate': 3400,     # Burn rate in kg/s
        'Isp': 300             # Specific impulse in seconds
    },
    {   # Stage 2
        'mass': 8000,
        'fuel_mass': 100000,
        'burn_rate': 600,
        'Isp': 400
    },
    {   # Stage 3 (optional)
        'mass': 3000,
        'fuel_mass': 50000,
        'burn_rate': 300,
        'Isp': 430
    }
]

# Simulation Parameters
SIMULATION = {
    'time_step': 1,              # Time step in seconds
    'total_time': 20000,         # Total simulation time in seconds
    'escape_threshold_multiplier': 40,  # Multiplier for escape threshold
    'vertical_ascent_duration': 90,     # Duration for vertical ascent in seconds
    'pitch_exponent_stage1': 0.8        # Pitch exponent for stage 1
}

# Thrust Event Parameters
THRUST_EVENTS = {
    'vertical_ascent': {
        'type': 'vertical_ascent'
    },
    'pitch_maneuver_stage1': {
        'type': 'pitch_maneuver',
        'pitch_exponent': 0.8
    },
    'pitch_maneuver_stage2': {
        'type': 'pitch_maneuver',
        'pitch_exponent': 1
    },
    'circularization': {
        'type': 'circularization'
    }
}

# Additional Parameters (e.g., drag coefficients)
PHYSICS = {
    'G': 6.67430e-11,          # Gravitational constant
    'drag': {
        'Cd': 0.5,              # Drag coefficient
        'A': 10,                # Cross-sectional area in m^2
        'rho0': 1.225,          # Sea level atmospheric density (kg/m^3)
        'h_scale': 8500         # Scale height in meters
    }
}

# classes/stage.py

class Stage:
    def __init__(self, mass, fuel_mass):
        self.mass = mass  # Dry mass of the stage
        self.fuel_mass = fuel_mass  # Fuel mass of the stage
        self.active = False  # Whether the stage's engine is currently active
        self.attached = True  # Whether the stage is still attached to the rocket

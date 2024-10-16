# classes/planet.py

import numpy as np

class Planet:
    def __init__(self, mass, radius):
        self.mass = mass
        self.radius = radius
        self.position = np.array([0, 0], dtype=float)  # Planet at origin

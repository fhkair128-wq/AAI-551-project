# material.py
"""
Defines the Material class for thermal properties of the plate.
"""

class Material:
    """
    Class representing thermal material properties.
    Attributes:
        k (float): thermal conductivity [W/m·K]
        rho (float): density [kg/m³]
        c (float): specific heat [J/kg·K]
        h (float): convective heat transfer coefficient [W/m²·K]
        q_gen (float): internal heat generation [W/m³]
        T_inf (float): ambient temperature [°C]
        L (float): half thickness of the plate [m]
    """
    def __init__(self, k, rho, c, h, q_gen, T_inf, L):
        try:
            if any(param <= 0 for param in [k, rho, c, h, L]):
                raise ValueError("All physical properties must be positive.")
        except TypeError:
            raise TypeError("Material parameters must be numeric values.")

        self.k = k
        self.rho = rho
        self.c = c
        self.h = h
        self.q_gen = q_gen
        self.T_inf = T_inf
        self.L = L
        self.alpha = k / (rho * c)  # thermal diffusivity

    def __str__(self):
        return (f"Material Properties:\n"
                f"  k = {self.k} W/m·K\n"
                f"  ρ = {self.rho} kg/m³\n"
                f"  c = {self.c} J/kg·K\n"
                f"  h = {self.h} W/m²·K\n"
                f"  q''' = {self.q_gen} W/m³\n"
                f"  T∞ = {self.T_inf} °C\n"
                f"  L = {self.L} m\n"
                f"  α = {self.alpha:.3e} m²/s")

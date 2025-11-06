# heatplate.py
"""
Defines the HeatPlate class to solve 1D unsteady heat conduction using implicit FDM.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from material import Material


class HeatPlate:
    """
    Simulates transient 1D heat conduction in a plate with internal heat generation.
    Composition: contains a Material object.
    """

    def __init__(self, material: Material, n_nodes, dt, t_final):
        self.mat = material
        self.n = n_nodes
        self.dt = dt
        self.t_final = t_final

        self.dx = self.mat.L / self.n
        self.r = self.mat.alpha * self.dt / self.dx ** 2

        # Initialize temperature field
        self.T = np.full((int(self.t_final / self.dt) + 1, self.n + 1),
                         self.mat.T_inf)

        # Metadata storage (mutable object)
        self.metadata = {"method": "Implicit FDM", "status": "Initialized"}

    def assemble_matrix(self):
        """Assemble coefficient matrix A for implicit method."""
        n = self.n
        A = np.zeros((n + 1, n + 1))

        # Boundary node at x = 0 (symmetry)
        A[0, 0] = 1 + 2 * self.r
        A[0, 1] = -2 * self.r

        # Interior nodes
        for i in range(1, n):
            A[i, i - 1] = -self.r
            A[i, i] = 1 + 2 * self.r
            A[i, i + 1] = -self.r

        # Boundary node at x = L (convection)
        A[n, n] = 1 + 2 * self.r + 2 * self.mat.h * self.dt / (self.dx * self.mat.rho * self.mat.c)
        A[n, n - 1] = -2 * self.r

        return A

    def solve_implicit(self):
        """Solve transient heat conduction using implicit FDM."""
        A = self.assemble_matrix()
        b = np.zeros(self.n + 1)

        for t in range(int(self.t_final / self.dt)):
            # Right-hand side vector
            b.fill(self.mat.T_inf + (self.mat.q_gen * self.dt) / (self.mat.rho * self.mat.c))

            # Apply boundary condition at x = L
            b[-1] = (self.mat.T_inf +
                     (2 * self.mat.h * self.mat.T_inf) / (self.dx * self.mat.rho * self.mat.c) +
                     (self.mat.q_gen * self.dt) / (self.mat.rho * self.mat.c))

            # Solve linear system
            x = np.linalg.solve(A, b)
            self.T[t + 1, :] = x

        self.metadata["status"] = "Completed"
        return self.T

    def plot_results(self):
        """Plot temperature distribution and convective heat flux vs. time."""
        time_axis = np.arange(0, self.t_final + self.dt, self.dt)
        x_axis = np.linspace(0, self.mat.L, self.n + 1)

        # Temperature vs time at specific positions
        plt.figure()
        indices = [0, int(self.n / 3), int(2 * self.n / 3), self.n]
        for i in indices:
            plt.plot(time_axis, self.T[:, i], label=f"x={x_axis[i]*100:.1f} cm")
        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (°C)")
        plt.title("Temperature at Different Positions vs Time")
        plt.legend()
        plt.grid(True)

        # Temperature vs position at selected times
        plt.figure()
        times_to_plot = [0.1, 0.25, 0.5, 0.75, 1.0]  # fractions of t_final
        for frac in times_to_plot:
            t_index = int(frac * len(time_axis)) - 1
            plt.plot(x_axis * 100, self.T[t_index, :], label=f"t={time_axis[t_index]:.0f}s")
        plt.xlabel("Position (cm)")
        plt.ylabel("Temperature (°C)")
        plt.title("Temperature Distribution at Different Times")
        plt.legend()
        plt.grid(True)

        # Convective heat flux vs time
        q_conv = self.mat.h * (self.T[:, -1] - self.mat.T_inf)
        plt.figure()
        plt.plot(time_axis, q_conv)
        plt.xlabel("Time (s)")
        plt.ylabel("Convective Heat Flux (W/m²)")
        plt.title("Convective Heat Flux vs Time")
        plt.grid(True)

        plt.show()

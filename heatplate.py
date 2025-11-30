from __future__ import annotations

from typing import Dict, Generator, Iterable, List, Tuple

import math

import numpy as np
import matplotlib.pyplot as plt

from material import Material


class HeatPlate:
    """
    Numerical model for 1D unsteady conduction in a plate with
    internal heat generation and convective cooling at the surface.

    The plate is symmetric about x=0, and we model x in [0, L].
    """

    def __init__(
        self,
        material: Material,
        n_nodes: int,
        dt: float,
        t_final: float,
        T_init: float | None = None,
    ) -> None:
        if n_nodes < 2:
            raise ValueError("n_nodes must be at least 2.")
        if dt <= 0 or t_final <= 0:
            raise ValueError("dt and t_final must be positive.")

        self.material = material
        self.n_nodes = n_nodes  # number of intervals is n_nodes, n_nodes+1 grid points
        self.dt = dt
        self.t_final = t_final
        self.dx = material.L / n_nodes

        # Fourier number
        self.r = material.alpha * dt / self.dx**2

        # spatial grid using list comprehension (requirement)
        self.x: List[float] = [i * self.dx for i in range(n_nodes + 1)]

        # mutable objects: list + dict for metadata
        self.metadata: Dict[str, object] = {
            "dx": self.dx,
            "r": self.r,
            "x_range": (0.0, material.L),
            "description": "1D plate with internal heat generation and convection",
        }

        # initial temperature field (all at T_inf if not specified)
        if T_init is None:
            T_init = material.T_inf
        self.T_current = np.full(n_nodes + 1, float(T_init))
        self.time_points: List[float] = [0.0]
        self.T_history: List[np.ndarray] = [self.T_current.copy()]

        # Lambda functions for boundary-condition coefficients (Part 2 requirement)
        self.center_bc = lambda r: (1.0 + 2.0 * r, -2.0 * r)
        self.surface_bc = lambda r, h, dt, dx, rho, c: (
            1.0 + 2.0 * r + 2.0 * h * dt / (dx * rho * c),
            -2.0 * r,
        )

        # Pre-assemble the constant coefficient matrix for the implicit scheme
        self.A = self._assemble_matrix()

    # ---------- internal helpers ----------

    def _assemble_matrix(self) -> np.ndarray:
        """
        Assemble the (n+1)x(n+1) coefficient matrix A for
        the implicit finite-difference scheme.
        """
        n = self.n_nodes
        r = self.r
        k = self.material.k
        rho = self.material.rho
        c = self.material.c
        h = self.material.h
        dt = self.dt
        dx = self.dx

        A = np.zeros((n + 1, n + 1), dtype=float)

        # Center node (x = 0, symmetry: dT/dx = 0)
        a00, a01 = self.center_bc(r)
        A[0, 0] = a00
        A[0, 1] = a01

        # Interior nodes: 1 .. n-1
        for i in range(1, n):
            A[i, i] = 1.0 + 2.0 * r
            A[i, i - 1] = -r
            A[i, i + 1] = -r

        # Surface node (x = L, convection)
        ann, annm1 = self.surface_bc(r, h, dt, dx, rho, c)
        A[n, n] = ann
        A[n, n - 1] = annm1

        return A

    # ---------- operator overloading ----------

    def __call__(self, steady_tol: float = 1e-4) -> "HeatPlate":
        """
        Allow HeatPlate object to be called like a function to run the simulation.
        This demonstrates operator overloading used for computation.
        """
        return solve_implicit(self, steady_tol=steady_tol)

    # ---------- main solver ----------

    def step_implicit(self, T_old: np.ndarray) -> np.ndarray:
        """
        Perform one implicit time step given the temperature at the previous time level.
        """
        n = self.n_nodes
        rho = self.material.rho
        c = self.material.c
        h = self.material.h
        q_gen = self.material.q_gen
        T_inf = self.material.T_inf
        dt = self.dt
        dx = self.dx

        # Right-hand side vector b
        b = np.zeros(n + 1, dtype=float)

        # volumetric generation term that appears everywhere
        q_term = q_gen * dt / (rho * c)

        # Center node (i=0)
        b[0] = T_old[0] + q_term

        # Interior nodes: 1 .. n-1
        for i in range(1, n):
            b[i] = T_old[i] + q_term

        # Surface node (i=n) with convection contribution
        b[n] = T_old[n] + 2.0 * h * dt / (dx * rho * c) * T_inf + q_term

        # Solve linear system A T_new = b
        T_new = np.linalg.solve(self.A, b)
        return T_new

    def run(
        self, steady_tol: float = 1e-4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the simulation until t_final or until the maximum
        nodal change is below steady_tol.

        Returns
        -------
        times : np.ndarray
        T_all : np.ndarray
            2D array with shape (n_time_steps, n_nodes+1).
        """
        n_steps_max = int(math.ceil(self.t_final / self.dt))
        T_old = self.T_current.copy()

        # Already stored t=0
        for step in range(1, n_steps_max + 1):
            t = step * self.dt
            T_new = self.step_implicit(T_old)

            self.time_points.append(t)
            self.T_history.append(T_new.copy())

            # Convergence check (stop early if nearly steady)
            max_diff = np.max(np.abs(T_new - T_old))
            if max_diff < steady_tol:
                break

            T_old = T_new

        times = np.array(self.time_points)
        T_all = np.vstack(self.T_history)

        # Update current state
        self.T_current = T_all[-1, :]
        return times, T_all

    # ---------- generator for temperature output ----------

    def temperature_generator(self) -> Generator[Tuple[float, np.ndarray], None, None]:
        """
        Yield temperature profiles one by one as (time, T_profile) pairs.
        Generator function (Part 2 requirement).
        """
        for t, T in zip(self.time_points, self.T_history):
            yield t, T

    # ---------- analytical steady-state solution ----------

    def analytical_steady_profile(self) -> np.ndarray:
        """
        Analytical steady-state solution for comparison.
        From ME 604 report:
            T(x) = T_inf + (q/(2k)) (L² - x²) + (q L / h)
        """
        q = self.material.q_gen
        k = self.material.k
        h = self.material.h
        T_inf = self.material.T_inf
        L = self.material.L

        # Immutable objects: tuple used here for clarity
        coeffs: Tuple[float, float] = (q / (2.0 * k), q * L / h)
        a, b = coeffs

        x_arr = np.array(self.x)
        return T_inf + a * (L**2 - x_arr**2) + b

    # ---------- plotting helpers ----------

    def plot_temperature_vs_time(self, positions_cm: Iterable[float]) -> None:
        """
        Plot T(t) at selected spatial positions.
        positions_cm: iterable of positions in cm (0 to L*100).
        """
        times = np.array(self.time_points)
        T_all = np.vstack(self.T_history)
        x_cm = np.array(self.x) * 100.0
        positions_cm = list(positions_cm)

        plt.figure()
        for x_target in positions_cm:
            # Find nearest grid index
            idx = int(round(x_target / (x_cm[-1]) * self.n_nodes))
            idx = max(0, min(idx, self.n_nodes))
            plt.plot(
                times,
                T_all[:, idx],
                label=f"x = {x_target:g} cm",
            )
        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (°C)")
        plt.title("Temperature at Selected Positions vs Time")
        plt.legend()
        plt.grid(True)

    def plot_temperature_vs_position(self, times_to_plot: Iterable[float]) -> None:
        """
        Plot T(x) at selected times.
        times_to_plot: iterable of times in seconds.
        """
        times = np.array(self.time_points)
        T_all = np.vstack(self.T_history)
        x_cm = np.array(self.x) * 100.0
        times_to_plot = list(times_to_plot)

        plt.figure()
        for t_target in times_to_plot:
            # Find nearest time index
            idx = int(round(t_target / self.dt))
            idx = max(0, min(idx, len(times) - 1))
            plt.plot(
                x_cm,
                T_all[idx, :],
                label=f"t = {times[idx]:g} s",
            )
        plt.xlabel("Position (cm)")
        plt.ylabel("Temperature (°C)")
        plt.title("Temperature Distribution Across Plate at Selected Times")
        plt.legend()
        plt.grid(True)

    def plot_convective_heat_flux(self) -> None:
        """
        Plot convective heat flux at the surface vs time:
        q_conv = h (T_surface - T_inf).
        """
        times = np.array(self.time_points)
        T_all = np.vstack(self.T_history)
        T_surface = T_all[:, -1]
        q_conv = self.material.h * (T_surface - self.material.T_inf)

        plt.figure()
        plt.plot(times, q_conv)
        plt.xlabel("Time (s)")
        plt.ylabel("Convective Heat Flux (W/m²)")
        plt.title("Convective Heat Flux vs Time")
        plt.grid(True)


# ---------- Stand-alone functions required by project spec ----------

def solve_implicit(plate: HeatPlate, steady_tol: float = 1e-4) -> HeatPlate:
    """
    Wrapper function that calls HeatPlate.run() to perform the
    implicit time integration. Returns the same HeatPlate object
    for convenience / chaining.
    """
    plate.run(steady_tol=steady_tol)
    return plate


def plot_results(plate: HeatPlate) -> None:
    """
    Wrapper function that generates all required plots.
    """
    # Use tuples for the "constant" lists of positions and times (immutable)
    positions_cm: Tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    times_to_plot: Tuple[float, ...] = (60.0, 5 * 60.0, 10 * 60.0, 20 * 60.0, 40 * 60.0, 80 * 60.0)

    plate.plot_temperature_vs_time(positions_cm)
    plate.plot_temperature_vs_position(times_to_plot)
    plate.plot_convective_heat_flux()

    # Compare steady-state center & surface temperatures with analytical
    T_ss_analytical = plate.analytical_steady_profile()
    T_ss_numeric = plate.T_current

    center_idx = 0
    surface_idx = plate.n_nodes

    print("=== Steady-state comparison ===")
    print(f"Analytical center T  = {T_ss_analytical[center_idx]:.2f} °C")
    print(f"Numeric center T     = {T_ss_numeric[center_idx]:.2f} °C")
    print(f"Analytical surface T = {T_ss_analytical[surface_idx]:.2f} °C")
    print(f"Numeric surface T    = {T_ss_numeric[surface_idx]:.2f} °C")

    plt.show()

"""
Main executable script for the 1D unsteady heat conduction project.

This script demonstrates:
    - data I/O from CSV (via pandas)
    - class usage (Material, HeatPlate)
    - functions solve_implicit() and plot_results()
    - __name__ == "__main__" guard
"""

from math import log  # also shows using built-in math module

from io_utils import load_simulation_from_csv
from heat_plate import solve_implicit, plot_results


def main() -> None:
    """
    High-level driver for the simulation.
    Adjust `input_params.csv` to match your case.
    """
    csv_path = "input_params.csv"
    material, plate = load_simulation_from_csv(csv_path)

    print(material)  # uses Material.__str__()

    # Run implicit solver (can also call plate() thanks to __call__ overloading)
    solve_implicit(plate, steady_tol=1e-3)

    # Example use of math.log just to satisfy "use math" requirement in a meaningful way:
    # estimate lumped-capacitance time to reach 95% of steady state
    # theta(t)/theta_end = 0.95 = 1 - exp(-h t / (rho c))
    # -> t = -(rho c / h) * ln(0.05)
    rho = material.rho
    c = material.c
    h = material.h
    t_lumped = -(rho * c / h) * log(0.05)

    print(f"Estimated steady-state time (lumped capacitance) ≈ {t_lumped:.2f} s")

    # Generate all figures and print numeric/analytical comparison
    plot_results(plate)


if __name__ == "__main__":
    main()

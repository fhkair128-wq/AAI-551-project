from dataclasses import dataclass

@dataclass
class Material:
    """
    Store material and boundary properties for the 1D plate with
    internal heat generation and convective cooling.
    Units should be:
        k      : W/(m·K)
        rho    : kg/m³
        c      : J/(kg·K)
        h      : W/(m²·K)
        q_gen  : W/m³ (volumetric generation)
        T_inf  : °C
        L      : m  (half thickness of the plate)
    """
    k: float
    rho: float
    c: float
    h: float
    q_gen: float
    T_inf: float
    L: float

    def __post_init__(self) -> None:
        # Basic validation with exceptions
        if self.k <= 0 or self.rho <= 0 or self.c <= 0:
            raise ValueError("k, rho and c must all be positive.")
        if self.L <= 0:
            raise ValueError("Plate half-thickness L must be positive.")
        if self.h < 0:
            raise ValueError("Convective coefficient h cannot be negative.")

    @property
    def alpha(self) -> float:
        """Return thermal diffusivity α = k/(ρc)."""
        return self.k / (self.rho * self.c)

    def __mul__(self, factor: float) -> "Material":
        """
        Operator overloading:
        scale the internal heat generation q_gen by a numeric factor,
        returning a *new* Material object (used to study different loads).
        """
        if not isinstance(factor, (int, float)):
            raise TypeError("Material can only be multiplied by a scalar.")
        return Material(
            k=self.k,
            rho=self.rho,
            c=self.c,
            h=self.h,
            q_gen=self.q_gen * factor,
            T_inf=self.T_inf,
            L=self.L,
        )

    __rmul__ = __mul__

    def __str__(self) -> str:
        """Neatly formatted description of all properties."""
        lines = [
            "Material properties:",
            f"  k      = {self.k:.3g} W/(m·K)",
            f"  rho    = {self.rho:.3g} kg/m³",
            f"  c      = {self.c:.3g} J/(kg·K)",
            f"  h      = {self.h:.3g} W/(m²·K)",
            f"  q_gen  = {self.q_gen:.3g} W/m³",
            f"  T_inf  = {self.T_inf:.3g} °C",
            f"  L      = {self.L:.3g} m (half thickness)",
            f"  alpha  = {self.alpha:.3e} m²/s",
        ]
        return "\n".join(lines)

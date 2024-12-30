import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import norm
from typing import Tuple, Dict, Optional, Callable


class MeanFieldAnalyzer:
    """
    Implements mean field theory calculations for minimalRNN analysis.
    Paper reference: "Dynamical Isometry and a Mean Field Theory of RNNs:
    Gating Enables Signal Propagation in Recurrent Neural Networks"
    """

    def __init__(
            self,
            sigma_w: float = 6.88,  # From paper
            sigma_v: float = 1.39,  # From paper
            R: float = 0.46,  # Input variance
            mu_b: float = 0.0  # Bias mean
    ):
        self.sigma_w = sigma_w
        self.sigma_v = sigma_v
        self.R = R
        self.mu_b = mu_b

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid implementation."""
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            return np.where(
                x >= 0,
                1 / (1 + np.exp(-x)),
                np.exp(x) / (1 + np.exp(x))
            )

    @staticmethod
    def sigmoid_prime(x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid."""
        s = MeanFieldAnalyzer.sigmoid(x)
        return s * (1 - s)

    def gaussian_integral(
            self,
            func: Callable[[np.ndarray], np.ndarray],
            mean: float,
            variance: float,
            bounds: Tuple[float, float] = (-10, 10),
            num_points: int = 1000
    ) -> float:
        """
        Compute integral of function under Gaussian measure.

        Args:
            func: Function to integrate
            mean: Mean of Gaussian measure
            variance: Variance of Gaussian measure
            bounds: Integration bounds
            num_points: Number of points for numerical integration

        Returns:
            Integral value
        """
        x = np.linspace(bounds[0], bounds[1], num_points)
        z = np.sqrt(max(variance, 1e-10)) * x + mean
        pdf = norm.pdf(x)
        return trapezoid(func(z) * pdf, x)

    def compute_expectations(
            self,
            q_star: float,
            mu_b: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Compute E[u²] and E[u'²] for given q*.

        Args:
            q_star: Fixed point variance
            mu_b: Optional override for bias mean

        Returns:
            E_u_sq: Expected squared gate value
            E_uprime_sq: Expected squared gate derivative
        """
        if mu_b is None:
            mu_b = self.mu_b

        def gate_squared(x):
            return self.sigmoid(x) ** 2

        def gate_derivative_squared(x):
            return self.sigmoid_prime(x) ** 2

        E_u_sq = self.gaussian_integral(gate_squared, mu_b, q_star)
        E_uprime_sq = self.gaussian_integral(gate_derivative_squared, mu_b, q_star)

        return E_u_sq, E_uprime_sq

    def compute_Q_star(self, E_u_sq: float) -> float:
        """
        Compute Q* using equation 7 from paper.

        Args:
            E_u_sq: Expected squared gate value

        Returns:
            Q*: Fixed point hidden state variance
        """
        denominator = 1 - E_u_sq
        if abs(denominator) < 1e-10:
            raise ValueError("Denominator too close to zero in Q* calculation")
        return self.R * E_u_sq / denominator

    def compute_chi_1(
            self,
            Q_star: float,
            E_uprime_sq: float
    ) -> float:
        """
        Compute χ₁ (signal propagation coefficient).

        Args:
            Q_star: Fixed point hidden state variance
            E_uprime_sq: Expected squared gate derivative

        Returns:
            χ₁: Signal propagation coefficient
        """
        return self.sigma_w ** 2 * (Q_star + self.R) * E_uprime_sq

    def verify_parameters(
            self,
            q_star: float,
            mu_b: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Verify all parameter relationships and compute key quantities.

        Args:
            q_star: Fixed point variance to verify
            mu_b: Optional override for bias mean

        Returns:
            Dictionary containing verification results
        """
        # Compute expectations
        E_u_sq, E_uprime_sq = self.compute_expectations(q_star, mu_b)

        # Compute Q*
        Q_star = self.compute_Q_star(E_u_sq)

        # Compute χ₁
        chi_1 = self.compute_chi_1(Q_star, E_uprime_sq)

        # Verify variance equation
        q_star_calc = Q_star * self.sigma_w ** 2 + self.R * self.sigma_v ** 2

        return {
            'q_star': q_star,
            'Q_star': Q_star,
            'chi_1': chi_1,
            'E_u_sq': E_u_sq,
            'E_uprime_sq': E_uprime_sq,
            'q_star_calc': q_star_calc,
            'q_star_error': abs(q_star_calc - q_star),
            'chi_1_target_error': abs(chi_1 - 0.589)  # Based on our findings
        }


if __name__ == "__main__":
    # Example usage and basic verification
    analyzer = MeanFieldAnalyzer()

    # Test with paper's approximate q* value
    result = analyzer.verify_parameters(16.0)

    print("\nParameter Verification Results:")
    print("-" * 40)
    for key, value in result.items():
        print(f"{key:15s} = {value:.6f}")
import numpy as np
from scipy.integrate import trapezoid, dblquad
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from scipy.stats import norm
import matplotlib
matplotlib.use("TkAgg")  # Or use another compatible backend



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


@dataclass
class NetworkParameters:
    """Container for network parameters"""
    sigma_w: float
    sigma_v: float
    R: float
    mu_b: float
    q_star: float


class ExtendedMeanFieldAnalyzer:
    """Extension of MeanFieldAnalyzer with additional analysis capabilities"""

    def __init__(self, params: NetworkParameters):
        self.params = params
        self.base_analyzer = MeanFieldAnalyzer(
            sigma_w=params.sigma_w,
            sigma_v=params.sigma_v,
            R=params.R,
            mu_b=params.mu_b
        )

    def compute_cosine_similarity(self, Sigma: float) -> float:
        """
        Compute cosine similarity evolution for given input correlation
        Args:
            Sigma: Input correlation coefficient
        Returns:
            Predicted cosine similarity
        """

        def integrand(z1: float, z2: float) -> float:
            u1 = np.sqrt(self.params.q_star) * z1 + self.params.mu_b
            u2 = np.sqrt(self.params.q_star) * (Sigma * z1 + np.sqrt(1 - Sigma ** 2) * z2) + self.params.mu_b
            s1 = self.base_analyzer.sigmoid(u1)
            s2 = self.base_analyzer.sigmoid(u2)
            return s1 * s2 * np.exp(-0.5 * (z1 ** 2 + z2 ** 2)) / (2 * np.pi)

        result, _ = dblquad(integrand, -5, 5, lambda x: -5, lambda x: 5)
        return result

    def analyze_stability_region(self, mu_b_range: np.ndarray,
                                 sigma_w_range: np.ndarray) -> np.ndarray:
        """
        Analyze stability across parameter ranges
        Returns:
            2D array of chi_1 values
        """
        stability_map = np.zeros((len(mu_b_range), len(sigma_w_range)))

        for i, mu_b in enumerate(mu_b_range):
            for j, sigma_w in enumerate(sigma_w_range):
                analyzer = MeanFieldAnalyzer(
                    sigma_w=sigma_w,
                    sigma_v=self.params.sigma_v,
                    R=self.params.R,
                    mu_b=mu_b
                )
                result = analyzer.verify_parameters(self.params.q_star)
                stability_map[i, j] = result['chi_1']

        return stability_map

    def plot_stability_analysis(self, mu_b_range: np.ndarray,
                                sigma_w_range: np.ndarray):
        """Generate stability analysis plots"""
        stability_map = self.analyze_stability_region(mu_b_range, sigma_w_range)

        plt.figure(figsize=(10, 8))
        plt.imshow(stability_map, extent=[sigma_w_range[0], sigma_w_range[-1],
                                          mu_b_range[0], mu_b_range[-1]],
                   aspect='auto', origin='lower')
        plt.colorbar(label='χ₁')
        plt.xlabel('σw')
        plt.ylabel('μb')
        plt.title('Stability Analysis (χ₁)')

        # Add contour for χ₁ = 0.589 (paper's working value)
        plt.contour(sigma_w_range, mu_b_range, stability_map,
                    levels=[0.589], colors='r', linestyles='--')

        return stability_map

    @staticmethod
    def compare_vanilla_minimal(vanilla_params: NetworkParameters,
                                minimal_params: NetworkParameters,
                                sequence_length: int = 100):
        """Compare gradient propagation between vanilla and minimal RNN"""
        vanilla = MeanFieldAnalyzer(
            sigma_w=vanilla_params.sigma_w,
            sigma_v=vanilla_params.sigma_v,
            R=vanilla_params.R,
            mu_b=vanilla_params.mu_b
        )
        minimal = MeanFieldAnalyzer(
            sigma_w=minimal_params.sigma_w,
            sigma_v=minimal_params.sigma_v,
            R=minimal_params.R,
            mu_b=minimal_params.mu_b
        )

        v_result = vanilla.verify_parameters(vanilla_params.q_star)
        m_result = minimal.verify_parameters(minimal_params.q_star)

        # Compute gradient norm evolution
        steps = np.arange(sequence_length)
        v_grads = v_result['chi_1'] ** steps
        m_grads = m_result['chi_1'] ** steps

        plt.figure(figsize=(10, 6))
        plt.semilogy(steps, v_grads, label='Vanilla RNN')
        plt.semilogy(steps, m_grads, label='MinimalRNN')
        plt.xlabel('Sequence Length')
        plt.ylabel('Gradient Norm (log scale)')
        plt.title('Gradient Propagation Analysis')
        plt.legend()
        plt.grid(True)

        return v_grads, m_grads


# Example usage
if __name__ == "__main__":
    params = NetworkParameters(
        sigma_w=6.88,
        sigma_v=1.39,
        R=0.46,
        mu_b=0.0,
        q_star=16.0
    )

    analyzer = ExtendedMeanFieldAnalyzer(params)

    # Analyze stability region
    mu_b_range = np.linspace(-4, 8, 50)
    sigma_w_range = np.linspace(0.5, 10, 50)
    print("Starting stability analysis...")

    stability_map = analyzer.plot_stability_analysis(mu_b_range, sigma_w_range)
    print("finishing stability analysis...")
    plt.savefig("stability_analysis.png")



    # Compare vanilla and minimal RNN
    vanilla_params = NetworkParameters(sigma_w=1.0, sigma_v=0.025, R=0.46, mu_b=0.0, q_star=1.0)
    v_grads, m_grads = ExtendedMeanFieldAnalyzer.compare_vanilla_minimal(
        vanilla_params, params
    )
    plt.show()  # Ensure plot is displayed

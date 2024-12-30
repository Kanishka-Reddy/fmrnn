import numpy as np
from typing import Callable, Tuple, Optional
from dataclasses import dataclass


@dataclass
class IntegrationResult:
    """Container for integration results with error estimates."""
    value: float
    error_estimate: float
    converged: bool


class NumericalIntegrator:
    """
    Implements robust numerical integration strategies for mean field calculations.
    """

    def __init__(self, rtol: float = 1e-8, atol: float = 1e-8):
        self.rtol = rtol
        self.atol = atol

    def adaptive_gaussian_integral(
            self,
            func: Callable[[np.ndarray], np.ndarray],
            mean: float,
            variance: float,
            initial_points: int = 1000,
            max_points: int = 10000,
            bounds: Tuple[float, float] = (-10, 10)
    ) -> IntegrationResult:
        """
        Adaptively compute integral with error estimation and convergence checking.

        Args:
            func: Function to integrate
            mean: Mean of Gaussian measure
            variance: Variance of Gaussian measure
            initial_points: Initial number of integration points
            max_points: Maximum number of points to try
            bounds: Integration bounds

        Returns:
            IntegrationResult containing value, error estimate, and convergence flag
        """
        # Compute with initial points
        result1 = self._fixed_point_gaussian_integral(
            func, mean, variance, initial_points, bounds
        )

        # Compute with double points
        result2 = self._fixed_point_gaussian_integral(
            func, mean, variance, initial_points * 2, bounds
        )

        # Check convergence
        error = abs(result2 - result1)
        tol = self.atol + self.rtol * abs(result2)

        if error < tol:
            return IntegrationResult(result2, error, True)

        # If not converged, try with more points
        points = initial_points * 4
        while points <= max_points:
            result3 = self._fixed_point_gaussian_integral(
                func, mean, variance, points, bounds
            )

            error = abs(result3 - result2)
            tol = self.atol + self.rtol * abs(result3)

            if error < tol:
                return IntegrationResult(result3, error, True)

            result2 = result3
            points *= 2

        # Failed to converge
        return IntegrationResult(result2, error, False)

    @staticmethod
    def _fixed_point_gaussian_integral(
            func: Callable[[np.ndarray], np.ndarray],
            mean: float,
            variance: float,
            num_points: int,
            bounds: Tuple[float, float]
    ) -> float:
        """
        Compute integral with fixed number of points.
        """
        x = np.linspace(bounds[0], bounds[1], num_points)
        z = np.sqrt(max(variance, 1e-10)) * x + mean
        pdf = 1 / np.sqrt(2 * np.pi) * np.exp(-x ** 2 / 2)
        y = func(z) * pdf

        # Trapezoidal rule with endpoint corrections
        dx = x[1] - x[0]
        integral = dx * (np.sum(y[1:-1]) + 0.5 * (y[0] + y[-1]))

        return integral

    def monte_carlo_gaussian_integral(
            self,
            func: Callable[[np.ndarray], np.ndarray],
            mean: float,
            variance: float,
            num_samples: int = 10000,
            seed: Optional[int] = None
    ) -> IntegrationResult:
        """
        Monte Carlo integration for validation and comparison.

        Args:
            func: Function to integrate
            mean: Mean of Gaussian measure
            variance: Variance of Gaussian measure
            num_samples: Number of Monte Carlo samples
            seed: Random seed for reproducibility

        Returns:
            IntegrationResult containing value and error estimate
        """
        rng = np.random.RandomState(seed)

        # Generate samples from standard normal
        x = rng.randn(num_samples)

        # Transform to desired distribution
        z = np.sqrt(max(variance, 1e-10)) * x + mean

        # Compute function values
        y = func(z)

        # Compute mean and standard error
        value = np.mean(y)
        error = np.std(y) / np.sqrt(num_samples)

        return IntegrationResult(value, error, True)


def test_integrator():
    """Basic tests for the integrator."""
    integrator = NumericalIntegrator()

    # Test with simple Gaussian integral (should be 1)
    def test_func(x):
        return np.ones_like(x)

    result = integrator.adaptive_gaussian_integral(test_func, 0, 1)
    print(f"Test integral result: {result.value:.6f} (should be ≈ 1)")
    print(f"Error estimate: {result.error_estimate:.2e}")
    print(f"Converged: {result.converged}")

    # Compare with Monte Carlo
    mc_result = integrator.monte_carlo_gaussian_integral(test_func, 0, 1)
    print(f"\nMonte Carlo result: {mc_result.value:.6f}")
    print(f"Monte Carlo error estimate: {mc_result.error_estimate:.2e}")


if __name__ == "__main__":
    test_integrator()
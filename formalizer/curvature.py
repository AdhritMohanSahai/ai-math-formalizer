import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable

@dataclass
class CurvatureResult:
    """Holds the geometric curvature output across the functional space."""
    x_points: jnp.ndarray
    kappa: jnp.ndarray
    max_curvature: float

class LogarithmicCurvature:
    """
    Computes the logarithmic curvature of mathematical spaces using
    JAX automatic differentiation for infinite-precision calculus.
    """
    def __init__(self):
        # Initialize any spatial smoothing constraints here if needed later
        pass

    def analyze(self, func: Callable, x_domain: jnp.ndarray) -> CurvatureResult:
        """
        Evaluates the exact analytical curvature kappa for a given function.
        """
        # 1. Use JAX autograd to instantly generate derivative functions
        # We use vmap (vectorizing map) so these functions can process entire arrays at once
        df = jax.vmap(jax.grad(func))
        ddf = jax.vmap(jax.grad(jax.grad(func)))
        
        # 2. Compute exact derivatives over the domain
        f_prime = df(x_domain)
        f_double_prime = ddf(x_domain)

        # 3. Apply the formal differential geometric curvature formula
        # kappa = |f''(x)| / (1 + (f'(x))^2)^(3/2)
        numerator = jnp.abs(f_double_prime)
        denominator = jnp.power(1.0 + jnp.square(f_prime), 1.5)
        kappa = numerator / denominator

        return CurvatureResult(
            x_points=x_domain,
            kappa=kappa,
            max_curvature=float(jnp.max(kappa))
        )

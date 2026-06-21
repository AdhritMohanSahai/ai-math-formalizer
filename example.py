import jax.numpy as jnp
from formalizer.growth import AsymptoticClassifier
from formalizer.curvature import LogarithmicCurvature

# 1. Define the mathematical function for autograd analysis
def target_function(x):
    """The function y = x * log(x)"""
    return x * jnp.log(x)

# Define the analytical domain (start slightly above 0 to avoid log(0) errors)
x_domain = jnp.linspace(1.1, 100.0, 500)
y_values = target_function(x_domain)

print("--- AI-MATH-FORMALIZER DIAGNOSTICS ---")

# 2. Run the Asymptotic Growth Classifier
classifier = AsymptoticClassifier()
growth_result = classifier.analyze(x_domain, y_values)
print(f"Predicted Growth Bound: {growth_result.bound}")

# 3. Run the Logarithmic Curvature Operator
curvature_operator = LogarithmicCurvature()
curve_result = curvature_operator.analyze(target_function, x_domain)
print(f"Maximum Functional Curvature: {curve_result.max_curvature:.4f}")
print("--------------------------------------")

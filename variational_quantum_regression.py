"""
Variational Quantum Regression
Converted and patched from variational-quantum-regression.ipynb
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Global data setup
X_DATA = np.arange(0, 8, 1)
# Adding a slight curve to make optimization interesting
Y_DATA = X_DATA + np.random.uniform(-0.5, 0.5, 8)
Y_NORM = np.linalg.norm(Y_DATA)
Y_NORMALIZED = Y_DATA / Y_NORM


def quantum_inner_prod(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes exact theoretical inner product via Qiskit Statevector."""
    if len(vec1) != len(vec2):
        raise ValueError('Lengths of states are not equal')

    N = len(vec1)
    nqubits = math.ceil(np.log2(N))

    # Concatenate and normalize for state preparation
    vec = np.concatenate((vec1, vec2)).astype(np.float64) / np.sqrt(2.0)
    vec = vec / np.linalg.norm(vec)

    circ = QuantumCircuit(nqubits + 1)
    circ.initialize(vec.tolist(), range(nqubits + 1))
    circ.h(nqubits)

    # Qiskit 1.0+ standard statevector execution
    state = Statevector(circ)
    amplitudes = np.real(state.data)

    p0 = np.sum(amplitudes[:N] ** 2)
    return float(2 * p0 - 1)


def calculate_cost_function(parameters: list[float]) -> float:
    """
    Cost function for linear regression: y = ax + b
    Tests the fidelity of the ansatz against the target Y vector.
    """
    a, b = parameters

    # Compute ansatz
    ansatz = a * X_DATA + b
    ansatz_norm = np.linalg.norm(ansatz)

    if math.isclose(ansatz_norm, 0.0):
        return 1.0  # Max penalty for zero vector

    ansatz_normalized = ansatz / ansatz_norm

    # Quantum circuit tests similarity
    y_ansatz = (ansatz_norm / Y_NORM) * quantum_inner_prod(Y_NORMALIZED, ansatz_normalized)

    # Cost to minimize
    return (1 - y_ansatz) ** 2


def main():
    print("Starting variational quantum linear regression...")
    x0 = [0.5, 0.5]  # Initial guesses for [a, b]

    optimizers = ["BFGS", "COBYLA", "Nelder-Mead"]
    results = {}

    for opt in optimizers:
        print(f"Running optimizer: {opt}...")
        res = minimize(
            calculate_cost_function,
            x0=x0,
            method=opt,
            options={'maxiter': 200},
            tol=1e-6
        )
        results[opt] = res['x']
        print(f"  Result [a, b]: {res['x']}")

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_DATA, Y_DATA, color='black', label='Original Data')

    xfit = np.linspace(min(X_DATA), max(X_DATA), 100)

    colors = ['blue', 'red', 'green']
    for idx, opt in enumerate(optimizers):
        a, b = results[opt]
        plt.plot(xfit, a * xfit + b, label=f'{opt} Fit', color=colors[idx], linestyle='--')

    plt.legend()
    plt.title("Variational Quantum Regression (Linear Fit)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)

    print("\nDisplaying plot...")
    plt.show()


if __name__ == "__main__":
    main()
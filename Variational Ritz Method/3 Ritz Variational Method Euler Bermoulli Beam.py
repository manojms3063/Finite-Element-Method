import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

N = 10
X = np.linspace(0, 1, N)
h = X[1] - X[0]
x0 = 0
x1 = 1
L = x1 - x0
x = sp.Symbol('x')
E = 1
I = 1
q0 = 1
M0 = 1
F0 = 1

# Construct stiffness matrix K using second derivatives
K = np.zeros((N, N))
for i in range(1, N + 1):
    for j in range(1, N + 1):
        p_i = x**(i + 1)
        p_j = x**(j + 1)
        d2p_i = sp.diff(p_i, x, 2)
        d2p_j = sp.diff(p_j, x, 2)
        K[i - 1, j - 1] = float(sp.integrate(E * I * d2p_i * d2p_j, (x, x0, x1)))

# Construct force vector F
F = np.zeros(N)
for i in range(1, N + 1):
    p_i = x**(i + 1)
    F[i - 1] = (
        q0 * float(sp.integrate(p_i, (x, x0, x1)))
        + (L) ** (i + 1) * F0
        - (i + 1) * (L) ** i * M0
    )

# Solve for coefficients
c = np.linalg.solve(K, F)
print("Ritz Coefficients:", c)

# Construct approximate solution
y = 0
for i in range(1, N + 1):
    y += c[i - 1] * (x ** i)

# Convert symbolic y(x) to a numerical function
y_func = sp.lambdify(x, y, 'numpy')

# Evaluate solution
print("Solution evaluated at X:", y_func(X))

# Plotting only approximate solution
x_plot = np.linspace(x0, x1, N)  # use more points for smooth curve
plt.figure(figsize=(10, 6))

plt.plot(
    x_plot,
    y_func(x_plot),
    'b--o',
    label='Ritz Approximate Solution',
    markersize=5,
    linewidth=2
)

plt.rcParams['font.family'] = 'Times New Roman'
plt.xlabel("x", fontsize=12)
plt.ylabel("y(x)", fontsize=12)
plt.legend(loc='best', fontsize=10)  # Legend with improved positioning
plt.grid(True, linestyle='--', alpha=0.6)  # Dashed grid lines with transparency
plt.tight_layout()  # Prevent label cutoff
# Save the figure with 500 DPI
plt.savefig('Problem 3.png', dpi=700)
plt.show()

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

N = 10
X = np.linspace(0, 1, N)
# h = X[1]-X[0]
x0 = 0
x1 = 1
x = sp.Symbol('x')
# p = x * (1 - x)
# dp = sp.diff(p, x)

# # Basis functions and evaluation
# for i in range(1,N+1):
#     p = x**(i) - x**(i+1)
#     # print(p)
#     # Evaluate current p at X points
#     P = sp.lambdify(x, p, 'numpy')
#     # print(f"p_{i} evaluated at X:", P(X))

K = np.zeros((N, N))  
for i in range(1, N+1):
    for j in range(1, N+1):
        p_i = x**(i)
        p_j = x**(j)
        dp_i = sp.diff(p_i, x)
        dp_j = sp.diff(p_j, x)
        K[i-1,j-1] = float(sp.integrate((dp_i * dp_j)-(p_i*p_j), (x, x0, x1)))
# print(K)

F = np.zeros(N)
for i in range(1, N+1):
    p_i = x**(i)
    F[i-1] = -float(sp.integrate(x**2*p_i,(x, x0, x1)))+1
# print(F)
c = np.linalg.solve(K,F)
print(c)


y = 0
for i in range(1,N+1):
    y += c[i-1]*(x**(i))  # Sum all basis functions with coefficients

# Convert to numerical function for evaluation
y_func = sp.lambdify(x, y, 'numpy')
print("Solution evaluated at X:", y_func(X))

# Exact solution
y_exact = ((2*sp.cos(1-x)-sp.sin(x))/sp.cos(1)) + (x**2-2)
y_exact = sp.lambdify(x, y_exact , 'numpy')
error = np.abs( y_func(X) - y_exact(X))
max_abs_error = np.max(error)

print(f"Maximum Absolute Error: {max_abs_error}")

# Plotting
x_plot = np.linspace(x0, x1, N)
plt.figure(figsize=(10, 6))

# Approximate solution (blue dashed line with circle markers)
plt.plot(
    x_plot, 
    y_func(x_plot), 
    'b--o',         # 'b'=blue, '--'=dashed, 'o'=circle markers
    label='Approximate Solution',
    markersize=12,   # Adjust marker size
    linewidth=1.5   # Adjust line width
)

# Exact solution (red solid line with square markers)
plt.plot(
    x_plot, 
    y_exact(x_plot), 
    'r-s',          # 'r'=red, '-'=solid, 's'=square markers
    label='Exact Solution',
    markersize=7,
    linewidth=1.5
)

plt.rcParams['font.family'] = 'Times New Roman'
plt.xlabel("x", fontsize=12)
plt.ylabel("y(x)", fontsize=12)
plt.legend(loc='best', fontsize=10)  # Legend with improved positioning
plt.grid(True, linestyle='--', alpha=0.6)  # Dashed grid lines with transparency
plt.tight_layout()  # Prevent label cutoff
# Save the figure with 500 DPI
# plt.savefig('Problem 2.png', dpi=700)
plt.show()
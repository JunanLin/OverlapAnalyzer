import numpy as np
import sympy as sp
from numpy.polynomial import chebyshev
from numpy.polynomial import Polynomial
from scipy.optimize import minimize, NonlinearConstraint
import matplotlib.pyplot as plt

def f(x, k, x0, x1):
    return np.piecewise(x, 
                        [x < x0, (x0 <= x) & (x <= x1), x > x1],
                        [0, lambda x: k*(x-x0), k*(x1-x0)])
def objective(params, M):
    k = params[0]
    c = params[1:]
    return np.sum(c / k * M)

# Constraint function: h(k, c_i, x)
def constraint_function(params, x, x0, x1):
    k = params[0]
    c = params[1:]
    # Determine which form of h to use based on x
    if x < x0:
        return np.sum(c * x ** np.arange(len(c)))
    elif x0 <= x < x1:
        return np.sum(c * x ** np.arange(len(c))) - k * (x - x0)
    elif x1 <= x:
        return np.sum(c * x ** np.arange(len(c))) - k * (x1 - x0)

def vectorized_constraint(params, x_points, x0, x1):
    return np.array([constraint_function(params, x, x0, x1) for x in x_points])

def chebyshev_approximation(f, a, b, degree):
    """
    Approximate the function f over the interval [a, b] using Chebyshev polynomials of a given degree.
    
    Args:
    f: function to approximate
    a: start of the interval
    b: end of the interval
    degree: degree of the Chebyshev polynomial
    
    Returns:
    Approximation function that evaluates the Chebyshev approximation on any input.
    """
    
    # Rescale x from [a, b] to [-1, 1]
    def rescale_x(x):
        return 2 * (x - a) / (b - a) - 1

    # Rescale x from [-1, 1] back to [a, b]
    def rescale_back(t):
        return 0.5 * (t + 1) * (b - a) + a

    # Generate Chebyshev nodes in the interval [-1, 1]
    nodes = np.cos(np.pi * (np.arange(degree + 1) + 0.5) / (degree + 1))
    
    # Map nodes to the interval [a, b]
    nodes_mapped = rescale_back(nodes)
    
    # Evaluate the function at these nodes
    f_values = f(nodes_mapped)
    
    # Get the Chebyshev coefficients by fitting to the values
    coeffs = chebyshev.chebfit(nodes, f_values, degree)
    
    # Return the approximation function
    def chebyshev_approx(x):
        # Rescale x to [-1, 1]
        t = rescale_x(x)
        # Evaluate the Chebyshev series
        return chebyshev.chebval(t, coeffs)

    return chebyshev_approx, coeffs

def poly_form_of_chebyshev_approximation(f, a, b, degree):
    """
    Compute the polynomial in standard form of the Chebyshev approximation.
    Return: Numpy Polynomial object in standard form.
    """
    # First, convert the Chebyshev coefficients (in t) to the standard polynomial in t
    cheb_approx, cheb_coeffs = chebyshev_approximation(f, a, b, degree)
    poly_coeffs_t = chebyshev.cheb2poly(cheb_coeffs)
    
    # The transformation is t = (2*(x - a))/(b - a) - 1
    # So we need to substitute t = (2*(x - a))/(b - a) - 1 into the polynomial

    # Create the polynomial P(t) in terms of t
    P_t = Polynomial(poly_coeffs_t)
    
    # Define the linear transformation: t(x) = (2*(x - a)) / (b - a) - 1
    T_x = Polynomial([-(a + b) / (b - a), 2 / (b - a)])  # -1 + (2/(b - a)) * (x - a)
    
    # Compose P(t) with T_x, i.e., P(T_x(x))
    P_x = P_t(T_x)
    
    # Return the resulting polynomial in x
    return P_x

def plot_results(f_exact, polynomial, x0, x1, x_max, title, O_exact, O_LB):
        # Discretize the x range
        x_values = np.linspace(x0, x_max, 500)

        # Evaluate the polynomial
        poly_values = polynomial(x_values)

        # Evaluate the piecewise linear function
        exact_values = f_exact(x_values)

        # Plot the polynomial
        plt.plot(x_values, poly_values, label='Polynomial $\\sum_i c_i x^i$', color='blue')

        # Plot the piecewise linear function
        plt.plot(x_values, exact_values, label='Exact function', linestyle='--', color='red')

        # Mark x0, x1, x_max on the x-axis
        plt.axvline(x=x0, color='gray', linestyle=':')
        plt.axvline(x=x1, color='gray', linestyle=':')
        plt.axvline(x=x_max, color='gray', linestyle=':')

        # Add labels and legend
        plt.title(title+ f', Degree={polynomial.degree()}' + r', $O_{exact}=$' + f'{O_exact:.3f}' + r', $O_{LB}=$' + f'{O_LB:.3f}')
        plt.xlabel('Energy')
        plt.ylabel('f(E)')
        plt.legend()

        # Show the plot
        plt.grid(True)
        plt.show()
        # plt.savefig(title + '.pdf')

class PolyMajorLB():
    """
    Using a polynomial majorisation of the piecewise sloped function, compute a lower bound for the overlap.
    The degree of the polynomial is determined by the number of moments.
    Args:
    """
    def __init__(self, H_moments, k_init, k_lower_bound, E0, E1, Emax, LL_shift, num_x_points, plot_title, O_exact, print_iter=False, visualize=True):
        local_vars = locals()  # Dictionary of local variables
        for name, value in local_vars.items():
            if name != 'self':  # Exclude 'self' from being set as an attribute
                setattr(self, name, value)
        self.poly_degree = len(self.H_moments) - 1
        self.x_points = np.concatenate([np.linspace(E0, E1, num_x_points[0]), np.linspace(E1+0.01, Emax, num_x_points[1])])
        self.nonlinear_constraint = NonlinearConstraint(
            lambda params: np.array([constraint_function(params, x, E0, E1) for x in self.x_points]),
            lb=0,  # Lower bound (h(k, c_i, x) >= 0)
            ub=np.inf  # Upper bound (no upper limit)
        )
        self.bounds = [(self.k_lower_bound, None)] + [(None, None) for _ in range(self.poly_degree+1)]  # k is strictly positive, c_i are unbounded
        self.polynomials = poly_form_of_chebyshev_approximation(lambda x: f(x, k_init, E0, E1), E0-LL_shift, Emax, self.poly_degree)
        self.init_params = np.insert(self.polynomials.coef, 0, k_init)
    
    def optimize(self):
        # Define the callback function
        def print_iteration(xk, state):
            print(f"Current parameters: k = {xk[0]:.5f}, c = {xk[1:]}")

        # Modify the minimize function call
        result = minimize(objective, self.init_params, args=(self.H_moments,), 
                        constraints=[self.nonlinear_constraint], bounds=self.bounds, 
                        method='trust-constr', callback=print_iteration if self.print_iter else None)

        # Print the results
        if result.success:
            print("Optimal parameters found:")
            print("k =", result.x[0])
            print("c_i =", result.x[1:])
            LB = 1-np.dot(self.H_moments,result.x[1:])/(result.x[0]*(self.E1-self.E0))
            print("Overlap Lower Bound: ", LB)
            if self.visualize:
                plot_results(lambda x: f(x, result.x[0], E0, E1), Polynomial(result.x[1:]), self.E0, self.E1, self.Emax, self.plot_title, self.O_exact, LB)
        else:
            print("Optimization failed:", result.message)

        return result, LB

if __name__ == "__main__":
    hlist = np.array([ 1.00000000e+00, -1.57561648e+00,  2.60888740e+00, -4.47442375e+00,
        7.87516943e+00, -1.41324921e+01,  2.57354796e+01, -4.73813394e+01,
        8.79466788e+01, -1.64223685e+02,  3.08002987e+02, -5.79503324e+02,
        1.09283856e+03, -2.06431600e+03,  3.90404094e+03, -7.38966679e+03,
        1.39959675e+04, -2.65199968e+04,  5.02669682e+04, -9.52998279e+04,
        1.80706686e+05, -3.42696233e+05,  6.49955119e+05, -1.23278195e+06,
        2.33835700e+06, -4.43559460e+06,  8.41406080e+06, -1.59613472e+07,
        3.02790060e+07, -5.74407969e+07,  1.08969525e+08, -2.06725793e+08,
        3.92182976e+08, -7.44023843e+08,  1.41152542e+09, -2.67789808e+09,
        5.08045590e+09, -9.63861153e+09,  1.82864453e+10, -3.46934127e+10,
        6.58214700e+10])
    E0 = -1.8977806459898725
    E1 = -1.8649403599042444
    Emax = -0.3541130013531074
    O_exact = 0.4817748
    k = 0.1
    cheb_degree = 16
    M = hlist[:cheb_degree+1]

    poly = PolyMajorLB(M, k, 1e-6, E0, E1, Emax, 0, [10, 30], r'$H_4$ (STO-3G), d=2.0', O_exact)
    opt_result, LB = poly.optimize()
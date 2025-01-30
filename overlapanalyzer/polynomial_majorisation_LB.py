import numpy as np
import scipy as sp
import copy
import pandas as pd
from numpy.polynomial import chebyshev
from numpy.polynomial import Polynomial
from scipy.optimize import minimize, NonlinearConstraint, linprog
from scipy.special import eval_chebyt
import matplotlib.pyplot as plt
# Uncomment for production plotting with LaTeX (slower)
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.sans-serif": "Computer Modern Roman",
#     "font.size": 17,
# })
from overlapanalyzer.utils import exp_val_higher_moment
from overlapanalyzer.contour_integration import getContourDataFromEndPoints
from overlapanalyzer.alg_LinearOp import linear_solver
from overlapanalyzer.eigen import non_degenerate_values, select_right_multiplicity, truncate_by_ovlp_threshold
from overlapanalyzer.polynomial_estimates import rescale_E

def f_piecewise(x, x0, x1):
    return np.piecewise(x, 
                        [x < x0, (x0 <= x) & (x <= x1), x > x1],
                        [0, lambda x: (x-x0), (x1-x0)])

def f_logistic(x, x_c, w):
    return 1/(1+np.exp(-w*(x-x_c)))

def find_logistic_w(x0, x1, threshold=0.95):
    """
    Find the omega parameter for the logistics function f(x)=1/(1+exp(-omega (x-x_c))) based on the given x0 and x1.
    Threshold is the value of f at x1.
    """
    return -2*np.log(1/threshold - 1)/(x1-x0)

# Constraint function: h(k, c_i, x)
def constraint_function_piecewise(c, x, x0, x1):
    if x < x0:
        return np.sum(c * x ** np.arange(len(c)))
    elif x0 <= x < x1:
        return np.sum(c * x ** np.arange(len(c))) - (x - x0)
    elif x1 <= x:
        return np.sum(c * x ** np.arange(len(c))) - (x1 - x0)


def vectorized_constraint_piecewise(params, x_points, x0, x1):
    return np.array([constraint_function_piecewise(params, x, x0, x1) for x in x_points])

def constraint_function_logistic(params, x, x_c, w):
    c = params
    return np.sum(c * x ** np.arange(len(c))) - f_logistic(x, x_c, w)

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

def chebyshev_matrix_from_degree(degree):
    # Initialize the transformation matrix
    transform_matrix = np.zeros((degree + 1, degree + 1))
    
    # Populate the matrix by creating each Chebyshev polynomial in the power basis
    for i in range(degree + 1):
        # Create the i-th Chebyshev polynomial (T_i)
        cheb_poly = chebyshev.Chebyshev.basis(i).convert(kind=np.polynomial.Polynomial)
        # Get the coefficients for T_i in terms of power basis (1, a, a^2, ...)
        transform_matrix[i, :len(cheb_poly.coef)] = cheb_poly.coef
        
    return transform_matrix

def f_piecewise_concave(x, left_low, left_up, right_low, right_up, v2, v1):
    return np.piecewise(x, 
                    [x < left_low, (left_low <= x) & (x <= left_up), (left_up < x) & (x <= right_low), (right_low < x) & (x <= right_up), x > right_up],
                    [v2, lambda x: x*(v1-v2)/(left_up - left_low) + v2-left_low*(v1-v2)/(left_up - left_low), v1, lambda x: x*(v2-v1)/(right_up - right_low) + v1-right_low*(v2-v1)/(right_up - right_low), v2])

# def f_function(x, v1_range, v2, v1, left_v2=None, right_v2=None):
#     """
#     A piecewise function with optional linear interpolation at the boundaries.
#     Supports scalar or array inputs for x.

#     Parameters:
#     - x: float or array-like, the input values at which the function is evaluated.
#     - v1_range: tuple, (v1_start, v1_end) defining where the function evaluates to v1.
#     - v2: float, the value of the function outside of v1_range.
#     - v1: float, the value of the function within v1_range.
#     - left_v2: float, optional, the value at x < v1_range[0] to create a linear interpolation.
#     - right_v2: float, optional, the value at x > v1_range[1] to create a linear interpolation.

#     Returns:
#     - np.ndarray, the values of the function at x.
#     """
#     x = np.asarray(x)  # Ensure x is a NumPy array for vectorized operations
#     v1_start, v1_end = v1_range

#     # Initialize the result with v2 values
#     result = np.full_like(x, v2, dtype=np.float64)

#     # Region: Linear interpolation before v1_start
#     if left_v2 is not None:
#         left_mask = (x >= left_v2) & (x < v1_start)
#         result[left_mask] = v2 + (v1 - v2) * (x[left_mask] - left_v2) / (v1_start - left_v2)

#     # Region: v1 within v1_range
#     v1_mask = (x >= v1_start) & (x <= v1_end)
#     result[v1_mask] = v1

#     # Region: Linear interpolation after v1_end
#     if right_v2 is not None:
#         right_mask = (x > v1_end) & (x <= right_v2)
#         result[right_mask] = v1 + (v2 - v1) * (x[right_mask] - v1_end) / (right_v2 - v1_end)

#     return result

def ITE_UB(E0, exp_H, exp_H2):
    return np.exp(-(exp_H - E0)**2 / (2*(exp_H2 - exp_H**2)))

def gen_b_ub_from_length(target_indices, b_values, L):
    """
    Generate the b_ub vector for linprog constraints from exact constraints.
    Each b value only corresponds to a single target index.
    :param target_indices: list of target indices.
    :param b_values: list of b values for the constraints. Must match the length of target_indices.
    :param L: length of the vector. Must be greater than the maximum index in target_indices.
    """
    b_ub = np.zeros(L)
    for i, index in enumerate(target_indices):
        b_ub[index] = b_values[i]
    return b_ub

def gen_gaps_from_values(value_list):
    return [(value_list[i] - value_list[i-1]) for i in range(1, len(value_list))]

def gen_bound_pairs(value_list, bound_extension, E0_LB=None, Emax_UB = None):
    """
    Generate a list of (lower, upper) bound pairs from a list of values.
    If E0_LB or Emax_UB is provided, they will be used as LB/UB for the first/last value.
    Otherwise, the E0-E1 and Emax-(Emax-1) gaps will be used.
    :param value_list: list of values, assumed to be sorted from lowest to highest.
    :param bound_extension: the extension factor for the bounds. Must be between 0 and 1/2.
    :param E0_LB: lower bound for the first value. If None, the E0-E1 gap will be used.
    :param Emax_UB: upper bound for the last value. If None, the Emax-(Emax-1) gap will be used.
    """
    gaps = np.array(gen_gaps_from_values(value_list)) * bound_extension

    if E0_LB is None:
        E0_LB = value_list[0] - gaps[0]
    elif E0_LB >= value_list[0]:
        raise ValueError("The provided E0_LB must be less than the first value.")
    if Emax_UB is None:
        Emax_UB = value_list[-1] + gaps[-1]
    elif Emax_UB <= value_list[-1]:
        raise ValueError("The provided Emax_UB must be greater than the last value.")
    return np.array([(E0_LB, value_list[0]+gaps[0])] + [(value_list[i]-gaps[i-1], value_list[i]+gaps[i]) for i in range(1, len(value_list) - 1)] + [(value_list[-1]-gaps[-1], Emax_UB)])

def compute_boundaries(target_indices, L):
    boundaries = []
    target_indices = sorted(target_indices)  # Ensure indices are sorted
    previous_index = 0

    for index in target_indices:
        if previous_index < index:  # Handle ranges before the target index
            boundaries.append((previous_index, index - 1))
        boundaries.append((index, index))  # Add the target index as its own boundary
        previous_index = index + 1

    if previous_index < L:  # Handle the final range after the last target index
        boundaries.append((previous_index, L - 1))

    return boundaries

def process_bound_pairs(bound_pairs, target_indices, dense_info=(None, None)):
    """
    Keep the bound pairs corresponding to the target indices, and merge the rest that are neighbours to each other.
    If dense_info is provided, and they are at the correct locations, then they will be added as additional separation points.
    :param bound_pairs: numpy array of (lower, upper) bound pairs.
    :param target_indices: list of target indices.
    :param dense_info: tuple (dense_lb, dense_ub) defining the dense region. Either entry can be None.
    """
    if 0 in target_indices and dense_info[0] is not None:
        raise ValueError("dense_info[0] cannot be used since 0 is in target_indices.")
    nonzero_b_indices = target_indices.copy() if dense_info[0] is None else np.array(target_indices)+1
    def _insert_lower_bound(pair_list, bound):
        pair_list.insert(1,(bound, pair_list[0][1]))
        pair_list[0] = (pair_list[0][0], bound)
    def _insert_upper_bound(pair_list, bound):
        pair_list.insert(-1, (pair_list[-1][0], bound))
        pair_list[-1] = (bound, pair_list[-1][1])
        
    boundaries = compute_boundaries(target_indices, len(bound_pairs))
    new_bound_pairs = [(bound_pairs[lower][0], bound_pairs[upper][1]) for lower, upper in boundaries]
    incl_endpoints = [True for _ in range(len(new_bound_pairs))]
    # The following lines define what "correct locations" mean for the dense region
    if dense_info[0] is not None and new_bound_pairs[0][0] < dense_info[0] < new_bound_pairs[target_indices[0]][1]:
        _insert_lower_bound(new_bound_pairs, dense_info[0])
        incl_endpoints.insert(0, False)
    if dense_info[1] is not None and new_bound_pairs[target_indices[-1]][0] < dense_info[1] < new_bound_pairs[-1][1]:
        _insert_upper_bound(new_bound_pairs, dense_info[1])
        incl_endpoints.insert(-1, False)
    return new_bound_pairs, incl_endpoints, nonzero_b_indices

def gen_constraints_recipe(bound_pairs, target_b_values, incl_endpoints_list, num_x_points):
    """
    Generate the constraint recipe table for the linprog constraints.
    :param bound_pairs: list of (lower, upper) bound pairs.
    :param target_b_values: list of target b values for the constraints. Must match the length of target_indices.
    :param incl_endpoints_list: list of booleans indicating whether the endpoints are included in the region.
    :param num_x_points: tuple of (num_0, num_1, ...) for the number of points in each region. Must match the number of regions.
    Output: a dictionary with the following entries:
    - Region: (lower bound, upper bound) of the region
    - b_values: b values for the region
    - num_points: number of discretization points in the region
    - include_endpoints: whether the endpoints are included in the region
    """
    return {"Region": bound_pairs, "b_values": target_b_values, "num_points": num_x_points, "include_endpoints": incl_endpoints_list}

def save_constraints_dict_as_csv(constraints_dict, filename):
    """
    Save the constraints dictionary as a CSV file.
    """
    df = pd.DataFrame(constraints_dict)
    df.to_csv(filename, index=False)

def load_constraints_dict_from_csv(filename):
    """
    Load the constraints dictionary from a CSV file.
    """
    df = pd.read_csv(filename)
    return df.to_dict()

class PB_linprog_params_from_eig:
    """
    Prepares the parameters for linprog based on the eigenvalue results.
    :param eig_results: dictionary of eigenvalue results.
    :param target_indices: list of target indices.
    :param b_values: list of b values for the constraints. Must match the length of target_indices.
    :param poly_degree: degree of the polynomial.
    :param use_exact_energies: whether to use exact energies as constraints.
    :param rescale: whether to rescale the energies.
    :param bound_extension: extension factor for the bounds. Only used when use_exact_energies=False.
    :param dense_info: tuple (dense_lb, dense_ub) defining the dense region. Only used when use_exact_energies=False.
    :param num_x_points: tuple of (num_0, num_1, ...) for the number of points in each region. Only used when use_exact_energies=False, and len must match definitions target_indices & dense_info.
    :param Energy_constraint_limits: tuple (Emin, Emax) defining the energy constraint limits; energies outside the range will not be used as constraints.
    """
    def __init__(self, eig_results, target_indices, use_exact_energies, use_all_evals=False,
                 use_cheb=True, rescale=True, bound_extension=None, dense_info=(None, None), num_x_points=None, Energy_constraint_limits=(float('-inf'), float('inf'))):
        # Store inputs as attributes
        local_vars = locals()
        for name, value in local_vars.items():
            if name != 'self':
                setattr(self, name, value)

        # Rescale parameters and overlaps
        self.E0_rescale, self.Emax_rescale = eig_results['rescale_values']
        self.exact_overlaps = eig_results['overlaps'] if use_all_evals else eig_results['overlaps_truncated']
        self.unscaled_energies = eig_results['exact_energies_no_degen'] if use_all_evals else eig_results['exact_energies_truncated']
        self.exact_mask = (np.array(self.unscaled_energies) >= Energy_constraint_limits[0]) & (np.array(self.unscaled_energies) <= Energy_constraint_limits[1])
        self.exact_energies = self._to_x(self.unscaled_energies)
        self.moments = np.array(eig_results['moments_rescaled']) if rescale else np.array(eig_results['moments'])
        self.max_poly_degree = len(self.moments) - 1

        self.constraint_pts = self._get_exact_constraints() if use_exact_energies else self._get_dense_constraints()
        self.A_ub_full = np.polynomial.chebyshev.chebvander(self.constraint_pts, self.max_poly_degree) if use_cheb else np.vander(self.constraint_pts, self.max_poly_degree + 1, increasing=True)
        # self.M = np.array(self.moments[:poly_degree + 1])

        # Get range information and mask energies above overlap threshold
        # self.range_info = self._to_x(Energy_constraint_limits)
        # self.truncated_energies, self.truncated_overlaps = truncate_by_ovlp_threshold(self.exact_energies, self.all_overlaps, ovlp_threshold)

        # Validate inputs
        if not use_exact_energies and (bound_extension is None or dense_info is None or num_x_points is None):
            raise ValueError("(bound_extension, dense_size, num_x_points) are required when use_exact_energies=False.")

        # Below, we use a symmetric bound extension for the ground and highest excited states in gen_bound_pairs; can be changed in future
        if not use_exact_energies:
            self.bound_pairs, self.incl_endpoints, self.nonzero_b_indices = process_bound_pairs(gen_bound_pairs(self.unscaled_energies, bound_extension), target_indices, dense_info)

    def _to_x(self, E):
        """Rescale energies if rescale=True."""
        if self.rescale:
            return rescale_E(np.array(E), self.E0_rescale, self.Emax_rescale)
        else:
            return np.array(E)

    def _get_exact_constraints(self):
        """
        Get exact energies within limits as constraint points.
        """
        exact_energies = np.array(self.unscaled_energies)
        energies_within_limits = exact_energies[self.exact_mask]
        return self._to_x(energies_within_limits)

    def _get_dense_constraints(self):
        """
        Generate dense constraints based on bound pairs and number of discretization points.
        """
        if len(self.num_x_points) != len(self.bound_pairs):
            raise ValueError("num_x_points must match the number of bound pairs! Check the number of constraint regions.")
        energy_constraints = []
        for i, (lower, upper) in enumerate(self.bound_pairs):
            if lower > self.Energy_constraint_limits[0] and upper < self.Energy_constraint_limits[1]:
                energy_constraints.extend(np.linspace(lower, upper, self.num_x_points[i], endpoint=self.incl_endpoints[i]))
        # energy_constraints = np.concatenate(energy_constraints)
        return self._to_x(energy_constraints)

    def get_A_ub(self, poly_degree):
        """Return A_ub."""
        return self.A_ub_full[:,:poly_degree + 1]

    def get_moment_values(self, poly_degree):
        """
        Return computed moment values.
        Note: need to compute the chebyshev moments by applying the transformation matrix.
        """
        if self.use_cheb:
            return (chebyshev_matrix_from_degree(self.max_poly_degree) @ self.moments)[:poly_degree + 1]
        else:
            return self.moments[:poly_degree + 1]
    
    def get_b_ub(self, b_list):
        """Return b_ub values."""
        if self.use_exact_energies:        
            full_b = gen_b_ub_from_length(self.target_indices, b_list, len(self.unscaled_energies))
            return full_b[self.exact_mask]
        else:
            b_recipe = np.zeros(len(self.bound_pairs))
            for i, index in enumerate(self.nonzero_b_indices):
                b_recipe[index] = b_list[i]
            # Repeat the b values for the number of points in each region
            return np.repeat(b_recipe, self.num_x_points)
    
    def get_constraint_pts(self):
        return self.constraint_pts
    
    def get_all_for_PolyBounds(self, poly_degree, b_list):
        values = self.get_moment_values(poly_degree)
        A_ub = self.get_A_ub(poly_degree)
        b_ub = self.get_b_ub(b_list)
        return values, A_ub, b_ub

class PolyBounds():
    """
    Obtaining lower and upper bounds on the overlap using polynomial majorisation/minorisation.
    Parameters
    ----------
    linprog_params: PB_linprog_params object
    """
    def __init__(self, linprog_params, **kwargs):
        self.linprog_params = linprog_params
        print("Exact overlap: ", linprog_params.pre_constraint_info['exact_ovlp'])
        # if len(self.H_moments) > 2 and true_GS:
        #     self.ite_ub = ITE_UB(kwargs.get('exact_energies')[0], H_moments[1], H_moments[2])
        #     print("ITE Upper Bound: ", self.ite_ub)
    def optimize(self, bound=1e20, method='highs', linprog_options={}):
        self.bounds = []
        lb_input, ub_input = self.linprog_params.get_linprog_params()
        self.LB_result = linprog(lb_input['values'], A_ub=lb_input['A_ub'], b_ub=lb_input['b_ub'], bounds=(-bound, bound), method=method, options=linprog_options)
        self.UB_result = linprog(ub_input['values'], A_ub=ub_input['A_ub'], b_ub=ub_input['b_ub'], bounds=(-bound, bound), method=method, options=linprog_options)
        print(f"Poly degree: {self.linprog_params.poly_degree}")
        for bound_name in ['LB', 'UB']:
            result = self.LB_result if bound_name == 'LB' else self.UB_result
            if result.success:
                print(f"{bound_name} Optimal parameters found:")
                print("c_i =", result.x)
                bound = np.dot(self.linprog_params.get_moment_values(),result.x) / self.linprog_params.v1
                self.bounds.append(bound)
                print(f"Overlap {bound_name}: ", bound)
            else:
                print(f"{bound_name} Optimization failed:", result.message)
                self.bounds.append(np.nan)
        # if self.visualize:
        #     self.plot_results(show_constraint_points=self.kwargs.get('show_constraint_points'), n_points=self.kwargs.get('n_plot_points'), linewidth=self.kwargs.get('linewidth'))
        return self.LB_result, self.UB_result, self.bounds
    def plot_results(self, legend_anchor=(1,0.2), show_spectrum=True, show_constraint_points=False, n_points=5000, limit_adjust=(0,0), linewidth=1, plot_title=None):
        # Discretize the x range
        plot_x_values = self.linprog_params.constraint_pts
        # x_values_full = np.linspace(self.range_info[0], self.range_info[1], n_points)
        LB_polynomial = chebyshev.Chebyshev(self.LB_result.x) if self.linprog_params.use_cheb else Polynomial(self.LB_result.x)
        LB_polynomial_values = LB_polynomial(plot_x_values)
        UB_polynomial = chebyshev.Chebyshev(self.UB_result.x) if self.linprog_params.use_cheb else Polynomial(self.UB_result.x)
        UB_polynomial_values = UB_polynomial(plot_x_values)
        if self.linprog_params.use_exact_constraints:
            f_values = f_function(plot_x_values, tuple(self.linprog_params.v1_range), self.linprog_params.v2, self.linprog_params.v1)
        else:
            f_values = f_function(plot_x_values, tuple(self.linprog_params.v1_range), self.linprog_params.v2, self.linprog_params.v1, self.linprog_params.left_v2, self.linprog_params.right_v2)
        # ax1=plt.gca()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        # Plot the polynomial
        ax1.plot(plot_x_values, LB_polynomial_values, label='Minorizing Poly', color='blue', linewidth=linewidth)
        ax1.plot(plot_x_values, UB_polynomial_values, label='Majorizing Poly', color='green', linewidth=linewidth)

        # Plot the exact piecewise linear function
        ax1.plot(plot_x_values, f_values, label=r'$f_0(E)$', linestyle='--', color='red', linewidth=linewidth)
        exact_energies = self.linprog_params.exact_energies_no_degen
        exact_overlaps = self.linprog_params.all_overlaps
        if show_spectrum:
            ax2 = ax1.twinx()
            # non_degen_evals = non_degenerate_values(eig_results['exact_energies'], degens)
            ax2.stem(exact_energies, exact_overlaps, linefmt='grey', markerfmt=' ', basefmt='C0', label='Exact spectrum')
            ax2.set_ylabel('Overlap')
            # Mark E0_LB, Emax_UB on the x-axis
            ax1.axvline(x=exact_energies[0], color='gray', linestyle=':')
            ax1.axvline(x=exact_energies[-1], color='gray', linestyle=':')
        if show_constraint_points:
            ax1.scatter(self.linprog_params.constraint_pts, [0 for _ in range(len(self.linprog_params.constraint_pts))], color='pink', marker='|', zorder=4)
            # print("Constraint points: ", self.constraint_pts)
            
        # Add labels and legend
        if plot_title is not None:
            ax1.set_title(plot_title+ f', Degree={self.linprog_params.poly_degree}' + r', $O_{exact}=$' + f'{self.linprog_params.pre_constraint_info['exact_ovlp']:.3f}' + r', $O_{LB}=$' + f'{self.bounds[0]:.3f}' + r', $O_{UB}=$' + f'{self.bounds[1]:.3f}')
        else:
            ax1.set_xlabel('Energy')
        ax1.set_ylabel('Function Value')
        # plt.ylabel('f(E)')
        fig.legend(loc='lower right', bbox_to_anchor=legend_anchor, bbox_transform=ax1.transAxes)
        plt.tight_layout()
        # plt.grid(True)
        plt.show()

class PolyBounds_new():
    """
    Obtaining lower and upper bounds on the overlap using polynomial majorisation/minorisation.
    :param values: list of values defining the objective function
    :param A_ub: matrix A_ub for linprog
    :param b_ub: vector b_ub for linprog
    :param P_exact: exact overlap value
    """
    def __init__(self, values, A_ub, b_ub, P_exact=None):
        self.values = values
        self.A_ub = A_ub
        self.b_ub = b_ub
        self.poly_degree = len(values) - 1
        if P_exact is not None:
            print("Exact overlap: ", P_exact)
    def optimize(self, bound=1e20, method='highs', linprog_options={}):
        print("Condition number for A: ", np.linalg.cond(self.A_ub))
        self.LB_result = linprog(-self.values, A_ub=self.A_ub, b_ub=self.b_ub, bounds=(-bound, bound), method=method, options=linprog_options)
        self.UB_result = linprog(self.values, A_ub=-self.A_ub, b_ub=-self.b_ub, bounds=(-bound, bound), method=method, options=linprog_options)
        self.bounds = []
        print(f"Poly degree: {self.poly_degree}")
        for bound_name in ['LB', 'UB']:
            result = self.LB_result if bound_name == 'LB' else self.UB_result
            if result.success:
                print(f"{bound_name} Optimal parameters found:")
                print("c_i =", result.x)
                bound = np.dot(self.values,result.x)
                self.bounds.append(bound)
                print(f"Overlap {bound_name}: ", bound)
            else:
                print(f"{bound_name} Optimization failed:", result.message)
                self.bounds.append(np.nan)
        return self.LB_result, self.UB_result, self.bounds
    def plot_results(self, constraint_pts, n_plot_points=1000, used_cheb=True, legend_anchor=(1,0.2), show_constraint_points=False, limit_adjust=(0,0), linewidth=1, show_spectrum=False, show_lowest_highest=False, constraint_recipe_obj=None, plot_title=None):
        """
        constraint_pts: the constraint points used in the optimization.
        """
        poly_plot_points = np.linspace(constraint_pts[0], constraint_pts[-1], n_plot_points)
        LB_polynomial = chebyshev.Chebyshev(self.LB_result.x) if used_cheb else Polynomial(self.LB_result.x)
        LB_polynomial_values = LB_polynomial(poly_plot_points)
        UB_polynomial = chebyshev.Chebyshev(self.UB_result.x) if used_cheb else Polynomial(self.UB_result.x)
        UB_polynomial_values = UB_polynomial(poly_plot_points)
        f_values = self.b_ub
        # ax1=plt.gca()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        # Plot the polynomial
        ax1.plot(poly_plot_points, LB_polynomial_values, label='Minorizing Poly', color='blue', linewidth=linewidth)
        ax1.plot(poly_plot_points, UB_polynomial_values, label='Majorizing Poly', color='green', linewidth=linewidth)

        # Plot the exact piecewise linear function
        ax1.plot(constraint_pts, f_values, label=r'$f_0(E)$', linestyle='--', color='red', linewidth=linewidth)
        # Overlay the spectrum information
        if show_spectrum:
            ax2 = ax1.twinx()
            exact_energies = constraint_recipe_obj.exact_energies
            exact_overlaps = constraint_recipe_obj.exact_overlaps
            # non_degen_evals = non_degenerate_values(eig_results['exact_energies'], degens)
            ax2.stem(exact_energies, exact_overlaps, linefmt='grey', markerfmt=' ', basefmt='C0', label='Exact spectrum')
            ax2.set_ylabel('Overlap')
        # Mark E0_LB, Emax_UB on the x-axis
        if show_lowest_highest:
            ax1.axvline(x=constraint_recipe_obj.exact_energies[0], color='gray', linestyle=':')
            ax1.axvline(x=constraint_recipe_obj.exact_energies[-1], color='gray', linestyle=':')
        if show_constraint_points:
            ax1.scatter(constraint_pts, [0 for _ in range(len(constraint_pts))], color='pink', marker='|', zorder=4)
            # print("Constraint points: ", self.constraint_pts)
            
        # Add labels and legend
        if plot_title is not None:
            ax1.set_title(plot_title+ f', Degree={self.poly_degree}' + r', $O_{exact}=$' + f'{self.O_exact:.3f}' + r', $O_{LB}=$' + f'{self.bounds[0]:.3f}' + r', $O_{UB}=$' + f'{self.bounds[1]:.3f}')
            ax1.set_xlabel('Energy')
        ax1.set_ylabel('Function Value')
        # plt.ylabel('f(E)')
        fig.legend(loc='lower right', bbox_to_anchor=legend_anchor, bbox_transform=ax1.transAxes)
        plt.tight_layout()
        # plt.grid(True)
        plt.show()

def moments_from_distr(evals, overlaps, degree):
    """
    Compute the first n moments of a Hamiltonian given its eigenvalues and overlaps.
    Formula: moments = sum(overlap * eval^i) / sum(overlap)
    """
    moments = np.array([np.dot(evals**i, overlaps) for i in range(degree+1)])
    # if return_cheb:
    #     moments = chebyshev_matrix_from_degree(degree) @ moments
    return moments

def random_distr(num_evals, major_ovlp=0.6, max_ovlp_index=1, energy_bounds=(-1,1), seed=None):
    """
    Generate a random distribution of eigenvalues and overlaps.
    There should be one dominant overlap equal to max_ovlp, and the rest are random.
    The overlap values should sum to 1.
    """
    if seed is not None:
        np.random.seed(seed)
    evals = np.sort(np.random.uniform(energy_bounds[0], energy_bounds[1], num_evals))
    small_probs = np.random.dirichlet([1] * (num_evals - 1)) * (1-major_ovlp)
    overlaps = np.insert(small_probs, max_ovlp_index, major_ovlp)
    return (evals, overlaps)

class MultStateBounds:
    """
    Resolves the bounds for multiple states using an "outer" linear program loop.
    """
    def __init__(self, F, l, u):
        if F.shape[0] != len(l) or F.shape[0] != len(u):
            raise ValueError(f"# rows in F: {F.shape[1]} must match the length of l: {len(l)} and u: {len(u)}.")
        if F.shape[0] < F.shape[1]:
            raise ValueError(f"F must have more rows {F.shape[0]} than columns {F.shape[1]}.")
        self.F = F
        self.l = l
        self.u = u
    def solve(self):
        self.lb_P_list = []
        self.ub_P_list = []
        A_ub = np.vstack((self.F, -self.F))
        b_ub = np.hstack((self.u, -self.l))
        for i in range(self.F.shape[1]):
            unit_vector = np.zeros(self.F.shape[1])
            unit_vector[i] = 1
            ms_lb = linprog(unit_vector, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1))
            ms_ub = linprog(-unit_vector, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1))
            self.lb_P_list.append(np.dot(unit_vector, ms_lb.x))
            self.ub_P_list.append(np.dot(unit_vector, ms_ub.x)) # Note the cancellation of negative sign
        return self.lb_P_list, self.ub_P_list

def get_MS_l_u(constraint_recipe:PB_linprog_params_from_eig, F:np.array, poly_degree:int):
    """
    Generate the l and u vectors for the MultStateBounds class.
    """
    l = []
    u = []
    for b_values in F:
        values = constraint_recipe.get_moment_values(poly_degree)
        A_ub = constraint_recipe.get_A_ub(poly_degree)
        b_ub = constraint_recipe.get_b_ub(b_values)
        poly_bounds = PolyBounds_new(values, A_ub, b_ub)
        poly_bounds.optimize()
        l.append(poly_bounds.bounds[0])
        u.append(poly_bounds.bounds[1])
    return np.array(l), np.array(u)

class CauchyLB():
    """
    Using the Cauchy integral formula to directly compute a lower bound for the overlap,
    by calculating the expectation of <f(H)>.
    Junan 20241017: UNFINISHED
    """
    def __init__(self, H, v, E0, E1, Emax, num_z_points, plot_title, O_exact, print_iter=False, visualize=True):
        local_vars = locals()  # Dictionary of local variables
        for name, value in local_vars.items():
            if name != 'self':  # Exclude 'self' from being set as an attribute
                setattr(self, name, value)
        gap = E1 - E0
        self.contour = getContourDataFromEndPoints(E0-gap/2, E0+gap/2, num_z_points)
    def run(self):
        # Junan 20240929: UNFINISHED
        computed_QF = []
        for z in self.contour['Points']:
            MTI = copy.copy(-self.H) # This is the matrix to be inverted
            identity = sp.sparse.diags([z for _ in range(MTI.shape[0])])
            MTI += identity
            my_solver = linear_solver(MTI, self.v, self.v)
            state, n_iter = my_solver.gmres_solve(tol=1e-7, restart=30, maxiter=100)
            computed_QF.append(np.real_if_close(np.vdot(self.v, state)))
        
def stddev_and_var(H, vs, threshold=1e-10):
    """
    Returns the standard deviation and variance of the Hamiltonian H wrt vectors vs.
    vs is a collection of column vectors.
    """
    variance = []
    for i in range(vs.shape[1]):
        first_two_moments = exp_val_higher_moment(H, vs[:, i:i+1], 2, return_all=True)
        this_var = first_two_moments[2] - first_two_moments[1]**2
        if this_var < 0:
            if -this_var < threshold:
                print("Small negative variance, casting to 0.")
                this_var = 0
            else:
                print("Negative variance detected.")
                print("Variance is", this_var)
                print("Threshold is", threshold)
                print("Returning 0 variance, proceed with caution.")
                this_var = 0
        variance.append(this_var)
    
    variance = np.array(variance)
    return np.sqrt(variance), variance
    
def K_matrix(eps, ritz:np.array, std_devs:np.array):
    """
    Return the matrix used in Pollak−Martinazzo theory for obtaining energy lower bounds.
    """
    Ritz_vals = np.append(ritz, eps + np.sum(std_devs**2/(ritz-eps)))
    L = len(ritz)
    K = np.diag(Ritz_vals)
    K[:-1, L] = std_devs.T
    K[L, :-1] = std_devs
    return K

def visualize_LBs(exact_energies, Ritz_energies, K_evals, epsilon):
    plt.figure()
    # Plot the individual exact energies as points on the x-axis
    plt.scatter(exact_energies, np.zeros_like(exact_energies), marker='|', color='blue', label='Exact energies')
    # Plot the Ritz energies as points on the x-axis
    plt.scatter(Ritz_energies, np.zeros_like(Ritz_energies), marker='|', color='red', label='Ritz energies')
    # Plot the K eigenvalues as points on the x-axis
    plt.scatter(K_evals, np.zeros_like(K_evals), marker='|', color='purple', label='K eigenvalues')
    # Plot the value of epsilon as a point on the x-axis
    plt.scatter(epsilon, 0, marker='|', color='green', label='Value of epsilon')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test = MultStateBounds(np.array([[1, 1], [1, 1/2]]), np.array([0.4, 0.35]), np.array([0.8, 0.6]))
    lb_list, ub_list = test.solve()
    print("Lower bounds: ", lb_list)
    print("Upper bounds: ", ub_list)
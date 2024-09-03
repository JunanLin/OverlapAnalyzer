import os
import numpy as np
import json
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import legendre
from numpy.polynomial.chebyshev import chebval, Chebyshev, cheb2poly
from openfermion import (
    get_sparse_operator,
    get_ground_state,
    QubitOperator
)
from overlapanalyzer.utils import exp_val_higher_moment

def ITE_est(H, phi, E0=None, Ev=None):
    """
    Implement the overlap estimation using imaginary time evolution (ITE) method.
    Ref: http://arxiv.org/abs/2306.02620
    Currently supporting FermionOperator/QubitOperator for H and uses matrix operations.
    Need to change code if need to utilize more efficient ways of H operations.
    """
    # Convert H to sparse matrix
    H_mat = get_sparse_operator(H)
    # Check if phi is a vector
    if not isinstance(phi, np.ndarray):
        raise TypeError("phi must be a numpy array")
    # Check if phi is normalized
    if not np.isclose(np.linalg.norm(phi), 1):
        raise ValueError("phi must be normalized")
    # Check if E0 is given
    if E0 is None:
        print("No ground state energy given. Calculating exact ground state energy.")
        E0 = get_ground_state(H_mat)[0]
    
    # Hphi = H_mat @ phi
    exp_vals = exp_val_higher_moment(H_mat, phi, 2, return_all=True)
    Exp_H2 = exp_vals[2] # Exp. value of H^2
    if Ev is None:
        Ev = exp_vals[1] # Exp. value of H
    sig_sq = Exp_H2 - Ev**2 # Variance of H
    I_gamma = (Ev-E0)**2/(2*sig_sq) # Overlap index, note the factor of 4 instead of 2 as shown in the paper

    return np.exp(-I_gamma)

def Legendre_step_est(Emin, Emax, Ec, H_shifted_n_exp_val):
    """
    Estimates overlap by approximating the stepwise function {1 if E_min<E<E_c, 0 if E_c<E<E_max} with Legendre polynomials, and replaces 
    powers of E by expectation values (moments) of H^n.
    The estimator is written as: sum_{n=0}^{n_{max}} b_n <P_n(H)> where 
    b_n = (2n+1)/2 sum_{j=0}^{n} p_n[j]/(n+1-j) (E_c-E_min)^j, which are coefficients of the expansion
    """
    assert type(H_shifted_n_exp_val) == np.ndarray
    def _x(E, Emin, Emax):
        return (2*E - Emin - Emax)/(Emax - Emin)
    
    def bn_list_Legendre(Emin, Emax, Ec, nmax):
        bn=[]
        for n in range(0, nmax+1):
            pn = legendre(n).c
            coeff_int = np.array([pn[i]/(n+1-i) for i in range(0,n+1)])
            x_list = np.array([_x(Ec, Emin, Emax)**j - _x(Emin, Emin, Emax)**j for j in range(n+1, 0, -1)])
            bn.append((2*n+1)/(2)*np.dot(coeff_int, x_list))
        return np.array(bn)
    
    nmax = len(H_shifted_n_exp_val)-1
    bn = bn_list_Legendre(Emin, Emax, Ec, nmax)
    pn_list = np.array([np.dot(legendre(n).c, H_shifted_n_exp_val[-1:-(n+2):-1]) for n in range(0, nmax+1)]) # arranges x in reverse order since legendre polynomial coeffs are in decreasing order
    combined_list = np.multiply(bn, pn_list)
    # return a partial sum of the combined list: a numpy array with elements [combined_list[0], combined_list[0]+combined_list[1], ...]
    return np.cumsum(combined_list)


# Define the Chebyshev Polynomials
def cheb(n): 
    return Chebyshev((0,) * (n) + (1,))

# Define the linear combination of Chebyshev Polynomials
def linear_combination_chebyshev(x, coefficients):
    return chebval(x, coefficients)

# Define the step-function h(x)
def h(x, xc):
    return 1 if x < xc else 0

# Define the absolute value function of the linear combination
def abs_linear_combination(x, coefficients):
    # return np.abs(linear_combination_chebyshev(x, coefficients) - h(x, xc)) 
    return np.abs(linear_combination_chebyshev(x, coefficients))

# Define the maximize function
def maximize(f, coefficients, x0, bounds):
    result = minimize(lambda x: -f(x, coefficients), x0, bounds=bounds)
    return -result.fun, result.x

def an_list(xc, nmax):
    """
    Return the Chebyshev expansion coefficients of the Heaviside step function with a jump at xc in [-1,1].
    """
    return [-2/np.pi*np.arctan(np.sqrt((1-xc)/(1+xc)))+1] + [-2/(n*np.pi)*np.sin(n*np.arccos(xc)) for n in range(1, nmax)]

def E_to_x(E, Emin, Emax):
    """ A mapping from E to x in [-1, 1]. """
    return (2*E - Emin - Emax)/(Emax - Emin)

def Chebyshev_step_est(H:QubitOperator, phi, Lower, Upper, E0, E1, Ec, nmax):
    """
    Estimates overlap by approximating the stepwise function {1 if E_min<E<E_c, 0 if E_c<E<E_max} with Chebyshev polynomials, and replaces 
    powers of E by expectation values (moments) of x^n.
    The estimator is written as: sum_{n=0}^{n_{max}} b_n <P_n(H)> where 
    b_n = (2n+1)/2 sum_{j=0}^{n} p_n[j]/(n+1-j) (E_c-E_min)^j, which are coefficients of the expansion
    Returns:
    """
    x_of_H = 1/(Upper - Lower) * (2*H - QubitOperator('', Lower + Upper))
    x_moment_values = np.array(exp_val_higher_moment(get_sparse_operator(x_of_H), phi, nmax, return_all=True))
    # x_moment_values = E_to_x(H_exp_vals, Lower, Upper)
    x0 = E_to_x(E0, Lower, Upper)
    x1 = E_to_x(E1, Lower, Upper)
    xc = E_to_x(Ec, Lower, Upper)
    cn = an_list(xc, nmax)
    x0list = np.array([chebval(x0, cn[:n]) for n in range(1,nmax+1)]) # Chebyshev polynomial evaluated at x0
    # arranges x in reverse order since Chebyshev polynomial coeffs are in decreasing order
    pn_list = np.array([np.dot(cheb2poly(cheb(n).coef), x_moment_values[:n+1]) for n in range(0, nmax)])
    # print("pn_list: ", pn_list)
    # print("cn: ", cn)
    combined_list = np.multiply(cn, pn_list)
    overlap_estimates_unscaled = np.cumsum(combined_list)
    delta_list = np.array([maximize(abs_linear_combination, cn[:n], x1+0.05, [(x1,1)])[0] for n in range(1, nmax+1)])
    p0_lower_bound = (overlap_estimates_unscaled - delta_list)/(x0list - delta_list)
    p0_lower_bound_positive = np.where(p0_lower_bound > 0, p0_lower_bound, 0)
    return overlap_estimates_unscaled/x0list, p0_lower_bound, p0_lower_bound_positive

class PolynomialEstimator:
    def __init__(self, H:QubitOperator, phi, Lower, Upper, E0, E1, Ec, nmax):
        self.H = H
        self.phi = phi
        self.Lower = Lower
        self.Upper = Upper
        self.E0 = E0
        self.E1 = E1
        self.Ec = Ec
        self.nmax = nmax
        self.estimation = None
        self.bounds = None
        self.bounds_positive = None

    def run(self, method):
        self.method = method
        if method == "ITE":
            self.estimation = self.calculate_ite_estimation()
        elif method == "Legendre":
            self.estimation = self.calculate_legendre_step_estimation().tolist()
        elif method == "Chebyshev":
            est, bnd, bnd_pos = self.calculate_chebyshev_step_estimation()
            self.estimation = est.tolist()
            self.bounds = bnd.tolist()
            self.bounds_positive = bnd_pos.tolist()
        else:
            raise ValueError("Invalid estimation method")

    def calculate_ite_estimation(self):
        # Untested
        return ITE_est(self.H, self.phi)

    def calculate_legendre_step_estimation(self):
        # Untested
        H_shifted_n_exp_val = exp_val_higher_moment(get_sparse_operator(self.H), self.phi, self.nmax + 1, return_all=True)
        return Legendre_step_est(self.Lower, self.Upper, self.Ec, H_shifted_n_exp_val)

    def calculate_chebyshev_step_estimation(self):
        return Chebyshev_step_est(self.H, self.phi, self.Lower, self.Upper, self.E0, self.E1, self.Ec, self.nmax)
    
    def save_results_to_json(self, exact_overlap, savedir, filename):
        dict_save = {
            "method": self.method,
            "overlaps": self.estimation,
            "bounds": self.bounds,
            "bounds_positive": self.bounds_positive,
            "exact_overlap": exact_overlap
        }
        dict_save["absolute_overlap_error"] = (np.array(dict_save["overlaps"]) - exact_overlap).tolist()
        dict_save["relative_overlap_error"] = (np.abs(dict_save["absolute_overlap_error"])/exact_overlap).tolist()
        dict_save["matvec"] = ([i//2 for i in range(2,self.nmax+1)] + [(self.nmax+1)//2])
        with open(os.path.join(savedir, filename + f"_{self.method}.json"), "w") as f:
            json.dump(dict_save, f)
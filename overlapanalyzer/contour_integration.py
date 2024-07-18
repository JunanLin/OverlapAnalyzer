import numpy as np
from .eigen import evals_no_degen

def _z(c, r, angle):
    return c + r * np.exp(1j * angle)

def get_contour(c, r, N, method='gauss'):
    if method == 'trapezoidal':
        delta_theta = -2 * np.pi / N
    #    return np.array([_z(c, r, delta_theta/2 + delta_theta * i) for i in range(N)])
        return np.array([_z(c, r, delta_theta * i) for i in range(N)])
    elif method == 'gauss':
        x, w = np.polynomial.legendre.leggauss(N)
        theta = np.pi * (x + 1) / 2  # Mapping to [0, pi]
        prefactor = r * np.exp(1j * theta)
        # prefactor = r * np.exp( - 1j * np.pi * (x - 1) / 2.)
        return {"Points": c + prefactor, "Prefactor": prefactor, "Weights": w}

def get_contour_with_eigsh_info(w, degen, idx, N, method='gauss'):
    """
    Calculate integration contour for a given list of eigenvalues w, degeneracies degen, and index idx.
    The contour centers at the eigenvalue corresponding to the subspace labelled by idx, and has a radius such that only that eigenvalue is enclosed.
    """
    w_no_degen = evals_no_degen(w, degen)
    c = w_no_degen[idx]
    if idx == 0:
        r = (w_no_degen[1] - w_no_degen[0]) / 2
    elif idx == len(w_no_degen) - 1:
        r = (w_no_degen[-1] - w_no_degen[-2]) / 2
    else:
        diff_left = w_no_degen[idx] - w_no_degen[idx - 1]
        diff_right = w_no_degen[idx + 1] - w_no_degen[idx]
        r = min(diff_left, diff_right) / 2
    print(f"Selected range: ({c-r}, {c+r})")
    return c-r, c+r, get_contour(c, r, N, method)

def gauss_integration(func, c, r, N_func_eval, pointwise=False):
    x, w = np.polynomial.legendre.leggauss(N_func_eval)

    # Map Gauss-Legendre points to the contour
    theta = np.pi * (x + 1) / 2
    prefactor = r * np.exp(1j * theta)
    points = c + prefactor

    # Compute the integral approximation
    if pointwise:
        fz = np.array([func(p,index) for index, p in enumerate(points)])
    else:
        fz = np.array([func(p,0) for p in points])
    integral = np.sum(w * prefactor * fz)
    return -1j * np.pi * (integral + np.conjugate(integral)) / 2

def gauss_integration_test(func, c, r, N_func_eval):
    x, w = np.polynomial.legendre.leggauss(N_func_eval)

    # Map Gauss-Legendre points to the contour
    theta = np.pi * (x + 1) / 2
    prefactor = r * np.exp(1j * theta)
    points = c + prefactor
    fz = np.array([func(p) for p in points])
    integral = np.sum(w * prefactor * fz)
    return 1j * np.pi * (integral + np.conjugate(integral)) / 2

def sum_gauss_points(contour, computed_QF):
    integral = np.sum(contour["Weights"] * contour["Prefactor"] * computed_QF)
    return np.real_if_close((integral + np.conjugate(integral)) / 4.)


def get_contour_ellipse(c, a, b, N):
    x, w = np.polynomial.legendre.leggauss(N)
    theta = np.pi * (x + 1) / 2  # Mapping to [0, pi]
    prefactor_points = a * np.cos(theta) + b * 1j * np.sin(theta)
    prefactor_derivative = - a * np.sin(theta) + b * 1j * np.cos(theta)
    return {"Points": c + prefactor_points, "Prefactor": prefactor_derivative, "Weights": w}

def gauss_integration_ellipse(func, c, a, b, N_func_eval):
    contour_info = get_contour_ellipse(c, a, b, N_func_eval)
    points = contour_info["Points"]
    prefactor = contour_info["Prefactor"]
    w = contour_info["Weights"]

    fz = np.array([func(p) for p in points])
    integral = np.sum(w * prefactor * fz)
    return np.pi * (integral - np.conjugate(integral)) / 2

def sum_gauss_points_ellipse(contour, computed_QF):
    integral = np.sum(contour["Weights"] * contour["Prefactor"] * computed_QF)
    return np.real_if_close((integral - np.conjugate(integral)) / (4j))

def test_int_methods():
    import pandas as pd
    import datetime
    # Example usage:

    def f(z):
        # vec = 1/np.sqrt(10)*np.array([-3, 1]) # Eigenvector
        vec = 1/np.sqrt(1)*np.array([0, 1])
        A = np.array([[z-3, -2],[-2,z-3]])
        return vec.T @ (np.linalg.inv(A) @ vec)

    center = 1
    a = 1
    b = 4
    # num_points = 8
    # print("Contour: ", get_contour(center, a, num_points))
    # print("Contour: ", get_contour_ellipse_half(center, a, b,num_points))

    # print("Gauss-Legendre, circle:", gauss_integration_test(f, center, a, num_points)/(2*np.pi*1j))
    # print("Gauss-Legendre, ellipse:", gauss_integration_ellipse(f, center, a, b, num_points)/(2*np.pi*1j))
    
    radius = 1
    df = pd.DataFrame(columns=['DateTime', 'Method', 'Abs_Error', 'Num_Points'])

    for num_points in [4, 6, 8, 10, 12, 14, 16]:
        trapezoidal_approximation = numerical_contour_integration(f, center, radius, num_points, "trapezoidal")
        simpsons_approximation = numerical_contour_integration(f, center, radius, num_points, "simpsons")
        gauss_approximation = numerical_contour_integration(f, center, radius, num_points, "gauss")
        trapezoidal_row = pd.DataFrame([{'DateTime': datetime.datetime.now(), 'Method': 'Trapezoidal', 'Abs_Error': np.abs(trapezoidal_approximation/(-2 * np.pi * 1j) - 1/2),'Num_Points': num_points}])
        simpsons_row = pd.DataFrame([{'DateTime': datetime.datetime.now(), 'Method': 'Simpsons', 'Abs_Error': np.abs(simpsons_approximation/(-2 * np.pi * 1j) - 1/2),'Num_Points': num_points}])
        gauss_row = pd.DataFrame([{'DateTime': datetime.datetime.now(), 'Method': 'Gauss', 'Abs_Error': np.abs(gauss_approximation/(-2 * np.pi * 1j) - 1/2),'Num_Points': num_points}])
        df = pd.concat([df, trapezoidal_row, simpsons_row, gauss_row], ignore_index=True)
        # print("Approximation using the Trapezoidal rule:", np.real_if_close(trapezoidal_approximation/(2 * np.pi * 1j)))
        # print("Approximation using Simpson's rule:", np.real_if_close(simpsons_approximation/(2 * np.pi * 1j)))
        # print("Approximation using Gauss-Legendre Polynomials:", np.real_if_close(gauss_approximation/(2 * np.pi * 1j)))
    # Save to csv
    df.to_csv("contour_integration.csv", index=False)
    return None

# Below are old code, should avoid using them
def numerical_contour_integration(func, c, r, N_func_eval, method="gauss"):
    if method == "gauss":
        # Generate Gauss-Legendre weights and points
        x, w = np.polynomial.legendre.leggauss(N_func_eval)

        # Map Gauss-Legendre points to the contour
        prefactor = r * np.exp( - 1j * np.pi * (x - 1) / 2.)
        points = c + prefactor

        # Compute the integral approximation
        fz = np.array([func(p) for p in points])
        integral = np.sum(w * prefactor * fz)
        return - 1j * np.pi * (integral + np.conjugate(integral)) / 2.
    else:
        N = (N_func_eval - 1)
        if method == "simpsons": 
            assert N_func_eval % 2 == 0 # Used to be 1
            # N = (N_func_eval - 1) // 2
            N = N_func_eval // 2

        delta_theta = - np.pi / N
        if method == "simpsons":
            f_z = np.array([func(_z(c, r,delta_theta * i / 2.)) for i in range(2 * N_func_eval)])
        elif method == "trapezoidal":
            f_z = np.array([func(_z(c, r,delta_theta * i)) for i in range(N_func_eval)])

        # assert len(f_z) == N_func_eval

        total_sum = 0.
        for i in range(N):

            if method == "simpsons":
                factor = 1 / 6.0
                integrand = f_z[2 * i] + 4 * f_z[2 * i+1] + f_z[2 * i+2]

            elif method == "trapezoidal":
                factor = 1 / 2.0
                integrand = f_z[i] + f_z[i+1]

            total_sum += factor * integrand * (_z(c,r,(i+1)*delta_theta) - _z(c,r,i*delta_theta))

        return  ( (total_sum) - np.conjugate(total_sum) )

if __name__ == "__main__":
    print(get_contour_with_eigsh_info([3,3,4,5,6,7,7], [2,1,1,1,2], 4, 8))
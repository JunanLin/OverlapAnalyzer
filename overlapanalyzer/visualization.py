import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import numpy as np
from openfermion.linalg import get_sparse_operator
from openfermion import load_operator
from scipy.sparse import diags
from scipy.sparse import csr_matrix, csc_matrix
from overlapanalyzer.read_ham import find_files, quick_load
from overlapanalyzer.polynomial_estimates import chebval, E_to_x, an_list

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Computer Modern Roman",
    "font.size": 16,
})

def set_diag_to_zero(sparse_matrix):
    """
    Takes a sparse matrix and returns a new sparse matrix with diagonal elements set to 0.

    Args:
    sparse_matrix (csr_matrix): A scipy CSR sparse matrix.

    Returns:
    csr_matrix: A new sparse matrix with diagonal elements set to 0.
    """
    if not isinstance(sparse_matrix, (csr_matrix, csc_matrix)):
        raise ValueError("Input must be a CSR/CSC sparse matrix")

    # Extract the diagonal elements
    diagonal = sparse_matrix.diagonal()

    # Create a sparse matrix of the diagonal
    diagonal_matrix = diags(diagonal, 0, shape=sparse_matrix.shape)

    # Subtract the diagonal matrix from the original matrix
    new_matrix = sparse_matrix - diagonal_matrix
    return new_matrix


def visualize_complex_matrix(matrix, save_path):
    """
    Visualizes a given matrix using a colormap.

    Args:
    matrix (np.array): A 2D numpy array representing the matrix to be visualized.

    Returns:
    None
    """
    magnitude_matrix = np.abs(matrix)
    plt.imshow(magnitude_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    # plt.title("Matrix Visualization")
    plt.savefig(save_path, format='pdf')
    # plt.show()
    plt.close()


def visualize_chebyshev(Lower, Upper, E0, E1, Ec=None, chebyshev_degree=10):
    x_values = np.linspace(-1, 1, 400)
    plt.figure(figsize=(10, 8))
    if Ec is None:
        Ec = (E0+E1)/2
    x0 = E_to_x(E0, Lower, Upper)
    x1 = E_to_x(E1, Lower, Upper)
    xc = E_to_x(Ec, Lower, Upper)
    cheb_coeffs = an_list(xc, chebyshev_degree)
    for n in range(1, chebyshev_degree):  # Starting from 1 because cheb_coeffs[:0] would be an empty slice
        y_values = chebval(x_values, cheb_coeffs[:n])
        plt.plot(x_values, y_values, label=f'n = {n-1}')
    plt.axvline(x=x0, color='r', linestyle=':', label='x0')
    plt.axvline(x=xc, color='g', linestyle=':', label='xc')
    plt.axvline(x=x1, color='b', linestyle=':', label='x1')

    plt.title('Chebyshev Approximation of Step Function')
    plt.xlabel('x')
    # plt.ylabel('Chebval(x, cheb_coeffs[:n])')
    plt.legend()
    plt.grid(True)
    # plt.savefig("Chebyshev_polynomials.pdf")
    plt.show()


def plot_PolyDeg_vs_LB_multi(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[4, 1])
    fig.subplots_adjust(hspace=0.05)  # adjust space between Axes
    # ax = plt.gca()
    legend_elements = [Line2D([0], [0], color='grey', lw=1.5),
                       Line2D([0], [0], color='grey', linestyle='-.', lw=1.5),
                       Line2D([0], [0], color='grey', linestyle=':', lw=1.5)]

    for i, key in enumerate(data.keys()):
        if i == 0:
            current_lowest_Eckart = data[key]['Eckart']
        color = ax1._get_lines._cycler_items[i]['color']
        x_data = np.array([i for i in range(1, len(data[key]['all_Overlap_LBs'])+1)])
        ax1.plot(x_data, data[key]['all_Overlap_LBs'], marker='*', color=color, label='Lower Bound')
        ax1.axhline(y=data[key]['exact_Overlap'], linestyle='-.', color=color, label=r'$P_{exact}$')
        ax1.axhline(y=data[key]['Eckart'], linestyle=':', color=color, label='Eckart')
        ax2.plot(x_data, data[key]['all_Overlap_LBs'], marker='*', color=color, label=f'{key} LBs')
        ax2.axhline(y=data[key]['exact_Overlap'], linestyle='-.', color=color, label=f'{key}'+r' $P_{exact}$')
        ax2.axhline(y=data[key]['Eckart'], linestyle=':', color=color, label=f'{key} Eckart')
        if data[key]['Eckart'] < current_lowest_Eckart:
            current_lowest_Eckart = data[key]['Eckart']
    
    ax1.set_ylim(-0.05, 1)
    ax2.set_ylim(current_lowest_Eckart-0.02, current_lowest_Eckart+0.02)
    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = .3  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    plt.xlabel('Polynomial degree')
    ax1.set_ylabel('Overlap value')
    ax1.legend(legend_elements, ['Lower Bound', r'$P_{exact}$', 'Eckart Bound'])
    plt.show()

def plot_Bounds_vs_PolyDeg_Multi_new(data_dict_list, exactCons_data_dict_list, MaxDegree_list):
    # Font size variables
    label_fontsize = 18
    molecule_fontsize = 18
    label_padding = 3  # Padding for y-axis labels

    fig, axs = plt.subplots(len(data_dict_list), 2, figsize=(12, 8), sharex=True, sharey=True)

    # Increase spacing between subplots
    fig.subplots_adjust(wspace=0.4)  # Adjust horizontal spacing

    # Molecule names for labeling subplots
    molecule_names = [r'H$_4$', r'H$_4$', r'H$_2$O', r'H$_2$O']

    for row, data_dict in enumerate(data_dict_list):
        exactCons_data_dict = exactCons_data_dict_list[row]
        
        for key in [0, 1]:
            max_deg = MaxDegree_list[row][key]
            x_data = np.array([i for i in range(1, max_deg)])
            axs[row, key].xaxis.set_major_locator(MaxNLocator(integer=True))

            # Plot red markers (eigenvalue bounds)
            data1 = np.array(data_dict[str(key)]['Overlap_LBs'][:max_deg-1])
            mask = ~np.isnan(data1) 
            axs[row, key].plot(x_data[mask], data1[mask], marker='.', linestyle=':', label='Lower Bound (Approx.)', color='red')
            data2 = np.array(data_dict[str(key)]['Overlap_UBs'][:max_deg-1])
            mask = ~np.isnan(data2) 
            axs[row, key].plot(x_data[mask], data2[mask], marker='v', linestyle=':', label='Upper Bound (Approx.)', color='red')

            # Plot green markers (exact eigenvalues as constraints)
            data3 = np.array(exactCons_data_dict[str(key)]['Overlap_LBs'][:max_deg-1])
            mask = ~np.isnan(data3)
            axs[row, key].plot(x_data[mask], data3[mask], marker='.', linestyle=':', label='Lower Bound (Exact)', color='green')
            data4 = np.array(exactCons_data_dict[str(key)]['Overlap_UBs'][:max_deg-1])
            mask = ~np.isnan(data4)
            axs[row, key].plot(x_data[mask], data4[mask], marker='v', linestyle=':', label='Upper Bound (Exact)', color='green')

            # Plot blue horizontal line (exact overlap)
            axs[row, key].axhline(y=data_dict[str(key)]['exact_ovlp'], linestyle='-', label='Exact Overlap', color='blue')

            # Set subplot limits
            axs[row, key].set_ylim(-0.05, 1.0)
            axs[row, key].set_yticks([0.0, 0.5, 1.0])

            # Adjust x-axis limits
            if key == 0:  # Left plots
                axs[row, key].set_xlim(0, 21)
            else:  # Right plots
                axs[row, key].set_xlim(0, max_deg)

            # Add molecule label to top right corner
            axs[row, key].text(
                0.95, 0.95, molecule_names[row * 2 + key],
                transform=axs[row, key].transAxes,
                fontsize=molecule_fontsize,
                verticalalignment='top', horizontalalignment='right'
            )

    # Y-axis labels with increased padding
    axs[0, 0].set_ylabel(r'$P_0$', fontsize=label_fontsize, labelpad=label_padding)
    axs[1, 0].set_ylabel(r'$P_0$', fontsize=label_fontsize, labelpad=label_padding)
    axs[0, 1].set_ylabel(r'$P_1$', fontsize=label_fontsize, labelpad=label_padding)
    axs[1, 1].set_ylabel(r'$P_1$', fontsize=label_fontsize, labelpad=label_padding)

    # X-axis labels
    axs[1, 0].set_xlabel('Polynomial Degree', fontsize=label_fontsize)
    axs[1, 1].set_xlabel('Polynomial Degree', fontsize=label_fontsize)

    # Force y-axis tick labels to show on all plots
    for ax in axs.flat:
        ax.tick_params(axis='y', which='both', labelleft=True)

    # Place legend in the top-right subplot
    axs[0, 1].legend(loc='best', fontsize=10)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.show()


def plot_Bounds_vs_PolyDeg(data, molname):
    for i, key in enumerate(data.keys()):
        if data[key]['Eckart'] > 0:
            x_data = np.array([i for i in range(1, len(data[key]['Overlap_LBs'])+1)])
            plt.plot(x_data, data[key]['Overlap_LBs'], marker='.', linestyle=':', label=f'Lower Bound', color='red')
            plt.plot(x_data, data[key]['Overlap_UBs'], marker='v', linestyle=':', label='Upper Bound', color='red')
            plt.axhline(y=data[key]['exact_Overlap'], linestyle='-', label=r' $P_{exact}$', color='blue')
            plt.axhline(y=data[key]['Eckart'], linestyle='-.', label='Eckart Bound', color='blue')
            plt.xticks(x_data)
            plt.xlabel('Polynomial degree')
            plt.ylabel('Overlap value')
            plt.legend()
            # plt.title(molname + f', R = {key}')
            plt.show()
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[4, 1])
            fig.subplots_adjust(hspace=0.05)  # adjust space between Axes
            # ax = plt.gca()


            x_data = np.array([i for i in range(1, len(data[key]['Overlap_LBs'])+1)])
            ax1.plot(x_data, data[key]['Overlap_LBs'], marker='.', linestyle=':', color='red', label='Lower Bound')
            ax1.plot(x_data, data[key]['Overlap_UBs'], marker='v', linestyle=':', color='red', label='Upper Bound')
            ax1.axhline(y=data[key]['exact_Overlap'], linestyle='-', color='blue', label=r'$P_{exact}$')
            ax1.axhline(y=data[key]['Eckart'], linestyle='-.', color='blue', label='Eckart Bound')
            # ax2.plot(x_data, data[key]['Overlap_LBs'], marker='.', linestyle=':', color=color)
            # ax2.axhline(y=data[key]['exact_Overlap'], linestyle='-', color=color)
            ax2.axhline(y=data[key]['Eckart'], linestyle='-.', color='blue')

            
            ax1.set_ylim(-0.05, 1.05)
            ax2.set_ylim(data[key]['Eckart']-0.02, data[key]['Eckart']+0.02)
            # hide the spines between ax and ax2
            ax1.spines.bottom.set_visible(False)
            ax2.spines.top.set_visible(False)
            ax1.xaxis.tick_top()
            ax1.tick_params(labeltop=False)  # don't put tick labels at the top
            ax1.set_xticks([])
            ax1.set_xticks([], minor=True)
            ax2.xaxis.tick_bottom()

            d = .3  # proportion of vertical to horizontal extent of the slanted line
            kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                        linestyle="none", color='k', mec='k', mew=1, clip_on=False)
            ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
            ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
            ax2.set_xticks(x_data)
            ax2.set_xlabel('Polynomial degree')
            ax1.set_ylabel('Overlap value')
            ax1.legend()
            # ax1.set_title(molname + f', R = {key}')
            plt.show()

def plot_Bounds_vs_PolyDeg_Multi(data_dict_list, exactCons_data_dict_list, MaxDegree_list):
    fig, axs = plt.subplots(len(data_dict_list),2)

    for row, data_dict in enumerate(data_dict_list):
        exactCons_data_dict = exactCons_data_dict_list[row]
        for key in [0, 1]:
            max_deg = MaxDegree_list[row][key]
            x_data = np.array([i for i in range(1, max_deg)])
            axs[row, key].xaxis.set_major_locator(MaxNLocator(integer=True))

            data1 = np.array(data_dict[str(key)]['Overlap_LBs'][:max_deg-1])
            mask = ~np.isnan(data1) 
            axs[row, key].plot(x_data[mask], data1[mask], marker='.', linestyle=':', label='Lower Bound', color='red')
            data2 = np.array(data_dict[str(key)]['Overlap_UBs'][:max_deg-1])
            mask = ~np.isnan(data2) 
            axs[row, key].plot(x_data[mask], data2[mask], marker='v', linestyle=':', label='Upper Bound', color='red')
            data3 = np.array(exactCons_data_dict[str(key)]['Overlap_LBs'][:max_deg-1])
            mask = ~np.isnan(data3)
            axs[row, key].plot(x_data[mask], data3[mask], marker='.', linestyle=':', label='Lower Bound', color='green')
            data4 = np.array(exactCons_data_dict[str(key)]['Overlap_UBs'][:max_deg-1])
            mask = ~np.isnan(data4)
            axs[row, key].plot(x_data[mask], data4[mask], marker='v', linestyle=':', label='Upper Bound', color='green')
            axs[row, key].axhline(y=data_dict[str(key)]['exact_ovlp'], linestyle='-', label=r' $P_{exact}$', color='blue')
            # plt.axhline(y=data_dict[str(key)]['Eckart'], linestyle='-.', label='Eckart Bound', color='blue')
            # axs[row, key].set_xticks(x_data)
            # axs[row, key].xlabel('Polynomial degree')
            # axs[row, key].ylabel('Overlap value')
            # axs[row, key].legend()
            # plt.title(molname + f', R = {key}')
    plt.tight_layout()
    plt.show()

def plot_inaccuracy_vs_LB(data):
    for i, key in enumerate(data.keys()):
        inaccuracies = np.array(data[key]['inaccuracies'])*2/data[key]['gap']
        O_exact = data[key]['O_exact']
        for pol_deg in data[key]["Overlap_LBs"].keys():
            y_data = np.abs(np.array(data[key]["Overlap_LBs"][pol_deg]) - O_exact) / O_exact
            plt.semilogy(inaccuracies, y_data, marker='.', linestyle=':', label=pol_deg)
        plt.legend()
        plt.ylabel('Relative error')
        plt.xlabel('Energy Inaccuracy')
        plt.show()
        plt.close()
        O_exact = data[key]['O_exact']
        for pol_deg in data[key]["Overlap_UBs"].keys():
            y_data = np.abs(np.array(data[key]["Overlap_UBs"][pol_deg]) - O_exact) / O_exact
            plt.semilogy(inaccuracies, y_data, marker='.', linestyle=':', label=pol_deg)
        plt.legend()
        plt.ylabel('Relative error')
        plt.xlabel('Energy Inaccuracy')
        plt.show()
        plt.close()
        
if __name__ == "__main__":

    dir = 'hamiltonian_gen_test/h2o/4e4o'
    load_ham_type = 'fer'
    ham_filenames = find_files(dir+'/ham_'+load_ham_type,".data") # list of filenames ending with ".data"

    for i, filename in enumerate(ham_filenames):
        H_loaded_q = load_operator(data_directory=dir+'/ham_'+load_ham_type, file_name=filename, plain_text=True)
        H_diag_0 = set_diag_to_zero(get_sparse_operator(H_loaded_q))
        visualize_complex_matrix(H_diag_0.toarray(), save_path=dir+'/ham_'+load_ham_type+'/'+filename[:-5]+'.pdf')
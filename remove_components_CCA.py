import numpy as np

def remove_comps_CCA(raw_data, no_exclude_comps, W_st, A_st):
    # Apply obtained weights to raw dataset (W dimensions n_channels x n_components) - matrix multiplication
    CCA_data = raw_data.T @ W_st[:, no_exclude_comps:]  # n_times, n_components
    CCA_data = CCA_data.T  # n_components, n_times

    # Reconstruct data without the first x components (highest correlations)
    for_recon = A_st[:, no_exclude_comps:]
    # CCA_data: n_components x n_times, for_recon: n_channels, n_components
    reconstructed_data = np.tensordot(CCA_data, for_recon, axes=(0, 1))  # n_times, n_channels

    return reconstructed_data
import numpy as np

def remove_comps_DSS(raw_data, no_chans, no_exclude_comps, todss, fromdss):
    # Apply obtained weights to raw dataset and then reconstruct in the sensor space
    keep = np.ones(no_chans)
    keep[:no_exclude_comps] = 0  # don't keep first Nc components
    E = np.diag(keep)

    # raw_data: n_chans x n_times
    # todss: n_components, n_chans
    # fromdss: n_components, n_chans
    # reconstructed, shape=(n_times, n_chans)
    reconstructed_data = raw_data.T @ todss @ E @ fromdss

    return reconstructed_data
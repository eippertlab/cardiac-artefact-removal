# Taking the peak-peak value in the potential window of interest, and seeing how this varies across trials both before
# and after CCA - Prepared, ICA, and SSP 5
# Using the coefficient of variation as the metric


import mne
import numpy as np
import h5py
from scipy.io import loadmat
from SNR_functions import *
import pandas as pd
from invert import invert
from scipy.stats import variation
from reref_data import rereference_data
import matplotlib.pyplot as plt


def get_cropped_epochs(raw_data, epoch_interval, baseline_interval, interest_window, trigger):
    es, e_ids = mne.events_from_annotations(raw_data)  # Get events and ids
    e_id_dict = {key: value for key, value in e_ids.items() if key == trigger}  # Extract relevant
    epos = mne.Epochs(raw_data, es, event_id=e_id_dict, tmin=epoch_interval[0], tmax=epoch_interval[1],
                      baseline=tuple(baseline_interval), preload=True)  # Create epochs
    epos = epos.crop(tmin=interest_window[0], tmax=interest_window[1])  # Crop

    return epos

def get_coeffofvariation(epos, ch):
    data = epos.get_data(picks=ch)  # n_epochs, n_channels, n_times
    data = np.squeeze(data)  # Remove channel dimension as we only have one
    peak_peak_amp = np.ptp(data, axis=1, keepdims=True)  # Returns peak-peak val of each epoch (2000, 1)
    # Then get variance of these peak-peak vals
    coeff_var = variation(peak_peak_amp, axis=0)[0]  # Just one number

    return coeff_var


if __name__ == '__main__':
    which_method = {'Prep': True,
                    'ICA': True,
                    'SSP5': True}

    subjects = np.arange(1, 37)  # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]
    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000

    # Contains information on which CCA component to pick and how to flip
    xls = pd.ExcelFile('/data/p_02569/Components_CCA.xls')
    df_CCA = pd.read_excel(xls, 'Dataset 1')
    df_CCA.set_index('Subject', inplace=True)

    # Contains information on which CCA component to pick and how to flip
    xls = pd.ExcelFile('/data/p_02569/Components_DSS.xls')
    df_DSS = pd.read_excel(xls, 'Dataset 1')
    df_DSS.set_index('Subject', inplace=True)

    for i in np.arange(0, len(which_method)):
        method = list(which_method.keys())[i]
        if which_method[method]:  # If this method is true, go through with the rest
            #############################################################################################
            # CCA corrected data
            #############################################################################################
            class save_Var():
                def __init__(self):
                    pass

            # Instantiate class
            savevar = save_Var()

            # Matrix of dimensions no.subjects
            var_med = np.zeros((len(subjects), 1))
            var_tib = np.zeros((len(subjects), 1))
            for subject in subjects:
                for cond_name in cond_names:
                    # Get the correct condition variables
                    if cond_name == 'tibial':
                        trigger_name = 'Tibial - Stimulation'
                        potential_window = [12 / 1000, 32 / 1000]
                    elif cond_name == 'median':
                        trigger_name = 'Median - Stimulation'
                        potential_window = [8 / 1000, 18 / 1000]

                    subject_id = f'sub-{str(subject).zfill(3)}'

                    # Get the right file path
                    if method == 'Prep':
                        save_path = f"/data/pt_02569/tmp_data/prepared_py_cca/"
                        file_path = f"/data/pt_02569/tmp_data/prepared_py_cca/{subject_id}/"
                        file_name = f'noStimart_sr1000_{cond_name}_withqrs.fif'
                    elif method == 'ICA':
                        save_path = f"/data/pt_02569/tmp_data/baseline_ica_py_cca/"
                        file_path = f"/data/pt_02569/tmp_data/baseline_ica_py_cca/{subject_id}/"
                        file_name = f'clean_baseline_ica_auto_{cond_name}.fif'
                    elif method == 'SSP5':
                        save_path = f"/data/pt_02569/tmp_data/ssp_py_cca/"
                        file_path = f"/data/pt_02569/tmp_data/ssp_py_cca/{subject_id}/5 projections/"
                        file_name = f'ssp_cleaned_{cond_name}.fif'

                    epochs = mne.read_epochs(f"{file_path}{file_name}", preload=True)
                    channel = df_CCA.loc[subject_id, f"{method}_{cond_name}"]
                    inv = df_CCA.loc[subject_id, f"{method}_{cond_name}_inv"]
                    if inv == 'inv':
                        epochs.apply_function(invert, picks=channel)

                    epochs = epochs.crop(tmin=potential_window[0], tmax=potential_window[1])
                    var = get_coeffofvariation(epochs, channel)
                    if cond_name == 'median':
                        var_med[subject - 1, 0] = var
                    elif cond_name == 'tibial':
                        var_tib[subject - 1, 0] = var

            # Save to file
            savevar.var_med = var_med
            savevar.var_tib = var_tib
            dataset_keywords = [a for a in dir(savevar) if not a.startswith('__')]

            fn = f"{save_path}variance.h5"
            with h5py.File(fn, "w") as outfile:
                for keyword in dataset_keywords:
                    outfile.create_dataset(keyword, data=getattr(savevar, keyword))


            #############################################################################################
            # DSS corrected data
            #############################################################################################
            class save_Var():
                def __init__(self):
                    pass

            # Instantiate class
            savevar = save_Var()

            # Matrix of dimensions no.subjects
            var_med = np.zeros((len(subjects), 1))
            var_tib = np.zeros((len(subjects), 1))

            for subject in subjects:
                for cond_name in cond_names:
                    # Get the correct condition variables
                    if cond_name == 'tibial':
                        trigger_name = 'Tibial - Stimulation'
                        potential_window = [12 / 1000, 32 / 1000]
                    elif cond_name == 'median':
                        trigger_name = 'Median - Stimulation'
                        potential_window = [8 / 1000, 18 / 1000]

                    subject_id = f'sub-{str(subject).zfill(3)}'

                    # Get the right file path
                    if method == 'Prep':
                        save_path = f"/data/pt_02569/tmp_data/prepared_py_dss/"
                        file_path = f"/data/pt_02569/tmp_data/prepared_py_dss/{subject_id}/"
                        file_name = f'noStimart_sr1000_{cond_name}_withqrs.fif'
                    elif method == 'ICA':
                        save_path = f"/data/pt_02569/tmp_data/baseline_ica_py_dss/"
                        file_path = f"/data/pt_02569/tmp_data/baseline_ica_py_dss/{subject_id}/"
                        file_name = f'clean_baseline_ica_auto_{cond_name}.fif'
                    elif method == 'SSP5':
                        save_path = f"/data/pt_02569/tmp_data/ssp_py_dss/"
                        file_path = f"/data/pt_02569/tmp_data/ssp_py_dss/{subject_id}/5 projections/"
                        file_name = f'ssp_cleaned_{cond_name}.fif'

                    epochs = mne.read_epochs(f"{file_path}{file_name}", preload=True)
                    channel = df_DSS.loc[subject_id, f"{method}_{cond_name}"]
                    inv = df_DSS.loc[subject_id, f"{method}_{cond_name}_inv"]
                    if inv == 'inv':
                        epochs.apply_function(invert, picks=channel)

                    epochs = epochs.crop(tmin=potential_window[0], tmax=potential_window[1])
                    var = get_coeffofvariation(epochs, channel)
                    if cond_name == 'median':
                        var_med[subject - 1, 0] = var
                    elif cond_name == 'tibial':
                        var_tib[subject - 1, 0] = var

            # Save to file
            savevar.var_med = var_med
            savevar.var_tib = var_tib
            dataset_keywords = [a for a in dir(savevar) if not a.startswith('__')]

            fn = f"{save_path}variance.h5"
            with h5py.File(fn, "w") as outfile:
                for keyword in dataset_keywords:
                    outfile.create_dataset(keyword, data=getattr(savevar, keyword))

            #############################################################################################
            # Deal with non-cca corrected data
            #############################################################################################
            class save_Var():
                def __init__(self):
                    pass

            # Instantiate class
            savevar = save_Var()

            # Matrix of dimensions no.subjects
            var_med = np.zeros((len(subjects), 1))
            var_tib = np.zeros((len(subjects), 1))

            for subject in subjects:
                for cond_name in cond_names:
                    # Get correct condition information
                    if cond_name == 'tibial':
                        trigger_name = 'Tibial - Stimulation'
                        potential_window = [12 / 1000, 32 / 1000]
                        channel = 'L1'
                    elif cond_name == 'median':
                        trigger_name = 'Median - Stimulation'
                        potential_window = [8 / 1000, 18 / 1000]
                        channel = 'SC6'

                    subject_id = f'sub-{str(subject).zfill(3)}'

                    # Get the right file path
                    if method == 'Prep':
                        save_path = f"/data/pt_02569/tmp_data/prepared_py/"
                        file_path = f"/data/pt_02569/tmp_data/prepared_py/{subject_id}/"
                        file_name = f'noStimart_sr1000_{cond_name}_withqrs.fif'
                    elif method == 'ICA':
                        save_path = f"/data/pt_02569/tmp_data/baseline_ica_py/"
                        file_path = f"/data/pt_02569/tmp_data/baseline_ica_py/{subject_id}/"
                        file_name = f'clean_baseline_ica_auto_{cond_name}.fif'
                    elif method == 'SSP5':
                        save_path = f"/data/pt_02569/tmp_data/ssp_py/"
                        file_path = f"/data/pt_02569/tmp_data/ssp_py/{subject_id}/5 projections/"
                        file_name = f'ssp_cleaned_{cond_name}.fif'

                    raw = mne.io.read_raw_fif(f"{file_path}{file_name}", preload=True)
                    epochs = get_cropped_epochs(raw, iv_epoch, iv_baseline, potential_window, trigger_name)
                    var = get_coeffofvariation(epochs, channel)

                    if cond_name == 'median':
                        var_med[subject - 1, 0] = var
                    elif cond_name == 'tibial':
                        var_tib[subject - 1, 0] = var

            # Save to file
            savevar.var_med = var_med
            savevar.var_tib = var_tib
            dataset_keywords = [a for a in dir(savevar) if not a.startswith('__')]

            fn = f"{save_path}variance.h5"
            with h5py.File(fn, "w") as outfile:
                for keyword in dataset_keywords:
                    outfile.create_dataset(keyword, data=getattr(savevar, keyword))


    ########################## Print to screen #################################
    for cca_flag in [True, False]:
        keywords = ['var_med', 'var_tib']
        if cca_flag:
            input_paths = ["/data/pt_02569/tmp_data/prepared_py_cca/",
                           "/data/pt_02569/tmp_data/baseline_ica_py_cca/",
                           "/data/pt_02569/tmp_data/ssp_py_cca/"]
        else:
            input_paths = ["/data/pt_02569/tmp_data/prepared_py/",
                           "/data/pt_02569/tmp_data/baseline_ica_py/",
                           "/data/pt_02569/tmp_data/ssp_py/"]

        names = ['Prepared', 'ICA', 'SSP']

        if cca_flag:
            print("\n")
            print('CCA Corrected Results')
            for i in np.arange(0, len(input_paths)):
                input_path = input_paths[i]
                name = names[i]
                fn = f"{input_path}variance.h5"
                with h5py.File(fn, "r") as infile:
                    # Get the data
                    var_med = infile[keywords[0]][()]
                    var_tib = infile[keywords[1]][()]

                average_med = np.nanmean(var_med, axis=0)
                average_tib = np.nanmean(var_tib, axis=0)

                if name == 'SSP':
                    for n in np.arange(0, 1):
                        print(f'Var {name} Median {n + 5}: {average_med[n]:.4f}')
                        print(f'Var {name} Tibial {n + 5}: {average_tib[n]:.4f}')
                else:
                    print(f'Var {name} Median: {average_med[0]:.4f}')
                    print(f'Var {name} Tibial: {average_tib[0]:.4f}')
        else:
            print("\n")
            print('Prior to CCA Results')
            for i in np.arange(0, len(input_paths)):
                input_path = input_paths[i]
                name = names[i]
                fn = f"{input_path}variance.h5"
                # All have shape (24, 1) bar SSP which is (24, 16)
                with h5py.File(fn, "r") as infile:
                    # Get the data
                    var_med = infile[keywords[0]][()]
                    var_tib = infile[keywords[1]][()]

                average_med = np.nanmean(var_med, axis=0)
                average_tib = np.nanmean(var_tib, axis=0)

                if name == 'SSP':
                    for n in np.arange(0, 1):
                        print(f'Var {name} Median {n + 5}: {average_med[n]:.4f}')
                        print(f'Var {name} Tibial {n + 5}: {average_tib[n]:.4f}')
                else:
                    print(f'Var {name} Median: {average_med[0]:.4f}')
                    print(f'Var {name} Tibial: {average_tib[0]:.4f}')

        keywords = ['var_med', 'var_tib']
        input_paths = ["/data/pt_02569/tmp_data/prepared_py_dss/",
                       "/data/pt_02569/tmp_data/baseline_ica_py_dss/",
                       "/data/pt_02569/tmp_data/ssp_py_dss/"]
        names = ['Prepared', 'ICA', 'SSP']
        print("\n")
        print('DSS Corrected Results')
        for i in np.arange(0, len(input_paths)):
            input_path = input_paths[i]
            name = names[i]
            fn = f"{input_path}variance.h5"
            with h5py.File(fn, "r") as infile:
                # Get the data
                var_med = infile[keywords[0]][()]
                var_tib = infile[keywords[1]][()]

            average_med = np.nanmean(var_med, axis=0)
            average_tib = np.nanmean(var_tib, axis=0)

            if name == 'SSP':
                for n in np.arange(0, 1):
                    print(f'Var {name} Median {n + 5}: {average_med[n]:.4f}')
                    print(f'Var {name} Tibial {n + 5}: {average_tib[n]:.4f}')
            else:
                print(f'Var {name} Median: {average_med[0]:.4f}')
                print(f'Var {name} Tibial: {average_tib[0]:.4f}')
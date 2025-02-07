# Compute SNR of the data for each method after CCA or DSS was applied
# The SNR was estimated by dividing the evoked response peak amplitude
# (absolute value) by the standard deviation of the LEP waveform in
# the pre-stimulus interval
# https://www.sciencedirect.com/science/article/abs/pii/S105381190901297X

import mne
import numpy as np
import h5py
import pandas as pd
from scipy.io import loadmat
from SNR_functions import *
from invert import invert

if __name__ == '__main__':
    reduced_epochs = False  # Use a smaller number of epochs to calculate the SNR, standard is False
    reduced_window = False  # Smaller window about expected peak, standard is False

    subjects = np.arange(1, 37)  # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']
    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    iv_baseline = cfg['iv_baseline'][0] / 1000

    # Contains information on which CCA component to pick and how to flip
    xls = pd.ExcelFile('/data/p_02569/Components_CCA.xls')
    df_CCA = pd.read_excel(xls, 'Dataset 1')
    df_CCA.set_index('Subject', inplace=True)

    # Contains information on which CCA component to pick and how to flip
    xls = pd.ExcelFile('/data/p_02569/Components_DSS.xls')
    df_DSS = pd.read_excel(xls, 'Dataset 1')
    df_DSS.set_index('Subject', inplace=True)

    # Loop through methods and save as required
    which_method = {'Prep': True,
                    'ICA': True,
                    'SSP5': True}

    for i in np.arange(0, len(which_method)):
        method = list(which_method.keys())[i]
        if which_method[method]:  # If this method is true, go through with the rest
            #############################################################################################
            # CCA corrected data
            #############################################################################################
            class save_SNR():
                def __init__(self):
                    pass

            # Instantiate class
            savesnr = save_SNR()

            snr_med = np.zeros((len(subjects), 1))
            snr_tib = np.zeros((len(subjects), 1))

            for subject in subjects:
                for cond_name in cond_names:
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

                    evoked = epochs.pick(channel).average()
                    snr = calculate_SNR_evoked_ccadss(evoked, cond_name, iv_baseline, channel)

                    # Now have one snr related to each subject and condition
                    if cond_name == 'median':
                        snr_med[subject - 1, 0] = snr
                    elif cond_name == 'tibial':
                        snr_tib[subject - 1, 0] = snr

            savesnr.snr_med = snr_med
            savesnr.snr_tib = snr_tib
            dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]

            fn = f"{save_path}snr.h5"
            with h5py.File(fn, "w") as outfile:
                for keyword in dataset_keywords:
                    outfile.create_dataset(keyword, data=getattr(savesnr, keyword))


            #############################################################################################
            # DSS corrected data
            #############################################################################################
            class save_SNR():
                def __init__(self):
                    pass

            # Instantiate class
            savesnr = save_SNR()

            snr_med = np.zeros((len(subjects), 1))
            snr_tib = np.zeros((len(subjects), 1))

            for subject in subjects:
                for cond_name in cond_names:
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

                    evoked = epochs.pick(channel).average()
                    snr = calculate_SNR_evoked_ccadss(evoked, cond_name, iv_baseline, channel)

                    # Now have one snr related to each subject and condition
                    if cond_name == 'median':
                        snr_med[subject - 1, 0] = snr
                    elif cond_name == 'tibial':
                        snr_tib[subject - 1, 0] = snr

            savesnr.snr_med = snr_med
            savesnr.snr_tib = snr_tib
            dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]

            fn = f"{save_path}snr.h5"
            with h5py.File(fn, "w") as outfile:
                for keyword in dataset_keywords:
                    outfile.create_dataset(keyword, data=getattr(savesnr, keyword))

    ############### Print to Screen numbers #################
    keywords = ['snr_med', 'snr_tib']
    input_paths_cca = {'Prep': "/data/pt_02569/tmp_data/prepared_py_cca/",
                       'ICA': "/data/pt_02569/tmp_data/baseline_ica_py_cca/",
                       'SSP': "/data/pt_02569/tmp_data/ssp_py_cca/"}

    input_paths_dss = {'Prep': "/data/pt_02569/tmp_data/prepared_py_dss/",
                       'ICA': "/data/pt_02569/tmp_data/baseline_ica_py_dss/",
                       'SSP': "/data/pt_02569/tmp_data/ssp_py_dss/"}

    print("\nCCA corrected data")
    for i in np.arange(0, len(input_paths_cca)):
        name = list(input_paths_cca.keys())[i]
        input_path = input_paths_cca[name]
        fn = f"{input_path}snr.h5"
        with h5py.File(fn, "r") as infile:
            # Get the data
            snr_med = infile[keywords[0]][()]
            snr_tib = infile[keywords[1]][()]

        average_med = np.nanmean(snr_med, axis=0)
        average_tib = np.nanmean(snr_tib, axis=0)

        print(f'SNR {name} Median: {average_med[0]:.4f}')
        print(f'SNR {name} Tibial: {average_tib[0]:.4f}')

    print("\nDSS corrected data")
    for i in np.arange(0, len(input_paths_dss)):
        name = list(input_paths_dss.keys())[i]
        input_path = input_paths_dss[name]
        fn = f"{input_path}snr.h5"
        with h5py.File(fn, "r") as infile:
            # Get the data
            snr_med = infile[keywords[0]][()]
            snr_tib = infile[keywords[1]][()]

        average_med = np.nanmean(snr_med, axis=0)
        average_tib = np.nanmean(snr_tib, axis=0)

        print(f'SNR {name} Median: {average_med[0]:.4f}')
        print(f'SNR {name} Tibial: {average_tib[0]:.4f}')


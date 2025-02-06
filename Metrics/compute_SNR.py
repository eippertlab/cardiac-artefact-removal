# Compute SNR of the data for each method
# The SNR was estimated by dividing the evoked response peak amplitude
# (absolute value) by the standard deviation of the LEP waveform in
# the pre-stimulus interval
# https://www.sciencedirect.com/science/article/abs/pii/S105381190901297X

import mne
import numpy as np
import h5py
from scipy.io import loadmat
from SNR_functions import *
from reref_data import rereference_data
import pickle
from replace_data import replace_data
from remove_components_CCA import remove_comps_CCA
from remove_components_DSS import remove_comps_DSS

if __name__ == '__main__':
    reduced_epochs = False  # Use a smaller number of epochs to calculate the SNR, standard is False
    reduced_window = False  # Smaller window about expected peak, standard is False
    ant_ref = False  # Use the data that has been anteriorly referenced instead for SSP
    choose_limited = False  # Use ICA with limited components removed

    subjects = np.arange(1, 37)  # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]
    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    # Loop through methods and save as required
    which_method = {'Prep': False,
                    'PCA': False,
                    'PCA Tukey': False,
                    'ICA': False,
                    'ICA-Anterior': False,
                    'ICA-Separate': False,
                    'SSP': False,
                    'CCA_heart': False,
                    'DSS_heart': False}

    for i in np.arange(0, len(which_method)):
        method = list(which_method.keys())[i]
        if which_method[method]:  # If this method is true, go through with the rest
            class save_SNR():
                def __init__(self):
                    pass

            # Instantiate class
            savesnr = save_SNR()

            # Treat SSP separately - has extra loop for projections
            if method == 'SSP':
                snr_med = np.zeros((len(subjects), len(np.arange(1, 21))))  # 5, 21
                snr_tib = np.zeros((len(subjects), len(np.arange(1, 21))))  # 5, 21
                chan_med = []
                chan_tib = []

                for subject in subjects:
                    for cond_name in cond_names:
                        if cond_name == 'tibial':
                            trigger_name = 'Tibial - Stimulation'
                        elif cond_name == 'median':
                            trigger_name = 'Median - Stimulation'

                        subject_id = f'sub-{str(subject).zfill(3)}'

                        # Want the SNR for each projection tried from 1 to 20
                        for n in np.arange(1, 21):  # (5, 21)
                            # Load SSP projection data
                            input_path = "/data/pt_02569/tmp_data/ssp_py/" + subject_id
                            savename = input_path + "/" + str(n) + " projections/"
                            if ant_ref:
                                raw = mne.io.read_raw_fif(f"{savename}ssp_cleaned_{cond_name}_antRef.fif")
                            else:
                                raw = mne.io.read_raw_fif(f"{savename}ssp_cleaned_{cond_name}.fif")
                            evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
                            snr, chan = calculate_SNR_evoked(evoked, cond_name, iv_baseline, reduced_window)

                            # Now have one snr for relevant channel in each subject + condition
                            if cond_name == 'median':
                                snr_med[subject - 1, n - 1] = snr
                                chan_med.append(chan)
                            elif cond_name == 'tibial':
                                snr_tib[subject - 1, n - 1] = snr
                                chan_tib.append(chan)

                # Save to file to compare to matlab - only for debugging
                savesnr.snr_med = snr_med
                savesnr.snr_tib = snr_tib
                savesnr.chan_med = chan_med
                savesnr.chan_tib = chan_tib
                dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]
                if reduced_window:
                    if reduced_epochs:
                        if ant_ref:
                            fn = f"/data/pt_02569/tmp_data/ssp_py/snr_reduced_ant_smallwin.h5"
                        else:
                            fn = f"/data/pt_02569/tmp_data/ssp_py/snr_reduced_smallwin.h5"
                    else:
                        if ant_ref:
                            fn = f"/data/pt_02569/tmp_data/ssp_py/snr_ant_smallwin.h5"
                        else:
                            fn = f"/data/pt_02569/tmp_data/ssp_py/snr_smallwin.h5"
                else:
                    if reduced_epochs:
                        if ant_ref:
                            fn = f"/data/pt_02569/tmp_data/ssp_py/snr_reduced_ant.h5"
                        else:
                            fn = f"/data/pt_02569/tmp_data/ssp_py/snr_reduced.h5"
                    else:
                        if ant_ref:
                            fn = f"/data/pt_02569/tmp_data/ssp_py/snr_ant.h5"
                        else:
                            fn = f"/data/pt_02569/tmp_data/ssp_py/snr.h5"

                with h5py.File(fn, "w") as outfile:
                    for keyword in dataset_keywords:
                        outfile.create_dataset(keyword, data=getattr(savesnr, keyword))

            # CCA_heart and DSS_heart also separate to account for projections
            elif method == 'CCA_heart':
                snr_med = np.zeros((len(subjects), len(np.arange(1, 21))))
                snr_tib = np.zeros((len(subjects), len(np.arange(1, 21))))
                chan_med = []
                chan_tib = []

                for subject in subjects:
                    for cond_name in cond_names:
                        if cond_name == 'tibial':
                            trigger_name = 'Tibial - Stimulation'
                        elif cond_name == 'median':
                            trigger_name = 'Median - Stimulation'

                        subject_id = f'sub-{str(subject).zfill(3)}'

                        # Load prepared_data
                        input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/"
                        raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr1000_{cond_name}_withqrs.fif", preload=True)
                        raw_data = raw.get_data(picks=esg_chans)
                        input_path_CCA = f"/data/pt_02569/tmp_data/cca_heartart_py/{subject_id}/"

                        # Want the SNR for each selection of components removed from 1 to 20
                        for n in np.arange(1, 21):
                            # Load weights and spatial patterns
                            with open(f'{input_path_CCA}{cond_name}_heart_cca_Ast.pkl', 'rb') as f:
                                A_st = pickle.load(f)
                            with open(f'{input_path_CCA}{cond_name}_heart_cca_Wst.pkl', 'rb') as f:
                                W_st = pickle.load(f)
                            reconstructed_data = remove_comps_CCA(raw_data, n, W_st, A_st)
                            replace_kwargs = dict(
                                new_data=reconstructed_data.T  # n_channels, n_times
                            )
                            # Replace and save
                            clean_raw = raw.copy().apply_function(replace_data, picks=esg_chans, channel_wise=False,
                                                                  **replace_kwargs)
                            evoked = evoked_from_raw(clean_raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
                            snr, chan = calculate_SNR_evoked(evoked, cond_name, iv_baseline, reduced_window)

                            # Now have one snr for relevant channel in each subject + condition
                            if cond_name == 'median':
                                snr_med[subject - 1, n - 1] = snr
                                chan_med.append(chan)
                            elif cond_name == 'tibial':
                                snr_tib[subject - 1, n - 1] = snr
                                chan_tib.append(chan)

                savesnr.snr_med = snr_med
                savesnr.snr_tib = snr_tib
                savesnr.chan_med = chan_med
                savesnr.chan_tib = chan_tib
                dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]
                if reduced_window:
                    if reduced_epochs:
                        fn = f"/data/pt_02569/tmp_data/cca_heartart_py/snr_reduced_smallwin.h5"
                    else:
                        fn = f"/data/pt_02569/tmp_data/cca_heartart_py/snr_smallwin.h5"
                else:
                    if reduced_epochs:
                        fn = f"/data/pt_02569/tmp_data/cca_heartart_py/snr_reduced.h5"
                    else:
                        fn = f"/data/pt_02569/tmp_data/cca_heartart_py/snr.h5"

                with h5py.File(fn, "w") as outfile:
                    for keyword in dataset_keywords:
                        outfile.create_dataset(keyword, data=getattr(savesnr, keyword))

            elif method == 'DSS_heart':
                snr_med = np.zeros((len(subjects), len(np.arange(1, 21))))
                snr_tib = np.zeros((len(subjects), len(np.arange(1, 21))))
                chan_med = []
                chan_tib = []

                for subject in subjects:
                    for cond_name in cond_names:
                        if cond_name == 'tibial':
                            trigger_name = 'Tibial - Stimulation'
                        elif cond_name == 'median':
                            trigger_name = 'Median - Stimulation'

                        subject_id = f'sub-{str(subject).zfill(3)}'

                        # Load prepared_data
                        input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/"
                        raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr1000_{cond_name}_withqrs.fif", preload=True)
                        raw_data = raw.get_data(picks=esg_chans)
                        input_path_DSS = f"/data/pt_02569/tmp_data/dss_heartart_py/{subject_id}/"

                        # Want the SNR for each selection of components removed from 1 to 20
                        for n in np.arange(1, 21):
                            # Load weights and spatial patterns
                            with open(f'{input_path_DSS}{cond_name}_heart_todss.pkl', 'rb') as f:
                                todss = pickle.load(f)
                            with open(f'{input_path_DSS}{cond_name}_heart_fromdss.pkl', 'rb') as f:
                                fromdss = pickle.load(f)
                            reconstructed_data = remove_comps_DSS(raw_data, len(esg_chans), n, todss, fromdss)
                            replace_kwargs = dict(
                                new_data=reconstructed_data.T  # n_channels, n_times
                            )
                            # Replace and save
                            clean_raw = raw.copy().apply_function(replace_data, picks=esg_chans, channel_wise=False,
                                                                  **replace_kwargs)
                            evoked = evoked_from_raw(clean_raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
                            snr, chan = calculate_SNR_evoked(evoked, cond_name, iv_baseline, reduced_window)

                            # Now have one snr for relevant channel in each subject + condition
                            if cond_name == 'median':
                                snr_med[subject - 1, n - 1] = snr
                                chan_med.append(chan)
                            elif cond_name == 'tibial':
                                snr_tib[subject - 1, n - 1] = snr
                                chan_tib.append(chan)

                savesnr.snr_med = snr_med
                savesnr.snr_tib = snr_tib
                savesnr.chan_med = chan_med
                savesnr.chan_tib = chan_tib
                dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]
                if reduced_window:
                    if reduced_epochs:
                        fn = f"/data/pt_02569/tmp_data/dss_heartart_py/snr_reduced_smallwin.h5"
                    else:
                        fn = f"/data/pt_02569/tmp_data/dss_heartart_py/snr_smallwin.h5"
                else:
                    if reduced_epochs:
                        fn = f"/data/pt_02569/tmp_data/dss_heartart_py/snr_reduced.h5"
                    else:
                        fn = f"/data/pt_02569/tmp_data/dss_heartart_py/snr.h5"

                with h5py.File(fn, "w") as outfile:
                    for keyword in dataset_keywords:
                        outfile.create_dataset(keyword, data=getattr(savesnr, keyword))

            # All other methods
            else:
                snr_med = np.zeros((len(subjects), 1))
                snr_tib = np.zeros((len(subjects), 1))
                chan_med = []
                chan_tib = []

                for subject in subjects:
                    for cond_name in cond_names:
                        if cond_name == 'tibial':
                            trigger_name = 'Tibial - Stimulation'
                            nerve = 2
                        elif cond_name == 'median':
                            trigger_name = 'Median - Stimulation'
                            nerve = 1

                        subject_id = f'sub-{str(subject).zfill(3)}'

                        # Get the right file path
                        if method == 'Prep':
                            file_path = "/data/pt_02569/tmp_data/prepared_py/"
                            file_name = f'noStimart_sr1000_{cond_name}_withqrs.fif'
                        elif method == 'PCA':
                            file_path = "/data/pt_02569/tmp_data/ecg_rm_py/"
                            file_name = f'data_clean_ecg_spinal_{cond_name}_withqrs.fif'
                        elif method == 'PCA Tukey':
                            file_path = "/data/pt_02569/tmp_data/ecg_rm_py_tukey/"
                            file_name = f'data_clean_ecg_spinal_{cond_name}_withqrs.fif'
                        elif method == 'ICA':
                            file_path = "/data/pt_02569/tmp_data/baseline_ica_py/"
                            if choose_limited:
                                file_name = f'clean_baseline_ica_auto_{cond_name}_lim.fif'
                            else:
                                file_name = f'clean_baseline_ica_auto_{cond_name}.fif'
                        elif method == 'ICA-Anterior':
                            file_path = "/data/pt_02569/tmp_data/baseline_ica_py/"
                            file_name = f"anterior_clean_baseline_ica_auto_{cond_name}.fif"
                        elif method == 'ICA-Separate':
                            file_path = "/data/pt_02569/tmp_data/baseline_ica_py/"
                            file_name = f"separated_clean_baseline_ica_auto_{cond_name}.fif"

                        input_path = file_path + subject_id + "/"
                        raw = mne.io.read_raw_fif(f"{input_path}{file_name}", preload=True)

                        if ant_ref:
                            # anterior reference
                            if nerve == 1:
                                raw = rereference_data(raw, 'AC')
                            elif nerve == 2:
                                raw = rereference_data(raw, 'AL')

                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
                        snr, chan = calculate_SNR_evoked(evoked, cond_name, iv_baseline, reduced_window)

                        # Now have one snr related to each subject and condition
                        if cond_name == 'median':
                            snr_med[subject - 1, 0] = snr
                            chan_med.append(chan)
                        elif cond_name == 'tibial':
                            snr_tib[subject - 1, 0] = snr
                            chan_tib.append(chan)

                savesnr.snr_med = snr_med
                savesnr.snr_tib = snr_tib
                savesnr.chan_med = chan_med
                savesnr.chan_tib = chan_tib
                dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]

                if reduced_window:
                    if reduced_epochs:
                        if ant_ref:
                            fn = f"/{file_path}snr_reducedtrials_ant_smallwin.h5"
                        else:
                            fn = f"{file_path}snr_reduced_smallwin.h5"
                    else:
                        if ant_ref:
                            fn = f"{file_path}snr_ant_smallwin.h5"
                        else:
                            fn = f"{file_path}snr_smallwin.h5"
                else:
                    if reduced_epochs:
                        if ant_ref:
                            fn = f"{file_path}snr_reduced_ant.h5"
                        else:
                            fn = f"{file_path}snr_reduced.h5"
                    else:
                        if ant_ref:
                            fn = f"{file_path}snr_ant.h5"
                        else:
                            fn = f"{file_path}snr.h5"
                if method == 'ICA' and choose_limited:
                    fn = f"{file_path}snr_lim.h5"
                if method == 'ICA-Anterior':
                    fn = f"{file_path}snr_anteriorICA.h5"
                if method == 'ICA-Separate':
                    fn = f"{file_path}snr_separateICA.h5"
                with h5py.File(fn, "w") as outfile:
                    for keyword in dataset_keywords:
                        outfile.create_dataset(keyword, data=getattr(savesnr, keyword))

    ############### Print to Screen numbers #################
    keywords = ['snr_med', 'snr_tib']
    input_paths = {'Prep': "/data/pt_02569/tmp_data/prepared_py/",
                   'PCA': "/data/pt_02569/tmp_data/ecg_rm_py/",
                   'PCA Tukey': "/data/pt_02569/tmp_data/ecg_rm_py_tukey/",
                   'ICA': "/data/pt_02569/tmp_data/baseline_ica_py/",
                   'ICA-Anterior': "/data/pt_02569/tmp_data/baseline_ica_py/",
                   'ICA-Separate': "/data/pt_02569/tmp_data/baseline_ica_py/",
                   'SSP': "/data/pt_02569/tmp_data/ssp_py/",
                   'CCA_heart': "/data/pt_02569/tmp_data/cca_heartart_py/",
                   'DSS_heart': "/data/pt_02569/tmp_data/dss_heartart_py/"}

    print("\n")
    for i in np.arange(0, len(input_paths)):
        name = list(input_paths.keys())[i]
        input_path = input_paths[name]
        if name == 'ICA' and choose_limited:
            fn = f"{input_path}snr_lim.h5"
        elif name == 'ICA-Anterior':
            fn = f"{input_path}snr_anteriorICA.h5"
        elif name == 'ICA-Separate':
            fn = f"{input_path}snr_separateICA.h5"
        else:
            fn = f"{input_path}snr.h5"
        # All have shape (24, 1) bar SSP which is (36, 16)
        with h5py.File(fn, "r") as infile:
            # Get the data
            snr_med = infile[keywords[0]][()]
            snr_tib = infile[keywords[1]][()]

        average_med = np.nanmean(snr_med, axis=0)
        average_tib = np.nanmean(snr_tib, axis=0)

        if name in ['SSP', 'CCA_heart', 'DSS_heart']:
            for n in np.arange(0, 20):  # 0, 16
                # print(f'SNR {name} Median {n + 5}: {average_med[n]:.4f}')
                # print(f'SNR {name} Tibial {n + 5}: {average_tib[n]:.4f}')
                print(f'SNR {name} Median {n + 1}: {average_med[n]:.4f}')
                print(f'SNR {name} Tibial {n + 1}: {average_tib[n]:.4f}')
        else:
            print(f'SNR {name} Median: {average_med[0]:.4f}')
            print(f'SNR {name} Tibial: {average_tib[0]:.4f}')


# File to compute epochs for each subject and each method

import mne
import os
import numpy as np
import pickle
from scipy.io import loadmat
from Metrics.SNR_functions import evoked_from_raw
from replace_data import replace_data
from remove_components_CCA import remove_comps_CCA
from remove_components_DSS import remove_comps_DSS
import matplotlib.pyplot as plt

if __name__ == '__main__':
    reduced_trials = False  # If true, generate images with fewer triggers
    longer_time = True
    subjects = np.arange(1, 37)   # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    # Want 200ms before R-peak and 400ms after R-peak
    # Baseline is the 100ms period before the artefact occurs
    iv_baseline = [-300 / 1000, -200 / 1000]
    iv_epoch = [-400 / 1000, 600 / 1000]

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    methods = [False, False, False, True, True]
    method_names = ['Prep', 'PCA', 'ICA', 'CCA-heart', 'DSS-heart']  # Will treat SSP separately since there are multiple
    SSP = False

    # To use mne grand_average method, need to generate a list of evoked potentials for each subject
    for i in np.arange(0, len(methods)):  # Methods Applied
        if methods[i]:  # Allows us to apply to only methods of interest
            method = method_names[i]

            for cond_name in cond_names:  # Conditions (median, tibial)
                evoked_list = []

                if cond_name == 'tibial':
                    trigger_name = 'qrs'
                    channel = ['L1']

                elif cond_name == 'median':
                    trigger_name = 'qrs'
                    channel = ['SC6']

                for subject in subjects:  # All subjects
                    subject_id = f'sub-{str(subject).zfill(3)}'

                    if method == 'Prep':
                        input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/"
                        raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif"
                                                  , preload=True)
                        events, event_ids = mne.events_from_annotations(raw)
                        event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                        epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                            baseline=tuple(iv_baseline))
                        epochs.save(fname=input_path+f'epochs_{cond_name}_qrs.fif', overwrite=True)

                    elif method == 'PCA':
                        input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/"
                        fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        events, event_ids = mne.events_from_annotations(raw)
                        event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                        epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                            baseline=tuple(iv_baseline))
                        epochs.save(fname=input_path + f'epochs_{cond_name}_qrs.fif', overwrite=True)

                    elif method == 'ICA':
                        input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + "/"
                        fname = f"clean_baseline_ica_auto_{cond_name}.fif"
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        events, event_ids = mne.events_from_annotations(raw)
                        event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                        epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                            baseline=tuple(iv_baseline))
                        epochs.save(fname=input_path + f'epochs_{cond_name}_qrs.fif', overwrite=True)

                    elif method == 'CCA-heart':
                        if cond_name == 'median':
                            n = 9
                        elif cond_name == 'tibial':
                            n = 6
                        # Load prepared_data
                        input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/"
                        raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr1000_{cond_name}_withqrs.fif", preload=True)
                        raw_data = raw.get_data(picks=esg_chans)
                        input_path_CCA = f"/data/pt_02569/tmp_data/cca_heartart_py/{subject_id}/"

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
                        events, event_ids = mne.events_from_annotations(clean_raw)
                        event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                        epochs = mne.Epochs(clean_raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                            baseline=tuple(iv_baseline))
                        epochs.save(fname=input_path_CCA + f'epochs_{cond_name}_{n}_qrs.fif', overwrite=True)

                    elif method == 'DSS-heart':
                        if cond_name == 'median':
                            n = 9
                        elif cond_name == 'tibial':
                            n = 7
                        # Load prepared_data
                        input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/"
                        raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr1000_{cond_name}_withqrs.fif", preload=True)
                        raw_data = raw.get_data(picks=esg_chans)
                        input_path_DSS = f"/data/pt_02569/tmp_data/dss_heartart_py/{subject_id}/"

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
                        events, event_ids = mne.events_from_annotations(clean_raw)
                        event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                        epochs = mne.Epochs(clean_raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                            baseline=tuple(iv_baseline))
                        epochs.save(fname=input_path_DSS + f'epochs_{cond_name}_{n}_qrs.fif', overwrite=True)

    # Now deal with SSP plots - Just doing 5 & 6 for now
    if SSP:
        for n in np.arange(5, 7):  # Methods Applied
            for cond_name in cond_names:  # Conditions (median, tibial)
                evoked_list = []

                if cond_name == 'tibial':
                    trigger_name = 'qrs'
                    channel = 'L1'

                elif cond_name == 'median':
                    trigger_name = 'qrs'
                    channel = 'SC6'

                for subject in subjects:  # All subjects
                    subject_id = f'sub-{str(subject).zfill(3)}'

                    input_path = f"/data/pt_02569/tmp_data/ssp_py/{subject_id}/{n} projections/"
                    raw = mne.io.read_raw_fif(f"{input_path}ssp_cleaned_{cond_name}.fif", preload=True)
                    events, event_ids = mne.events_from_annotations(raw)
                    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                        baseline=tuple(iv_baseline))
                    epochs.save(fname=input_path + f'epochs_{cond_name}_qrs.fif', overwrite=True)


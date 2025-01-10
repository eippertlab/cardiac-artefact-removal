# Run auto method of ICA on the prepared data as a comparison
# From MNE package - removes components related to ECG channel

import os
import mne
import matplotlib.pyplot as plt
from scipy.io import loadmat
from get_conditioninfo import *
from reref_data import rereference_data


def run_ica(subject, condition, srmr_nr, sampling_rate, choose_limited):

    # Set paths
    subject_id = f'sub-{str(subject).zfill(3)}'
    save_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id  + '/' # Saving to baseline_ica_py
    input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + '/'  # Taking prepared data
    os.makedirs(save_path, exist_ok=True)

    # Get the condition information based on the condition read in
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name

    # load ESG data
    fname = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif'
    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

    # ICA - data already filtered at import stage so fine for ICA now
    # Minus 1 as we don't use ECG channel
    ica = mne.preprocessing.ICA(n_components=len(raw.ch_names) - 1, max_iter='auto', random_state=97)
    ica.fit(raw)

    raw.load_data()

    # Automatically choose ICA components
    ica.exclude = []
    # find which ICs match the ECG pattern
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name='ECG')
    if choose_limited:
        ica.exclude = ecg_indices[0:6]
    else:
        ica.exclude = ecg_indices

    # Just for visualising
    # ica.plot_overlay(raw.copy().drop_channels(['ECG']), exclude=ecg_indices, picks='eeg')
    # print(ica.exclude)
    # ica.plot_scores(ecg_scores)

    # Apply the ica we got from the filtered data onto the unfiltered raw
    ica.apply(raw)

    # Save raw data
    if choose_limited:
        fname = 'clean_baseline_ica_auto_' + cond_name + '_lim.fif'
        raw.save(os.path.join(save_path, fname), fmt='double', overwrite=True)
    else:
        fname = 'clean_baseline_ica_auto_' + cond_name + '.fif'
        raw.save(os.path.join(save_path, fname), fmt='double', overwrite=True)

        # Save ecg indices
        with open(f'{save_path}ecg_indices_{cond_name}.txt', 'w') as file:
            file.write('\n'.join(str(ecg_index) for ecg_index in ecg_indices))


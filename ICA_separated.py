# Run auto method of ICA on the prepared data separately for lumbar and cervical patches

import os
import mne
import matplotlib.pyplot as plt
from scipy.io import loadmat
from get_conditioninfo import *
from reref_data import rereference_data
from get_esg_channels import get_esg_channels


def run_ica_separatepatches(subject, condition, srmr_nr, sampling_rate):
    # Set paths
    subject_id = f'sub-{str(subject).zfill(3)}'
    save_path = "/data/pt_02569//tmp_data/baseline_ica_py/" + subject_id  + '/' # Saving to baseline_ica_py
    input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + '/'  # Taking prepared data
    os.makedirs(save_path, exist_ok=True)

    # Get the condition information based on the condition read in
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    nerve = cond_info.nerve

    # Get respective channel types
    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()
    # load cleaned ESG data
    fname = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif'
    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

    # ICA
    # Perform ica separately for lumbar and cervical channels
    # Only process one patch for median or tibial
    if cond_name == 'median':
        channels = cervical_chans
    elif cond_name == 'tibial':
        channels = lumbar_chans

    channels.append('ECG')  # Needed as we use ECG to find bad channels

    raw_patch = raw.copy().pick_channels(channels)

    # Drop the channels from the patch we will process - we'll add these back later, we do this so for later processing
    # steps, even the separated ICA has the same number of channels, but only those in the relevant patch have been
    # ICA corrected
    raw = raw.drop_channels(channels)

    ica = mne.preprocessing.ICA(n_components=len(raw_patch.ch_names) - 1, max_iter='auto',
                                random_state=97)
    ica.fit(raw_patch)

    raw_patch.load_data()

    # Automatically choose ICA components
    ica.exclude = []
    # find which ICs match the ECG pattern
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw_patch, ch_name='ECG')
    ica.exclude = ecg_indices

    # Just for visualising
    # ica.plot_overlay(raw_patch.copy().drop_channels(['ECG']), exclude=ecg_indices, picks='eeg')
    # print(ica.exclude)
    # ica.plot_scores(ecg_scores)

    # Apply the ica
    ica.apply(raw_patch)

    raw.add_channels([raw_patch])  # Add back the channels I removed

    # Save data
    fname = 'separated_clean_baseline_ica_auto_' + cond_name + '.fif'
    raw.save(os.path.join(save_path, fname), fmt='double', overwrite=True)

    # Save ecg indices
    with open(f'{save_path}separated_ecg_indices_{cond_name}.txt', 'w') as file:
        file.write('\n'.join(str(ecg_index) for ecg_index in ecg_indices))


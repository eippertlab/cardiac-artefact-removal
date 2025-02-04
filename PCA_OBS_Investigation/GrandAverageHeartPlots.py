# Script to create plots of the grand averages evoked responses of the heartbeat across participants for each stimulation

import mne
import os
import numpy as np
from scipy.io import loadmat
from Metrics.SNR_functions import evoked_from_raw
import matplotlib.pyplot as plt


if __name__ == '__main__':
    reduced_trials = False  # Should always be false in this script
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
    # Want 200ms before and 400ms after the R-peak in our epoch - need baseline outside this
    iv_epoch = [-300 / 1000, 400 / 1000]

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    image_path = "/data/p_02569/Images/GrandAverageHeartPlots_PCAComCon_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

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

            input_path = "/data/pt_02569/tmp_data/ecg_rm_py_comcon/" + subject_id + "/"
            fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
            raw = mne.io.read_raw_fif(input_path + fname, preload=True)
            evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_trials)
            evoked.reorder_channels(esg_chans)
            evoked_list.append(evoked)


        averaged = mne.grand_average(evoked_list, interpolate_bads=False, drop_bads=False)
        relevant_channel = averaged.pick_channels([channel])
        plt.plot(relevant_channel.times, relevant_channel.data[0, :]*10**6, label='Evoked Grand Average')
        plt.ylabel('Amplitude [\u03BCV]')
        plt.xlabel('Time [s]')
        plt.xlim([-200/1000, 400/1000])
        plt.title(f"Method: PCA Complex Conjugate, Condition: {trigger_name}, Channel: {channel}")
        if reduced_trials:
            fname = f"PCAComplexConjugate_{trigger_name}_{channel}_reducedtrials.png"
        else:
            fname = f"PCAComplexConjugate_{trigger_name}_{channel}.png"
        plt.legend(loc='upper right')
        plt.savefig(image_path+fname)
        plt.clf()

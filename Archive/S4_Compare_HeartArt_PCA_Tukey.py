# Script to create plots of the grand averages heart evoked responses of the heartbeat across participants

import mne
import os
import numpy as np
from scipy.io import loadmat
from Metrics.SNR_functions import evoked_from_raw
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    reduced_trials = False  # Should always be false in this script
    if reduced_trials:
        raise RuntimeError('reduced_trials must be false')
    subjects = np.arange(1, 37)   # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    iv_epoch = [-400/1000, 400/1000]
    iv_baseline = [-400/1000, -300/1000]

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    image_path = "/data/p_02569/Images/PCA_Tukey_HeartPlots_Compare_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    for cond_name in cond_names:  # Conditions (median, tibial)
        evoked_list_pca = []
        evoked_list_pca_tukey = []

        if cond_name == 'tibial':
            trigger_name = 'qrs'
            channel = 'L1'
            full_name = 'Tibial Nerve Stimulation'

        elif cond_name == 'median':
            trigger_name = 'qrs'
            channel = 'SC6'
            full_name = 'Median Nerve Stimulation'

        for subject in subjects:  # All subjects
            subject_id = f'sub-{str(subject).zfill(3)}'

            # Get the PCA data
            input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + '/'
            fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
            raw = mne.io.read_raw_fif(input_path + fname, preload=True)
            evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_trials)
            evoked.reorder_channels(esg_chans)
            evoked_list_pca.append(evoked)

            # Get the PCA Tukey data
            input_path = "/data/pt_02569/tmp_data/ecg_rm_py_tukey/" + subject_id + '/'
            fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
            raw = mne.io.read_raw_fif(input_path + fname, preload=True)
            evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_trials)
            evoked.reorder_channels(esg_chans)
            evoked_list_pca_tukey.append(evoked)

        averaged_pca = mne.grand_average(evoked_list_pca, interpolate_bads=False, drop_bads=False)
        relevant_channel_pca = averaged_pca.pick_channels([channel])
        averaged_pca_tukey = mne.grand_average(evoked_list_pca_tukey, interpolate_bads=False, drop_bads=False)
        relevant_channel_pca_tukey = averaged_pca_tukey.pick_channels([channel])

        plt.figure()
        plt.plot(relevant_channel_pca.times, relevant_channel_pca.data[0, :] * 10 ** 6, label='PCA-OBS', color='blue')
        plt.plot(relevant_channel_pca_tukey.times, relevant_channel_pca_tukey.data[0, :] * 10 ** 6,
                 label='PCA-OBS Tukey', color='red')
        plt.ylabel('Amplitude (\u03BCV)')
        plt.xlabel('Time (s)')
        plt.xlim([-200 / 1000, 400 / 1000])
        plt.title(f"Heart Artefact Grand Average\n"
                  f"{full_name}")
        fname = f"HeartArtComp_{trigger_name}_{channel}.png"
        # plt.legend(loc='upper right')
        plt.savefig(image_path + fname)
        plt.savefig(image_path + fname + f".pdf", bbox_inches='tight', format="pdf")
        # plt.show()
        plt.clf()

# Want to look at how detrend affects the interpolation period

import mne
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from scipy.signal import detrend
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    pal = sns.color_palette(n_colors=4)
    subjects = np.arange(31, 32)  # 1 through 36 to access subject data
    cond_names = ['tibial', 'median']
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

    image_path = "/data/p_02569/Images/SingleSubject_PCA-OBS_Dataset1_ComplexConjugate/"
    os.makedirs(image_path, exist_ok=True)

    for cond_name in cond_names:  # Conditions (median, tibial)

        if cond_name == 'tibial':
            trigger_name = 'Tibial - Stimulation'
            channel = 'L1'

        elif cond_name == 'median':
            trigger_name = 'Median - Stimulation'
            channel = 'SC6'

        for subject in subjects:  # All subjects
            subject_id = f'sub-{str(subject).zfill(3)}'

            ################################################################################
            # Uncleaned
            ###############################################################################
            input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + '/'
            fname = f"epochs_{cond_name}.fif"
            epochs = mne.read_epochs(input_path + fname, preload=True)
            epo_data = np.squeeze(epochs.get_data(picks=channel), axis=1)
            plt.figure()
            plt.plot(epochs.times, epo_data.T.mean(axis=1))
            plt.axvline(-0.007, color='red')
            plt.axvline(0.007, color='red')

            ################################################################################
            # PCA-OBS
            ###############################################################################
            input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + '/'
            fname = f"epochs_{cond_name}.fif"
            epochs = mne.read_epochs(input_path + fname, preload=True)
            epo_data = np.squeeze(epochs.get_data(picks=channel), axis=1)
            plt.figure()
            plt.plot(epochs.times, epo_data.T.mean(axis=1))
            plt.axvline(-0.007, color='red')
            plt.axvline(0.007, color='red')
            plt.show()
            exit()
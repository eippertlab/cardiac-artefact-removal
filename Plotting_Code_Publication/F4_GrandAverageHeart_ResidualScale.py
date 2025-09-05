# Script to create plots of the grand averages evoked responses of the heartbeat across participants for each stimulation

import mne
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    pal = sns.color_palette(n_colors=10)
    subjects = np.arange(1, 37)   # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    image_path = "/data/p_02569/Images_ResidualArtefact/GrandAverageHeartYY_CleanedOnly_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    for cond_name in cond_names:  # Conditions (median, tibial)
        evoked_list_pca = []
        evoked_list_ica = []
        evoked_list_ssp5 = []
        evoked_list_cca = []
        evoked_list_dss = []

        if cond_name == 'tibial':
            trigger_name = 'qrs'
            channel = 'L1'

        elif cond_name == 'median':
            trigger_name = 'qrs'
            channel = 'SC6'

        for subject in subjects:  # All subjects
            subject_id = f'sub-{str(subject).zfill(3)}'

            input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + '/'
            fname = f"epochs_{cond_name}_qrs.fif"
            epochs = mne.read_epochs(input_path + fname, preload=True)
            evoked = epochs.average()
            evoked.reorder_channels(esg_chans)
            evoked_list_pca.append(evoked)

            input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + '/'
            fname = f"epochs_{cond_name}_qrs.fif"
            epochs = mne.read_epochs(input_path + fname, preload=True)
            evoked = epochs.average()
            evoked.reorder_channels(esg_chans)
            evoked_list_ica.append(evoked)

            input_path = f"/data/pt_02569/tmp_data/ssp_py/{subject_id}/5 projections/"
            fname = f"epochs_{cond_name}_qrs.fif"
            epochs = mne.read_epochs(input_path + fname, preload=True)
            evoked = epochs.average()
            evoked.reorder_channels(esg_chans)
            evoked_list_ssp5.append(evoked)

            input_path = f"/data/pt_02569/tmp_data/cca_heartart_py/{subject_id}/"
            if cond_name == 'median':
                n = 9
            elif cond_name == 'tibial':
                n = 6
            fname = f"epochs_{cond_name}_{n}_qrs.fif"
            epochs = mne.read_epochs(input_path + fname, preload=True)
            evoked = epochs.average()
            evoked.reorder_channels(esg_chans)
            evoked_list_cca.append(evoked)

            input_path = f"/data/pt_02569/tmp_data/dss_heartart_py/{subject_id}/"
            if cond_name == 'median':
                n = 9
            elif cond_name == 'tibial':
                n = 7
            fname = f"epochs_{cond_name}_{n}_qrs.fif"
            epochs = mne.read_epochs(input_path + fname, preload=True)
            evoked = epochs.average()
            evoked.reorder_channels(esg_chans)
            evoked_list_dss.append(evoked)

        averaged_pca = mne.grand_average(evoked_list_pca, interpolate_bads=False, drop_bads=False)
        relevant_channel_pca = averaged_pca.pick_channels([channel])

        averaged_ica = mne.grand_average(evoked_list_ica, interpolate_bads=False, drop_bads=False)
        relevant_channel_ica = averaged_ica.pick_channels([channel])

        averaged_ssp5 = mne.grand_average(evoked_list_ssp5, interpolate_bads=False, drop_bads=False)
        relevant_channel_ssp5 = averaged_ssp5.pick_channels([channel])

        averaged_cca = mne.grand_average(evoked_list_cca, interpolate_bads=False, drop_bads=False)
        relevant_channel_cca = averaged_cca.pick_channels([channel])

        averaged_dss = mne.grand_average(evoked_list_dss, interpolate_bads=False, drop_bads=False)
        relevant_channel_dss = averaged_dss.pick_channels([channel])

        # Want 1 row, 3 column subplot
        fig, [ax1, ax2, ax3, ax4, ax5] = plt.subplots(1, 5, figsize=[18, 6])

        # PCA
        ax1.plot(relevant_channel_pca.times, relevant_channel_pca.data[0, :]*10**6, label='PCA-OBS',
                 color=pal[1])
        ax1.set_ylabel('Artefact Amplitude (\u03BCV)')
        ax1.set_xlabel('Time (s)')
        if cond_name == 'median':
            ax1.set_title('PCA-OBS')
        # ax1.spines['left'].set_color('blue')
        # ax1.tick_params(axis='y', colors='blue')

        # ICA
        ax2.plot(relevant_channel_ica.times, relevant_channel_ica.data[0, :] * 10 ** 6, label='ICA',
                 color=pal[2])
        ax2.set_xlabel('Time (s)')
        if cond_name == 'median':
            ax2.set_title('ICA')
        # ax2.spines['left'].set_color('orange')
        # ax2.tick_params(axis='y', colors='orange')

        # SSP5
        ax3.plot(relevant_channel_ssp5.times, relevant_channel_ssp5.data[0, :] * 10 ** 6, label='SSP',
                 color=pal[3])
        ax3.set_xlabel('Time (s)')
        if cond_name == 'median':
            ax3.set_title('SSP')
        # ax30.spines['right'].set_color('blue')

        # CCA
        ax4.plot(relevant_channel_cca.times, relevant_channel_cca.data[0, :] * 10 ** 6, label='CCA-cardiac',
                 color=pal[4])
        ax4.set_xlabel('Time (s)')
        if cond_name == 'median':
            ax4.set_title('CCA-cardiac')
        # ax30.spines['right'].set_color('blue')

        # DSS
        ax5.plot(relevant_channel_dss.times, relevant_channel_dss.data[0, :] * 10 ** 6, label='DSS-cardiac',
                 color=pal[9])
        ax5.set_xlabel('Time (s)')
        if cond_name == 'median':
            ax5.set_title('DSS-cardiac')

        ax1.set_xlim([-200 / 1000, 400 / 1000])
        ax2.set_xlim([-200 / 1000, 400 / 1000])
        ax3.set_xlim([-200 / 1000, 400 / 1000])
        ax4.set_xlim([-200 / 1000, 400 / 1000])
        ax5.set_xlim([-200 / 1000, 400 / 1000])

        fname = f"CardiacTimeCourse__{channel}.png"

        plt.tight_layout()
        plt.savefig(image_path+fname)
        plt.savefig(image_path + fname + '.pdf', bbox_inches='tight', format="pdf")



# Script to create plots of the single subject evoked responses of the heartbeat
# for each stimulation and cleaning method

import mne
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_axes_aligner import align
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    pal = sns.color_palette(n_colors=10)
    # subjects = np.arange(1, 37)   # 1 through 36 to access subject data
    subjects = [10]  # , 6, 20, 31
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

    image_path = "/data/p_02569/Images/SingleSubjectHeartYY_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    for cond_name in cond_names:  # Conditions (median, tibial)
        if cond_name == 'tibial':
            trigger_name = 'qrs'
            channel = 'L1'

        elif cond_name == 'median':
            trigger_name = 'qrs'
            channel = 'SC6'

        for subject in subjects:  # All subjects
            subject_id = f'sub-{str(subject).zfill(3)}'

            input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + '/'
            fname = f"epochs_{cond_name}_qrs.fif"
            epochs = mne.read_epochs(input_path+fname, preload=True)
            evoked_prep = epochs.average()
            evoked_prep.reorder_channels(esg_chans)

            input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + '/'
            fname = f"epochs_{cond_name}_qrs.fif"
            epochs = mne.read_epochs(input_path + fname, preload=True)
            evoked_pca = epochs.average()
            evoked_pca.reorder_channels(esg_chans)

            input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + '/'
            fname = f"epochs_{cond_name}_qrs.fif"
            epochs = mne.read_epochs(input_path + fname, preload=True)
            evoked_ica = epochs.average()
            evoked_ica.reorder_channels(esg_chans)

            input_path = f"/data/pt_02569/tmp_data/ssp_py/{subject_id}/5 projections/"
            fname = f"epochs_{cond_name}_qrs.fif"
            epochs = mne.read_epochs(input_path + fname, preload=True)
            evoked_ssp5 = epochs.average()
            evoked_ssp5.reorder_channels(esg_chans)

            input_path = f"/data/pt_02569/tmp_data/cca_heartart_py/{subject_id}/"
            if cond_name == 'median':
                n = 9
            elif cond_name == 'tibial':
                n = 6
            fname = f"epochs_{cond_name}_{n}_qrs.fif"
            epochs = mne.read_epochs(input_path + fname, preload=True)
            evoked_cca = epochs.average()
            evoked_cca.reorder_channels(esg_chans)

            input_path = f"/data/pt_02569/tmp_data/dss_heartart_py/{subject_id}/"
            if cond_name == 'median':
                n = 9
            elif cond_name == 'tibial':
                n = 7
            fname = f"epochs_{cond_name}_{n}_qrs.fif"
            epochs = mne.read_epochs(input_path + fname, preload=True)
            evoked_dss = epochs.average()
            evoked_dss.reorder_channels(esg_chans)

            relevant_channel_prep = evoked_prep.pick_channels([channel])
            relevant_channel_pca = evoked_pca.pick_channels([channel])
            relevant_channel_ica = evoked_ica.pick_channels([channel])
            relevant_channel_ssp5 = evoked_ssp5.pick_channels([channel])
            relevant_channel_cca = evoked_cca.pick_channels([channel])
            relevant_channel_dss = evoked_dss.pick_channels([channel])

            # Want 1 row, 3 column subplot
            # Want left y-axis to relate to cleaned heart artefact
            # Want right y-axis to relate to uncleaned heart artefact
            fig, [ax1, ax2, ax3, ax4, ax5] = plt.subplots(1, 5, figsize=[18, 6])
            ax10 = ax1.twinx()
            ax20 = ax2.twinx()
            ax30 = ax3.twinx()
            ax40 = ax4.twinx()
            ax50 = ax5.twinx()
            ax1.get_shared_y_axes().join(ax1, ax2, ax3, ax4, ax5)
            ax10.get_shared_y_axes().join(ax10, ax20, ax30, ax40, ax50)

            # PCA
            ax1.plot(relevant_channel_pca.times, relevant_channel_pca.data[0, :]*10**6, label='PCA-OBS',
                     color=pal[1])
            ax1.set_ylabel('Cleaned Artefact Amplitude (\u03BCV)')
            ax1.set_xlabel('Time (s)')
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            if cond_name == 'median':
                ax1.set_title('PCA-OBS')
            ax10.plot(relevant_channel_prep.times, relevant_channel_prep.data[0, :]*10**6, label='Uncleaned',
                      linewidth=0.5, linestyle='dashed', color='blue')
            ax10.set_yticklabels([])

            # ICA
            ax2.plot(relevant_channel_ica.times, relevant_channel_ica.data[0, :] * 10 ** 6, label='ICA',
                     color=pal[2])
            ax2.set_xlabel('Time (s)')
            if cond_name == 'median':
                ax2.set_title('ICA')
            ax2.set_yticklabels([])
            ax20.plot(relevant_channel_prep.times, relevant_channel_prep.data[0, :] * 10 ** 6, label='Uncleaned',
                      linewidth=0.5, linestyle='dashed', color='blue')
            ax20.set_yticklabels([])

            # SSP5
            ax3.plot(relevant_channel_ssp5.times, relevant_channel_ssp5.data[0, :] * 10 ** 6, label='SSP',
                     color=pal[3])
            ax3.set_xlabel('Time (s)')
            if cond_name == 'median':
                ax3.set_title('SSP')
            ax3.set_yticklabels([])
            ax30.plot(relevant_channel_prep.times, relevant_channel_prep.data[0, :] * 10 ** 6, label='Uncleaned',
                      linewidth=0.5, linestyle='dashed', color='blue')
            ax30.set_yticklabels([])

            # CCA
            ax4.plot(relevant_channel_cca.times, relevant_channel_cca.data[0, :] * 10 ** 6, label='CCA',
                     color=pal[4])
            ax4.set_xlabel('Time (s)')
            if cond_name == 'median':
                ax4.set_title('CCA-cardiac')
            ax4.set_yticklabels([])
            ax40.plot(relevant_channel_prep.times, relevant_channel_prep.data[0, :] * 10 ** 6, label='Uncleaned',
                      linewidth=0.5, linestyle='dashed', color='blue')
            ax40.set_yticklabels([])

            # DSS
            ax5.plot(relevant_channel_dss.times, relevant_channel_dss.data[0, :] * 10 ** 6, label='DSS',
                     color=pal[9])
            ax5.set_xlabel('Time (s)')
            if cond_name == 'median':
                ax5.set_title('DSS-cardiac')
            ax5.set_yticklabels([])
            ax50.plot(relevant_channel_prep.times, relevant_channel_prep.data[0, :] * 10 ** 6, label='Uncleaned',
                      linewidth=0.5, linestyle='dashed', color='blue')
            ax50.set_ylabel('Uncleaned Artefact Amplitude (\u03BCV)')
            ax50.yaxis.label.set_color('blue')
            ax50.tick_params(axis='y', colors='blue')

            ax1.set_xlim([-200/1000, 400/1000])
            ax2.set_xlim([-200/1000, 400/1000])
            ax3.set_xlim([-200/1000, 400/1000])
            ax4.set_xlim([-200/1000, 400/1000])
            ax5.set_xlim([-200/1000, 400/1000])

            fname = f"{subject_id}_CardiacTimeCourse_{channel}.png"

            # Align y-axes
            if cond_name == 'median':
                align.yaxes(ax1, 0, ax10, 0, 0.65)
            else:
                align.yaxes(ax1, 0, ax10, 0, 0.35)
            plt.tight_layout()
            plt.savefig(image_path+fname)
            plt.savefig(image_path + fname + '.pdf', bbox_inches='tight', format="pdf")


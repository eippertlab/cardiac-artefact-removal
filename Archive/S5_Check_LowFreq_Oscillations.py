# To see if oscillations seen in uncleaned data can be attributed to the heart artefact
# Take 2 subjects - subject 6
# Get fundamental freq of heart beat
# Check power at these frequencies for uncleaned, after PCA_OBS, ICA and SSP6
# Subtract cleaned from uncleaned to see how methods affect frequency content

import mne
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
from Metrics.inps_yasa import get_harmonics
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    pal = sns.color_palette(n_colors=4)
    subjects = [6, 20]
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]
    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000

    esg_chans = ['S35', 'S240', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S40', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S40', 'S340', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S140', 'S33', 'S3', 'AL', 'L40', 'S6',
                 'S23']

    image_path = "/data/p_02569/Images/LowFreqOscillations_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    ############################################################################################
    # Single Subject Level
    ############################################################################################
    for subject in subjects:
        subject_id = f'sub-{str(subject).zfill(3)}'

        for cond_name in cond_names:
            if cond_name == 'tibial':
                trigger_name = 'Tibial - Stimulation'
                channel = ['L1']

            elif cond_name == 'median':
                trigger_name = 'Median - Stimulation'
                channel = ['SC6']

            ##########################################################################################
            # Time Course and corresponding Power Spectrum
            ###########################################################################################
            count = 0
            fig, axes = plt.subplots(4, 2, figsize=[6, 18])
            axes_unflat = axes
            axes = axes.flatten()

            ##########################################################################
            # Uncleaned Data
            ##########################################################################
            input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + '/'
            raw_prep = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif"
                                           , preload=True)
            raw_prep = raw_prep.pick_channels(channel)
            events, event_ids = mne.events_from_annotations(raw_prep)
            event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
            epochs_prep = mne.Epochs(raw_prep, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                    baseline=tuple(iv_baseline))

            heart_frequencies = get_harmonics(raw_prep, trigger='qrs', sample_rate=1000)
            evoked = epochs_prep.average()

            # Plot
            axes[count].plot(evoked.times, np.mean(evoked.data[:, :], axis=0) * 10 ** 6, color=pal[0])
            axes[count].set_ylabel('SEP Amplitude (\u03BCV)')
            axes[count].set_xlabel('Time (s)')
            axes[count].set_title('Time Courses')
            count += 1
            fourier_transform = np.fft.rfft(raw_prep.get_data().reshape(-1))
            abs_fourier_transform = np.abs(fourier_transform)
            power_spectrum_prep = np.square(abs_fourier_transform)
            frequency_prep = np.linspace(0, sampling_rate / 2, len(power_spectrum_prep))
            axes[count].plot(frequency_prep, power_spectrum_prep, color=pal[0])
            axes[count].set_xlim([0, 60])
            axes[count].set_ylabel('Power (\u03BCV\u00b2/Hz)')
            axes[count].set_xlabel('Frequency (Hz)')
            axes[count].set_title('Power Spectra')
            count += 1

            ###################################################################
            # PCA_OBS
            ###################################################################
            input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + '/'
            fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
            raw_pca = mne.io.read_raw_fif(input_path + fname, preload=True)
            raw_pca = raw_pca.pick_channels(channel)
            events, event_ids = mne.events_from_annotations(raw_pca)
            event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
            epochs_pca = mne.Epochs(raw_pca, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                     baseline=tuple(iv_baseline))

            # Plot
            axes[count].plot(epochs_pca.average().times, np.mean(epochs_pca.average().data[:, :], axis=0) * 10 ** 6,
                             color=pal[1])
            axes[count].set_ylabel('SEP Amplitude (\u03BCV)')
            axes[count].set_xlabel('Time (s)')
            count += 1
            fourier_transform = np.fft.rfft(raw_pca.get_data().reshape(-1))
            abs_fourier_transform = np.abs(fourier_transform)
            power_spectrum_pca = np.square(abs_fourier_transform)
            frequency_pca = np.linspace(0, sampling_rate / 2, len(power_spectrum_pca))
            axes[count].plot(frequency_pca, power_spectrum_pca, color=pal[1])
            axes[count].set_xlim([0, 60])
            axes[count].set_ylabel('Power (\u03BCV\u00b2/Hz)')
            axes[count].set_xlabel('Frequency (Hz)')
            count += 1

            ##########################################################################
            # SSP6
            ##########################################################################
            input_path = f"/data/pt_02569/tmp_data/ssp_py/{subject_id}/6 projections/"
            raw_ssp6 = mne.io.read_raw_fif(f"{input_path}ssp_cleaned_{cond_name}.fif", preload=True)
            # raw_ssp6 = raw_ssp6.pick_channels(channel)
            events, event_ids = mne.events_from_annotations(raw_ssp6)
            event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
            epochs_ssp6 = mne.Epochs(raw_ssp6, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                     baseline=tuple(iv_baseline), preload=True)

            # Plot
            axes[count].plot(epochs_ssp6.pick_channels(channel).average().times,
                             np.mean(epochs_ssp6.pick_channels(channel).average().data[:, :], axis=0) * 10 ** 6,
                             color=pal[3])
            axes[count].set_ylabel('SEP Amplitude (\u03BCV)')
            axes[count].set_xlabel('Time (s)')
            count += 1
            fourier_transform = np.fft.rfft(raw_ssp6.pick_channels(channel).get_data().reshape(-1))
            abs_fourier_transform = np.abs(fourier_transform)
            power_spectrum_ssp6 = np.square(abs_fourier_transform)
            frequency_ssp6 = np.linspace(0, sampling_rate / 2, len(power_spectrum_ssp6))
            axes[count].plot(frequency_ssp6, power_spectrum_ssp6, color=pal[3])
            axes[count].set_xlim([0, 60])
            axes[count].set_ylabel('Power (\u03BCV\u00b2/Hz)')
            axes[count].set_xlabel('Frequency (Hz)')
            count += 1

            ############################################################################################
            # Difference in Power Spectra
            ############################################################################################
            diff_pca = power_spectrum_pca-power_spectrum_prep
            axes[count].plot(frequency_prep, diff_pca, color='black')
            axes[count].set_title('PCA - Uncleaned')
            axes[count].set_xlim([0, 60])
            axes[count].set_ylabel('Power (\u03BCV\u00b2/Hz)')
            axes[count].set_xlabel('Frequency (Hz)')
            count += 1
            diff_ssp = power_spectrum_ssp6 - power_spectrum_prep
            axes[count].plot(frequency_prep, diff_ssp, color='black')
            axes[count].set_title('SSP - Uncleaned')
            axes[count].set_xlim([0, 60])
            axes[count].set_ylabel('Power (\u03BCV\u00b2/Hz)')
            axes[count].set_xlabel('Frequency (Hz)')

            ###########
            # Format row and headers
            ##########
            rows = ['Uncleaned', 'PCA-OBS', 'SSP', 'Power Spectra']
            pad = 5
            for ax, row in zip(axes_unflat[:, 0], rows):
                if row in rows[:-1]:
                    ax.set_xlim([-0.1, 0.3])
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                            xycoords=ax.yaxis.label, textcoords='offset points',
                            size='large', ha='right', va='center', rotation=90)

            plt.suptitle(f'Subject {subject}', x=0.55)
            plt.tight_layout()
            plt.subplots_adjust(left=0.17, top=0.95)
            plt.savefig(image_path+f"{subject_id}_{cond_name}.png")
            plt.savefig(image_path+f"{subject_id}_{cond_name}.pdf", bbox_inches='tight', format="pdf")
            # plt.show()


    # #############################################################################################
    # # Average across all subjects
    # ############################################################################################
    # for cond_name in cond_names:  # Conditions (median, tibial)
    #     evoked_list_prep = []
    #     evoked_list_pca = []
    #     evoked_list_ica = []
    #     evoked_list_ssp6 = []
    #
    #     if cond_name == 'tibial':
    #         trigger_name = 'Tibial - Stimulation'
    #         channel = 'L1'
    #
    #     elif cond_name == 'median':
    #         trigger_name = 'Median - Stimulation'
    #         channel = 'SC6'
    #
    #     for subject in subjects:  # All subjects
    #         subject_id = f'sub-{str(subject).zfill(3)}'
    #
    #         ################################################################################
    #         # Uncleaned
    #         ###############################################################################
    #         input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + '/'
    #         fname = f"epochs_{cond_name}.fif"
    #         epochs = mne.read_epochs(input_path+fname, preload=True)
    #         evoked = epochs.average()
    #         evoked.reorder_channels(esg_chans)
    #         evoked_list_prep.append(evoked)
    #
    #         ##############################################################################
    #         # PCA_OBS
    #         ##############################################################################
    #         input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + '/'
    #         fname = f"epochs_{cond_name}.fif"
    #         epochs = mne.read_epochs(input_path + fname, preload=True)
    #         evoked = epochs.average()
    #         evoked.reorder_channels(esg_chans)
    #         evoked_list_pca.append(evoked)
    #
    #         ##############################################################################
    #         # ICA
    #         ##############################################################################
    #         input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + '/'
    #         fname = f"epochs_{cond_name}.fif"
    #         epochs = mne.read_epochs(input_path + fname, preload=True)
    #         evoked = epochs.average()
    #         evoked.reorder_channels(esg_chans)
    #         evoked_list_ica.append(evoked)
    #
    #         #############################################################################
    #         # SSP 6
    #         #############################################################################
    #         input_path = f"/data/pt_02569/tmp_data/ssp_py/{subject_id}/6 projections/"
    #         fname = f"epochs_{cond_name}.fif"
    #         epochs = mne.read_epochs(input_path + fname, preload=True)
    #         evoked = epochs.average()
    #         evoked.reorder_channels(esg_chans)
    #         evoked_list_ssp6.append(evoked)
    #

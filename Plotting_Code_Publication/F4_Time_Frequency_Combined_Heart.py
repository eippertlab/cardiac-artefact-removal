# Script to create plots of the grand averages evoked responses of the heartbeat across participants for each stimulation

import mne
import os
import numpy as np
from Metrics.SNR_functions import evoked_from_raw
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_axes_aligner import align
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    subjects = np.arange(1, 37)   # 1 through 36 to access subject data
    # subjects = [1]
    freqs = np.arange(5., 250., 3.)
    # freqs = np.arange(5., 1000., 3.)
    fmin, fmax = freqs[[0, -1]]
    cond_names = ['median', 'tibial']
    sampling_rate = 1000
    iv_baseline = [-300 / 1000, -200 / 1000]
    iv_epoch = [-300 / 1000, 450 / 1000]

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    image_path = "/data/p_02569/Images/TFR_Heart_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    for cond_name in cond_names:  # Conditions (median, tibial)
        evoked_list_prep = []
        evoked_list_pca = []
        evoked_list_ica = []
        evoked_list_ssp6 = []

        if cond_name == 'tibial':
            trigger_name = 'qrs'
            channel = ['L1']

        elif cond_name == 'median':
            trigger_name = 'qrs'
            channel = ['SC6']

        for subject in subjects:  # All subjects
            subject_id = f'sub-{str(subject).zfill(3)}'

            input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + '/'
            fname = f"epochs_{cond_name}_qrs.fif"
            epochs = mne.read_epochs(input_path+fname, preload=True)
            evoked = epochs.average()
            evoked.reorder_channels(esg_chans)
            evoked = evoked.pick_channels(channel).crop(tmin=iv_epoch[0], tmax=iv_epoch[1])
            # fname = f"noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif"
            # raw = mne.io.read_raw_fif(input_path + fname, preload=True)
            # evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
            power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
            evoked_list_prep.append(power)

            input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + '/'
            fname = f"epochs_{cond_name}_qrs.fif"
            epochs = mne.read_epochs(input_path + fname, preload=True)
            evoked = epochs.average()
            # fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
            # raw = mne.io.read_raw_fif(input_path + fname, preload=True)
            # evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
            evoked.reorder_channels(esg_chans)
            evoked = evoked.pick_channels(channel).crop(tmin=iv_epoch[0], tmax=iv_epoch[1])
            power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
            evoked_list_pca.append(power)

            input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + '/'
            fname = f"epochs_{cond_name}_qrs.fif"
            epochs = mne.read_epochs(input_path + fname, preload=True)
            evoked = epochs.average()
            # fname = f"clean_baseline_ica_auto_{cond_name}.fif"
            # raw = mne.io.read_raw_fif(input_path + fname, preload=True)
            # evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
            evoked.reorder_channels(esg_chans)
            evoked = evoked.pick_channels(channel).crop(tmin=iv_epoch[0], tmax=iv_epoch[1])
            power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
            evoked_list_ica.append(power)

            input_path = f"/data/pt_02569/tmp_data/ssp_py/{subject_id}/6 projections/"
            fname = f"epochs_{cond_name}_qrs.fif"
            epochs = mne.read_epochs(input_path + fname, preload=True)
            evoked = epochs.average()
            # fname = f"ssp_cleaned_{cond_name}.fif"
            # raw = mne.io.read_raw_fif(input_path + fname, preload=True)
            # evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
            evoked.reorder_channels(esg_chans)
            evoked = evoked.pick_channels(channel).crop(tmin=iv_epoch[0], tmax=iv_epoch[1])
            power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
            evoked_list_ssp6.append(power)

        averaged_prep = mne.grand_average(evoked_list_prep, interpolate_bads=False, drop_bads=False)
        averaged_pca = mne.grand_average(evoked_list_pca, interpolate_bads=False, drop_bads=False)
        averaged_ica = mne.grand_average(evoked_list_ica, interpolate_bads=False, drop_bads=False)
        averaged_ssp6 = mne.grand_average(evoked_list_ssp6, interpolate_bads=False, drop_bads=False)

        tmin = -0.2
        tmax = 0.4
        vmin = -400
        vmax = -240
        # if cond_name == 'tibial':
        #     vmin = -400
        #     vmax = -240
        # else:
        #     vmin = -400
        #     vmax = -250
        # fig, ax = plt.subplots(1, 5, figsize=[18, 6], gridspec_kw={"width_ratios": [10, 10, 10, 10, 1]})
        fig, ax = plt.subplots(1, 4, figsize=[24, 6], constrained_layout=True)
        averaged_prep.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
                          axes=ax[0], show=False, colorbar=False, dB=True,
                          tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax)
        averaged_pca.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
                          axes=ax[1], show=False, colorbar=False, dB=True,
                          tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax)
        averaged_ica.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
                          axes=ax[2], show=False, colorbar=False, dB=True,
                          tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax)
        averaged_ssp6.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
                          axes=ax[3], show=False, colorbar=False, dB=True,
                          tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax)
        # Add axis for colorbar display
        fig.subplots_adjust(bottom=0.25)
        cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05])
        cb = fig.colorbar(ax[0].images[-1], cax=cbar_ax, shrink=0.6, orientation='horizontal')

        # cb = fig.colorbar(ax[0].images[-1], cax=ax[-1])  # Take colourbar from first image and puts it in last axis
        cb.set_label('Power (dB)')
        if cond_name == 'median':
            ax[0].set_title('Uncleaned')
            ax[1].set_title('PCA-OBS')
            ax[2].set_title('ICA')
            ax[3].set_title('SSP')
        ax[1].set_yticklabels([])
        ax[1].set_ylabel(None)
        ax[2].set_yticklabels([])
        ax[2].set_ylabel(None)
        ax[3].set_yticklabels([])
        ax[3].set_ylabel(None)
        # if cond_name == 'median':
        #     plt.suptitle(f"Time-Frequency Representation of the Cardiac Artefact\n"
        #                  f"Cervical Spinal Cord")
        # else:
        #     plt.suptitle(f"Time-Frequency Representation of the Cardiac Artefact\n"
        #                  f"Lumbar Spinal Cord")
        fname = f"Heart_{cond_name}"
        # fname = f"Heart_{cond_name}_HigherFreq"
        plt.savefig(image_path+fname+'.png')
        plt.savefig(image_path + fname + '.pdf', bbox_inches='tight', format="pdf")
        plt.show()

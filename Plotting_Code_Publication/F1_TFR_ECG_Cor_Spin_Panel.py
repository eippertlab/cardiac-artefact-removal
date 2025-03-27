# Script to plot the time-frequency decomposition in dB about the heartbeat
# TFRs of the heart artefact in ECG, Spinal and Cortical Channels
# Plot as a panel, with the ECG and Spinal time courses in a bottom panel

import mne
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from Metrics.SNR_functions import evoked_from_raw
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    freqs = np.arange(5., 250., 3.)
    fmin, fmax = freqs[[0, -1]]
    subjects = np.arange(1, 37)
    cond_names = ['tibial', 'median']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    # Want 200ms before R-peak and 400ms after R-peak
    # Baseline is the 100ms period before the artefact occurs
    iv_baseline = [-300 / 1000, -200 / 1000]
    # Want 200ms before and 500ms after the R-peak in our epoch - need baseline outside this
    iv_epoch = [-300 / 1000, 500 / 1000]

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    image_path = "/data/p_02569/Images/TimeFrequencyPlots_HeartPanel_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    channel_types = ['ECG', 'Spinal', 'Cortical']
    # dict = {}

    # To use mne grand_average method, need to generate a list of evoked potentials for each subject
    for cond_name in cond_names:
        fig = plt.figure(figsize=(21, 9))
        gs = fig.add_gridspec(2, 4, width_ratios=[10, 10, 10, 0.35], height_ratios=[2, 1])
        ax_ecg = fig.add_subplot(gs[0, 0])
        ax_spinal = fig.add_subplot(gs[0, 1])
        ax_cortical = fig.add_subplot(gs[0, 2])
        cbar_ax = fig.add_subplot(gs[0:2, 3])
        ax_time = fig.add_subplot(gs[1, 0:3])

        for channel_type in channel_types:
            # Conditions (median, tibial)
            evoked_list = []  # Holds all the TFRs
            time_list = []  # Holds the actual evoked for each subject

            if cond_name == 'tibial':
                full_name = 'Tibial Nerve Stimulation'
                trigger_name = 'qrs'
                if channel_type == 'ECG':
                    channel = ['ECG']
                    ax = ax_ecg
                elif channel_type == 'Spinal':
                    channel = ['L1']
                    ax = ax_spinal
                elif channel_type == 'Cortical':
                    channel = ['Cz']
                    ax = ax_cortical

            elif cond_name == 'median':
                full_name = 'Median Nerve Stimulation'
                trigger_name = 'qrs'
                if channel_type == 'ECG':
                    channel = ['ECG']
                    ax = ax_ecg
                elif channel_type == 'Spinal':
                    channel = ['SC6']
                    ax = ax_spinal
                elif channel_type == 'Cortical':
                    channel = ['CP4']
                    ax = ax_cortical

            for subject in subjects:  # All subjects
                subject_id = f'sub-{str(subject).zfill(3)}'

                if channel_type == 'ECG' or channel_type == 'Cortical':
                    input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + '/'
                    raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs_eeg.fif",
                                              preload=True)
                elif channel_type == 'Spinal':
                    input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + '/'
                    raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif",
                                              preload=True)

                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                evoked = evoked.pick_channels(channel)
                if channel_type == 'ECG':
                    evoked = evoked.set_channel_types({'ECG': 'eeg'})  # Needed to do transform
                time_list.append(evoked)
                power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
                # evoked_list.append(evoked)
                evoked_list.append(power)

            averaged = mne.grand_average(evoked_list, interpolate_bads=False, drop_bads=False)
            averaged_time = mne.grand_average(time_list, interpolate_bads=False, drop_bads=False)

            tmin = -0.2
            tmax = 0.4
            vmin = -400
            # vmax = -175
            vmax = -140

            # power = mne.time_frequency.tfr_stockwell(relevant_channel, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
            # TFR plots
            averaged.plot(picks=[0], baseline=iv_baseline, mode='mean', cmap='jet',
                          axes=ax, show=False, colorbar=False, dB=True,
                          tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax)

            # Time Plots
            if channel_type in ['ECG', 'Spinal', 'Cortical']:
                if channel_type == 'ECG':
                    style = None
                elif channel_type == 'Cortical':
                    style= 'dotted'
                else:
                    style = 'dashed'
                ax_time.plot(evoked.times, averaged_time.get_data().reshape(-1)*10**6, color='black', linestyle=style)
                ax_time.set_xlim([-0.3, 0.5])
                ax_time.set_xlabel('Time (s)')
                ax_time.set_ylabel('Amplitude (\u03BCV)')

            # Labelling
            if channel_type == 'ECG':
                ax.set_title('ECG Channel')
            elif channel_type == 'Spinal':
                ax.set_title('Spinal Channel')
                ax.set_ylabel(None)
            elif channel_type == 'Cortical':
                ax.set_title('Cortical Channel')
                ax.set_ylabel(None)

        cb = fig.colorbar(ax_ecg.images[-1], cax=cbar_ax)
        cb.set_label('Amplitude (dB)')
        fname = f"{cond_name}_CardiacTFRandTimeCourse_dB"
        plt.tight_layout()
        fig.savefig(image_path+fname+'.png')
        plt.savefig(image_path + fname + '.pdf', bbox_inches='tight', format="pdf")
        plt.clf()
# Script to plot the time-frequency decomposition in dB about the heartbeat
# Can't do this for CCA corrected data as it is already epoched about the spinal triggers
# https://mne.tools/stable/auto_examples/time_frequency/time_frequency_simulated.html#morlet-wavelets

import mne
import os
import numpy as np
from scipy.io import loadmat
from Metrics.SNR_functions import evoked_from_raw
import matplotlib.pyplot as plt

if __name__ == '__main__':
    spinal = False  # If true plots SEPs, if false plots QRS
    if spinal:
        freqs = np.arange(5., 400., 3.)
        fmin, fmax = freqs[[0, -1]]
    else:
        freqs = np.arange(5., 250., 3.)
        fmin, fmax = freqs[[0, -1]]
    subjects = np.arange(1, 37)
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    if spinal:
        iv_epoch = cfg['iv_epoch'][0] / 1000
        iv_baseline = cfg['iv_baseline'][0] / 1000
    else:
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

    image_path = "/data/p_02569/Images/TimeFrequencyPlots_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    methods = [True, True, False]
    method_names = ['Prep', 'PCA', 'ICA']  # Will treat SSP separately since there are multiple
    ssp = False  # Using files from merging mixed nerve with digits

    # To use mne grand_average method, need to generate a list of evoked potentials for each subject
    for i in np.arange(0, len(methods)):  # Methods Applied
        if methods[i]:  # Allows us to apply to only methods of interest
            method = method_names[i]

            for cond_name in cond_names:  # Conditions (median, tibial)
                evoked_list = []

                if cond_name == 'tibial':
                    full_name = 'Tibial Nerve Stimulation'
                    if spinal:
                        trigger_name = 'Tibial - Stimulation'
                    else:
                        trigger_name = 'qrs'
                    channel = ['L1']

                elif cond_name == 'median':
                    full_name = 'Median Nerve Stimulation'
                    if spinal:
                        trigger_name = 'Median - Stimulation'
                    else:
                        trigger_name = 'qrs'
                    channel = ['SC6']

                for subject in subjects:  # All subjects
                    subject_id = f'sub-{str(subject).zfill(3)}'

                    if method == 'Prep':
                        input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/"
                        fname = f"noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif"
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                        evoked.reorder_channels(esg_chans)
                        evoked = evoked.pick_channels(channel)
                        power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
                        # evoked_list.append(evoked)
                        evoked_list.append(power)

                    elif method == 'PCA':
                        input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/"
                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)

                        if spinal:
                            events, event_dict = mne.events_from_annotations(raw)
                            tstart_esg = -0.007
                            tmax_esg = 0.007
                            mne.preprocessing.fix_stim_artifact(raw, events=events, event_id=event_dict[trigger_name],
                                                                tmin=tstart_esg,
                                                                tmax=tmax_esg, mode='linear', stim_channel=None)
                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                        evoked.reorder_channels(esg_chans)
                        evoked = evoked.pick_channels(channel)
                        power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
                        # evoked_list.append(evoked)
                        evoked_list.append(power)

                    elif method == 'ICA':
                        input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + "/"
                        fname = f"clean_baseline_ica_auto_{cond_name}.fif"
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                        evoked.reorder_channels(esg_chans)
                        evoked = evoked.pick_channels(channel)
                        power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
                        # evoked_list.append(evoked)
                        evoked_list.append(power)

                averaged = mne.grand_average(evoked_list, interpolate_bads=False, drop_bads=False)
                # relevant_channel = averaged.pick_channels(channel)

                if spinal:
                    tmin = -0.1
                    tmax = 0.1
                    vmin = -380
                    vmax = -250
                else:
                    tmin = -0.2
                    tmax = 0.4
                    vmin = -400
                    vmax = -260
                fig, ax = plt.subplots(1, 1)
                # power = mne.time_frequency.tfr_stockwell(relevant_channel, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
                averaged.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
                              axes=ax, show=False, colorbar=True, dB=True,
                              tmin=tmin, tmax=tmax, vmax=vmax, vmin=vmin)
                im = ax.images
                cb = im[-1].colorbar
                cb.set_label('Amplitude [dB]')
                if method == 'Prep':
                    plt.title(f"Channel: {channel[0]}\n"
                              f"{full_name}")
                else:
                    plt.title(f"Method: {method}, Condition: {trigger_name}")
                if spinal:
                    fname = f"{method}_{trigger_name}_{cond_name}_dB.png"
                else:
                    fname = f"{method}_{trigger_name}_{cond_name}_dB.png"
                fig.savefig(image_path+fname)
                plt.clf()

    # Now deal with SSP plots - Just doing 5 to 6 for now
    if ssp:  # Actually using segregated ssp again
        for n in np.arange(5, 7):  # Methods Applied
            method = f'SSP_{n}'
            for cond_name in cond_names:  # Conditions (median, tibial)
                evoked_list = []

                if cond_name == 'tibial':
                    full_name = 'Tibial Nerve Stimulation'
                    if spinal:
                        trigger_name = 'Tibial - Stimulation'
                    else:
                        trigger_name = 'qrs'
                    channel = ['L1']

                elif cond_name == 'median':
                    full_name = 'Median Nerve Stimulation'
                    if spinal:
                        trigger_name = 'Median - Stimulation'
                    else:
                        trigger_name = 'qrs'
                    channel = ['SC6']

                for subject in subjects:  # All subjects
                    subject_id = f'sub-{str(subject).zfill(3)}'

                    input_path = f"/data/pt_02569/tmp_data/ssp_py/{subject_id}/{n} projections/"
                    raw = mne.io.read_raw_fif(f"{input_path}ssp_cleaned_{cond_name}.fif", preload=True)
                    evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                    evoked.reorder_channels(esg_chans)
                    evoked = evoked.pick_channels(channel)
                    power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
                    # evoked_list.append(evoked)
                    evoked_list.append(power)

                averaged = mne.grand_average(evoked_list, interpolate_bads=False, drop_bads=False)
                # relevant_channel = averaged.pick_channels(channel)
                if spinal:
                    tmin = -0.1
                    tmax = 0.1
                    vmin = -380
                    vmax = -250
                else:
                    tmin = -0.2
                    tmax = 0.4
                    vmin = -400
                    vmax = -260
                fig, ax = plt.subplots(1, 1)
                # power = mne.time_frequency.tfr_stockwell(relevant_channel, fmin=fmin, fmax=fmax, width=1.1)
                averaged.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
                              axes=ax, show=False, colorbar=True, dB=True,
                              tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax)
                im = ax.images
                cb = im[-1].colorbar
                cb.set_label('Amplitude [dB]')
                plt.title(f"Method: {method}, Condition: {trigger_name}")
                if spinal:
                    fname = f"{method}_{trigger_name}_{cond_name}_dB.png"
                else:
                    fname = f"{method}_{trigger_name}_{cond_name}_dB.png"

                fig.savefig(image_path + fname)
                plt.clf()

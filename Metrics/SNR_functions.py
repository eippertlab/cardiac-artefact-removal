# Holds functions necessary to form evoked responses and compute relevant SNR metrics

import mne
import numpy as np


# Create epochs and evoked response
def evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs):
    events, event_ids = mne.events_from_annotations(raw)
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                        baseline=tuple(iv_baseline))

    if reduced_epochs and trigger_name == 'Median - Stimulation':
        epochs = epochs[900:1100]
    elif reduced_epochs and trigger_name == 'Tibial - Stimulation':
        epochs = epochs[800:1200]

    # evoked = epochs.average()  # This line as is drops the ECG channel
    evoked = epochs.average(picks='all')  # Keeps ECG channel

    return evoked


# Create epochs and evoked response
def evoked_from_fourth_raw(raw, iv_epoch, iv_baseline, trigger_name):
    events, event_ids = mne.events_from_annotations(raw)
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                        baseline=tuple(iv_baseline))
    epochs = epochs[1::4]  # Take every fourth trial
    # print(np.shape(epochs.get_data()))
    evoked = epochs.average(picks='all')  # Keeps ECG channel

    return evoked


# Create epochs and evoked response - no baseline correction
def evoked_from_raw_nobaseline(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs):
    events, event_ids = mne.events_from_annotations(raw)
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                        baseline=None)

    if reduced_epochs and trigger_name == 'Median - Stimulation':
        epochs = epochs[900:1100]
    elif reduced_epochs and trigger_name == 'Tibial - Stimulation':
        epochs = epochs[800:1200]

    evoked = epochs.average(picks='all')

    return evoked


# Takes an evoked response and calculates the SNR
def calculate_SNR_evoked(evoked, cond_name, iv_baseline, reduced_window):
    # Drop reference and ECG from channels from data
    if 'TH6' in evoked.ch_names:
        evoked.drop_channels(['TH6'])
    if 'AL' in evoked.ch_names:
        evoked.drop_channels(['AL'])
    if 'AC' in evoked.ch_names:
        evoked.drop_channels(['AC'])
    if 'ECG' in evoked.ch_names:
        evoked.drop_channels(['ECG'])

    # Want to only check channels relevant to potential being triggered
    # Tibial centred around 22ms - take 10ms on either end
    # Median centred around 13ms - take 5ms on either end
    # Also option to use a reduced time window on either side of expected latency
    if cond_name == 'tibial':
        channels = ['S23', 'L1', 'S31']
        if reduced_window:
            start = 20/1000
            end = 24/1000
        else:
            start = 12 / 1000
            end = 32 / 1000

    elif cond_name == 'median':
        channels = ['S6', 'SC6', 'S14']
        if reduced_window:
            start = 11/1000
            end = 15/1000
        else:
            start = 8 / 1000
            end = 18 / 1000

    # Want to select the channel with the maximal signal evoked in the correct time window out of the relevant channels
    # Initialise values
    peak_amplitude = 0
    peak_channel = 'L4'
    peak_latency = 0

    # Cycle through channels of interest
    for ch in channels:
        evoked_channel = evoked.copy().pick_channels([ch])

        # Extract relevant data in time of interest - to check if we even have negative value here
        time_idx = evoked_channel.time_as_index([start, end])
        data = evoked_channel.data[0, time_idx[0]:time_idx[1]]

        # Check data in channel actually has neg values - Extract negative peaks
        if np.any(data < 0):
            _, latency, amplitude = evoked_channel.get_peak(ch_type=None, tmin=start, tmax=end, mode='neg',
                                                            time_as_index=False, merge_grads=False,
                                                            return_amplitude=True)
        # If there are no negative values, insert dummys
        else:
            latency = np.nan
            amplitude = np.nan
            ch = 'L4'

        # Update peak amplitude if needed, is amplitude is np.nan this will be false - no updating
        if abs(amplitude) > peak_amplitude:
            peak_amplitude = abs(amplitude)
            peak_channel = ch
            peak_latency = latency

    # If at the end the peak amplitude hasn't updated from 0, stick in NaN
    if peak_amplitude == 0:
        peak_amplitude = np.nan
        peak_channel = 'L4'
        peak_latency = np.nan

    # Now get the std in the baseline period of the selected channel
    peak_evoked = evoked.copy().pick_channels([peak_channel])
    iv_baseline_idx = peak_evoked.time_as_index([iv_baseline[0], iv_baseline[1]])
    base = peak_evoked.data[0, iv_baseline_idx[0]:iv_baseline_idx[1]]  # only one channel, baseline time period
    stan_dev = np.std(base, axis=0)

    # Compute and save snr
    snr = abs(peak_amplitude) / abs(stan_dev)

    return snr, peak_channel



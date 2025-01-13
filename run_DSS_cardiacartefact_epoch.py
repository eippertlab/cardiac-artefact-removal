# Script to actually run CCA on the data to remove the cardiac artefact by zeroing the 2 components with the highest
# correlation
# Using the meet package https://github.com/neurophysics/meet.git to run the CCA


import os
import mne
import numpy as np
from meet import spatfilt
from scipy.io import loadmat
from get_conditioninfo import get_conditioninfo
from replace_data import replace_data
from meegkit.dss import dss1
from meegkit.utils.matrix import fold, unfold
import matplotlib.pyplot as plt
import matplotlib as mpl


def run_DSS_heart(subject, condition, srmr_nr, no_exclude_comps):
    # Want 200ms before R-peak and 400ms after R-peak
    # Baseline is the 100ms period before the artefact occurs
    iv_baseline = [-300 / 1000, -200 / 1000]
    iv_epoch = [-400 / 1000, 600 / 1000]
    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    iv_epoch_sep = cfg['iv_epoch'][0] / 1000
    iv_baseline_sep = cfg['iv_baseline'][0] / 1000

    # Set variables
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    trigger_name_qrs = 'qrs'
    subject_id = f'sub-{str(subject).zfill(3)}'

    input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/"
    fname = f'noStimart_sr1000_{cond_name}_withqrs.fif'
    save_path = "/data/pt_02569/tmp_data/dss_heartart_py/" + subject_id + "/"
    fname_save = f'{cond_name}_heart_dss_{no_exclude_comps}removed.fif'
    os.makedirs(save_path, exist_ok=True)

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    if cond_name == 'median':
        best_chs = ['S6', 'SC6', 'S14']
        pot_time = 0.013
    else:
        best_chs = ['S23', 'L1', 'S31']
        pot_time = 0.022

    raw = mne.io.read_raw_fif(input_path + fname, preload=True)
    raw_data = raw.get_data(picks=esg_chans)  # n_chans, n_times

    # now create epochs based on the heart triggers
    events, event_ids = mne.events_from_annotations(raw)
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name_qrs}
    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1]-1/1000,
                        baseline=tuple(iv_baseline), preload=True)

    ##### Single trial matrix
    epo_data = np.swapaxes(epochs.get_data(picks=esg_chans), 0, 2)  # n_trials, n_chans, n_times
    # DSS needs n_samples, n_chans, n_trials so we swap axes

    # Run DSS
    # Some correlate ESG channel data with ECG trace in raw data - this would give us just 1 component (we have one ECG
    # trace), instead, try to do it similarly to SEP maximisation approach
    # Running with default values
    todss, fromdss, pwr_raw, pwr_avg = dss1(epo_data)
    print(pwr_raw)
    print(pwr_avg)

    # Apply obtained weights to raw dataset (todss dimensions n_comp x n_chans) - matrix multiplication
    # raw_data: n_chans x n_times, todss: n_components, n_chans
    DSS_data = np.tensordot(raw_data, todss[no_exclude_comps:, :], axes=(0, 1))

    # Reconstruct data without the first x components (highest power)
    # DSS_data: n_times x n_components, fromdss: n_components, n_channels
    reconstructed_data = np.tensordot(DSS_data, fromdss[no_exclude_comps:, :], axes=(1, 0))  # n_times, n_channels

    #######################  Replace data in the original raw so we keep annotations etc ####################
    replace_kwargs = dict(
        new_data = reconstructed_data.T # n_channels, n_times
    )
    # Replace and save
    clean_raw = raw.copy().apply_function(replace_data, picks=esg_chans, channel_wise=False, **replace_kwargs)
    clean_raw.save(os.path.join(save_path, fname_save), fmt='double', overwrite=True)

    #######################  Image check ####################
    # create epochs based on the heart triggers
    events, event_ids = mne.events_from_annotations(raw.pick(esg_chans))
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name_qrs}
    epochs_before = mne.Epochs(raw, events, picks=esg_chans, event_id=event_id_dict, tmin=iv_epoch[0],
                               tmax=iv_epoch[1], baseline=tuple(iv_baseline), preload=True)

    events, event_ids = mne.events_from_annotations(clean_raw.pick(esg_chans))
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name_qrs}
    epochs_after = mne.Epochs(clean_raw, events, picks=esg_chans, event_id=event_id_dict, tmin=iv_epoch[0],
                            tmax=iv_epoch[1], baseline=tuple(iv_baseline), preload=True)

    # create epochs based on the SEP triggers
    events, event_ids = mne.events_from_annotations(raw.pick(esg_chans))
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    epochs_before_sep = mne.Epochs(raw, events, picks=esg_chans, event_id=event_id_dict, tmin=iv_epoch_sep[0],
                               tmax=iv_epoch_sep[1], baseline=tuple(iv_baseline_sep), preload=True)

    events, event_ids = mne.events_from_annotations(clean_raw.pick(esg_chans))
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    epochs_after_sep = mne.Epochs(clean_raw, events, picks=esg_chans, event_id=event_id_dict, tmin=iv_epoch_sep[0],
                              tmax=iv_epoch_sep[1], baseline=tuple(iv_baseline_sep), preload=True)

    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    axes[0].plot(epochs_before.times, epochs_before.average().get_data().T*10**6)
    axes[0].set_title('Heartbeat, Before DSS')
    axes[1].plot(epochs_after.times, epochs_after.average().get_data().T*10**6)
    axes[1].set_title(f'Heartbeat, {no_exclude_comps} components')
    axes[2].plot(epochs_before_sep.times, epochs_before_sep.average(picks=best_chs).get_data().T*10**6)
    axes[2].set_title('SEP, Before DSS')
    axes[2].set_xlim([-0.1, 0.3])
    axes[2].axvline(pot_time, color='red', linewidth=0.5)
    axes[3].plot(epochs_after_sep.times, epochs_after_sep.average(picks=best_chs).get_data().T*10**6)
    axes[3].set_title(f'SEP, {no_exclude_comps} components')
    axes[3].set_xlim([-0.1, 0.3])
    axes[3].axvline(pot_time, color='red', linewidth=0.5)
    for axis in axes:
        axis.set_ylabel(u"Amplitude (\u03bcV)")
        axis.set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_DSS_heart(1, 3, 1, no_exclude_comps=2)

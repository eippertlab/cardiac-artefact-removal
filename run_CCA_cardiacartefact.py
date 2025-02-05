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
import matplotlib.pyplot as plt
import pickle


def run_CCA_heart(subject, condition, srmr_nr, plot_images):
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
    image_path = "/data/p_02569/Images/CCA_HeartArtComp/" + subject_id + "/"
    save_path = "/data/pt_02569/tmp_data/cca_heartart_py/" + subject_id + "/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)

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
    raw_data = raw.get_data(picks=esg_chans)

    # now create epochs based on the heart triggers
    events, event_ids = mne.events_from_annotations(raw)
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name_qrs}
    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1]-1/1000,
                        baseline=tuple(iv_baseline), preload=True)

    # Prepare matrices for cca
    ##### Average matrix
    epo_av = epochs.copy().average().data.T
    # Now want channels x observations matrix #np.shape()[0] gets number of trials
    # Epo av is no_times x no_channels (_x39)
    # Want to repeat this to form an array thats no. observations x no.channels
    # Need to repeat the array, no_trials/times amount along the y axis
    avg_matrix = np.tile(epo_av, (int((np.shape(epochs.get_data())[0])), 1))
    avg_matrix = avg_matrix.T  # Need to transpose for correct form for function - channels x observations

    ##### Single trial matrix
    epo_cca_data = epochs.get_data(picks=esg_chans)
    epo_data = epochs.get_data(picks=esg_chans)

    # 0 to access number of epochs, 1 to access number of channels
    # channels x observations
    # Need to transpose to get it in the form CCA wants
    st_matrix = np.swapaxes(epo_cca_data, 1, 2).reshape(-1, epo_cca_data.shape[1]).T
    st_matrix_long = np.swapaxes(epo_data, 1, 2).reshape(-1, epo_data.shape[1]).T

    # Run CCA
    # Some correlate ESG channel data with ECG trace in raw data - this would give us just 1 component (we have one ECG
    # trace), instead, try to do it similarly to SEP maximisation approach
    W_avg, W_st, r = spatfilt.CCA_data(avg_matrix, st_matrix)

    # Spatial Patterns
    A_st = np.cov(st_matrix) @ W_st

    # Save A_st and W_st just once so we can transform back whenever needed
    rfile = open(save_path + f'{cond_name}_heart_cca_Ast.pkl', 'wb')
    pickle.dump(A_st, rfile)
    rfile.close()

    rfile = open(save_path + f'{cond_name}_heart_cca_Wst.pkl', 'wb')
    pickle.dump(W_st, rfile)
    rfile.close()

    if plot_images:
        for no_exclude_comps in np.arange(1, 21):
            # fname_save = f'{cond_name}_heart_cca_{no_exclude_comps}removed.fif'

            # Apply obtained weights to raw dataset (W dimensions n_channels x n_components) - matrix multiplication
            CCA_data = raw_data.T @ W_st[:, no_exclude_comps:]  # n_times, n_components
            CCA_data = CCA_data.T  # n_components, n_times

            # Reconstruct data without the first x components (highest correlations)
            for_recon = A_st[:, no_exclude_comps:]
            # CCA_data: n_components x n_times, for_recon: n_channels, n_components
            reconstructed_data = np.tensordot(CCA_data, for_recon, axes=(0, 1))  # n_times, n_channels

            #######################  Replace data in the original raw so we keep annotations etc ####################
            replace_kwargs = dict(
                new_data = reconstructed_data.T # n_channels, n_times
            )
            # Replace and save
            clean_raw = raw.copy().apply_function(replace_data, picks=esg_chans, channel_wise=False, **replace_kwargs)
            # clean_raw.save(os.path.join(save_path, fname_save), fmt='double', overwrite=True)

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
            axes[0].set_title('Heartbeat, Before CCA')
            axes[1].plot(epochs_after.times, epochs_after.average().get_data().T*10**6)
            axes[1].set_title(f'Heartbeat, {no_exclude_comps} components')
            axes[2].plot(epochs_before_sep.times, epochs_before_sep.average(picks=best_chs).get_data().T*10**6)
            axes[2].set_title('SEP, Before CCA')
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
            plt.savefig(image_path+f"{cond_name}_{no_exclude_comps}removed.png")
            plt.close()
    # plt.show()

if __name__ == '__main__':
    run_CCA_heart(1, 2, 1, no_exclude_comps=6)

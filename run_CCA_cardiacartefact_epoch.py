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
from Archive.Plotting_Code.IsopotentialFunctions import mrmr_esg_isopotentialplot
import matplotlib.pyplot as plt
import matplotlib as mpl


def run_CCA(subject, condition, srmr_nr, no_exclude_comps):
    plot_graphs = True
    # Want 200ms before R-peak and 400ms after R-peak
    # Baseline is the 100ms period before the artefact occurs
    iv_baseline = [-300 / 1000, -200 / 1000]
    iv_epoch = [-400 / 1000, 600 / 1000]

    # Set variables
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = 'qrs'
    subject_id = f'sub-{str(subject).zfill(3)}'

    input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/"
    fname = f'noStimart_sr1000_{cond_name}_withqrs.fif'
    save_path = "/data/pt_02569/tmp_data/cca_heartart_py/" + subject_id + "/"
    fname_save = f'{cond_name}_heart_cca_removed.fif'
    os.makedirs(save_path, exist_ok=True)

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    raw = mne.io.read_raw_fif(input_path + fname, preload=True)
    raw_data = raw.get_data(picks=esg_chans)

    # now create epochs based on the heart triggers
    events, event_ids = mne.events_from_annotations(raw)
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
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
    print(r)

    # Apply obtained weights to raw dataset (W dimensions n_channels x n_components) - matrix multiplication
    CCA_data = raw_data.T @ W_st[:, no_exclude_comps:]  # n_times, n_components
    CCA_data = CCA_data.T  # n_components, n_times

    # Spatial Patterns
    A_st = np.cov(st_matrix) @ W_st

    # Reconstruct data without the first 2 components (highest correlations)
    for_recon = A_st[:, no_exclude_comps:]
    # CCA_data: n_components x n_times, for_recon: n_channels, n_components
    reconstructed_data = np.tensordot(CCA_data, for_recon, axes=(0, 1))  # n_times, n_channels

    #######################  Replace data in the original raw so we keep annotations etc ####################
    replace_kwargs = dict(
        new_data = reconstructed_data.T # n_channels, n_times
    )
    # Replace and save
    clean_raw = raw.copy().apply_function(replace_data, picks=esg_chans, channel_wise=False, **replace_kwargs)
    clean_raw.save(os.path.join(save_path, fname_save), fmt='double', overwrite=True)

    #######################  Image check ####################
    # now create epochs based on the heart triggers
    events, event_ids = mne.events_from_annotations(raw.pick(esg_chans))
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    epochs_before = mne.Epochs(raw, events, picks=esg_chans, event_id=event_id_dict, tmin=iv_epoch[0],
                               tmax=iv_epoch[1], baseline=tuple(iv_baseline), preload=True)

    events, event_ids = mne.events_from_annotations(clean_raw.pick(esg_chans))
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    epochs_after = mne.Epochs(clean_raw, events, picks=esg_chans, event_id=event_id_dict, tmin=iv_epoch[0],
                            tmax=iv_epoch[1], baseline=tuple(iv_baseline), preload=True)

    epochs_before.average().plot()
    epochs_after.average().plot()
    plt.show()
    exit()

    # ################################ Plotting Graphs #######################################
    # figure_path_spatial = f'/data/p_02569/Images/ComponentIsopotentialPlots_Dataset1/{subject_id}/'
    # os.makedirs(figure_path_spatial, exist_ok=True)
    #
    # if plot_graphs:
    #     ####### Spinal Isopotential Plots for the first 4 components ########
    #
    #     fig = plt.figure()
    #     for icomp in np.arange(0, 4):  # Plot for each of four components
    #         plt.subplot(2, 2, icomp + 1, title=f'Component {icomp + 1}')
    #         if data_string == 'SSP':
    #             colorbar_axes = [-2, 2]
    #         elif data_string == 'Prep':
    #             colorbar_axes = [-10, 8]
    #             if cond_name == 'tibial':
    #                 colorbar_axes = [-20, 20]
    #         else:
    #             colorbar_axes = [-4, 4]
    #         chan_labels = epochs.ch_names
    #         colorbar = True
    #         time = 0.0
    #         mrmr_esg_isopotentialplot([subject], A_st[:, icomp], colorbar_axes, chan_labels, colorbar, time)
    #
    #     if data_string == 'SSP':
    #         plt.tight_layout()
    #         plt.savefig(figure_path_spatial + f'{data_string}_{n}_{cond_name}.png')
    #
    #     else:
    #         plt.tight_layout()
    #         plt.savefig(figure_path_spatial + f'{data_string}_{cond_name}.png')
    #     plt.close(fig)
    #
    #     ############ Time Course of First 4 components ###############
    #     # cca_epochs and cca_epochs_d both already baseline corrected before this point
    #     figure_path_time = f'/data/p_02569/Images/ComponentTimePlots_Dataset1/{subject_id}/'
    #     os.makedirs(figure_path_time, exist_ok=True)
    #
    #     fig = plt.figure()
    #     for icomp in np.arange(0, 4):
    #         if is_inverted[icomp] is True:
    #             plt.subplot(2, 2, icomp + 1, title=f'Component {icomp + 1}, inv, r={r[icomp]:.3f}')
    #         else:
    #             plt.subplot(2, 2, icomp + 1, title=f'Component {icomp + 1}, r={r[icomp]:.3f}')
    #         # Want to plot Cor1 - Cor4
    #         # Plot for the mixed nerve data
    #         # get_data returns (n_epochs, n_channels, n_times)
    #         data = cca_epochs.get_data(picks=[f'Cor{icomp + 1}'])
    #         to_plot = np.mean(data[:, 0, :], axis=0)
    #         plt.plot(cca_epochs.times, to_plot)
    #         plt.xlim([-0.025, 0.065])
    #         line_label = f"{sep_latency[0][0] / 1000}s"
    #         plt.axvline(x=sep_latency[0][0] / 1000, color='r', linewidth='0.6', label=line_label)
    #         plt.xlabel('Time [s]')
    #         plt.ylabel('Amplitude [A.U.]')
    #         plt.legend()
    #         if data_string == 'SSP':
    #             plt.tight_layout()
    #             plt.savefig(figure_path_time + f'{data_string}_{n}_{cond_name}.png')
    #
    #         else:
    #             plt.tight_layout()
    #             plt.savefig(figure_path_time + f'{data_string}_{cond_name}.png')
    #     plt.close(fig)
    #
    #     ######################## Plot image for cca_epochs ############################
    #     # cca_epochs and cca_epochs_d both already baseline corrected before this point
    #     figure_path_st = f'/data/p_02569/Images/ComponentSinglePlots_Dataset1/{subject_id}/'
    #     os.makedirs(figure_path_st, exist_ok=True)
    #
    #     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    #     axes = [ax1, ax2, ax3, ax4]
    #     cropped = cca_epochs.copy().crop(tmin=-0.025, tmax=0.065)
    #     cmap = mpl.colors.ListedColormap(["mediumblue", "deepskyblue", "lemonchiffon", "gold"])
    #
    #     for icomp in np.arange(0, 4):
    #         cropped.plot_image(picks=f'Cor{icomp + 1}', combine=None, cmap=cmap, evoked=False, show=False,
    #                            axes=axes[icomp], title=f'Component {icomp + 1}', colorbar=False, group_by=None,
    #                            vmin=-0.4, vmax=0.4, units=dict(eeg='V'), scalings=dict(eeg=1))
    #
    #     plt.tight_layout()
    #     fig.subplots_adjust(right=0.85)
    #     ax5 = fig.add_axes([0.9, 0.1, 0.01, 0.8])
    #     norm = mpl.colors.Normalize(vmin=-0.4, vmax=0.4)
    #     mpl.colorbar.ColorbarBase(ax5, cmap=cmap, norm=norm, spacing='proportional')
    #     # has to be as a list - starts with x, y coordinates for start and then width and height in % of figure width
    #     if data_string == 'SSP':
    #         plt.savefig(figure_path_st + f'{data_string}_{n}_{cond_name}.png')
    #
    #     else:
    #         plt.savefig(figure_path_st + f'{data_string}_{cond_name}.png')
    #     plt.close(fig)
    #     # plt.show()
    #
    #     ############################ Combine to one Image ##########################
    #     figure_path = f'/data/p_02569/Images/ComponentPlots_Dataset1/{subject_id}/'
    #     os.makedirs(figure_path, exist_ok=True)
    #
    #     if data_string == 'SSP':
    #         spatial = plt.imread(figure_path_spatial + f'{data_string}_{n}_{cond_name}.png')
    #         time = plt.imread(figure_path_time + f'{data_string}_{n}_{cond_name}.png')
    #         single_trial = plt.imread(figure_path_st + f'{data_string}_{n}_{cond_name}.png')
    #
    #     else:
    #         spatial = plt.imread(figure_path_spatial + f'{data_string}_{cond_name}.png')
    #         time = plt.imread(figure_path_time + f'{data_string}_{cond_name}.png')
    #         single_trial = plt.imread(figure_path_st + f'{data_string}_{cond_name}.png')
    #
    #     fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    #     axes[0, 0].imshow(time)
    #     axes[0, 0].axis('off')
    #     axes[0, 1].imshow(spatial)
    #     axes[0, 1].axis('off')
    #     axes[1, 0].imshow(single_trial)
    #     axes[1, 0].axis('off')
    #     axes[1, 1].axis('off')
    #
    #     plt.subplots_adjust(top=0.95, wspace=0, hspace=0)
    #
    #     if data_string == 'SSP':
    #         plt.suptitle(f'Subject {subject}, {data_string}_{n}, {cond_name}')
    #         plt.savefig(figure_path + f'{data_string}_{n}_{cond_name}.png')
    #
    #     else:
    #         plt.suptitle(f'Subject {subject}, {data_string}_{cond_name}')
    #         plt.savefig(figure_path + f'{data_string}_{cond_name}.png')
    #     plt.close(fig)

if __name__ == '__main__':
    run_CCA(1, 3, 1, no_exclude_comps=5)
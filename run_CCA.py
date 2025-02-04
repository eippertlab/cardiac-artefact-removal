# Script to actually run CCA on the data
# Using the meet package https://github.com/neurophysics/meet.git to run the CCA


import os
import mne
import numpy as np
from meet import spatfilt
from scipy.io import loadmat
from get_conditioninfo import get_conditioninfo
from get_esg_channels import get_esg_channels
from Archive.Plotting_Code.IsopotentialFunctions import mrmr_esg_isopotentialplot
import matplotlib.pyplot as plt
import matplotlib as mpl


def run_CCA(subject, condition, srmr_nr, data_string, n):
    plot_graphs = True
    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]
    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000

    interpol_window_esg = cfg['interpol_window_esg'][0]  # In ms

    # Set variables
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    subject_id = f'sub-{str(subject).zfill(3)}'

    potential_path = f"/data/p_02068/SRMR1_experiment/analyzed_data/esg/{subject_id}/"

    # Select the right files based on the data_string
    if data_string == 'PCA':
        input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id
        fname = f'data_clean_ecg_spinal_{cond_name}_withqrs.fif'
        save_path = "/data/pt_02569/tmp_data/ecg_rm_py_cca/" + subject_id
        os.makedirs(save_path, exist_ok=True)

    elif data_string == 'Prep':
        input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id
        fname = f'noStimart_sr1000_{cond_name}_withqrs.fif'
        save_path = "/data/pt_02569/tmp_data/prepared_py_cca/" + subject_id
        os.makedirs(save_path, exist_ok=True)

    elif data_string == 'ICA':
        input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id
        fname = f'clean_baseline_ica_auto_{cond_name}.fif'
        save_path = "/data/pt_02569/tmp_data/baseline_ica_py_cca/" + subject_id
        os.makedirs(save_path, exist_ok=True)

    elif data_string == 'SSP':
        input_path = "/data/pt_02569/tmp_data/ssp_py/" + subject_id + "/" + str(n) + " projections/"
        fname = f"ssp_cleaned_{cond_name}.fif"
        save_path = "/data/pt_02569/tmp_data/ssp_py_cca/" + subject_id + "/" + str(n) + " projections/"
        os.makedirs(save_path, exist_ok=True)

    else:
        print('Invalid Data String Name Entered')
        exit()

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23', 'TH6']

    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

    # now create epochs based on the trigger names
    events, event_ids = mne.events_from_annotations(raw)
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1]-1/1000,
                        baseline=tuple(iv_baseline), preload=True)

    # cca window size - Birgit created individual potential latencies for each subject
    halfwindow_size = 10/2
    fname_pot = 'potential_latency.mat'
    matdata = loadmat(potential_path + fname_pot)

    if cond_name == 'median':
        epochs = epochs.pick_channels(cervical_chans, ordered=True)
        esg_chans = cervical_chans
        sep_latency = matdata['med_potlatency']
        window_times = [7/1000, 37/1000]
    elif cond_name == 'tibial':
        epochs = epochs.pick_channels(lumbar_chans, ordered=True)
        esg_chans = lumbar_chans
        sep_latency = matdata['tib_potlatency']
        window_times = [7/1000, 47/1000]
    else:
        raise ValueError('Invalid condition name attempted for use')

    # Crop the epochs
    window = epochs.time_as_index(window_times)
    epo_cca = epochs.copy().crop(tmin=window_times[0], tmax=window_times[1], include_tmax=False)

    # Prepare matrices for cca
    ##### Average matrix
    epo_av = epo_cca.copy().average().data.T
    # Now want channels x observations matrix #np.shape()[0] gets number of trials
    # Epo av is no_times x no_channels (10x40)
    # Want to repeat this to form an array thats no. observations x no.channels (20000x40)
    # Need to repeat the array, no_trials/times amount along the y axis
    avg_matrix = np.tile(epo_av, (int((np.shape(epochs.get_data())[0])), 1))
    avg_matrix = avg_matrix.T  # Need to transpose for correct form for function - channels x observations

    ##### Single trial matrix
    epo_cca_data = epo_cca.get_data(picks=esg_chans)
    epo_data = epochs.get_data(picks=esg_chans)

    # 0 to access number of epochs, 1 to access number of channels
    # channels x observations
    no_times = int(window[1] - window[0])
    # Need to transpose to get it in the form CCA wants
    st_matrix = np.swapaxes(epo_cca_data, 1, 2).reshape(-1, epo_cca_data.shape[1]).T
    st_matrix_long = np.swapaxes(epo_data, 1, 2).reshape(-1, epo_data.shape[1]).T

    # Run CCA
    W_avg, W_st, r = spatfilt.CCA_data(avg_matrix, st_matrix)  # Getting the same shapes as matlab so far

    all_components = len(r)

    # Apply obtained weights to the long dataset (dimensions 40x9) - matrix multiplication
    CCA_concat = st_matrix_long.T @ W_st[:, 0:all_components]
    CCA_concat = CCA_concat.T

    # Spatial Patterns
    A_st = np.cov(st_matrix) @ W_st

    # Reshape - (900, 2000, 9)
    no_times_long = np.shape(epochs.get_data())[2]
    no_epochs = np.shape(epochs.get_data())[0]

    # Perform reshape
    CCA_comps = np.reshape(CCA_concat, (all_components, no_times_long, no_epochs), order='F')

    # Now we have CCA comps, get the data in the axes format MNE likes (n_epochs, n_channels, n_times)
    CCA_comps = np.swapaxes(CCA_comps, 0, 2)
    CCA_comps = np.swapaxes(CCA_comps, 1, 2)
    selected_components = all_components  # Just keeping all for now to avoid rerunning

    #######################  Epoch data class to store the information ####################
    data = CCA_comps[:, 0:selected_components, :]
    events = epochs.events
    event_id = epochs.event_id
    tmin = iv_epoch[0]
    sfreq = 1000

    ch_names = []
    ch_types = []
    for i in np.arange(0, all_components):
        ch_names.append(f'Cor{i+1}')
        ch_types.append('eeg')

    # Initialize an info structure
    info = mne.create_info(
        ch_names=ch_names,
        ch_types=ch_types,
        sfreq=sfreq
    )

    # Create and save
    cca_epochs = mne.EpochsArray(data, info, events, tmin, event_id)
    cca_epochs = cca_epochs.apply_baseline(baseline=tuple(iv_baseline))
    cca_epochs.save(os.path.join(save_path, fname), fmt='double', overwrite=True)

    ################################ Plotting Graphs #######################################
    figure_path_spatial = f'/data/p_02569/Images/ComponentIsopotentialPlots_Dataset1/{subject_id}/'
    os.makedirs(figure_path_spatial, exist_ok=True)

    if plot_graphs:
        ####### Spinal Isopotential Plots for the first 4 components ########

        fig = plt.figure()
        for icomp in np.arange(0, 4):  # Plot for each of four components
            plt.subplot(2, 2, icomp + 1, title=f'Component {icomp + 1}')
            if data_string == 'SSP':
                colorbar_axes = [-2, 2]
            elif data_string == 'Prep':
                colorbar_axes = [-10, 8]
                if cond_name == 'tibial':
                    colorbar_axes = [-20, 20]
            else:
                colorbar_axes = [-4, 4]
            chan_labels = epochs.ch_names
            colorbar = True
            time = 0.0
            mrmr_esg_isopotentialplot([subject], A_st[:, icomp], colorbar_axes, chan_labels, colorbar, time)

        if data_string == 'SSP':
            plt.tight_layout()
            plt.savefig(figure_path_spatial + f'{data_string}_{n}_{cond_name}.png')

        else:
            plt.tight_layout()
            plt.savefig(figure_path_spatial + f'{data_string}_{cond_name}.png')
        plt.close(fig)

        ############ Time Course of First 4 components ###############
        # cca_epochs and cca_epochs_d both already baseline corrected before this point
        figure_path_time = f'/data/p_02569/Images/ComponentTimePlots_Dataset1/{subject_id}/'
        os.makedirs(figure_path_time, exist_ok=True)

        fig = plt.figure()
        for icomp in np.arange(0, 4):
            plt.subplot(2, 2, icomp + 1, title=f'Component {icomp + 1}, r={r[icomp]:.3f}')
            # Want to plot Cor1 - Cor4
            # Plot for the mixed nerve data
            # get_data returns (n_epochs, n_channels, n_times)
            data = cca_epochs.get_data(picks=[f'Cor{icomp + 1}'])
            to_plot = np.mean(data[:, 0, :], axis=0)
            plt.plot(cca_epochs.times, to_plot)
            plt.xlim([-0.025, 0.065])
            line_label = f"{sep_latency[0][0] / 1000}s"
            plt.axvline(x=sep_latency[0][0] / 1000, color='r', linewidth='0.6', label=line_label)
            plt.xlabel('Time [s]')
            plt.ylabel('Amplitude [A.U.]')
            plt.legend()
            if data_string == 'SSP':
                plt.tight_layout()
                plt.savefig(figure_path_time + f'{data_string}_{n}_{cond_name}.png')

            else:
                plt.tight_layout()
                plt.savefig(figure_path_time + f'{data_string}_{cond_name}.png')
        plt.close(fig)

        ######################## Plot image for cca_epochs ############################
        # cca_epochs and cca_epochs_d both already baseline corrected before this point
        figure_path_st = f'/data/p_02569/Images/ComponentSinglePlots_Dataset1/{subject_id}/'
        os.makedirs(figure_path_st, exist_ok=True)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
        axes = [ax1, ax2, ax3, ax4]
        cropped = cca_epochs.copy().crop(tmin=-0.025, tmax=0.065)
        cmap = mpl.colors.ListedColormap(["mediumblue", "deepskyblue", "lemonchiffon", "gold"])

        for icomp in np.arange(0, 4):
            cropped.plot_image(picks=f'Cor{icomp + 1}', combine=None, cmap=cmap, evoked=False, show=False,
                               axes=axes[icomp], title=f'Component {icomp + 1}', colorbar=False, group_by=None,
                               vmin=-0.4, vmax=0.4, units=dict(eeg='V'), scalings=dict(eeg=1))

        plt.tight_layout()
        fig.subplots_adjust(right=0.85)
        ax5 = fig.add_axes([0.9, 0.1, 0.01, 0.8])
        norm = mpl.colors.Normalize(vmin=-0.4, vmax=0.4)
        mpl.colorbar.ColorbarBase(ax5, cmap=cmap, norm=norm, spacing='proportional')
        # has to be as a list - starts with x, y coordinates for start and then width and height in % of figure width
        if data_string == 'SSP':
            plt.savefig(figure_path_st + f'{data_string}_{n}_{cond_name}.png')

        else:
            plt.savefig(figure_path_st + f'{data_string}_{cond_name}.png')
        plt.close(fig)
        # plt.show()

        ############################ Combine to one Image ##########################
        figure_path = f'/data/p_02569/Images/ComponentPlots_Dataset1/{subject_id}/'
        os.makedirs(figure_path, exist_ok=True)

        if data_string == 'SSP':
            spatial = plt.imread(figure_path_spatial + f'{data_string}_{n}_{cond_name}.png')
            time = plt.imread(figure_path_time + f'{data_string}_{n}_{cond_name}.png')
            single_trial = plt.imread(figure_path_st + f'{data_string}_{n}_{cond_name}.png')

        else:
            spatial = plt.imread(figure_path_spatial + f'{data_string}_{cond_name}.png')
            time = plt.imread(figure_path_time + f'{data_string}_{cond_name}.png')
            single_trial = plt.imread(figure_path_st + f'{data_string}_{cond_name}.png')

        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        axes[0, 0].imshow(time)
        axes[0, 0].axis('off')
        axes[0, 1].imshow(spatial)
        axes[0, 1].axis('off')
        axes[1, 0].imshow(single_trial)
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')

        plt.subplots_adjust(top=0.95, wspace=0, hspace=0)

        if data_string == 'SSP':
            plt.suptitle(f'Subject {subject}, {data_string}_{n}, {cond_name}')
            plt.savefig(figure_path + f'{data_string}_{n}_{cond_name}.png')

        else:
            plt.suptitle(f'Subject {subject}, {data_string}_{cond_name}')
            plt.savefig(figure_path + f'{data_string}_{cond_name}.png')
        plt.close(fig)


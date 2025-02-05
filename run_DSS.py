# Script to actually run CCA on the data
# Using the meet package https://github.com/neurophysics/meet.git to run the CCA


import os
import mne
import numpy as np
from meegkit.utils import tscov
from meegkit.dss import dss0
import pickle
from scipy.stats import pearsonr
from replace_data import replace_data
from scipy.io import loadmat
from get_conditioninfo import get_conditioninfo
from get_esg_channels import get_esg_channels
from Plotting_Code_Publication.IsopotentialFunctions import mrmr_esg_isopotentialplot
import matplotlib.pyplot as plt
import matplotlib as mpl


def run_DSS(subject, condition, srmr_nr, data_string, n):
    selected_components = 4
    plot_graphs = True
    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000

    # Set variables
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    subject_id = f'sub-{str(subject).zfill(3)}'

    potential_path = f"/data/p_02068/SRMR1_experiment/analyzed_data/esg/{subject_id}/"

    # Select the right files based on the data_string
    if data_string == 'PCA':
        input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/"
        fname = f'data_clean_ecg_spinal_{cond_name}_withqrs.fif'
        save_path = "/data/pt_02569/tmp_data/ecg_rm_py_dss/" + subject_id + "/"
        os.makedirs(save_path, exist_ok=True)

    elif data_string == 'Prep':
        input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/"
        fname = f'noStimart_sr1000_{cond_name}_withqrs.fif'
        save_path = "/data/pt_02569/tmp_data/prepared_py_dss/" + subject_id + "/"
        os.makedirs(save_path, exist_ok=True)

    elif data_string == 'ICA':
        input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + "/"
        fname = f'clean_baseline_ica_auto_{cond_name}.fif'
        save_path = "/data/pt_02569/tmp_data/baseline_ica_py_dss/" + subject_id + "/"
        os.makedirs(save_path, exist_ok=True)

    elif data_string == 'SSP':
        input_path = "/data/pt_02569/tmp_data/ssp_py/" + subject_id + "/" + str(n) + " projections/"
        fname = f"ssp_cleaned_{cond_name}.fif"
        save_path = "/data/pt_02569/tmp_data/ssp_py_dss/" + subject_id + "/" + str(n) + " projections/"
        os.makedirs(save_path, exist_ok=True)

    else:
        raise ValueError('Invalid Data String Name Entered')

    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

    # now create epochs based on the trigger names
    events, event_ids = mne.events_from_annotations(raw)
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1]-1/1000,
                        baseline=tuple(iv_baseline), preload=True)

    # cca window size - Birgit created individual potential latencies for each subject
    fname_pot = 'potential_latency.mat'
    matdata = loadmat(potential_path + fname_pot)

    if cond_name == 'median':
        esg_chans = cervical_chans
        sep_latency = matdata['med_potlatency']
        window_times = [7/1000, 37/1000]
    elif cond_name == 'tibial':
        esg_chans = lumbar_chans
        sep_latency = matdata['tib_potlatency']
        window_times = [7/1000, 47/1000]
    else:
        raise ValueError('Invalid condition name attempted for use')

    raw = mne.io.read_raw_fif(input_path + fname, preload=True)
    raw_data = raw.get_data(picks=esg_chans)  # n_chans, n_times

    # now create epochs based on the stimulation triggers
    evoked = epochs.pick(esg_chans).average()
    evoked_data = np.swapaxes(evoked.get_data(picks=esg_chans), 0, 1)  # n_chans, n_times
    # DSS needs n_times, n_chans [, n_trials] so we swap axes

    # Run DSS - bias filter similar to SSP noise space
    # Compute original and biased covariance matrices
    c0, _ = tscov(raw_data.T)
    c1, _ = tscov(evoked_data)

    # SEP maximisation approach
    # Running with default values
    todss, fromdss, pwr_raw, pwr_bias = dss0(c0, c1)
    # todss is (n_chans, n_components)

    # Save todss and fromdss just once so we can transform back whenever needed
    rfile = open(save_path + f'{cond_name}_sep_fromdss.pkl', 'wb')
    pickle.dump(fromdss, rfile)
    rfile.close()

    rfile = open(save_path + f'{cond_name}_sep_todss.pkl', 'wb')
    pickle.dump(todss, rfile)
    rfile.close()

    # Clean Data
    cleaned_data = raw_data.T @ todss

    # Spatial pattern is cross correlation of component time series and raw sensor waveforms
    # Initialize an empty list to store correlation results
    sensor_correlations = []

    # Loop through each component time series
    for comp in cleaned_data.T[0:selected_components, :]:
        comp_correlations = []

        # Loop through each sensor waveform
        for sensor_waveform in raw_data:
            # Compute the cross-correlation between component and sensor waveform
            correlation, _ = pearsonr(sensor_waveform, comp)

            # Store the result for this component-sensor pair
            comp_correlations.append(correlation)

        # Append the list of correlations for this component to the main list
        sensor_correlations.append(comp_correlations)

    # Convert the results to a NumPy array and save
    sensor_correlations = np.array(sensor_correlations)  # 4, 18 (no_components, no_sensors)
    rfile = open(save_path + f'{cond_name}_sep_correlations.pkl', 'wb')
    pickle.dump(sensor_correlations, rfile)
    rfile.close()

    #######################  Epoch class to store information ####################
    # Replace data in the raw with the clean data so we can still use MNE methods to pull out the epochs
    replace_kwargs = dict(
        new_data=cleaned_data.T  # n_channels, n_times
    )
    # Replace and save
    clean_raw = raw.copy().apply_function(replace_data, picks=esg_chans, channel_wise=False, **replace_kwargs)

    events, event_ids = mne.events_from_annotations(clean_raw.pick(esg_chans))
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    epochs_clean_sep = mne.Epochs(clean_raw, events, picks=esg_chans, event_id=event_id_dict, tmin=iv_epoch[0],
                                  tmax=iv_epoch[1], baseline=tuple(iv_baseline), preload=True)

    # Get the epoch data, store and save
    epochs_clean_data = epochs_clean_sep.get_data()
    data = epochs_clean_data[:, 0:selected_components, :]
    events = epochs.events
    event_id = epochs.event_id
    tmin = iv_epoch[0]
    sfreq = 1000

    ch_names = []
    ch_types = []
    for i in np.arange(0, selected_components):
        ch_names.append(f'Cor{i + 1}')
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
    figure_path_spatial = f'/data/p_02569/Images/DSS_SEP/ComponentIsopotentialPlots_Dataset1/{subject_id}/'
    os.makedirs(figure_path_spatial, exist_ok=True)

    if plot_graphs:
        ####### Spinal Isopotential Plots for the first 4 components ########
        fig = plt.figure()
        for icomp in np.arange(0, 4):  # Plot for each of four components
            plt.subplot(2, 2, icomp + 1, title=f'Component {icomp + 1}')
            colorbar_axes = [-0.8*10**6, 0.8*10**6]  # To account for multiplication by 10**-6 inside function
            chan_labels = epochs.ch_names
            colorbar = True
            time = 0.0
            mrmr_esg_isopotentialplot([subject], sensor_correlations[icomp, :].T, colorbar_axes, chan_labels, colorbar, time)

        if data_string == 'SSP':
            plt.tight_layout()
            plt.savefig(figure_path_spatial + f'{data_string}_{n}_{cond_name}.png')

        else:
            plt.tight_layout()
            plt.savefig(figure_path_spatial + f'{data_string}_{cond_name}.png')
        plt.close(fig)

        ############ Time Course of First 4 components ###############
        # cca_epochs and cca_epochs_d both already baseline corrected before this point
        figure_path_time = f'/data/p_02569/Images/DSS_SEP/ComponentTimePlots_Dataset1/{subject_id}/'
        os.makedirs(figure_path_time, exist_ok=True)

        fig = plt.figure()
        for icomp in np.arange(0, 4):
            plt.subplot(2, 2, icomp + 1, title=f'Component {icomp + 1}')
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

        ############################ Combine to one Image ##########################
        figure_path = f'/data/p_02569/Images/DSS_SEP/ComponentPlots_Dataset1/{subject_id}/'
        os.makedirs(figure_path, exist_ok=True)

        if data_string == 'SSP':
            spatial = plt.imread(figure_path_spatial + f'{data_string}_{n}_{cond_name}.png')
            time = plt.imread(figure_path_time + f'{data_string}_{n}_{cond_name}.png')

        else:
            spatial = plt.imread(figure_path_spatial + f'{data_string}_{cond_name}.png')
            time = plt.imread(figure_path_time + f'{data_string}_{cond_name}.png')

        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        axes[0].imshow(time)
        axes[0].axis('off')
        axes[1].imshow(spatial)
        axes[1].axis('off')

        plt.subplots_adjust(top=0.95, wspace=0, hspace=0)

        if data_string == 'SSP':
            plt.suptitle(f'Subject {subject}, {data_string}_{n}, {cond_name}')
            plt.savefig(figure_path + f'{data_string}_{n}_{cond_name}.png')

        else:
            plt.suptitle(f'Subject {subject}, {data_string}_{cond_name}')
            plt.savefig(figure_path + f'{data_string}_{cond_name}.png')
        plt.close(fig)


if __name__ == '__main__':
    subject = 1
    condition = 3
    srmr_nr = 1
    data_string = 'PCA'
    n = 0
    run_DSS(subject, condition, srmr_nr, data_string, n)
# Script to Run the Isopotential functions
# Generates grand average topography after each method
# Uses a slightly different version of the code than is present in Spinal_Isopotential_Plots to generate the gifs

import numpy as np
import mne
import matplotlib.pyplot as plt
from Plotting_Code_Publication.IsopotentialFunctions import mrmr_esg_isopotentialplot
from scipy.io import loadmat
import os


if __name__ == '__main__':
    # methods = ['Prep', 'PCA', 'ICA', 'SSP']  # Methods to do
    methods = ['ICA', 'SSP']  # Methods to do
    n_proj = 6

    trigger_names = ['Median - Stimulation', 'Tibial - Stimulation']
    save_path = '/data/p_02569/SpatialPatterns_D1'
    os.makedirs(save_path, exist_ok=True)
    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]
    # epoch
    iv_epoch = [-200 / 1000, 700 / 1000]
    # let's say we epoch the data from -200 to 700 ms around the trigger of interest
    iv_baseline = [-100 / 1000, -10 / 1000]
    # subjects = [1, 6, 15, 30]
    subjects = np.arange(1, 37)

    for method in methods:
        # Just piloting with the classic PCA_OBS data
        for trigger_name in trigger_names:
            evoked_list = []
            for subj in subjects:
                if trigger_name == 'Median - Stimulation':
                    time_point = 13 / 1000
                    cond_name = 'median'
                else:
                    cond_name = 'tibial'
                    time_point = 22 / 1000

                subject_id = f'sub-{str(subj).zfill(3)}'

                if method == 'Prep':
                    data_path = '/data/pt_02569/tmp_data/prepared_py/' + subject_id + \
                                '/noStimart_sr1000_' + cond_name + '_withqrs.fif'

                elif method == 'PCA':
                    data_path = '/data/pt_02569/tmp_data/ecg_rm_py/' + subject_id + \
                                '/data_clean_ecg_spinal_' + cond_name + '_withqrs.fif'

                elif method == 'ICA':
                    data_path = '/data/pt_02569/tmp_data/baseline_ica_py/' + subject_id + \
                                '/clean_baseline_ica_auto_' + cond_name + '.fif'

                elif method == 'SSP':
                    data_path = "/data/pt_02569/tmp_data/ssp_py/" + subject_id + f"/{n_proj} projections/ssp_cleaned_" + cond_name + ".fif"

                # load some continuous data
                raw = mne.io.read_raw_fif(data_path, preload=True)
                events, event_ids = mne.events_from_annotations(raw)
                event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                epo = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                 baseline=tuple(iv_baseline))

                # Want each channel averaged across all epochs at a given time point
                evoked_list.append(epo.average())

            evoked = mne.grand_average(evoked_list)
            # plot as evolution
            fig = plt.figure()
            plt.title(f'Grand Average Topography, {method}, {trigger_name}')

            # time_idx = []
            # tmp = np.argwhere(epo.times >= time_point)
            # # sometimes when data is down sampled  find(epo.times == time_points(ii)) doesn't work
            # time_idx.append(tmp[0])
            # chanvalues = evoked.data[:, time_idx]

            chanvalues = evoked.crop(tmin=time_point - (2 / 1000), tmax=time_point + (2 / 1000)).data
            chanvalues = chanvalues.mean(axis=1)
            # Should then average across time points of interest
            chan_labels = evoked.ch_names
            if method == 'SSP' or method == 'ICA':
                colorbar_axes = [-0.5, 0.5]
            else:
                colorbar_axes = [-1, 1]
            # colorbar_axes = [-1, 1]
            subjects_4grid = np.arange(1, 37)  # subj  # Pass this instead of (1, 37) for 1 subjects
            # you can also base the grid on an several subjects
            # then the function takes the average over the channel positions of all those subjects
            colorbar = True
            mrmr_esg_isopotentialplot(subjects_4grid, chanvalues, colorbar_axes, chan_labels, colorbar,
                                      time_point)

            if method == 'SSP':
                filename = f'{save_path}/GrandAverage_SSP{n_proj}_{cond_name}.png'
            else:
                filename = f'{save_path}/GrandAverage_{method}_{cond_name}.png'

            plt.savefig(filename)
            plt.close()

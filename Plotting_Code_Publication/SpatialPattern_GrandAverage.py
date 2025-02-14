# Script to Run the Isopotential functions
# Generates grand average topography after each method
# Uses a slightly different version of the code than is present in Spinal_Isopotential_Plots to generate the gifs

import numpy as np
import mne
import matplotlib.pyplot as plt
from Plotting_Code_Publication.IsopotentialFunctions_2axis import mrmr_esg_isopotentialplot_cbar
from scipy.io import loadmat
import os
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    methods = ['ICA', 'DSS-cardiac']  # Methods to do

    trigger_names = ['Median - Stimulation', 'Tibial - Stimulation']
    save_path = '/data/p_02569/Images/Extra_SpatialPatterns_D1/'
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

                if method == 'ICA':
                    data_path = f"/data/pt_02569/tmp_data/ssp_py/{subject_id}/5 projections/epochs_{cond_name}.fif"

                elif method == 'DSS-cardiac':
                    if cond_name == 'median':
                        n = 9
                    elif cond_name == 'tibial':
                        n = 7
                    data_path = f"/data/pt_02569/tmp_data/dss_heartart_py/{subject_id}/epochs_{cond_name}_{n}.fif"

                # load some continuous data
                epo = mne.read_epochs(data_path, preload=True)

                # Want each channel averaged across all epochs at a given time point
                evoked_list.append(epo.average())

            evoked = mne.grand_average(evoked_list)
            # plot as evolution
            fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 0.05]})
            plt.title(f'Grand Average Topography, {method}, {trigger_name}')

            chanvalues = evoked.crop(tmin=time_point - (1 / 1000), tmax=time_point + (2 / 1000)).data
            chanvalues = chanvalues.mean(axis=1)
            # Should then average across time points of interest
            chan_labels = evoked.ch_names
            if cond_name == 'median':
                colorbar_axes = [-0.5, 0.5]
            else:
                colorbar_axes = [-0.2, 0.2]
            subjects_4grid = np.arange(1, 37)  # subj  # Pass this instead of (1, 37) for 1 subjects
            # you can also base the grid on an several subjects
            # then the function takes the average over the channel positions of all those subjects
            colorbar = True
            cf = mrmr_esg_isopotentialplot_cbar(subjects_4grid, chanvalues, colorbar_axes, chan_labels,
                                                colorbar, time_point, axes[0])
            ticks = [colorbar_axes[0], 0, colorbar_axes[1]]
            fig.colorbar(cf, cax=axes[1], label='Amplitude (\u03BCV)', ticks=ticks)

            if method == 'SSP':
                filename = f'{save_path}/GrandAverage_SSP5_{cond_name}.png'
            else:
                filename = f'{save_path}/GrandAverage_{method}_{cond_name}.png'

            plt.tight_layout()
            plt.savefig(filename)
            plt.savefig(filename+'.pdf', bbox_inches='tight', format="pdf")
            plt.close()

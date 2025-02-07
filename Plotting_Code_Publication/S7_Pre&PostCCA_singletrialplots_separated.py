######################## Plot image for cca_epochs and uncleaned data ############################

import mne
import numpy as np
from invert import invert
from transform import transform
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import pandas as pd
import random
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    reduced_trials = True
    if reduced_trials:
        no = 1000
        trial_indices = random.sample(range(1999), no)  # Need to be all unique to avoid selecting the same trials
        trial_indices.sort()  # Want in chronological order

    SSP_proj = 5
    subjects = [6]
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    # Contains information on which CCA component to pick
    xls = pd.ExcelFile('/data/p_02569/Components_CCA.xls')
    df_CCA = pd.read_excel(xls, 'Dataset 1')
    df_CCA.set_index('Subject', inplace=True)

    # Contains information on which CCA component to pick
    xls = pd.ExcelFile('/data/p_02569/Components_DSS.xls')
    df_DSS = pd.read_excel(xls, 'Dataset 1')
    df_DSS.set_index('Subject', inplace=True)

    for subject in subjects:
        for cond_name in cond_names:
            fig = plt.figure(figsize=(16, 9))
            gs = fig.add_gridspec(3, 4, width_ratios=[5, 5, 5, 0.25])
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])
            ax4 = fig.add_subplot(gs[1, 0])
            ax5 = fig.add_subplot(gs[1, 1])
            ax6 = fig.add_subplot(gs[1, 2])
            ax7 = fig.add_subplot(gs[2, 0])
            ax8 = fig.add_subplot(gs[2, 1])
            ax9 = fig.add_subplot(gs[2, 2])
            cbar_ax = fig.add_subplot(gs[0:3, 3])

            subject_id = f'sub-{str(subject).zfill(3)}'
            figure_path_st = f'/data/p_02569/Images/1ComponentSinglePlots_CCA_DSS_Dataset1/WithUncleaned_Separated/{subject_id}/'
            os.makedirs(figure_path_st, exist_ok=True)

            for method in ['Uncleaned_PreCCA', 'ICA_PreCCA', 'SSP_PreCCA', 'Uncleaned_CCA', 'ICA_CCA', 'SSP_CCA',
                           'Uncleaned_DSS', 'ICA_DSS', 'SSP_DSS']:
                if cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                    channel = 'SC6'
                    pot = 'N13'
                    pot_lat = 0.013
                elif cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                    channel = 'L1'
                    pot = 'N22'
                    pot_lat = 0.022

                if method == 'Uncleaned_PreCCA':
                    input_path = '/data/pt_02569/tmp_data/prepared_py/' + subject_id + "/"\
                                 '/epochs_' + cond_name + '.fif'
                    epochs = mne.read_epochs(input_path, preload=True)
                    tit = f"{method.replace('_PreCCA', '')}"  # Channel {channel}
                    ax = ax1

                elif method == 'ICA_PreCCA':
                    input_path = f"/data/pt_02569/tmp_data/baseline_ica_py/{subject_id}/epochs_{cond_name}.fif"
                    epochs = mne.read_epochs(input_path, preload=True)
                    tit = f"{method.replace('_PreCCA', '')}"
                    ax = ax2

                elif method == 'SSP_PreCCA':
                    input_path = "/data/pt_02569/tmp_data/ssp_py/" + subject_id + f"/{SSP_proj} projections/epochs_" + cond_name + ".fif"
                    epochs = mne.read_epochs(input_path, preload=True)
                    tit = f"{method.replace('_PreCCA', '')}"
                    ax = ax3

                elif method == 'Uncleaned_CCA':
                    input_path = "/data/pt_02569/tmp_data/prepared_py_cca/" + subject_id + "/"
                    epochs = mne.read_epochs(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif"
                                             , preload=True)
                    ax = ax4
                    channel = df_CCA.loc[subject_id, f"Prep_{cond_name}"]
                    tit = f'Uncleaned + CCA'
                    inv = df_CCA.loc[subject_id, f"Prep_{cond_name}_inv"]
                    if inv == 'inv':
                        epochs.apply_function(invert, picks=channel)

                elif method == 'ICA_CCA':
                    input_path = f"/data/pt_02569/tmp_data/baseline_ica_py_cca/{subject_id}/"
                    fname = f"clean_baseline_ica_auto_{cond_name}.fif"
                    epochs = mne.read_epochs(input_path + fname, preload=True)
                    ax = ax5
                    channel = df_CCA.loc[subject_id, f"ICA_{cond_name}"]
                    tit = f'ICA + CCA'
                    inv = df_CCA.loc[subject_id, f"ICA_{cond_name}_inv"]
                    if inv == 'inv':
                        epochs.apply_function(invert, picks=channel)

                elif method == 'SSP_CCA':
                    input_path = f"/data/pt_02569/tmp_data/ssp_py_cca/{subject_id}/{SSP_proj} projections/"
                    epochs = mne.read_epochs(f"{input_path}ssp_cleaned_{cond_name}.fif", preload=True)
                    ax = ax6
                    channel = df_CCA.loc[subject_id, f"SSP{SSP_proj}_{cond_name}"]
                    tit = f'SSP + CCA'
                    # tit = f'{method} + CCA, Component {str(channel)[-1]}'
                    inv = df_CCA.loc[subject_id, f"SSP{SSP_proj}_{cond_name}_inv"]
                    if inv == 'inv':
                        epochs.apply_function(invert, picks=channel)

                elif method == 'Uncleaned_DSS':
                    input_path = "/data/pt_02569/tmp_data/prepared_py_dss/" + subject_id + "/"
                    epochs = mne.read_epochs(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif"
                                             , preload=True)
                    ax = ax7
                    channel = df_DSS.loc[subject_id, f"Prep_{cond_name}"]
                    tit = f'Uncleaned + DSS'
                    inv = df_DSS.loc[subject_id, f"Prep_{cond_name}_inv"]
                    if inv == 'inv':
                        epochs.apply_function(invert, picks=channel)

                elif method == 'ICA_DSS':
                    input_path = f"/data/pt_02569/tmp_data/baseline_ica_py_dss/{subject_id}/"
                    fname = f"clean_baseline_ica_auto_{cond_name}.fif"
                    epochs = mne.read_epochs(input_path + fname, preload=True)
                    ax = ax8
                    channel = df_DSS.loc[subject_id, f"ICA_{cond_name}"]
                    tit = f'ICA + DSS'
                    inv = df_DSS.loc[subject_id, f"ICA_{cond_name}_inv"]
                    if inv == 'inv':
                        epochs.apply_function(invert, picks=channel)

                elif method == 'SSP_DSS':
                    input_path = f"/data/pt_02569/tmp_data/ssp_py_dss/{subject_id}/{SSP_proj} projections/"
                    epochs = mne.read_epochs(f"{input_path}ssp_cleaned_{cond_name}.fif", preload=True)
                    ax = ax9
                    channel = df_DSS.loc[subject_id, f"SSP{SSP_proj}_{cond_name}"]
                    tit = f'SSP + DSS'
                    inv = df_DSS.loc[subject_id, f"SSP{SSP_proj}_{cond_name}_inv"]
                    if inv == 'inv':
                        epochs.apply_function(invert, picks=channel)

                # fig, ax = plt.subplots(figsize=(5.2, 7))
                cropped = epochs.copy().crop(tmin=-0.025, tmax=0.065)

                # Z-score transform each trial
                cropped.apply_function(transform, picks=channel)

                if reduced_trials:
                    cropped = cropped[trial_indices]  # Select epochs of interest

                cmap = 'plasma'
                vmin = -1
                vmax = 1
                cropped.plot_image(picks=channel, combine=None, cmap=cmap, evoked=False, show=False,
                                   colorbar=False, group_by=None,
                                   vmin=vmin, vmax=vmax, axes=ax, title=tit, scalings=dict(eeg=1))
                ax.plot(pot_lat, -8, '^', clip_on=False, color='red')

            cb = fig.colorbar(ax6.images[-1], cax=cbar_ax)

            plt.tight_layout()
            if reduced_trials:
                plt.savefig(figure_path_st + f'{cond_name}_reducedtrials.png')
                plt.savefig(figure_path_st+f'{cond_name}_reducedtrials.pdf', bbox_inches='tight', format="pdf")
            else:
                plt.savefig(figure_path_st + f'{cond_name}.png')
                plt.savefig(figure_path_st+f'{cond_name}.pdf', bbox_inches='tight', format="pdf")
            plt.close(fig)
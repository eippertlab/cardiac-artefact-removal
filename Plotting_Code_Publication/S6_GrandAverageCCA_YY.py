# Script to create plots of the grand averages evoked responses across participants for each stimulation
# For CCA corrected data

import mne
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
from invert import invert
from mpl_axes_aligner import align
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

def align_yaxis_np(axes):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = np.array(axes)
    extrema = np.array([ax.get_ylim() for ax in axes])

    # reset for divide by zero issues
    for i in range(len(extrema)):
        if np.isclose(extrema[i, 0], 0.0):
            extrema[i, 0] = -1
        if np.isclose(extrema[i, 1], 0.0):
            extrema[i, 1] = 1

    # upper and lower limits
    lowers = extrema[:, 0]
    uppers = extrema[:, 1]

    # if all pos or all neg, don't scale
    all_positive = False
    all_negative = False
    if lowers.min() > 0.0:
        all_positive = True

    if uppers.max() < 0.0:
        all_negative = True

    if all_negative or all_positive:
        # don't scale
        return

    # pick "most centered" axis
    res = abs(uppers+lowers)
    min_index = np.argmin(res)

    # scale positive or negative part
    multiplier1 = abs(uppers[min_index]/lowers[min_index])
    multiplier2 = abs(lowers[min_index]/uppers[min_index])

    for i in range(len(extrema)):
        # scale positive or negative part based on which induces valid
        if i != min_index:
            lower_change = extrema[i, 1] * -1*multiplier2
            upper_change = extrema[i, 0] * -1*multiplier1
            if upper_change < extrema[i, 1]:
                extrema[i, 0] = lower_change
            else:
                extrema[i, 1] = upper_change

        # bump by 10% for a margin
        extrema[i, 0] *= 1.1
        extrema[i, 1] *= 1.1

    # set axes limits
    [axes[i].set_ylim(*extrema[i]) for i in range(len(extrema))]


if __name__ == '__main__':
    pal = sns.color_palette(n_colors=4)
    subjects = np.arange(1, 37)   # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    image_path = "/data/p_02569/Images/GrandAverageYY_CCA_DSS_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    for data_type in ['DSS', 'CCA']:
        if data_type == 'CCA':
            xls = pd.ExcelFile('/data/p_02569/Components_CCA.xls')
            df = pd.read_excel(xls, 'Dataset 1')
            df.set_index('Subject', inplace=True)
        elif data_type == 'DSS':
            xls = pd.ExcelFile('/data/p_02569/Components_DSS.xls')
            df = pd.read_excel(xls, 'Dataset 1')
            df.set_index('Subject', inplace=True)

        trials = [True, False, True, False]
        time = [False, True, True, False]

        for reduced_trials, shorter_timescale in zip(trials, time):
            for cond_name in cond_names:  # Conditions (median, tibial)
                evoked_list_prep = []
                evoked_list_prep_cca = []
                evoked_list_ica_cca = []
                evoked_list_ssp5_cca = []

                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                    spinal_channel = 'L1'

                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                    spinal_channel = 'SC6'

                for subject in subjects:  # All subjects
                    subject_id = f'sub-{str(subject).zfill(3)}'

                    ################################################################################
                    # Fully Uncleaned
                    ################################################################################
                    input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + '/'
                    fname = f"epochs_{cond_name}.fif"
                    epochs = mne.read_epochs(input_path + fname, preload=True)
                    if reduced_trials:
                        epochs = epochs[0::4]
                    epochs = epochs.pick_channels([spinal_channel])
                    evoked = epochs.average()
                    evoked_list_prep.append(evoked)

                    ################################################################################
                    # Uncleaned CCA + DSS
                    ###############################################################################
                    if data_type == 'CCA':
                        input_path = "/data/pt_02569/tmp_data/prepared_py_cca/" + subject_id + '/'
                    elif data_type == 'DSS':
                        input_path = "/data/pt_02569/tmp_data/prepared_py_dss/" + subject_id + '/'
                    epochs = mne.read_epochs(f"{input_path}noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif"
                                             , preload=True)
                    channel = df.loc[subject_id, f"Prep_{cond_name}"]
                    inv = df.loc[subject_id, f"Prep_{cond_name}_inv"]
                    if inv == 'inv':
                        epochs.apply_function(invert, picks=channel)
                    if reduced_trials:
                        epochs = epochs[0::4]
                    epochs = epochs.pick_channels([channel])
                    evoked = epochs.average()
                    data = evoked.data
                    evoked_list_prep_cca.append(data)

                    ##############################################################################
                    # ICA CCA + DSS
                    ##############################################################################
                    if data_type == 'CCA':
                        input_path = "/data/pt_02569/tmp_data/baseline_ica_py_cca/" + subject_id + '/'
                    elif data_type == 'DSS':
                        input_path = "/data/pt_02569/tmp_data/baseline_ica_py_dss/" + subject_id + '/'
                    fname = f"clean_baseline_ica_auto_{cond_name}.fif"
                    epochs = mne.read_epochs(input_path + fname, preload=True)
                    channel = df.loc[subject_id, f"ICA_{cond_name}"]
                    inv = df.loc[subject_id, f"ICA_{cond_name}_inv"]
                    if inv == 'inv':
                        epochs.apply_function(invert, picks=channel)
                    if reduced_trials:
                        epochs = epochs[0::4]
                    epochs = epochs.pick_channels([channel])
                    evoked = epochs.average()
                    data = evoked.data
                    evoked_list_ica_cca.append(data)

                    #############################################################################
                    # SSP 5 CCA + DSS
                    #############################################################################
                    if data_type == 'CCA':
                        input_path = f"/data/pt_02569/tmp_data/ssp_py_cca/{subject_id}/5 projections/"
                    elif data_type == 'DSS':
                        input_path = f"/data/pt_02569/tmp_data/ssp_py_dss/{subject_id}/5 projections/"
                    epochs = mne.read_epochs(f"{input_path}ssp_cleaned_{cond_name}.fif", preload=True)
                    channel = df.loc[subject_id, f"SSP5_{cond_name}"]
                    inv = df.loc[subject_id, f"SSP5_{cond_name}_inv"]
                    if inv == 'inv':
                        epochs.apply_function(invert, picks=channel)
                    if reduced_trials:
                        epochs = epochs[0::4]
                    epochs = epochs.pick_channels([channel])
                    evoked = epochs.average()
                    data = evoked.data
                    evoked_list_ssp5_cca.append(data)

                #################################################################################
                # Get grand averages
                #################################################################################
                # Can't use MNE grand average cause they don't have the same channels for CCA data
                relevant_channel_prep = mne.grand_average(evoked_list_prep, interpolate_bads=False, drop_bads=False)

                relevant_channel_prep_cca = np.mean(evoked_list_prep_cca, axis=0)

                relevant_channel_ica_cca = np.mean(evoked_list_ica_cca, axis=0)

                relevant_channel_ssp5_cca = np.mean(evoked_list_ssp5_cca, axis=0)

                # Want 1 row, 3 column subplot
                # Want left y-axis to relate to cleaned heart artefact
                # Want right y-axis to relate to uncleaned heart artefact
                fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=[18, 6])
                ax10 = ax1.twinx()
                ax20 = ax2.twinx()
                ax30 = ax3.twinx()
                ax1.get_shared_y_axes().join(ax1, ax2, ax3)  # Tie primary axes
                ax10.get_shared_y_axes().join(ax10, ax20, ax30)  # Tie secondary axes

                # Uncleaned
                if data_type == 'CCA':
                    ax1.plot(relevant_channel_prep.times[:-1], relevant_channel_prep_cca[0, :],
                             color=pal[0])
                elif data_type == 'DSS':
                    ax1.plot(relevant_channel_prep.times, relevant_channel_prep_cca[0, :],
                             color=pal[0])
                ax1.set_ylabel('Cleaned SEP Amplitude (AU)')
                ax1.set_xlabel('Time (s)')
                if cond_name == 'median':
                    ax1.set_title(f'Uncleaned + {data_type}')
                ax10.plot(relevant_channel_prep.times, relevant_channel_prep.data[0, :]*10**6,
                          linewidth=0.5, linestyle='dashed', color='black')
                ax10.set_yticklabels([])

                # ICA
                if data_type == 'CCA':
                    ax2.plot(relevant_channel_prep.times[:-1], relevant_channel_ica_cca[0, :],
                             color=pal[2])
                elif data_type == 'DSS':
                    ax2.plot(relevant_channel_prep.times, relevant_channel_ica_cca[0, :],
                             color=pal[2])
                ax2.set_xlabel('Time (s)')
                if cond_name == 'median':
                    ax2.set_title(f'ICA + {data_type}')
                ax2.set_yticklabels([])
                ax20.plot(relevant_channel_prep.times, relevant_channel_prep.data[0, :]*10**6,
                          linewidth=0.5, linestyle='dashed', color='black')
                ax20.set_yticklabels([])

                # SSP6
                if data_type == 'CCA':
                    ax3.plot(relevant_channel_prep.times[:-1], relevant_channel_ssp5_cca[0, :],
                             color=pal[3])
                elif data_type == 'DSS':
                    ax3.plot(relevant_channel_prep.times, relevant_channel_ssp5_cca[0, :],
                             color=pal[3])
                ax3.set_xlabel('Time (s)')
                if cond_name == 'median':
                    ax3.set_title(f'SSP + {data_type}')
                ax3.set_yticklabels([])
                ax30.plot(relevant_channel_prep.times, relevant_channel_prep.data[0, :]*10**6,
                          linewidth=0.5, linestyle='dashed', color='black')
                ax30.set_ylabel('Uncleaned SEP Amplitude (\u03BCV)')

                # Add vertical line at expected latency
                if cond_name == 'tibial':
                    ax1.axvline(x=22 / 1000, color='k', linewidth=0.7, label='22ms')
                    ax2.axvline(x=22 / 1000, color='k', linewidth=0.7, label='22ms')
                    ax3.axvline(x=22 / 1000, color='k', linewidth=0.7, label='22ms')

                elif cond_name == 'median':
                    ax1.axvline(x=13 / 1000, color='k', linewidth=0.7, label='13ms')
                    ax2.axvline(x=13 / 1000, color='k', linewidth=0.7, label='13ms')
                    ax3.axvline(x=13 / 1000, color='k', linewidth=0.7, label='13ms')

                if shorter_timescale:
                    ax1.set_xlim([-25 / 1000, 65 / 1000])
                    ax2.set_xlim([-25 / 1000, 65 / 1000])
                    ax3.set_xlim([-25 / 1000, 65 / 1000])
                else:
                    ax1.set_xlim([-100/1000, 300/1000])
                    ax2.set_xlim([-100/1000, 300/1000])
                    ax3.set_xlim([-100/1000, 300/1000])

                if reduced_trials and shorter_timescale:
                    fname = f"{data_type}_SEPTimeCourse_{cond_name}_reducedtrials_shorter"
                elif reduced_trials and not shorter_timescale:
                    fname = f"{data_type}_SEPTimeCourse_{cond_name}_reducedtrials"
                elif shorter_timescale and not reduced_trials:
                    fname = f"{data_type}_SEPTimeCourse_{cond_name}_shorter"
                else:
                    fname = f"{data_type}_SEPTimeCourse_{cond_name}"

                # # Align y-axes
                align_yaxis_np([ax1, ax10, ax2, ax20, ax3, ax30])

                plt.tight_layout()
                # plt.show()
                # exit()
                plt.savefig(image_path+fname+'.png')
                plt.savefig(image_path+fname+'.pdf', bbox_inches='tight', format="pdf")

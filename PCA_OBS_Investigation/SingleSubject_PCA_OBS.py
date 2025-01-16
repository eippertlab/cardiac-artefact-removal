# Script to create plots of single subject evoked responses in single participants for
# PCA-OBS

import mne
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from mpl_axes_aligner import align

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
    subjects = np.arange(1, 2)   # 1 through 36 to access subject data
    cond_names = ['tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    image_path = "/data/p_02569/Images/SingleSubject_PCA-OBS_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    filter = False

    for cond_name in cond_names:  # Conditions (median, tibial)

        if cond_name == 'tibial':
            trigger_name = 'Tibial - Stimulation'
            channel = 'L1'

        elif cond_name == 'median':
            trigger_name = 'Median - Stimulation'
            channel = 'SC6'

        for subject in subjects:  # All subjects
            subject_id = f'sub-{str(subject).zfill(3)}'

            ################################################################################
            # Uncleaned
            ###############################################################################
            input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + '/'
            fname = f"epochs_{cond_name}.fif"
            if filter:
                epochs = mne.read_epochs(input_path+fname, preload=True).filter(l_freq=30, h_freq=None, n_jobs=10)
            else:
                epochs = mne.read_epochs(input_path+fname, preload=True)
            evoked_uncleaned = epochs.average()
            evoked_uncleaned.reorder_channels(esg_chans)

            ##############################################################################
            # PCA_OBS
            ##############################################################################
            input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + '/'
            fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
            raw = mne.io.read_raw_fif(input_path + fname, preload=True)
            events, event_ids = mne.events_from_annotations(raw)
            event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
            epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                baseline=tuple(iv_baseline))
            if filter:
                epochs = epochs.filter(l_freq=30, h_freq=None)

            evoked_pca = epochs.average()
            evoked_pca.reorder_channels(esg_chans)

            #################################################################################
            # Get relevant channels
            #################################################################################
            relevant_channel_prep = evoked_uncleaned.pick_channels([channel])
            relevant_channel_pca = evoked_pca.pick_channels([channel])

            # Want 1 row, 3 column subplot
            # Want left y-axis to relate to cleaned heart artefact
            # Want right y-axis to relate to uncleaned heart artefact
            fig, ax1 = plt.subplots(1, 1, figsize=[18, 6])
            ax10 = ax1.twinx()

            # PCA
            ax1.plot(relevant_channel_pca.times, relevant_channel_pca.data[0, :]*10**6, label='PCA-OBS',
                     color=pal[1])
            ax1.set_ylabel('Cleaned SEP Amplitude (\u03BCV)')
            ax1.set_xlabel('Time (s)')
            if cond_name == 'median':
                ax1.set_title('PCA-OBS')
            ax1.spines['left'].set_color(pal[1])
            ax1.tick_params(axis='y', colors=pal[1])
            ax10.plot(relevant_channel_prep.times, relevant_channel_prep.data[0, :]*10**6, label='Uncleaned',
                      linewidth=0.5, linestyle='dashed', color='blue')
            ax10.set_yticklabels([])

            # Add vertical line at expected latency
            if cond_name == 'tibial':
                ax1.axvline(x=22 / 1000, color='g', linewidth=0.7, label='22ms')

            elif cond_name == 'median':
                ax1.axvline(x=13 / 1000, color='g', linewidth=0.7, label='13ms')

            ax1.axvline(x=-0.007, color='r', linewidth=0.7)
            ax1.axvline(x=0.007, color='r', linewidth=0.7)
            ax1.set_xlim([-100/1000, 300/1000])

            if cond_name == 'median':
                plt.suptitle(f"Subject {subject}, Channel {channel}")
            else:
                plt.suptitle(f"Subject {subject}, Channel {channel}")
            plt.tight_layout()
            plt.show()
            # if filter:
            #     fname = f"{subject_id}_SEPTimeCourse_{channel}_filtered30Hz.png"
            #     plt.savefig(image_path+fname)
            # else:
            #     fname = f"{subject_id}_SEPTimeCourse_{channel}.png"
            #     plt.savefig(image_path+fname)
            # plt.savefig(image_path+fname+'.pdf', bbox_inches='tight', format="pdf")



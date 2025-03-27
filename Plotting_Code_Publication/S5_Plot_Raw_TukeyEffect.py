# Script to plot effect of tukey window
# Know from plot.raw scrolling where an edge effect is seen
# Get event timing to mark on plot
# Then plot prepared, pca and pca tukey raw traces overlaid

from scipy.io import loadmat
import mne
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.offsetbox
from matplotlib.lines import Line2D
import numpy as np
import os
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

# Testing with random subjects atm
subjects = [20]  # 20
# subjects = np.arange(1, 2)  # (1, 37) # 1 through 36 to access subject data
cond_names = ['tibial']  # ['median']
sampling_rate = 1000

cfg_path = "/data/pt_02569/"  # Contains important info about experiment
cfg = loadmat(cfg_path + 'cfg.mat')
iv_epoch = cfg['iv_epoch'][0] / 1000
iv_baseline = cfg['iv_baseline'][0] / 1000
notch_freq = cfg['notch_freq'][0]
esg_bp_freq = cfg['esg_bp_freq'][0]

image_path = "/data/p_02569/Images/PCA_tukey_comparison_images/"
os.makedirs(image_path, exist_ok=True)

tmin = 760.5
tmax = 761.5

for subject in subjects:
    subject_id = f'sub-{str(subject).zfill(3)}'

    for cond_name in cond_names:
        if cond_name == 'tibial':
            trigger_name = 'qrs'
            channels = ['S23', 'L1', 'S31']
            full_name = 'Tibial Nerve Stimulation'
            # channels = ['L1']

        elif cond_name == 'median':
            trigger_name = 'qrs'
            channels = ['S6', 'SC6', 'S14']
            full_name = 'Median Nerve Stimulation'
            # channels = ['SC6']

        # ############################################################
        # # Uncleaned
        # ############################################################
        # # Load epochs resulting from PCA OBS cleaning - the raw data in this folder has not been rereferenced
        # input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + '/'
        # raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr1000_{cond_name}_withqrs.fif", preload=True)
        # # add reference channel to data
        # raw.pick_channels(channels)
        # data_uncleaned = raw.get_data(tmin=tmin, tmax=tmax)

        ##################################################################
        # PCA_OBS
        ##################################################################
        # Some editing here to avoid plotting fit_end and fit_start
        input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + '/'
        fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
        raw.pick_channels(channels)
        data_pcaobs = raw.get_data(tmin=tmin, tmax=tmax)


        #############################################################################
        # PCA_Tukey
        ############################################################################
        input_path = "/data/pt_02569/tmp_data/ecg_rm_py_tukey/" + subject_id + '/'
        fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
        data_tukey = raw.get_data(tmin=tmin, tmax=tmax)

        ################################################################################
        # Get timings of events in cropped region of data
        ################################################################################
        raw = raw.crop(tmin=tmin, tmax=tmax)
        events, event_ids = mne.events_from_annotations(raw)
        trigger_name = ['fit_end', 'fit_start', 'qrs', 'Tibial - Stimulation']
        event_id_dict = {key: value for key, value in event_ids.items() if key in trigger_name}
        event_times = events[:, 0]/1000
        event_labels = []
        # Gets all event labels in zone of interest
        for value in events[:, 2]:
            event_labels.append([k for k, v in event_id_dict.items() if v == value][0])

        ################################################################################
        # Want all on same plot
        ###############################################################################
        offsets = [0.1, 0.10005, 0.1001]
        # Data is in shape n_channels, n_times
        fig, axes = plt.subplots(1, 1)
        for count in np.arange(0, len(offsets)):
            if count == 0:
                label = 'PCA-OBS Tukey'
            else:
                label = None
            axes.plot(np.linspace(tmin, tmax, 1000), data_tukey[count, :] + offsets[count], color='red', label=label)

        for count in np.arange(0, len(offsets)):
            if count == 0:
                label = 'PCA-OBS'
            else:
                label = None
            axes.plot(np.linspace(tmin, tmax, 1000), data_pcaobs[count, :] + offsets[count], color='blue', label=label)
        plt.yticks(ticks=[0.1, 0.10005, 0.1001], labels=channels)

        for index in np.arange(0, len(event_labels)):
            if event_labels[index] == 'fit_end':
                plt.axvline(x=event_times[index], linewidth=0.7, color='k', linestyle='dashed',
                            label=event_labels[index])

        plt.xlabel('Time (s)')
        plt.title(f'Effect of a Tukey Window\n'
                  f'Participant {subject}, {full_name}')

        plt.legend(loc='lower right')
        # Add scale bar class
        class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
            """ size: length of bar in data units
                extent : height of bar ends in axes units """

            def __init__(self, size=1, extent=0.03, label="", loc=2, ax=None,
                         pad=0.4, borderpad=0.5, ppad=0, sep=2, prop=None,
                         frameon=True, linekw={}, **kwargs):
                if not ax:
                    ax = plt.gca()
                trans = ax.get_yaxis_transform()
                size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
                line = Line2D([0, 0], [size, 0], **linekw)
                hline1 = Line2D([-extent / 2., extent / 2.], [0, 0], **linekw)
                hline2 = Line2D([-extent / 2., extent / 2.], [size, size], **linekw)
                size_bar.add_artist(line)
                size_bar.add_artist(hline1)
                size_bar.add_artist(hline2)

                txt = matplotlib.offsetbox.TextArea(label, minimumdescent=False)
                self.vpac = matplotlib.offsetbox.VPacker(children=[size_bar, txt],
                                                         align="center", pad=ppad, sep=sep)
                matplotlib.offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad,
                                                                borderpad=borderpad, child=self.vpac, prop=prop,
                                                                frameon=frameon,
                                                                **kwargs)

        # loc is one of the four corners
        ob = AnchoredHScaleBar(size=10 * 10 ** -6, label="10 \u03BCV", loc=3, frameon=False,
                               pad=1, sep=4, linekw=dict(color="black"), )
        axes.add_artist(ob)
        plt.savefig(image_path + f"EffectOfWindow_{subject_id}_{cond_name}")
        plt.savefig(image_path + f"EffectOfWindow_{subject_id}_{cond_name}.pdf", bbox_inches='tight', format="pdf")

        plt.show()
        # raw.plot(duration=1, start=784, clipping=6, scalings=30e-6)
        # plt.show()

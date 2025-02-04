# Want a 3x4 plot matrix (Timecourse, TFR, Topography) for each of (Uncleaned, PCA-OBS, ICA, SSP6)

import numpy as np
import mne
import matplotlib.pyplot as plt
from IsopotentialFunctions_Axis import mrmr_esg_isopotentialplot
from IsopotentialFunctions_2axis import mrmr_esg_isopotentialplot_cbar
from scipy.io import loadmat
import os
import matplotlib as mpl
from scipy.stats import sem
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    #  First get the evoked list
    trigger_names = ['Tibial - Stimulation', 'Median - Stimulation']
    save_path = '/data/p_02569/Images/SEP_Combined_D1V2/'
    os.makedirs(save_path, exist_ok=True)

    methods = ['Uncleaned', 'PCA', 'ICA', 'SSP']

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]
    # epoch
    iv_epoch = [-200 / 1000, 700 / 1000]
    # let's say we epoch the data from -200 to 700 ms around the trigger of interest
    iv_baseline = [-100 / 1000, -10 / 1000]

    subjects = np.arange(1, 37)
    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    trials = [True, False, True, False]
    time = [False, True, True, False]

    for reduced_trials, shorter_timescale in zip(trials, time):
        for trigger_name in trigger_names:
            count_method = 0  # To cycle through the subplots
            count_row = 0
            fig, axes = plt.subplots(3, 5, figsize=[18, 14], gridspec_kw={'width_ratios': [15, 15, 15, 15, 1]})
            axes[0, 4].axis('off')

            for method in methods:
                evoked_list = []
                data_list = []
                evoked_list_pca_extra = []

                for subj in subjects:
                    subject_id = f'sub-{str(subj).zfill(3)}'

                    if trigger_name == 'Median - Stimulation':
                        time_point = 13 / 1000
                        cond_name = 'median'
                        channel = ['SC6']
                    else:
                        cond_name = 'tibial'
                        time_point = 22 / 1000
                        channel = ['L1']

                    if method == 'Uncleaned':
                        data_path = '/data/pt_02569/tmp_data/prepared_py/' + subject_id + \
                                    '/epochs_' + cond_name + '.fif'

                    elif method == 'PCA':
                        data_path_raw = '/data/pt_02569/tmp_data/ecg_rm_py/' + subject_id + \
                                    '/data_clean_ecg_spinal_' + cond_name + '_withqrs.fif'

                        data_path_epochs = '/data/pt_02569/tmp_data/ecg_rm_py/' + subject_id + \
                                        '/epochs_' + cond_name + '.fif'

                    elif method == 'ICA':
                        data_path = '/data/pt_02569/tmp_data/baseline_ica_py/' + subject_id + \
                                    '/epochs_' + cond_name + '.fif'

                    elif method == 'SSP':
                        data_path = "/data/pt_02569/tmp_data/ssp_py/" + subject_id + f"/6 projections/epochs_" + cond_name + ".fif"

                    # for PCA - want to interpolate the hump for the TFR plot
                    if method == 'PCA':
                        # Read raw so we can interpolate the hump at 0 - save as one type of PCA for topography
                        # This will need to be filtered and rereferenced
                        raw = mne.io.read_raw_fif(data_path_raw, preload=True)
                        events, event_dict = mne.events_from_annotations(raw)
                        tstart_esg = -0.007
                        tmax_esg = 0.007
                        mne.preprocessing.fix_stim_artifact(raw, events=events, event_id=event_dict[trigger_name],
                                                            tmin=tstart_esg,
                                                            tmax=tmax_esg, mode='linear', stim_channel=None)
                        events, event_ids = mne.events_from_annotations(raw)
                        event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                        epochs_interp = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                         baseline=tuple(iv_baseline), preload=True)
                        if reduced_trials:
                            epochs_interp = epochs_interp[0::4]
                        evoked_list_pca_extra.append(epochs_interp.average())

                        # Also read in the epochs already constructed and save in normal pca evoked list
                        epochs = mne.read_epochs(data_path_epochs, preload=True).reorder_channels(esg_chans)
                        if reduced_trials:
                            epochs = epochs[0::4]
                        if subj == 34:
                            evoked = epochs.average().crop(tmin=time_point, tmax=time_point + (2 / 1000))
                            data = evoked.data.mean(axis=1)
                            ch_idx = evoked.ch_names.index("S34")
                            data = np.insert(data, ch_idx, np.nan)
                            data_list.append(data)
                        else:
                            evoked = epochs.average().crop(tmin=time_point, tmax=time_point + (2 / 1000))
                            data = evoked.data.mean(axis=1)
                            data_list.append(data)
                        evoked_list.append(epochs.average())

                    else:
                        # For all others, read in the epochs we have constructed
                        epochs = mne.read_epochs(data_path, preload=True).reorder_channels(esg_chans)
                        # Want each channel averaged across all epochs at a given time point
                        if reduced_trials:
                            epochs = epochs[0::4]
                        if subj == 34:
                            evoked = epochs.average().crop(tmin=time_point, tmax=time_point + (2 / 1000))
                            data = evoked.data.mean(axis=1)
                            ch_idx = evoked.ch_names.index("S34")
                            data = np.insert(data, ch_idx, np.nan)
                            data_list.append(data)
                        else:
                            evoked = epochs.average().crop(tmin=time_point, tmax=time_point + (2 / 1000))
                            data = evoked.data.mean(axis=1)
                            data_list.append(data)
                        evoked_list.append(epochs.average())

                options = {0: 'Uncleaned',
                           1: 'PCA-OBS',
                           2: 'ICA',
                           3: 'SSP'
                           }
                ##########################################################################################
                # Plot the time course
                ##########################################################################################
                evoked_list_ch_data = [evoked.pick(channel).data*10**6 for evoked in evoked_list]
                grand_average = np.mean(evoked_list_ch_data, axis=0)
                error = sem(evoked_list_ch_data)
                upper = grand_average + error
                lower = grand_average - error
                axes[count_row, count_method].plot(epochs.times, grand_average.reshape(-1), color='black')
                axes[count_row, count_method].fill_between(epochs.times, lower.reshape(-1), upper.reshape(-1), alpha=0.3, color='black')
                if count_method == 0:
                    axes[count_row, count_method].set_ylabel('Amplitude (\u03BCV)')
                if shorter_timescale:
                    axes[count_row, count_method].set_xlim([-0.025, 0.065])
                else:
                    axes[count_row, count_method].set_xlim([-0.1, 0.3])
                if cond_name == 'tibial':
                    axes[count_row, count_method].axvline(x=22 / 1000, color='r', linewidth=0.5, label='22ms')
                elif cond_name == 'median':
                    axes[count_row, count_method].axvline(x=13 / 1000, color='r', linewidth=0.5, label='13ms')
                axes[count_row, count_method].set_xlabel('Time (s)')
                axes[count_row, count_method].spines[['right', 'top']].set_visible(False)
                axes[count_row, count_method].set_title(options[count_method])
                count_row += 1

                ################################################################################################
                # Plot the TFR
                ################################################################################################
                freqs = np.arange(5., 400., 3.)
                fmin, fmax = freqs[[0, -1]]
                power_list = []
                for index in np.arange(0, len(evoked_list)):
                    if method == 'PCA':
                        power = mne.time_frequency.tfr_stockwell(evoked_list_pca_extra[index], fmin=fmin, fmax=fmax,
                                                                 width=1.0, n_jobs=5)
                    else:
                        power = mne.time_frequency.tfr_stockwell(evoked_list[index], fmin=fmin, fmax=fmax, width=1.0,
                                                                 n_jobs=5)
                    power = power.pick_channels(channel)
                    # evoked_list.append(evoked)
                    power_list.append(power)
                if shorter_timescale:
                    tmin = -0.025
                    tmax = 0.065
                else:
                    tmin = -0.1
                    tmax = 0.3
                vmin = -380
                vmax = -280
                averaged = mne.grand_average(power_list, interpolate_bads=False, drop_bads=False)
                averaged.plot(channel, baseline=iv_baseline, mode='mean', cmap='jet',
                              axes=axes[count_row, count_method], show=False, colorbar=False, dB=True,
                              tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax)
                if count_method != 0:
                    axes[count_row, count_method].set_ylabel(None)
                cb = fig.colorbar(axes[count_row, count_method].images[-1], cax=axes[count_row, 4])
                cb.set_label('Power (dB)')
                count_row += 1

                ##########################################################################################
                # Plot the topography
                ##########################################################################################
                # evoked = mne.grand_average(evoked_list)
                # chanvalues = evoked.crop(tmin=time_point - (1 / 1000), tmax=time_point + (2 / 1000)).data
                # chanvalues = chanvalues.mean(axis=1)  # Get average in window of interest
                arrays = [np.array(x) for x in data_list]
                chanvalues = np.array([np.nanmean(k) for k in zip(*arrays)])

                # chan_labels = evoked.ch_names
                chan_labels = esg_chans
                colorbar_axes = [-0.5, 0.5]
                subjects_4grid = np.arange(1, 37)
                # then the function takes the average over the channel positions of all those subjects
                if count_method < 3:
                    colorbar = False
                    mrmr_esg_isopotentialplot(subjects_4grid, chanvalues, colorbar_axes, chan_labels, colorbar,
                                              time_point, axes[count_row, count_method])
                else:
                    colorbar = True
                    cf = mrmr_esg_isopotentialplot_cbar(subjects_4grid, chanvalues, colorbar_axes, chan_labels,
                                                        colorbar, time_point, axes[count_row, count_method])
                    ticks = [colorbar_axes[0], 0, colorbar_axes[1]]
                    fig.colorbar(cf, cax=axes[count_row, 4], label='Amplitude (\u03BCV)', ticks=ticks)
                axes[count_row, count_method].set_xticklabels([])
                axes[count_row, count_method].set_yticklabels([])
                axes[count_row, count_method].set_xticks([])
                axes[count_row, count_method].set_yticks([])
                axes[count_row, count_method].set_ylabel('Dummy')
                axes[count_row, count_method].yaxis.label.set_color('white')
                count_row = 0
                count_method += 1

            # Add row titles
            rows = ['Time Courses', 'Time-Frequency Representations', 'Spatial Topographies']
            pad = 5
            for ax, row in zip(axes[:,0], rows):
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                            xycoords=ax.yaxis.label, textcoords='offset points',
                            size='large', ha='right', va='center', rotation=90)

            # Final Formatting
            fig.align_ylabels()
            # plt.suptitle('Somatosensory Evoked Potentials', x=0.55)
            plt.tight_layout()
            plt.subplots_adjust(left=0.15, top=0.93)
            if reduced_trials and shorter_timescale:
                fname = f"GoodTopo_Combined_{cond_name}_reducedtrials_shorter"
            elif reduced_trials and not shorter_timescale:
                fname = f"GoodTopo_Combined_{cond_name}_reducedtrials"
            elif shorter_timescale and not reduced_trials:
                fname = f"GoodTopo_Combined_{cond_name}_shorter"
            else:
                fname = f"GoodTopo_Combined_{cond_name}"

            plt.savefig(save_path+fname+'.png')
            plt.savefig(save_path+fname+'.pdf', bbox_inches='tight', format="pdf")
            # plt.show()



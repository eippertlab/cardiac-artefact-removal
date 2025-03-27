# Just want to compare the evoked responses from PCA_OBS with and without a Tukey window

import mne
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.io import loadmat
from Metrics.SNR_functions import evoked_from_raw
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    reduced_epochs = False
    single_subject = False
    grand_average = True

    if single_subject:
        subjects = [1, 20]
    if grand_average:
        subjects = np.arange(1, 37)  # (1, 37) # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]
    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000

    if single_subject:
        for subject in subjects:
            subject_id = f'sub-{str(subject).zfill(3)}'
            figure_path = "/data/p_02569/Images/PCA_tukey_comparison_images/" + subject_id + "/"
            os.makedirs(figure_path, exist_ok=True)
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                    channels = ['S23', 'L1', 'S31']
                    full_name = 'Tibial Nerve Stimulation'
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                    channels = ['S6', 'SC6', 'S14']
                    full_name = 'Median Nerve Stimulation'

                # Load epochs resulting from SSP cleaning
                input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + '/'
                input_path_tuk = "/data/pt_02569/tmp_data/ecg_rm_py_tukey/" + subject_id + '/'
                fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                raw = mne.io.read_raw_fif(f"{input_path}{fname}")
                raw_tuk = mne.io.read_raw_fif(f"{input_path_tuk}{fname}")

                for ch in channels:
                    evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
                    evoked_tuk = evoked_from_raw(raw_tuk, iv_epoch, iv_baseline, trigger_name, reduced_epochs)

                    evoked_ch = evoked.pick_channels([ch])
                    evoked_tuk_ch = evoked_tuk.pick_channels([ch])
                    plt.figure()
                    plt.plot(evoked_ch.times, evoked_ch.data.reshape(-1)*10**6, label='PCA-OBS', color='blue')
                    plt.plot(evoked_tuk_ch.times, evoked_tuk_ch.data.reshape(-1)*10**6, label='PCA-OBS Tukey',
                             color='red')
                    plt.xlim([-0.025, 0.065])
                    plt.legend()
                    plt.title(f'Subject {subject}, Channel {ch}')
                    plt.ylabel('Amplitude (\u03BCV)')
                    plt.xlabel('Time (s)')

                    if cond_name == 'tibial':
                        plt.axvline(x=22 / 1000, linewidth=0.7, linestyle='dashed', color='k', label=None)
                    elif cond_name == 'median':
                        plt.axvline(x=13 / 1000, linewidth=0.7, linestyle='dashed', color='k', label=None)

                    if reduced_epochs:
                        plt.savefig(f"{figure_path}PCA_{ch}_{cond_name}_reduced.png")
                        print('Printing reduced')
                    else:
                        plt.savefig(f"{figure_path}PCA_{ch}_{cond_name}.png")
                        print('Printing')
                    plt.close()

            plt.show()

    if grand_average:
        figure_path = "/data/p_02569/Images/PCA_tukey_comparison_images/"
        os.makedirs(figure_path, exist_ok=True)
        # method_names = ['PCA', 'PCA MATLAB', 'PCA PCHIP', 'PCA Tukey', 'PCA Tukey PCHIP']
        method_names = ['PCA', 'PCA Tukey']

        for cond_name in cond_names:  # Conditions (median, tibial)
            evoked_list_pca = []
            evoked_list_mat = []
            evoked_list_pchip = []
            evoked_list_tukey = []
            evoked_list_tukey_pchip = []

            # Changing to pick channel when each is loaded, not after the evoked list is formed
            if cond_name == 'tibial':
                trigger_name = 'Tibial - Stimulation'
                channel = ['L1']
                full_name = 'Tibial Nerve Stimulation'

            elif cond_name == 'median':
                trigger_name = 'Median - Stimulation'
                channel = ['SC6']
                full_name = 'Median Nerve Stimulation'

            for subject in subjects:  # All subjects
                subject_id = f'sub-{str(subject).zfill(3)}'

                if 'PCA' in method_names:
                    input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + '/'
                    fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                    raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                    evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
                    evoked = evoked.pick_channels(channel, ordered=True)
                    evoked_list_pca.append(evoked)

                if 'PCA MATLAB' in method_names:
                    input_path = "/data/pt_02569/tmp_data/ecg_rm/" + subject_id + '/'
                    fname = f"cnt_clean_ecg_spinal_{cond_name}.set"
                    raw = mne.io.read_raw_eeglab(input_path + fname, preload=True)
                    evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
                    evoked = evoked.pick_channels(channel, ordered=True)
                    evoked_list_mat.append(evoked)

                if 'PCA PCHIP' in method_names:
                    input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + '/'
                    fname = f"data_clean_ecg_spinal_{cond_name}_withqrs_pchip.fif"
                    raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                    evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
                    evoked = evoked.pick_channels(channel, ordered=True)
                    evoked_list_pchip.append(evoked)

                if 'PCA Tukey' in method_names:
                    input_path = "/data/pt_02569/tmp_data/ecg_rm_py_tukey/" + subject_id + '/'
                    fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                    raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                    evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
                    evoked = evoked.pick_channels(channel, ordered=True)
                    evoked_list_tukey.append(evoked)

                if 'PCA Tukey PCHIP' in method_names:
                    input_path = "/data/pt_02569/tmp_data/ecg_rm_py_tukey/" + subject_id + '/'
                    fname = f"data_clean_ecg_spinal_{cond_name}_withqrs_pchip.fif"
                    raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                    evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)
                    evoked = evoked.pick_channels(channel, ordered=True)
                    evoked_list_tukey_pchip.append(evoked)

            fig = plt.figure()
            if 'PCA' in method_names:
                relevant_channel_pca = mne.grand_average(evoked_list_pca, interpolate_bads=False, drop_bads=False)
                plt.plot(relevant_channel_pca.times, np.mean(relevant_channel_pca.data[:, :], axis=0) * 10 ** 6,
                         label='PCA-OBS', color='blue')
            if 'PCA MATLAB' in method_names:
                relevant_channel_mat = mne.grand_average(evoked_list_mat, interpolate_bads=False, drop_bads=False)
                plt.plot(relevant_channel_mat.times, np.mean(relevant_channel_mat.data[:, :], axis=0) * 10 ** 6,
                         label='PCA MATLAB')
            if 'PCA PCHIP' in method_names:
                relevant_channel_pchip = mne.grand_average(evoked_list_pchip, interpolate_bads=False, drop_bads=False)
                plt.plot(relevant_channel_pchip.times, np.mean(relevant_channel_pchip.data[:, :], axis=0) * 10 ** 6,
                         label='PCA PCHIP')
            if 'PCA Tukey' in method_names:
                relevant_channel_tukey = mne.grand_average(evoked_list_tukey, interpolate_bads=False, drop_bads=False)
                plt.plot(relevant_channel_tukey.times, np.mean(relevant_channel_tukey.data[:, :], axis=0) * 10 ** 6,
                         label='PCA-OBS Tukey', color='red')
            if 'PCA Tukey PCHIP' in method_names:
                relevant_channel_tukey_pchip = mne.grand_average(evoked_list_tukey_pchip, interpolate_bads=False, drop_bads=False)
                plt.plot(relevant_channel_tukey_pchip.times,
                         np.mean(relevant_channel_tukey_pchip.data[:, :], axis=0) * 10 ** 6,
                         label='PCA Tukey PCHIP')

            # relevant_channel = averaged.pick_channels(channel)
            plt.ylabel('Amplitude (\u03BCV)')
            plt.xlabel('Time (s)')
            plt.xlim([-0.025, 0.065])
            if cond_name == 'tibial':
                plt.axvline(x=22 / 1000, color='k', linewidth=0.7, linestyle='dashed', label=None)
                # plt.ylim([-0.5, 1.3])
            elif cond_name == 'median':
                plt.axvline(x=13 / 1000, color='k', linewidth=0.7, linestyle='dashed', label=None)
                # plt.ylim([-0.8, 0.8])
            # plt.title(f"Grand Average Evoked Response\n"
            #           f"{full_name}")
            if reduced_epochs:
                fname = f"GrandAverage_{trigger_name}_reduced.png"
            else:
                fname = f"GrandAverage_{trigger_name}.png"
            if cond_name == 'tibial':
                plt.legend(loc='upper right')
            plt.savefig(figure_path + fname)
            plt.savefig(figure_path+fname+f".pdf", bbox_inches='tight', format="pdf")
            plt.show()
            plt.clf()

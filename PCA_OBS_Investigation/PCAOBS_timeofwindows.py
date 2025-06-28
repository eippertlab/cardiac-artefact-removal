# Want to get min, max and average time between R-peak and fit_start and fit_end
# disturbed by the PCA-OBS fitting algorithm

import mne
import os
import numpy as np
import pandas as pd
import pickle
from scipy.io import loadmat
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

if __name__ == '__main__':
    subjects = np.arange(1, 37)   # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    save_path_excel = "/data/p_02569/Images/WindowTiming_PCA-OBS_Dataset1/"
    os.makedirs(save_path_excel, exist_ok=True)

    df = pd.DataFrame(columns=['Subject', 'median_fit_start', 'median_fit_end', 'median_no_windows',
                               'tibial_fit_start', 'tibial_fit_end', 'tibial_no_windows'])
    # df['Subject'] = subjects
    df.set_index('Subject', inplace=True)

    trigger_fitstart = 'fit_start'
    trigger_fitend = 'fit_end'
    trigger_heart = 'qrs'
    median = []
    tibial = []

    for cond_name in cond_names:  # Conditions (median, tibial)

        if cond_name == 'tibial':
            channel = 'L1'

        elif cond_name == 'median':
            channel = 'SC6'

        for subject in subjects:  # All subjects
            subject_id = f'sub-{str(subject).zfill(3)}'

            ##############################################################################
            # PCA_OBS
            ##############################################################################
            input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + '/'
            fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
            raw = mne.io.read_raw_fif(input_path+fname, preload=True)
            sfreq = raw.info['sfreq']

            # fit_start and fit_end are triggers related to PCA-OBS windows
            events, event_ids = mne.events_from_annotations(raw)
            event_id_dict_heart = {key: value for key, value in event_ids.items() if key == trigger_heart}
            event_id_dict_fitstart = {key: value for key, value in event_ids.items() if key == trigger_fitstart }
            event_id_dict_fitend = {key: value for key, value in event_ids.items() if key == trigger_fitend}

            samples_heart = [x[0] for x in events if x[2]==event_id_dict_heart[trigger_heart]]
            samples_fitstart = [x[0] for x in events if x[2]==event_id_dict_fitstart[trigger_fitstart]]
            samples_fitend = [x[0] for x in events if x[2]==event_id_dict_fitend[trigger_fitend]]

            # Remove any qrs events that were not fit
            remove_beginning = True
            while remove_beginning:
                if samples_fitstart[0] > samples_heart[0]:
                    samples_heart = samples_heart[1:]
                else:
                    remove_beginning = False
            remove_end = True
            while remove_end:
                if samples_fitend[-1] < samples_heart [-1]:
                    samples_heart = samples_heart[:-1]
                else:
                    remove_end = False

            samples_tostarts = np.subtract(samples_heart, samples_fitstart)

            time_tostarts = [x/sfreq for x in samples_tostarts]
            window_edge = time_tostarts[0]   # All the same so just extract first

            if cond_name == 'median':
                median.append(window_edge)
            else:
                tibial.append(window_edge)

    print(median)
    print(tibial)
    print('\n')

    print('Median:')
    print(f"minimum: {np.min(median)}")
    print(f"maximum: {np.max(median)}")
    print(f"mean: {np.mean(median)}")
    print('\n')

    print('Tibial:')
    print(f"minimum: {np.min(tibial)}")
    print(f"maximum: {np.max(tibial)}")
    print(f"mean: {np.mean(tibial)}")
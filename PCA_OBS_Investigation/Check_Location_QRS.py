# Want to count how many times the fit_start and fit_end fall within the interpolation period and would thus be
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

    save_path_excel = "/data/p_02569/Images/SingleSubject_PCA-OBS_Dataset1/"
    save_path_epochs = "/data/p_02569/Images/SingleSubject_PCA-OBS_Dataset1/AffectedEpochs_QRS/"
    os.makedirs(save_path_excel, exist_ok=True)
    os.makedirs(save_path_epochs, exist_ok=True)

    df = pd.DataFrame(columns=['Subject', 'median_overlapqrs', 'median_totalqrs',
                               'tibial_overlapqrs', 'tibial_totalqrs'])
    # df['Subject'] = subjects
    df.set_index('Subject', inplace=True)

    tinterp = 0.007
    trigger_qrs = 'qrs'

    for cond_name in cond_names:  # Conditions (median, tibial)

        if cond_name == 'tibial':
            trigger_stimulation = 'Tibial - Stimulation'
            channel = 'L1'

        elif cond_name == 'median':
            trigger_stimulation = 'Median - Stimulation'
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

            samples_before_after = tinterp*sfreq

            # fit_start and fit_end are triggers related to PCA-OBS windows
            events, event_ids = mne.events_from_annotations(raw)
            event_id_dict_stimulation = {key: value for key, value in event_ids.items() if key == trigger_stimulation}
            event_id_dict_qrs = {key: value for key, value in event_ids.items() if key == trigger_qrs}

            samples_stimulation = [x[0] for x in events if x[2]==event_id_dict_stimulation[trigger_stimulation]]
            samples_qrs = [x[0] for x in events if x[2]==event_id_dict_qrs[trigger_qrs]]

            # Now we have extracted samples, check how many times fit_start and fit end are in the interpolation period
            count = 0
            epoch_indices = []
            for sample_win in samples_qrs:
                for sample_stim in samples_stimulation:
                    if abs(sample_win - sample_stim) <= samples_before_after:
                        count +=1
                        epoch_indices.append(samples_stimulation.index(sample_stim))
            df.at[f'{subject}', f'{cond_name}_overlapqrs'] = count
            df.at[f'{subject}', f'{cond_name}_totalqrs'] = len(samples_qrs)

            # Save epochs affected by qrs
            # Epoch indices are not necessarily all unique as fit_start of one window and fit_end of another window may
            # overlap the interpolation period of the same trial
            # print(epoch_indices)
            rfile = open(save_path_epochs + f'{subject_id}_{cond_name}_affectedepochs_qrs.pkl', 'wb')
            pickle.dump(epoch_indices, rfile)
            rfile.close()

    # print(df)
    with pd.ExcelWriter(save_path_excel+"QRS_Overlap.xlsx", engine='openpyxl') as writer:
        # use to_excel function and specify the sheet_name and index
        # to store the dataframe in specified sheet
        df.to_excel(writer, sheet_name="QRS_Overlap", index=True)

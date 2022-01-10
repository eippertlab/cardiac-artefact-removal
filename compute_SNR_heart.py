# Script to quantify the SNR of the heart artefact
# The signal in this case is the peak of the qrs waveform, and the noise is std in baseline period
# QRS complex usually has duration of 0.06 to 0.10 seconds - search this area for maximal positive peak
# Then noise calculated in 100ms period preceding the artefact
# SNR = peak of QRS/std of baseline period
# Doing in all channels for now, may reduce to only channels of interest

# Could potentially make sense to only look at the reduction in the heart artefact at relevant channels

import mne
import numpy as np
import h5py
from SNR_functions import calculate_heart_SNR_evoked, evoked_from_raw
from scipy.io import loadmat
from epoch_data import rereference_data
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Testing with just subject 1 at the moment
    subjects = np.arange(1, 37) # (1, 2) # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000
    iv_baseline = [-150/1000, -50/1000]
    iv_epoch = [-200/1000, 200/1000]

    # Set which to run
    calc_raw_snr = False
    calc_PCA_snr = False
    calc_post_ICA_snr = False
    calc_ICA_snr = True
    calc_SSP_snr = True
    reduced_window = True  # STILL NEED TO SET WINDOW IN FUNCTION FILE
    ant_ref = False  # Use the data that has been anteriorly referenced instead
    reduced_epochs = False  # Dummy variable - always false in this script as I don't reduce epochs

    # Run SNR calc on prepared data - heart artefact NOT removed here
    ############################################## Raw SNR Calculations ########################################
    if calc_raw_snr:
        # Declare class to hold ecg fit information
        class save_SNR():
            def __init__(self):
                pass

        # Instantiate class
        savesnr = save_SNR()

        # Matrix of dimensions no.subjects x no. projections
        snr_med = np.zeros((len(subjects), 1))
        snr_tib = np.zeros((len(subjects), 1))

        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                    nerve = 2
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                    nerve = 1

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Load data resulting from preparation script
                input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
                fname = f"noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)

                # add reference channel to data
                if ant_ref:
                    # anterior reference
                    if nerve == 1:
                        raw = rereference_data(raw, 'AC')
                    elif nerve == 2:
                        raw = rereference_data(raw, 'AL')
                else:
                    mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place

                cfg_path = "/data/pt_02569/"  # Contains important info about experiment
                cfg = loadmat(cfg_path + 'cfg.mat')
                notch_freq = cfg['notch_freq'][0]
                esg_bp_freq = cfg['esg_bp_freq'][0]
                raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                           iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

                raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, 'qrs', reduced_epochs)

                snr = calculate_heart_SNR_evoked(evoked, cond_name, iv_baseline)

                # Now have one snr related to each subject and condition
                if cond_name == 'median':
                    snr_med[subject - 1, 0] = snr
                elif cond_name == 'tibial':
                    snr_tib[subject - 1, 0] = snr

        # Save to file to compare to matlab - only for debugging
        savesnr.snr_med = snr_med
        savesnr.snr_tib = snr_tib
        dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]
        if reduced_window:
            if ant_ref:
                fn = f"/data/pt_02569/tmp_data/prepared_py/snr_heart_ant_smallwin.h5"
            else:
                fn = f"/data/pt_02569/tmp_data/prepared_py/snr_heart_smallwin.h5"
        else:
            if ant_ref:
                fn = f"/data/pt_02569/tmp_data/prepared_py/snr_heart_ant.h5"
            else:
                fn = f"/data/pt_02569/tmp_data/prepared_py/snr_heart.h5"

        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savesnr, keyword))
        # print(snr_med)
        # print(snr_tib)

    ############################################ PCA SNR Calculations ########################################
    if calc_PCA_snr:
        # Declare class to hold ecg fit information
        class save_SNR():
            def __init__(self):
                pass


        # Instantiate class
        savesnr = save_SNR()

        # Matrix of dimensions no.subjects x no. projections
        snr_med = np.zeros((len(subjects), 1))
        snr_tib = np.zeros((len(subjects), 1))

        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                    nerve = 2
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'
                    nerve = 1

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Data in this folder hasn't been filtered and rereferenced - do it here instead
                input_path = "/data/pt_02569/tmp_data/ecg_rm_py/" + subject_id + "/esg/prepro/"
                fname = f"data_clean_ecg_spinal_{cond_name}_withqrs.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                # add reference channel to data
                if ant_ref:
                    # anterior reference
                    if nerve == 1:
                        raw = rereference_data(raw, 'AC')
                    elif nerve == 2:
                        raw = rereference_data(raw, 'AL')
                else:
                    mne.add_reference_channels(raw, ref_channels=['TH6'], copy=False)  # Modifying in place

                cfg_path = "/data/pt_02569/"  # Contains important info about experiment
                cfg = loadmat(cfg_path + 'cfg.mat')
                notch_freq = cfg['notch_freq'][0]
                esg_bp_freq = cfg['esg_bp_freq'][0]
                raw.filter(l_freq=esg_bp_freq[0], h_freq=esg_bp_freq[1], n_jobs=len(raw.ch_names), method='iir',
                           iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

                raw.notch_filter(freqs=notch_freq, n_jobs=len(raw.ch_names), method='fir', phase='zero')

                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, 'qrs', reduced_epochs)

                snr = calculate_heart_SNR_evoked(evoked, cond_name, iv_baseline)

                # Now have one snr related to each subject and condition
                if cond_name == 'median':
                    snr_med[subject - 1, 0] = snr
                elif cond_name == 'tibial':
                    snr_tib[subject - 1, 0] = snr

        # Save to file to compare to matlab - only for debugging
        savesnr.snr_med = snr_med
        savesnr.snr_tib = snr_tib
        dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]
        if reduced_window:
            if ant_ref:
                fn = f"/data/pt_02569/tmp_data/ecg_rm_py/snr_heart_ant_smallwin.h5"
            else:
                fn = f"/data/pt_02569/tmp_data/ecg_rm_py/snr_heart_smallwin.h5"
        else:
            if ant_ref:
                fn = f"/data/pt_02569/tmp_data/ecg_rm_py/snr_heart_ant.h5"
            else:
                fn = f"/data/pt_02569/tmp_data/ecg_rm_py/snr_heart.h5"
        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savesnr, keyword))

        # print(snr_med)
        # print(snr_tib)

    ######################################## PCA + ICA SNR Calculations ########################################
    if calc_post_ICA_snr:
        # Declare class to hold ecg fit information
        class save_SNR():
            def __init__(self):
                pass

        # Instantiate class
        savesnr = save_SNR()

        # Matrix of dimensions no.subjects x no. projections
        snr_med = np.zeros((len(subjects), 1))
        snr_tib = np.zeros((len(subjects), 1))

        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Load data resulting from preparation script
                input_path = "/data/pt_02569/tmp_data/ica_py/" + subject_id + "/esg/prepro/"
                if ant_ref:
                    fname = f"clean_ica_auto_antRef_{cond_name}.fif"
                else:
                    fname = f"clean_ica_auto_{cond_name}.fif"
                raw = mne.io.read_raw_fif(input_path + fname)

                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, 'qrs', reduced_epochs)

                snr = calculate_heart_SNR_evoked(evoked, cond_name, iv_baseline)

                # Now have one snr related to each subject and condition
                if cond_name == 'median':
                    snr_med[subject - 1, 0] = snr
                elif cond_name == 'tibial':
                    snr_tib[subject - 1, 0] = snr

        # Save to file to compare to matlab - only for debugging
        savesnr.snr_med = snr_med
        savesnr.snr_tib = snr_tib
        dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]
        if reduced_window:
            if ant_ref:
                fn = f"/data/pt_02569/tmp_data/ica_py/snr_heart_ant_smallwin.h5"
            else:
                fn = f"/data/pt_02569/tmp_data/ica_py/snr_heart_smallwin.h5"
        else:
            if ant_ref:
                fn = f"/data/pt_02569/tmp_data/ica_py/snr_heart_ant.h5"
            else:
                fn = f"/data/pt_02569/tmp_data/ica_py/snr_heart.h5"
        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savesnr, keyword))
        # print(snr_med)
        # print(snr_tib)

    ######################################## ICA SNR Calculations ########################################
    if calc_ICA_snr:
        # Declare class to hold ecg fit information
        class save_SNR():
            def __init__(self):
                pass

        # Instantiate class
        savesnr = save_SNR()

        # Matrix of dimensions no.subjects x no. projections
        snr_med = np.zeros((len(subjects), 1))
        snr_tib = np.zeros((len(subjects), 1))

        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR
                # Load data resulting from preparation script
                input_path = "/data/pt_02569/tmp_data/baseline_ica_py/" + subject_id + "/esg/prepro/"
                if ant_ref:
                    fname = f"clean_baseline_ica_auto_antRef_{cond_name}.fif"
                else:
                    fname = f"clean_baseline_ica_auto_{cond_name}.fif"
                raw = mne.io.read_raw_fif(input_path + fname)

                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, 'qrs', reduced_epochs)

                snr = calculate_heart_SNR_evoked(evoked, cond_name, iv_baseline)

                # Now have one snr related to each subject and condition
                if cond_name == 'median':
                    snr_med[subject - 1, 0] = snr
                elif cond_name == 'tibial':
                    snr_tib[subject - 1, 0] = snr

        # Save to file to compare to matlab - only for debugging
        savesnr.snr_med = snr_med
        savesnr.snr_tib = snr_tib
        dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]
        if reduced_window:
            if ant_ref:
                fn = f"/data/pt_02569/tmp_data/baseline_ica_py/snr_heart_ant_smallwin.h5"
            else:
                fn = f"/data/pt_02569/tmp_data/baseline_ica_py/snr_heart_smallwin.h5"
        else:
            if ant_ref:
                fn = f"/data/pt_02569/tmp_data/baseline_ica_py/snr_heart_ant.h5"
            else:
                fn = f"/data/pt_02569/tmp_data/baseline_ica_py/snr_heart.h5"

        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savesnr, keyword))
        # print(snr_med)
        # print(snr_tib)

    ############################### SSP Projectors SNR #################################################
    if calc_SSP_snr:
        # Declare class to hold ecg fit information
        class save_SNR():
            def __init__(self):
                pass

        # Instantiate class
        savesnr = save_SNR()

        # Matrix of dimensions no.subjects x no. projections
        snr_med = np.zeros((len(subjects), len(np.arange(5, 21))))
        snr_tib = np.zeros((len(subjects), len(np.arange(5, 21))))

        for subject in subjects:
            for cond_name in cond_names:
                if cond_name == 'tibial':
                    trigger_name = 'Tibial - Stimulation'
                elif cond_name == 'median':
                    trigger_name = 'Median - Stimulation'

                subject_id = f'sub-{str(subject).zfill(3)}'

                # Want the SNR for each projection tried from 5 to 20
                for n in np.arange(5, 21):
                    # Load SSP projection data
                    input_path = "/data/p_02569/SSP/" + subject_id
                    savename = input_path + "/" + str(n) + " projections/"
                    if ant_ref:
                        raw = mne.io.read_raw_fif(f"{savename}ssp_cleaned_{cond_name}_antRef.fif")
                    else:
                        raw = mne.io.read_raw_fif(f"{savename}ssp_cleaned_{cond_name}.fif")

                    evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, 'qrs', reduced_epochs)

                    snr = calculate_heart_SNR_evoked(evoked, cond_name, iv_baseline)

                    # Now have one snr for relevant channel in each subject + condition
                    if cond_name == 'median':
                        snr_med[subject-1, n-5] = snr
                    elif cond_name == 'tibial':
                        snr_tib[subject - 1, n-5] = snr

        # Save to file to compare to matlab - only for debugging
        savesnr.snr_med = snr_med
        savesnr.snr_tib = snr_tib
        dataset_keywords = [a for a in dir(savesnr) if not a.startswith('__')]
        if reduced_window:
            if ant_ref:
                fn = f"/data/p_02569/SSP/snr_heart_ant_smallwin.h5"
            else:
                fn = f"/data/p_02569/SSP/snr_heart_smallwin.h5"
        else:
            if ant_ref:
                fn = f"/data/p_02569/SSP/snr_heart_ant.h5"
            else:
                fn = f"/data/p_02569/SSP/snr_heart.h5"

        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(savesnr, keyword))
        # print(snr_med)
        # print(snr_tib)

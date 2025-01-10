##############  File to compute the residual intensity after cleaning  #################
# BCG residual intensity was quantified by calculating the root mean square (RMS) of the EEG data,
# epoched over windows of 600 ms after the ECG peaks and averaged. The percentage of BCG residual intensity
# was defined as the ratio between RMS of EEG data after and before BCG correction, respectively.
# – From the adaptive optimal basis sets paper

from scipy.io import loadmat
import numpy as np
import mne
import h5py
from SNR_functions import evoked_from_raw

if __name__ == '__main__':
    choose_limited = False  # If true, use data where only top 4 components chosen - use FALSE, see main
    reduced_epochs = False  # Dummy variable - always false in this script as I don't reduce epochs

    # Define the channel names so they come out of each dataset the same
    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    subjects = np.arange(1, 37)  # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')

    # Want 200ms before R-peak and 400ms after R-peak
    # Baseline is the 100ms period before the artefact occurs
    iv_baseline = [-300 / 1000, -200 / 1000]
    # Want 200ms before and 400ms after the R-peak in our epoch - need baseline outside this
    iv_epoch = [-300 / 1000, 400 / 1000]
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    # Loop through methods and save as required
    which_method = {'Prep': False,
                    'PCA': False,
                    'PCA Tukey': False,
                    'ICA': False,
                    'ICA-Anterior': False,
                    'ICA-Separate': False,
                    'SSP': False}

    for i in np.arange(0, len(which_method)):
        method = list(which_method.keys())[i]
        if which_method[method]:  # If this method is true, go through with the rest
            class save_res():
                def __init__(self):
                    pass

            # Instantiate class
            saveres = save_res()

            if method == 'SSP':
                for n in np.arange(1, 21):  # 5, 7 most used
                    # Instantiate class
                    saveres = save_res()

                    # Matrix of dimensions no.subjects x no. channels
                    globals()[f"res_med_SSP_{n}"] = np.zeros((len(subjects), 39))
                    globals()[f"res_tib_SSP_{n}"] = np.zeros((len(subjects), 39))

                    for subject in subjects:
                        for cond_name in cond_names:
                            if cond_name == 'tibial':
                                trigger_name = 'qrs'
                                nerve = 2
                            elif cond_name == 'median':
                                trigger_name = 'qrs'
                                nerve = 1

                            subject_id = f'sub-{str(subject).zfill(3)}'

                            # Want the RMS of the data, load data
                            input_path = "/data/pt_02569/tmp_data/ssp_py/" + subject_id
                            savename = input_path + "/" + str(n) + " projections/"
                            raw = mne.io.read_raw_fif(f"{savename}ssp_cleaned_{cond_name}.fif")
                            evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)

                            # Now we have an evoked potential about the heartbeat
                            # Want to compute the RMS for each channel
                            res_chan_SSP = []
                            for ch in esg_chans:
                                # Pick a single channel
                                evoked_ch = evoked.copy().pick_channels([ch], ordered=False)
                                data = evoked_ch.data[0, 0:]  # Format n_channels x n_times
                                rms = np.sqrt(np.mean(data ** 2))
                                res_chan_SSP.append(rms)

                            # Now have rms for each subject, for each channel and condition
                            if cond_name == 'median':
                                globals()[f"res_med_SSP_{n}"][subject - 1, :] = res_chan_SSP
                            elif cond_name == 'tibial':
                                globals()[f"res_tib_SSP_{n}"][subject - 1, :] = res_chan_SSP

                    # Save to file
                    saveres.res_med = globals()[f"res_med_SSP_{n}"]
                    saveres.res_tib = globals()[f"res_tib_SSP_{n}"]
                    dataset_keywords = [a for a in dir(saveres) if not a.startswith('__')]

                    fn = f"/data/pt_02569/tmp_data/ssp_py/res_{n}.h5"

                    with h5py.File(fn, "w") as outfile:
                        for keyword in dataset_keywords:
                            outfile.create_dataset(keyword, data=getattr(saveres, keyword))
            else:
                # Matrix of dimensions no.subjects x no. channels
                res_med = np.zeros((len(subjects), 39))
                res_tib = np.zeros((len(subjects), 39))
                for subject in subjects:
                    for cond_name in cond_names:
                        if cond_name == 'tibial':
                            trigger_name = 'qrs'
                            nerve = 2
                        elif cond_name == 'median':
                            trigger_name = 'qrs'
                            nerve = 1

                        subject_id = f'sub-{str(subject).zfill(3)}'

                        # Get the right file path
                        if method == 'Prep':
                            file_path = "/data/pt_02569/tmp_data/prepared_py/"
                            file_name = f'noStimart_sr1000_{cond_name}_withqrs.fif'
                        elif method == 'PCA':
                            file_path = "/data/pt_02569/tmp_data/ecg_rm_py/"
                            file_name = f'data_clean_ecg_spinal_{cond_name}_withqrs.fif'
                        elif method == 'PCA Tukey':
                            file_path = "/data/pt_02569/tmp_data/ecg_rm_py_tukey/"
                            file_name = f'data_clean_ecg_spinal_{cond_name}_withqrs.fif'
                        elif method == 'ICA':
                            file_path = "/data/pt_02569/tmp_data/baseline_ica_py/"
                            if choose_limited:
                                file_name = f'clean_baseline_ica_auto_{cond_name}_lim.fif'
                            else:
                                file_name = f'clean_baseline_ica_auto_{cond_name}.fif'
                        elif method == 'ICA-Anterior':
                            file_path = "/data/pt_02569/tmp_data/baseline_ica_py/"
                            file_name = f"anterior_clean_baseline_ica_auto_{cond_name}.fif"
                        elif method == 'ICA-Separate':
                            file_path = "/data/pt_02569/tmp_data/baseline_ica_py/"
                            file_name = f"separated_clean_baseline_ica_auto_{cond_name}.fif"

                        input_path = file_path + subject_id + "/"
                        raw = mne.io.read_raw_fif(f"{input_path}{file_name}", preload=True)

                        evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs)

                        # Now we have an evoked potential about the heartbeat
                        # Want to compute the RMS for each channel
                        res_chan = []
                        for ch in esg_chans:
                            # Pick a single channel
                            evoked_ch = evoked.copy().pick_channels([ch], ordered=False)
                            data = evoked_ch.data[0, 0:]  # Format n_channels x n_times
                            rms = np.sqrt(np.mean(data ** 2))
                            res_chan.append(rms)

                        # Now have rms for each subject, for each channel and condition
                        if cond_name == 'median':
                            res_med[subject - 1, :] = res_chan
                        elif cond_name == 'tibial':
                            res_tib[subject - 1, :] = res_chan

                # Save to file
                saveres.res_med = res_med
                saveres.res_tib = res_tib
                dataset_keywords = [a for a in dir(saveres) if not a.startswith('__')]
                if method == 'ICA' and choose_limited:
                    fn = f"{file_path}res_lim.h5"
                elif method == 'ICA-Anterior':
                    fn = f"{file_path}res_anteriorICA.h5"
                elif method == 'ICA-Separate':
                    fn = f"{file_path}res_separateICA.h5"
                else:
                    fn = f"{file_path}res.h5"
                with h5py.File(fn, "w") as outfile:
                    for keyword in dataset_keywords:
                        outfile.create_dataset(keyword, data=getattr(saveres, keyword))

    ###################################################################################
    # Now we have all of the RMS values needed, we can calculate the residual intensity
    ###################################################################################
    input_paths = {'PCA': "/data/pt_02569/tmp_data/ecg_rm_py/",
                   'PCA Tukey': "/data/pt_02569/tmp_data/ecg_rm_py_tukey/",
                   'ICA': "/data/pt_02569/tmp_data/baseline_ica_py/",
                   'ICA-Anterior': "/data/pt_02569/tmp_data/baseline_ica_py/",
                   'ICA-Separate': "/data/pt_02569/tmp_data/baseline_ica_py/",
                   'SSP': "/data/pt_02569/tmp_data/ssp_py/"}

    # All files are 36x39 dimensions - n_subjects x n_channels
    keywords = ['res_med', 'res_tib']
    fn = f"/data/pt_02569/tmp_data/prepared_py/res.h5"
    with h5py.File(fn, "r") as infile:
        # Get the data
        res_med_prep = infile[keywords[0]][()]
        res_tib_prep = infile[keywords[1]][()]

    print("\n")
    print('All Channels Residual Intensity')
    print('Warning: For ICA separated, all channels are included but only the relevant patch has been processed via ICA,'
          'the numbers are therefore only accurate for the relevant channels computations')
    for i in np.arange(0, len(input_paths)):
        name = list(input_paths.keys())[i]
        input_path = input_paths[name]

        if name == 'SSP':
            # SSP
            for n in np.arange(1, 21):  # 5, 21
                fn = f"/data/pt_02569/tmp_data/ssp_py/res_{n}.h5"
                with h5py.File(fn, "r") as infile:
                    # Get the data
                    res_med = infile[keywords[0]][()]
                    res_tib = infile[keywords[1]][()]

                residual_med = (np.mean(res_med / res_med_prep, axis=tuple([0, 1]))) * 100
                residual_tib = (np.mean(res_tib / res_tib_prep, axis=tuple([0, 1]))) * 100

                print(f"Residual SSP Median {n}: {residual_med:.4f}%")
                print(f"Residual SSP Tibial {n}: {residual_tib:.4f}%")
        else:
            if name == 'ICA' and choose_limited:
                fn = f"{input_path}res_lim.h5"
            elif name == 'ICA-Anterior':
                fn = f"{input_path}res_anteriorICA.h5"
            elif name == 'ICA-Separate':
                fn = f"{input_path}res_separateICA.h5"
            else:
                fn = f"{input_path}res.h5"
            with h5py.File(fn, "r") as infile:
                # Get the data
                res_med = infile[keywords[0]][()]
                res_tib = infile[keywords[1]][()]

            residual_med = (np.mean(res_med / res_med_prep, axis=tuple([0, 1]))) * 100
            residual_tib = (np.mean(res_tib / res_tib_prep, axis=tuple([0, 1]))) * 100

            print(f'Residual {name} Median: {residual_med:.4f}%')
            print(f'Residual {name} Tibial: {residual_tib:.4f}%')

    ############################################################################################
    # Now look at residual intensity for just our channels of interest
    #     if cond_name == 'tibial':
    #         channels = ['S23', 'L1', 'S31']
    #     elif cond_name == 'median':
    #         channels = ['S6', 'SC6', 'S14']
    ###########################################################################################
    median_pos = []
    tibial_pos = []
    for channel in ['S23', 'L1', 'S31']:
        tibial_pos.append(esg_chans.index(channel))
    for channel in ['S6', 'SC6', 'S14']:
        median_pos.append(esg_chans.index(channel))

    print("\n")
    print('Relevant Channels Residual Intensity')
    for i in np.arange(0, len(input_paths)):
        name = list(input_paths.keys())[i]
        input_path = input_paths[name]

        if name == 'SSP':
            # SSP
            for n in np.arange(1, 21):  # 5, 21
                fn = f"/data/pt_02569/tmp_data/ssp_py/res_{n}.h5"
                with h5py.File(fn, "r") as infile:
                    # Get the data
                    res_med = infile[keywords[0]][()]
                    res_tib = infile[keywords[1]][()]

                residual_med = (np.mean(res_med[:, median_pos] / res_med_prep[:, median_pos], axis=tuple([0, 1]))) * 100
                residual_tib = (np.mean(res_tib[:, tibial_pos] / res_tib_prep[:, tibial_pos], axis=tuple([0, 1]))) * 100

                print(f"Residual SSP Median {n}: {residual_med:.4f}%")
                print(f"Residual SSP Tibial {n}: {residual_tib:.4f}%")
        else:
            if name == 'ICA' and choose_limited:
                fn = f"{input_path}res_lim.h5"
            elif name == 'ICA-Anterior':
                fn = f"{input_path}res_anteriorICA.h5"
            elif name == 'ICA-Separate':
                fn = f"{input_path}res_separateICA.h5"
            else:
                fn = f"{input_path}res.h5"
            with h5py.File(fn, "r") as infile:
                # Get the data
                res_med = infile[keywords[0]][()]
                res_tib = infile[keywords[1]][()]

            residual_med = (np.mean(res_med[:, median_pos] / res_med_prep[:, median_pos], axis=tuple([0, 1]))) * 100
            residual_tib = (np.mean(res_tib[:, tibial_pos] / res_tib_prep[:, tibial_pos], axis=tuple([0, 1]))) * 100

            print(f'Residual {name} Median: {residual_med:.4f}%')
            print(f'Residual {name} Tibial: {residual_tib:.4f}%')


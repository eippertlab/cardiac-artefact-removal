###########################################################
# Emma Bailey, 08/11/2021
# This is a wrapper script for the pipeline
###########################################################

import numpy as np
from rm_heart_artefact import rm_heart_artefact
from rm_heart_artefact_tukey import rm_heart_artefact_tukey
from SSP import apply_SSP
from import_data import import_data
from ICA import run_ica
from ICA_anterior import run_ica_anterior
from ICA_separated import run_ica_separatepatches
from run_CCA_cardiacartefact import run_CCA_heart
from run_DSS_cardiacartefact import run_DSS_heart
from run_CCA import run_CCA

if __name__ == '__main__':
    n_subjects = 36  # Number of subjects
    subjects = np.arange(1, 37)  # 1 through 36 to access subject data
    # subjects = [1]
    srmr_nr = 1  # Experiment Number
    conditions = [2, 3]  # Conditions of interest
    sampling_rate = 1000

    ######## Data import ############
    import_d = False
    pchip_interpolation = False  # If true import with pchip, otherwise use linear interpolation (False is standard in this project)

    ######## PCA-OBS #########
    heart_removal = False
    pchip = False  # Whether to use pchip prepared data or not (False is standard in this project)
    heart_removal_tukey = False  # Fitted artefact multiplied by tukey window

    ######### ICA ########
    ica = False
    ica_anterior = False  # Run ICA on anteriorly rereferenced data
    ica_separate_patches = False  # Run ICA on lumbar and cervical patches separately

    ######### SSP ########
    SSP_flag = False

    ######## CCA for cardiac artefact ########
    CCA_heart_flag = True

    ######## DSS for cardiac artefact ########
    DSS_heart_flag = True

    ######## CCA for signal enhancement ########
    CCA_flag = False

    ###############################################################################################################
    # Import Data from BIDS directory
    # Select channels to analyse
    # Remove stimulus artefact if needed
    # Downsample and concatenate blocks of the same conditions
    # Detect QRS events
    # Save the new data and the QRS events
    ##############################################################################################################
    if import_d:
        for subject in subjects:
            for condition in conditions:
                import_data(subject, condition, srmr_nr, sampling_rate, pchip_interpolation)

    ##############################################################################################################
    # To remove heart artifact via PCA_OBS
    ##############################################################################################################
    if heart_removal:
        for subject in subjects:
            for condition in conditions:
                rm_heart_artefact(subject, condition, srmr_nr, sampling_rate, pchip)
                # If pchip is true, uses data where stim artefact was removed by pchip

    ##############################################################################################################
    # To remove heart artifact via PCA_OBS and tukey window
    ##############################################################################################################
    if heart_removal_tukey:
        for subject in subjects:
            for condition in conditions:
                rm_heart_artefact_tukey(subject, condition, srmr_nr, sampling_rate, pchip)

    ##############################################################################################################
    # Run ICA on data
    ##############################################################################################################
    # choose_limited better false - SNR is worse if it's true, over 95% residual intensity and inps under 1.4
    if ica:
        for choose_limited in [True, False]: # If true only take the top 6 ICA components from find_bads_ecg
            for subject in subjects:
                for condition in conditions:
                    run_ica(subject, condition, srmr_nr, sampling_rate, choose_limited)

    if ica_anterior:
        for subject in subjects:
            for condition in conditions:
                run_ica_anterior(subject, condition, srmr_nr, sampling_rate)

    if ica_separate_patches:
        for subject in subjects:
            for condition in conditions:
                run_ica_separatepatches(subject, condition, srmr_nr, sampling_rate)

    ##############################################################################################################
    # To remove heart artifact using SSP method in MNE
    ##############################################################################################################
    if SSP_flag:
        for subject in subjects:
            for condition in conditions:
                apply_SSP(subject, condition, srmr_nr, sampling_rate)

    ##############################################################################################################
    # Run CCA on the data for cardiac artefact removal (runs images for removing 1-20 comps if True)
    ##############################################################################################################
    plot_images = False
    if CCA_heart_flag:
            for subject in subjects:
                for condition in conditions:
                    run_CCA_heart(subject, condition, srmr_nr, plot_images)

    ##############################################################################################################
    # Run DSS on the data for cardiac artefact removal (runs images for removing 1-20 comps if True)
    ##############################################################################################################
    plot_images = False
    if DSS_heart_flag:
        for subject in subjects:
            for condition in conditions:
                run_DSS_heart(subject, condition, srmr_nr, plot_images)

    ##############################################################################################################
    # Run CCA on the data for signal enhancement
    ##############################################################################################################
    data_strings = ['Prep', 'PCA']  # 'ICA' not used due to how decimated the signal is
    n = 5
    if CCA_flag:
        for data_string in data_strings:
            for subject in subjects:
                for condition in conditions:
                    run_CCA(subject, condition, srmr_nr, data_string, n)

        # Treat SSP separately
        data_string = 'SSP'
        for n in np.arange(5, 7):  # 21
            for subject in subjects:
                for condition in conditions:
                    run_CCA(subject, condition, srmr_nr, data_string, n)

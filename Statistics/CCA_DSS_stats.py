# File to compute 1x3 rmANOVAs
# 1st ANOVAs: If we are doing CCA or DSS - should the data be uncleaned, ica or ssp
# 2nd ANOVAs: If we are doing x method - should I leave the data, do CCA or do DSS
# Post-hoc permutation t-testing possible

import mne
import pandas as pd
import numpy as np
import h5py
from itertools import combinations
from pingouin import rm_anova
import os
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


if __name__ == '__main__':
    # Read in results files and format in a pandas dataframe
    # Set file locations
    method = 'CoV'  # can be CoV or SNR

    # Read in results files and format in a pandas dataframe
    # Set file locations
    if method == 'CoV':
        fname = 'variance.h5'
        keywords = ['var_med', 'var_tib']
    elif method == 'SNR':
        fname = 'snr.h5'
        keywords = ['snr_med', 'snr_tib']

    prep_path = '/data/pt_02569/tmp_data/prepared_py/'
    ica_path = '/data/pt_02569/tmp_data/baseline_ica_py/'
    ssp_path = '/data/pt_02569/tmp_data/ssp_py/'

    prep_path_cca = '/data/pt_02569/tmp_data/prepared_py_cca/'
    ica_path_cca = '/data/pt_02569/tmp_data/baseline_ica_py_cca/'
    ssp_path_cca = '/data/pt_02569/tmp_data/ssp_py_cca/'

    prep_path_dss = '/data/pt_02569/tmp_data/prepared_py_dss/'
    ica_path_dss = '/data/pt_02569/tmp_data/baseline_ica_py_dss/'
    ssp_path_dss = '/data/pt_02569/tmp_data/ssp_py_dss/'

    ################################# Make Dataframe ###################################
    # Have all options in one, then we can break into the smaller frames we need for each test
    file_paths = [prep_path, ica_path, ssp_path, prep_path_cca, ica_path_cca, ssp_path_cca, prep_path_dss,
                  ica_path_dss, ssp_path_dss]
    names = ['Uncleaned', 'ICA', 'SSP', 'Uncleaned_CCA', 'ICA_CCA', 'SSP_CCA', 'Uncleaned_DSS', 'ICA_DSS', 'SSP_DSS']

    # Pull each subjects value out
    count = 0
    if method == 'SNR':
        # Pull each subjects value out
        count = 0
        for file_path in file_paths:
            with h5py.File(file_path + fname, "r") as infile:
                if file_path == prep_path:
                    snr_med = infile[keywords[0]][()].reshape(-1)
                    data_med = {'Uncleaned': snr_med}
                    df_med = pd.DataFrame(data_med, index=np.arange(1, 37))

                    snr_tib = infile[keywords[1]][()].reshape(-1)
                    data_tib = {'Uncleaned': snr_tib}
                    df_tib = pd.DataFrame(data_tib, index=np.arange(1, 37))

                # For SSP all 20 projectors SNR are stored in one file - needs to be pulled separately
                elif file_path == ssp_path:
                    snr_med = infile[keywords[0]][()]
                    df_med[f'{names[count]}'] = snr_med[:, 4]
                    snr_tib = infile[keywords[1]][()]
                    df_tib[f'{names[count]}'] = snr_tib[:, 4]

                else:
                    # Get the data
                    snr_med = infile[keywords[0]][()].reshape(-1)
                    df_med[names[count]] = snr_med

                    snr_tib = infile[keywords[1]][()].reshape(-1)
                    df_tib[names[count]] = snr_tib

                count += 1

    elif method == 'CoV':
        count = 0
        for file_path in file_paths:
            with h5py.File(file_path + fname, "r") as infile:
                if file_path == prep_path:  # To start the dataframe
                    cov_med = infile[keywords[0]][()].reshape(-1)
                    data_med = {'Uncleaned': cov_med}
                    df_med = pd.DataFrame(data_med, index=np.arange(1, 37))

                    cov_tib = infile[keywords[1]][()].reshape(-1)
                    data_tib = {'Uncleaned': cov_tib}
                    df_tib = pd.DataFrame(data_tib, index=np.arange(1, 37))

                else:
                    # Get the data
                    cov_med = infile[keywords[0]][()].reshape(-1)
                    df_med[names[count]] = cov_med

                    cov_tib = infile[keywords[1]][()].reshape(-1)
                    df_tib[names[count]] = cov_tib

                count += 1

    ###################### Get mean and sem of columns ####################
    print('Median Means')
    print(df_med.mean())
    print('Median Standard Error')
    print(df_med.sem())
    print('Tibial Means')
    print(df_tib.mean())
    print('Tibial Standard Error')
    print(df_tib.sem())

    for test_idx in np.arange(1, 7):
        ###############################################################################################
        # Test 1:
        # Uncleaned, Uncleaned-CCA, Uncleaned-DSS
        ###############################################################################################
        if test_idx == 1:
            col_names = ['Uncleaned', 'Uncleaned_CCA', 'Uncleaned_DSS']

        ###############################################################################################
        # Test 2:
        # ICA, ICA-CCA, ICA-DSS
        ###############################################################################################
        if test_idx == 2:
            col_names = ['ICA', 'ICA_CCA', 'ICA_DSS']

        ###############################################################################################
        # Test 3:
        # SSP, SSP-CCA, SSP-DSS
        ###############################################################################################
        if test_idx == 3:
            col_names = ['SSP', 'SSP_CCA', 'SSP_DSS']

        ###############################################################################################
        # Test 4:
        # Uncleaned, ICA, SSP
        ###############################################################################################
        if test_idx == 4:
            col_names = ['Uncleaned', 'ICA', 'SSP']

        ###############################################################################################
        # Test 5:
        # Uncleaned-CCA, ICA-CCA, SSP-CCA
        ###############################################################################################
        if test_idx == 5:
            col_names = ['Uncleaned_CCA', 'ICA_CCA', 'SSP_CCA']

        ###############################################################################################
        # Test 6:
        # Uncleaned-DSS, ICA-DSS, SSP-DSS
        ###############################################################################################
        if test_idx == 6:
            col_names = ['Uncleaned_DSS', 'ICA_DSS', 'SSP_DSS']

        # ANOVA test for median and tibial
        for condition, df in zip(['median', 'tibial'], [df_med, df_tib]):
            df_test = df[col_names]  # Sub-select bits of interest
            aov = df_test.rm_anova()
            print('\n')
            print(f"Testing {method}, {col_names}, {condition}")
            print(aov)

            # If rm_anova was significant, perform post-hoc t-tests
            if aov['p-unc'].loc[aov.index[0]] < 0.05:
                ###################### Do median and tibial permutation statistics ##########################
                df_test = df_test.dropna()

                cc = list(combinations(df_test.columns, 2))  # All combinations
                # cc = [el for el in cc if el[0] == method]  # Just ones with certain method in first position
                df_comb = pd.concat([df_test[c[1]].sub(df_test[c[0]]) for c in cc], axis=1, keys=cc)
                df_comb.columns = pd.Series(cc).map('-'.join)
                arr = df_comb.to_numpy()
                # print(df_comb.describe())
                T_obs, p_values, H0 = mne.stats.permutation_t_test(arr, n_permutations=10000, n_jobs=36)

                formatted_pvals = {}
                colnames = df_comb.columns
                for index in np.arange(0, len(p_values)):
                    formatted_pvals.update({colnames[index]: p_values[index]})

                df_pvals = pd.DataFrame.from_dict(formatted_pvals, orient='index')
                print(f"{condition} Corrected P-Values")
                print(df_pvals)


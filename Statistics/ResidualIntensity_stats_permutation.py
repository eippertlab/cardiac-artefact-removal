# File to compute p-values via permutation t-testing for the Residual Intensity Results

import mne
import pandas as pd
import numpy as np
from pingouin import rm_anova
import h5py
from itertools import combinations
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


if __name__ == '__main__':
    # Read in results files and format in a pandas dataframe
    # Set file locations
    fname = 'res.h5'
    prep_path = '/data/pt_02569/tmp_data/prepared_py/'
    pca_path = '/data/pt_02569/tmp_data/ecg_rm_py/'
    ica_path = '/data/pt_02569/tmp_data/baseline_ica_py/'
    ssp_path = '/data/pt_02569/tmp_data/ssp_py/'
    ccaheart_path = '/data/pt_02569/tmp_data/cca_heartart_py/'
    dssheart_path = '/data/pt_02569/tmp_data/dss_heartart_py/'

    ################################# Make Dataframe ###################################
    # Necessary information to get relavant channel data
    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    median_pos = []
    tibial_pos = []
    for channel in ['S23', 'L1', 'S31']:
        tibial_pos.append(esg_chans.index(channel))
    for channel in ['S6', 'SC6', 'S14']:
        median_pos.append(esg_chans.index(channel))

    ################################# Make Dataframe ###################################
    # Set up dataframe for stats
    file_paths = [prep_path, pca_path, ica_path, ssp_path, ccaheart_path, dssheart_path]
    names = ['Prep', 'PCA', 'ICA', 'SSP', 'CCA-heart', 'DSS-heart']
    # Pull each subjects value out
    keywords = ['res_med', 'res_tib']
    count = 0
    for file_path in file_paths:
        if file_path == ssp_path:
            fname_med = 'res_5.h5'
            fname_tib = 'res_5.h5'
        elif file_path == ccaheart_path:
            fname_med = 'res_9.h5'
            fname_tib = 'res_6.h5'
        elif file_path == dssheart_path:
            fname_med = 'res_9.h5'
            fname_tib = 'res_7.h5'
        else:
            fname_med = 'res.h5'
            fname_tib = 'res.h5'

        # Need the prep values to do all the divisions thereafter
        if file_path == prep_path:  # Just extract values from file to use later
            with h5py.File(file_path + fname_med, "r") as infile:
                val_prep_med = infile[keywords[0]][()]
            with h5py.File(file_path + fname_tib, "r") as infile:
                val_prep_tib = infile[keywords[1]][()]

        elif file_path == pca_path:  # Start the dataframe
            # Get the data
            with h5py.File(file_path + fname_med, "r") as infile:
                val_med = infile[keywords[0]][()]
                res_med_current = (np.mean(val_med[:, median_pos] / val_prep_med[:, median_pos],
                                           axis=1)) * 100
                data_med = {'PCA': res_med_current}
                df_med = pd.DataFrame(data_med, index=np.arange(1, 37))

            with h5py.File(file_path + fname_tib, "r") as infile:
                val_tib = infile[keywords[1]][()]
                res_tib_current = (np.mean(val_tib[:, tibial_pos] / val_prep_tib[:, tibial_pos],
                                           axis=1)) * 100
                data_tib = {'PCA': res_tib_current}
                df_tib = pd.DataFrame(data_tib, index=np.arange(1, 37))

        else:
            with h5py.File(file_path + fname_med, "r") as infile:
                val_med = infile[keywords[0]][()]
                res_med_current = (np.mean(val_med[:, median_pos] / val_prep_med[:, median_pos],
                                           axis=1)) * 100
                df_med[names[count]] = res_med_current

            with h5py.File(file_path + fname_tib, "r") as infile:
                val_tib = infile[keywords[1]][()]
                res_tib_current = (np.mean(val_tib[:, tibial_pos] / val_prep_tib[:, tibial_pos],
                                           axis=1)) * 100
                df_tib[names[count]] = res_tib_current

        count += 1


    #################################### Dataframe of Differences #################################
    ###################### Get mean and sem of columns ####################
    print('Median Means')
    print(df_med.mean())
    print('Median Standard Error')
    print(df_med.sem())
    print('Tibial Means')
    print(df_tib.mean())
    print('Tibial Standard Error')
    print(df_tib.sem())

    # ANOVA test for median and tibial
    for condition, df in zip(['median', 'tibial'], [df_med, df_tib]):
        aov = df.rm_anova()
        print('\n')
        print(condition)
        print(aov)

        # If rm_anova was significant, perform post-hoc t-tests
        if aov['p-unc'].loc[aov.index[0]] < 0.05:
            ###################### Do median and tibial permutation statistics ##########################
            df = df.dropna()

            cc = list(combinations(df.columns, 2))  # All combinations
            # cc = [el for el in cc if el[0] == method]  # Just ones with certain method in first position
            df_comb = pd.concat([df[c[1]].sub(df[c[0]]) for c in cc], axis=1, keys=cc)
            df_comb.columns = pd.Series(cc).map('-'.join)
            arr = df_comb.to_numpy()
            # print(df_comb.describe())
            T_obs, p_values, H0 = mne.stats.permutation_t_test(arr, n_permutations=2000, n_jobs=36)

            formatted_pvals = {}
            colnames = df_comb.columns
            for index in np.arange(0, len(p_values)):
                formatted_pvals.update({colnames[index]: p_values[index]})

            df_pvals = pd.DataFrame.from_dict(formatted_pvals, orient='index')
            print(f"{condition} Corrected P-Values")
            print(df_pvals)
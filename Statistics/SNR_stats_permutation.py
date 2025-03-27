# File to compute p-values via permutation t-testing for the SNR results

import mne
import pandas as pd
import numpy as np
import h5py
from pingouin import rm_anova, ptests
from itertools import combinations
import os
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


if __name__ == '__main__':
    # Read in results files and format in a pandas dataframe
    # Set file locations
    fname = 'snr.h5'
    prep_path = '/data/pt_02569/tmp_data/prepared_py/'
    pca_path = '/data/pt_02569/tmp_data/ecg_rm_py/'
    ica_path = '/data/pt_02569/tmp_data/baseline_ica_py/'
    ssp_path = '/data/pt_02569/tmp_data/ssp_py/'
    ccaheart_path = '/data/pt_02569/tmp_data/cca_heartart_py/'
    dssheart_path = '/data/pt_02569/tmp_data/dss_heartart_py/'

    ################################# Make Dataframe ###################################
    file_paths = [prep_path, pca_path, ica_path, ssp_path, ccaheart_path, dssheart_path]
    names = ['Prep', 'PCA', 'ICA', 'SSP', 'CCA-heart', 'DSS-heart']
    # Pull each subjects value out
    keywords = ['snr_med', 'snr_tib']
    count = 0
    for file_path in file_paths:
        with h5py.File(file_path+fname, "r") as infile:
            if file_path == prep_path:
                snr_med = infile[keywords[0]][()].reshape(-1)
                data_med = {'Prep': snr_med}
                df_med = pd.DataFrame(data_med, index=np.arange(1, 37))

                snr_tib = infile[keywords[1]][()].reshape(-1)
                data_tib = {'Prep': snr_tib}
                df_tib = pd.DataFrame(data_tib, index=np.arange(1, 37))

            elif file_path in [ssp_path, ccaheart_path, dssheart_path]:
                # These have shape (n_subjects, n_projectors) - need to select correct no. of projectors
                if file_path == ssp_path:
                    snr_med = infile[keywords[0]][()]
                    df_med[f'{names[count]}'] = snr_med[:, 4]
                    snr_tib = infile[keywords[1]][()]
                    df_tib[f'{names[count]}'] = snr_tib[:, 4]

                elif file_path == ccaheart_path:
                    snr_med = infile[keywords[0]][()]
                    df_med[f'{names[count]}'] = snr_med[:, 8]
                    snr_tib = infile[keywords[1]][()]
                    df_tib[f'{names[count]}'] = snr_tib[:, 5]

                elif file_path == dssheart_path:
                    snr_med = infile[keywords[0]][()]
                    df_med[f'{names[count]}'] = snr_med[:, 8]
                    snr_tib = infile[keywords[1]][()]
                    df_tib[f'{names[count]}'] = snr_tib[:, 6]

            else:
                # Get the data
                snr_med = infile[keywords[0]][()].reshape(-1)
                df_med[names[count]] = snr_med

                snr_tib = infile[keywords[1]][()].reshape(-1)
                df_tib[names[count]] = snr_tib

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

    #################################### Dataframe of Differences #################################
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
            T_obs, p_values, H0 = mne.stats.permutation_t_test(arr, n_permutations=10000, n_jobs=36)

            formatted_pvals = {}
            colnames = df_comb.columns
            for index in np.arange(0, len(p_values)):
                formatted_pvals.update({colnames[index]: p_values[index]})

            df_pvals = pd.DataFrame.from_dict(formatted_pvals, orient='index')
            print(f"{condition} Corrected P-Values")
            print(df_pvals)


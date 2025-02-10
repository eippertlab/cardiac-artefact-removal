# File to compute p-values via permutation t-testing for the CCA and DSS CoV results

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
    data_type = 'DSS-SEP'  # CCA-SEP or DSS-SEP

    # Read in results files and format in a pandas dataframe
    # Set file locations
    fname = 'variance.h5'

    if data_type == 'CCA-SEP':
        prep_path = '/data/pt_02569/tmp_data/prepared_py_cca/'
        ica_path = '/data/pt_02569/tmp_data/baseline_ica_py_cca/'
        ssp_path = '/data/pt_02569/tmp_data/ssp_py_cca/'

    elif data_type == 'DSS-SEP':
        prep_path = '/data/pt_02569/tmp_data/prepared_py_dss/'
        ica_path = '/data/pt_02569/tmp_data/baseline_ica_py_dss/'
        ssp_path = '/data/pt_02569/tmp_data/ssp_py_dss/'

    ################################# Make Dataframe ###################################
    file_paths = [prep_path, ica_path, ssp_path]
    names = ['Prep', 'ICA', 'SSP']

    # Pull each subjects value out
    keywords = ['var_med', 'var_tib']
    count = 0
    for file_path in file_paths:
        with h5py.File(file_path + fname, "r") as infile:
            if file_path == prep_path:
                snr_med = infile[keywords[0]][()].reshape(-1)
                data_med = {'Prep': snr_med}
                df_med = pd.DataFrame(data_med, index=np.arange(1, 37))

                snr_tib = infile[keywords[1]][()].reshape(-1)
                data_tib = {'Prep': snr_tib}
                df_tib = pd.DataFrame(data_tib, index=np.arange(1, 37))

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
            T_obs, p_values, H0 = mne.stats.permutation_t_test(arr, n_permutations=2000, n_jobs=36)

            formatted_pvals = {}
            colnames = df_comb.columns
            for index in np.arange(0, len(p_values)):
                formatted_pvals.update({colnames[index]: p_values[index]})

            df_pvals = pd.DataFrame.from_dict(formatted_pvals, orient='index')
            print(f"{condition} Corrected P-Values")
            print(df_pvals)


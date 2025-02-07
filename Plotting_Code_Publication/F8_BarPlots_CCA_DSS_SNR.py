# Get the standard error of the mean across subjects for coeff of variation and plot


import pandas as pd
import matplotlib.pyplot as plt
import h5py
import colorsys
import numpy as np
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    fname = 'snr.h5'
    keywords = ['snr_med', 'snr_tib']

    names = ['Uncleaned', 'ICA', 'SSP']
    cca_paths = ['/data/pt_02569/tmp_data/prepared_py_cca/', '/data/pt_02569/tmp_data/baseline_ica_py_cca/',
                 '/data/pt_02569/tmp_data/ssp_py_cca/']
    dss_paths = ['/data/pt_02569/tmp_data/prepared_py_dss/', '/data/pt_02569/tmp_data/baseline_ica_py_dss/',
                 '/data/pt_02569/tmp_data/ssp_py_dss/']
    paths = ['/data/pt_02569/tmp_data/prepared_py/', '/data/pt_02569/tmp_data/baseline_ica_py/',
                 '/data/pt_02569/tmp_data/ssp_py/']

    ########################################################################################
    # Extract Relevant Values
    ########################################################################################
    df_med_uncleaned = pd.DataFrame(index=np.arange(1, 37))
    df_tib_uncleaned = pd.DataFrame(index=np.arange(1, 37))
    df_med_cca = pd.DataFrame(index=np.arange(1, 37))
    df_tib_cca = pd.DataFrame(index=np.arange(1, 37))
    df_med_dss = pd.DataFrame(index=np.arange(1, 37))
    df_tib_dss = pd.DataFrame(index=np.arange(1, 37))

    data_type_counter = 0
    for file_paths in [paths, cca_paths, dss_paths]:
        file_counter = 0
        for file_path in file_paths:
            name = names[file_counter]
            with h5py.File(file_path + fname, "r") as infile:
                if file_path == '/data/pt_02569/tmp_data/ssp_py/':
                    snr_med = infile[keywords[0]][()][:, 5]
                    snr_tib = infile[keywords[1]][()][:, 5]
                else:
                    snr_med = infile[keywords[0]][()]
                    snr_tib = infile[keywords[1]][()]

                if data_type_counter == 0:
                    df_med_uncleaned[name] = snr_med
                    df_tib_uncleaned[name] = snr_tib
                elif data_type_counter == 1:
                    df_med_cca[name] = snr_med
                    df_tib_cca[name] = snr_tib
                elif data_type_counter == 2:
                    df_med_dss[name] = snr_med
                    df_tib_dss[name] = snr_tib

            file_counter += 1
        data_type_counter += 1

    # ###################### Get mean and sem of columns ####################
    print('\nCCA')
    print('Median Means')
    print(df_med_cca.mean())
    print('Median Standard Error')
    print(df_med_cca.sem())
    print('Tibial Means')
    print(df_tib_cca.mean())
    print('Tibial Standard Error')
    print(df_tib_cca.sem())

    print('\nDSS')
    print('Median Means')
    print(df_med_dss.mean())
    print('Median Standard Error')
    print(df_med_dss.sem())
    print('Tibial Means')
    print(df_tib_dss.mean())
    print('Tibial Standard Error')
    print(df_tib_dss.sem())

    print('\nPrior')
    print('Median Means')
    print(df_med_uncleaned.mean())
    print('Median Standard Error')
    print(df_med_uncleaned.sem())
    print('Tibial Means')
    print(df_tib_uncleaned.mean())
    print('Tibial Standard Error')
    print(df_tib_uncleaned.sem())

    for df in [df_med_cca, df_tib_cca, df_med_dss, df_tib_dss, df_med_uncleaned, df_tib_uncleaned]:
        df.rename({'Prepared':'Uncleaned'}, axis=1, inplace=True)

    # Get dataframe in form easy to plot
    # Means
    df_mean = pd.DataFrame(columns=['Method', 'Median_uncleaned', 'Median_cca', 'Median_dss',
                                    'Tibial_uncleaned', 'Tibial_cca', 'Tibial_dss'])
    df_mean['Method'] = names
    df_sem = pd.DataFrame(columns=['Method', 'Median_uncleaned', 'Median_cca', 'Median_dss',
                                    'Tibial_uncleaned', 'Tibial_cca', 'Tibial_dss'])
    df_sem['Method'] = names
    for name, df_check in zip(['Median_uncleaned', 'Median_cca', 'Median_dss',
                                'Tibial_uncleaned', 'Tibial_cca', 'Tibial_dss'],
                               [df_med_uncleaned, df_med_cca, df_med_dss,
                                df_tib_uncleaned, df_tib_cca, df_tib_dss]):
        df_mean[name] = df_check.mean().values
        df_sem[name] = df_check.sem().values
    df_mean.set_index('Method', inplace=True)
    df_sem.set_index('Method', inplace=True)

    colours = ['darkgreen', 'seagreen', 'lightgreen']

    for condition in ['Median', 'Tibial']:
        # CCA coeff of variation plots
        methods_toplot = names
        columns = [f'{condition}_uncleaned', f'{condition}_cca', f'{condition}_dss']
        df_cropped_mean = df_mean.loc[methods_toplot, columns]
        df_cropped_sem = df_sem.loc[methods_toplot, columns]
        fig, ax = plt.subplots(1, 1)
        ax.set_xticklabels(methods_toplot)
        ax.set_ylabel('Signal-to-Noise Ratio (AU)')
        ax.set_title(f'{condition} Nerve Stimulation')
        df_cropped_mean.plot.bar(ax=ax, rot=15, color=colours, sharex=True, legend=0, yerr=df_cropped_sem, capsize=4)
        plt.legend(['Before CCA/DSS', 'After CCA', 'After DSS'], loc='upper right')
        plt.tight_layout()

        plt.savefig(f'/data/pt_02569/ResultsComparison/Images/SNR_{condition}_D1.png')
        plt.savefig(f'/data/pt_02569/ResultsComparison/Images/SNR_{condition}_D1.pdf', bbox_inches='tight', format="pdf")
    plt.show()

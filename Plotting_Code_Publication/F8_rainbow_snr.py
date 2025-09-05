# Generate rainbow plots of SEP SNR

import pandas as pd
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    ######################### SEP SNR #########################
    keywords = ['snr_med', 'snr_tib']
    input_paths = ["/data/pt_02569/tmp_data/prepared_py/",
                   "/data/pt_02569/tmp_data/ecg_rm_py/",
                   "/data/pt_02569/tmp_data/baseline_ica_py/",
                   "/data/pt_02569/tmp_data/ssp_py/",
                   "/data/pt_02569/tmp_data/cca_heartart_py/",
                   "/data/pt_02569/tmp_data/dss_heartart_py/"]

    names = ['Prepared', 'PCA', 'ICA', 'SSP', 'CCA', 'DSS']
    snr_list_med = {}
    snr_list_tib = {}
    for i, input_path in enumerate(input_paths):
        name = names[i]
        fn = f"{input_path}snr.h5"

        # All have shape (36, 1) bar SSP which is (36, n_proj)
        with h5py.File(fn, "r") as infile:
            if name in ['SSP', 'CCA', 'DSS']:
                snr_med = infile[keywords[0]][()]
                snr_tib = infile[keywords[1]][()]
                for i, (data_med, data_tib) in enumerate(zip(snr_med.T, snr_tib.T)):
                    snr_list_med[f'{name}_{i+1}'] = data_med
                    snr_list_tib[f'{name}_{i+1}'] = data_tib
            else:
                # Get the data
                snr_med = infile[keywords[0]][()].reshape(-1)
                snr_tib = infile[keywords[1]][()].reshape(-1)
                snr_list_med[name] = snr_med
                snr_list_tib[name] = snr_tib

        i += 1

    df_med = pd.DataFrame(snr_list_med)
    df_med.rename(columns={'Prepared': 'Uncleaned', 'SSP_5': 'SSP', 'PCA': 'PCA-OBS', 'CCA_9': 'CCA-cardiac', 'DSS_9': 'DSS-cardiac'}, inplace=True)
    df_med = df_med[['Uncleaned', 'PCA-OBS', 'ICA', 'SSP', 'CCA-cardiac', 'DSS-cardiac']]
    df_tib = pd.DataFrame(snr_list_tib)
    df_tib.rename(columns={'Prepared': 'Uncleaned', 'SSP_5': 'SSP', 'PCA': 'PCA-OBS', 'CCA_6': 'CCA-cardiac', 'DSS_7': 'DSS-cardiac'}, inplace=True)
    df_tib = df_tib[['Uncleaned', 'PCA-OBS', 'ICA', 'SSP', 'CCA-cardiac', 'DSS-cardiac']]

    df_med_long = df_med.melt(var_name='Method', value_name='SNR (A.U.)')
    df_tib_long = df_tib.melt(var_name='Method', value_name='SNR (A.U.)')

    dy = "SNR (A.U.)"
    dx = "Method"
    ort = "v"
    pallette = sns.color_palette(n_colors=10)
    pal = []
    for idx, color in enumerate(pallette):
        if idx in [0, 1, 2, 3, 4, 9]:
            pal.append(color)
    i = 0
    conditons = ['median', 'tibial']
    for df in [df_med_long, df_tib_long]:
        cond_name = conditons[i]
        i += 1
        f, ax = plt.subplots(figsize=(8, 8))
        ax = pt.half_violinplot(x=dx, y=dy, data=df, palette=pal, bw=.2, cut=0.,
                                scale="area", width=.6, inner=None, orient=ort,
                                linewidth=0.0)
        ax = sns.stripplot(x=dx, y=dy, data=df, palette=pal, edgecolor="white",
                           size=3, jitter=1, zorder=0, orient=ort)
        # ax = sns.boxplot(x=dx, y=dy, data=df, color="black", width=.15, zorder=10,
        #                  showcaps=True, boxprops={'facecolor': 'none', "zorder": 10},
        #                  showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10},
        #                  saturation=1, orient=ort)
        # ax = ax.set_ylim[(0, 25)]
        plt.ylim([0, 45])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        if cond_name == 'median':
            plt.savefig(f'/data/pt_02569/ResultsComparison/Images/Rainbow_{cond_name}')
            plt.savefig(f'/data/pt_02569/ResultsComparison/Images/Rainbow_{cond_name}.pdf', bbox_inches='tight',
                        format="pdf")
        else:
            plt.savefig(f'/data/pt_02569/ResultsComparison/Images/Rainbow_{cond_name}')
            plt.savefig(f'/data/pt_02569/ResultsComparison/Images/Rainbow_{cond_name}.pdf', bbox_inches='tight',
                        format="pdf")
    plt.show()

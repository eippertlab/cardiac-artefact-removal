# Look at elbow plots for SNR, RI and INSPR to see where effect starts to fall off
# Checking for SSP, CCA cardiac and DSS cardiac


import h5py
import matplotlib.pyplot as plt
from math import log10
from kneed import KneeLocator
import numpy as np
import pandas as pd
import matplotlib as mpl
import os
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    input_paths = {'SSP': "/data/pt_02569/tmp_data/ssp_py/",
                   'CCA_heart': "/data/pt_02569/tmp_data/cca_heartart_py/",
                   'DSS_heart': "/data/pt_02569/tmp_data/dss_heartart_py/"
                   }
    image_path = "/data/p_02569/Images/ElbowPlots/"
    os.makedirs(image_path, exist_ok=True)

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
    projectors = np.arange(1, 21)

    for method in ['SSP', 'CCA_heart', 'DSS_heart']:
        #######################################################################
        # SNR
        #######################################################################
        keywords = ['snr_med', 'snr_tib']
        input_path = input_paths[method]
        fn = f"{input_path}snr.h5"
        # (36, 16)
        with h5py.File(fn, "r") as infile:
            # Get the data
            snr_med = np.nanmean(infile[keywords[0]][()], axis=0)
            snr_tib = np.nanmean(infile[keywords[1]][()], axis=0)

        #######################################################################
        # INPSR
        #######################################################################
        # All files are 36x39 dimensions - n_subjects x n_channels
        keywords = ['pow_med', 'pow_tib']
        inps_med = []
        inps_tib = []
        fn = f"/data/pt_02569/tmp_data/prepared_py/inps_yasa.h5"
        with h5py.File(fn, "r") as infile:
            # Get the data
            pow_med_prep = infile[keywords[0]][()]
            pow_tib_prep = infile[keywords[1]][()]

            input_path = input_paths[method]

        for n in np.arange(1, 21):
            fn = f"{input_path}inps_yasa_{n}.h5"
            with h5py.File(fn, "r") as infile:
                # Get the data
                pow_med = infile[keywords[0]][()]
                pow_tib = infile[keywords[1]][()]

            # shape (n_subjects, n_rel_ch) - average INPSR over relevant channels for each subject and then log convert
            all_subj_inpsr_med = np.mean(pow_med_prep[:, median_pos] / pow_med[:, median_pos], axis=1)
            all_subj_inpsr_tib = np.mean(pow_tib_prep[:, tibial_pos] / pow_tib[:, tibial_pos], axis=1)
            log_inpsr_med = [log10(inps) for inps in all_subj_inpsr_med]
            log_inpsr_tib = [log10(inps) for inps in all_subj_inpsr_tib]
            inps_med.append(np.mean(log_inpsr_med))
            inps_tib.append(np.mean(log_inpsr_tib))

        #######################################################################
        # RI
        #######################################################################
        keywords = ['res_med', 'res_tib']
        residual_med = []
        residual_tib = []
        fn = f"/data/pt_02569/tmp_data/prepared_py/res.h5"
        with h5py.File(fn, "r") as infile:
            # Get the data
            res_med_prep = infile[keywords[0]][()]
            res_tib_prep = infile[keywords[1]][()]

        for n in np.arange(1, 21):  # 5, 21
            fn = f"{input_path}res_{n}.h5"
            with h5py.File(fn, "r") as infile:
                # Get the data
                res_med = infile[keywords[0]][()]
                res_tib = infile[keywords[1]][()]

            residual_med.append((np.mean(res_med[:, median_pos] / res_med_prep[:, median_pos], axis=tuple([0, 1]))) * 100)
            residual_tib.append((np.mean(res_tib[:, tibial_pos] / res_tib_prep[:, tibial_pos], axis=tuple([0, 1]))) * 100)

        # Images
        fig, axes = plt.subplots(1, 3, figsize=[18, 6])
        axes = axes.flatten()
        plt.suptitle(f"{method}, median")
        axes[0].scatter(projectors, snr_med)
        axes[0].axvline(list(snr_med).index(np.max(snr_med))+1, color='red', label=f"{list(snr_med).index(np.max(snr_med))+1}")
        axes[0].legend()
        axes[0].set_xlabel('Number of Projectors')
        axes[0].set_ylabel('Signal-to-Noise Ratio (AU)')
        axes[0].set_title('Signal-to-Noise Ratio')
        axes[0].set_xticks(projectors)
        axes[1].scatter(projectors, residual_med)
        kn = KneeLocator(projectors, residual_med, curve='convex', direction='decreasing')
        axes[1].axvline(kn.knee, color='red', label=f"{kn.knee}")
        axes[1].legend()
        axes[1].set_xlabel('Number of Projectors')
        axes[1].set_ylabel('Residual Intensity (%)')
        axes[1].set_title('Residual Intensity')
        axes[1].set_xticks(projectors)
        axes[2].scatter(projectors, inps_med)
        kn = KneeLocator(projectors, inps_med, curve='concave', direction='increasing')
        axes[2].axvline(kn.knee, color='red', label=f"{kn.knee}")
        axes[2].legend()
        axes[2].set_xlabel('Number of Projectors')
        axes[2].set_ylabel('Log(INPSR) (AU)')
        axes[2].set_title('Improved Normalised Power Spectrum Ratio')
        axes[2].set_xticks(projectors)
        plt.tight_layout()
        plt.savefig(image_path + f"{method}_median.png")
        plt.savefig(image_path + f"{method}_median" + '.pdf', bbox_inches='tight', format="pdf")

        fig, axes = plt.subplots(1, 3, figsize=[18, 6])
        axes = axes.flatten()
        plt.suptitle(f"{method}, tibial")
        axes[0].scatter(projectors, snr_tib)
        axes[0].axvline(list(snr_tib).index(np.max(snr_tib)) + 1, color='red', label=f"{list(snr_tib).index(np.max(snr_tib)) + 1}")
        axes[0].legend()
        axes[0].set_xlabel('Number of Projectors')
        axes[0].set_ylabel('Signal-to-Noise Ratio (AU)')
        axes[0].set_title('Signal-to-Noise Ratio')
        axes[0].set_xticks(projectors)
        axes[1].scatter(projectors, residual_tib)
        kn = KneeLocator(projectors, residual_tib, curve='convex', direction='decreasing')
        axes[1].axvline(kn.knee, color='red', label=f"{kn.knee}")
        axes[1].legend()
        axes[1].set_xlabel('Number of Projectors')
        axes[1].set_ylabel('RI (%)')
        axes[1].set_title('Residual Intensity')
        axes[1].set_xticks(projectors)
        axes[2].scatter(projectors, inps_tib)
        kn = KneeLocator(projectors, inps_tib, curve='concave', direction='increasing')
        axes[2].axvline(kn.knee, color='red', label=f"{kn.knee}")
        axes[2].legend()
        axes[2].set_xlabel('Number of Projectors')
        axes[2].set_ylabel('Log(INPSR) (AU)')
        axes[2].set_title('Improved Normalised Power Spectrum Ratio')
        axes[2].set_xticks(projectors)
        plt.legend()
        plt.tight_layout()
        plt.savefig(image_path + f"{method}_tibial.png")
        plt.savefig(image_path + f"{method}_tibial" + '.pdf', bbox_inches='tight', format="pdf")
        # plt.show()
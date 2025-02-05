# Look at elbow plots for SNR, RI and INSPR to see where effect starts to fall off
# Checking for SSP, CCA cardiac and DSS cardiac


import numpy as np
import h5py
import matplotlib.pyplot as plt
from math import log
import pandas as pd

if __name__ == '__main__':
    input_paths = {'SSP': "/data/pt_02569/tmp_data/ssp_py/",
                   'CCA_heart': "/data/pt_02569/tmp_data/cca_heartart_py/",
                   'DSS_heart': "/data/pt_02569/tmp_data/dss_heartart_py/"
                   }
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

            inps_med.append(np.mean(pow_med_prep[:, median_pos] / pow_med[:, median_pos], axis=tuple([0, 1])))
            inps_tib.append(np.mean(pow_tib_prep[:, tibial_pos] / pow_tib[:, tibial_pos], axis=tuple([0, 1])))

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

        inps_med_log = [log(inps) for inps in inps_med]
        inps_tib_log = [log(inps) for inps in inps_tib]

        # Images
        fig, axes = plt.subplots(1, 3)
        axes = axes.flatten()
        plt.suptitle(f"{method}, median")
        axes[0].scatter(projectors, snr_med)
        axes[0].set_xlabel('No. Projectors')
        axes[0].set_ylabel('Value')
        axes[0].set_title('SNR')
        axes[0].set_xticks(projectors)
        axes[1].scatter(projectors, residual_med)
        axes[1].set_xlabel('No. Projectors')
        axes[1].set_ylabel('Value')
        axes[1].set_title('RI')
        axes[1].set_xticks(projectors)
        axes[2].scatter(projectors, inps_med_log)
        axes[2].set_xlabel('No. Projectors')
        axes[2].set_ylabel('Log Value')
        axes[2].set_title('INPSR')
        axes[2].set_xticks(projectors)
        plt.tight_layout()

        fig, axes = plt.subplots(1, 3)
        axes = axes.flatten()
        plt.suptitle(f"{method}, tibial")
        axes[0].scatter(projectors, snr_tib)
        axes[0].set_xlabel('No. Projectors')
        axes[0].set_ylabel('Value')
        axes[0].set_title('SNR')
        axes[0].set_xticks(projectors)
        axes[1].scatter(projectors, residual_tib)
        axes[1].set_xlabel('No. Projectors')
        axes[1].set_ylabel('Value')
        axes[1].set_title('RI')
        axes[1].set_xticks(projectors)
        axes[2].scatter(projectors, inps_tib_log)
        axes[2].set_xlabel('No. Projectors')
        axes[2].set_ylabel('Log Value')
        axes[2].set_title('INPSR')
        axes[2].set_xticks(projectors)
        plt.tight_layout()
        plt.show()
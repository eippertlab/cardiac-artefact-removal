# Script to create plots of the grand averages evoked responses of the heartbeat across participants for each stimulation

import mne
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_axes_aligner import align
import seaborn as sns
import matplotlib as mpl
from PIL import Image
import glob
from matplotlib.ticker import FormatStrFormatter
import time
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    pal = sns.color_palette(n_colors=10)
    subjects = np.arange(1, 37)   # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 1000

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    method = 'DSS'  # Can be CCA or DSS
    image_path = f"/data/p_02569/Images/AverageGen_{method}_Dataset1/"
    os.makedirs(image_path, exist_ok=True)

    gen_images = True
    gen_gif = True

    for cond_name in cond_names:  # Conditions (median, tibial)
        evoked_list = []

        if cond_name == 'tibial':
            trigger_name = 'qrs'
            channel = 'L1'

        elif cond_name == 'median':
            trigger_name = 'qrs'
            channel = 'SC6'

        for subject in subjects:  # All subjects
            subject_id = f'sub-{str(subject).zfill(3)}'

            if gen_images:
                if method == 'CCA':
                    input_path = f"/data/pt_02569/tmp_data/cca_heartart_py/{subject_id}/"
                    if cond_name == 'median':
                        n = 9
                    elif cond_name == 'tibial':
                        n = 6
                    fname = f"epochs_{cond_name}_{n}_qrs.fif"
                    epochs = mne.read_epochs(input_path + fname, preload=True)

                elif method == 'DSS':
                    input_path = f"/data/pt_02569/tmp_data/dss_heartart_py/{subject_id}/"
                    if cond_name == 'median':
                        n = 9
                    elif cond_name == 'tibial':
                        n = 7
                    fname = f"epochs_{cond_name}_{n}_qrs.fif"
                    epochs = mne.read_epochs(input_path + fname, preload=True)

                evoked = epochs.average()
                evoked.pick([channel])
                evoked_list.append(evoked)

                current_average = mne.grand_average(evoked_list, interpolate_bads=False, drop_bads=False)

                # Subplot for both CCA and DSS - left side is always current subject, right side is building average
                fig, axes = plt.subplots(1, 2, figsize=[12, 8])
                axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                axes[0].plot(evoked.times, evoked.data[0, :] * 10 ** 6, color=pal[0])
                axes[0].set_xlabel('Time (s)')
                axes[0].set_ylabel('Amplitude (\u03BCV)')
                axes[0].set_title(f'{subject_id}')
                axes[0].set_xlim([-0.2, 0.4])

                axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                axes[1].plot(current_average.times, current_average.data[0, :] * 10 ** 6, color=pal[2])
                axes[1].set_xlabel('Time (s)')
                axes[1].set_ylabel('Amplitude (\u03BCV)')
                axes[1].set_title(f'n={len(evoked_list)}')
                axes[1].set_xlim([-0.2, 0.4])

                fname = f"{cond_name}_{subject_id}.png"
                plt.tight_layout()
                plt.savefig(image_path+fname)

        if gen_gif:
            # create an empty list called images
            images = []

            # get all the images in the 'images for gif' folder
            filenames = sorted(glob.glob(f'{image_path}{cond_name}*.png'))
            for filename in filenames:  # loop through all png files in the folder
                im = Image.open(filename)  # open the image

                # create extra copies (to make the gif spend longer on the most recent data)
                for x in range(0, 3):
                    images.append(im)

            # save as a gif
            images[0].save(f'{image_path}{cond_name}.gif',
                           save_all=True, append_images=images[1:], optimize=False, duration=500, loop=0)


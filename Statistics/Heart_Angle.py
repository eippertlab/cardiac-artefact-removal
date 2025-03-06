# Want to get the distance from each stimulation point, to the R-peak before
# Get this in degrees
# Want an average 'angle' per participant and condition
# Then want to plot the average participant angle

import os
import numpy as np
import mne
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
from pycircstat.tests import rayleigh
import seaborn as sns
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.sandbox.stats.multicomp import multipletests


if __name__ == '__main__':
    plot_images = False
    np.set_printoptions(suppress=True)
    cond_names = ['median', 'tibial']
    save_path = '/data/p_02569/Heart_Angle_D1/'
    save_path_single = '/data/p_02569/Heart_Angle_D1/SingleSubject/'
    colors = ['darkblue', 'deepskyblue']
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path_single, exist_ok=True)

    subjects = np.arange(1, 37)
    sampling_rate = 1000

    df_corrected = pd.DataFrame()
    df_corrected['Subjects'] = subjects

    df = pd.DataFrame()
    df['Subjects'] = subjects

    for cond_name in cond_names:
        subject_averages = []
        p_values = []
        if cond_name == 'median':
            trigger_name = 'Median - Stimulation'
        elif cond_name == 'tibial':
            trigger_name = 'Tibial - Stimulation'

        all_degrees = []
        for subj in subjects:
            subject_id = f'sub-{str(subj).zfill(3)}'

            input_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + '/'
            raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr1000_{cond_name}_withqrs.fif", preload=True)

            # Extract the stimulus event times
            events, event_ids = mne.events_from_annotations(raw)
            event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
            stim_events = [event.tolist() for event in events if event[2] == event_id_dict[trigger_name]]
            stim_times = np.asarray([stim_event[0]/sampling_rate for stim_event in stim_events])

            # Get the R-peak timings
            input_path_m = "/data/pt_02569/Python_Cardiac/QRS_Timing/"+subject_id + '/'
            fname_m = f"raw_{sampling_rate}_spinal_{cond_name}"
            matdata = loadmat(input_path_m + fname_m + '.mat')
            QRS_times = matdata['QRSevents'][0]/sampling_rate

            time_differences = []
            degree_differences = []
            for stim_time in stim_times:
                # Find time of R-peak before stimulus onset
                peak_time = QRS_times[QRS_times < stim_time].max()
                # As an index
                pos = QRS_times[QRS_times < stim_time].argmax()

                # Calcuate difference between stimulus onset and previous R-peak
                diff2peak = stim_time - peak_time
                time_differences.append(diff2peak)

                # Calculate relative position of the stimulus onset on the cardiac cycle
                stim_degree = 360*diff2peak/(QRS_times[pos+1]-QRS_times[pos])
                degree_differences.append(stim_degree)
                all_degrees.append(stim_degree)

            # # Now get the average degrees for this subject
            # if plot_images:
                # if subj in [1, 6, 10, 19, 24, 32]:
                #     r = 2
                #     average_radians = np.deg2rad(np.asarray(degree_differences))
                #     ax = plt.subplot(111, projection='polar')
                #     for rad in average_radians:
                #         plt.polar(rad, r, 'g.', linewidth=4)
                #     ax.set_theta_zero_location('N')
                #     ax.set_theta_direction(-1)  # clockwise
                #     ax.grid(False)
                #     ax.set_rticks([])
                #     ax.spines['polar'].set_visible(False)
                #     ax.plot(np.linspace(0, 2 * np.pi, 100), np.ones(100) * r, color='black', linewidth=0.5)
                #     plt.ylim([0, r + 0.25])
                #     plt.title(f"{subject_id}, {cond_name}")
                #     plt.show()

            # Perform rayleigh test
            pval, z = rayleigh(np.deg2rad(np.asarray(degree_differences)))
            p_values.append(pval)

            if plot_images:
                # Plot histogram for single subjects
                if cond_name == 'median':
                    color = colors[0]
                else:
                    color = colors[1]
                fig, ax = plt.subplots()
                n, bins, patches = plt.hist(np.asarray(degree_differences), range=tuple((0, 360)), bins=12, facecolor=color,
                                            edgecolor='#e0e0e0', linewidth=1.0, alpha=0.7,
                                            weights=np.ones_like(np.asarray(degree_differences)) * 100 / len(
                                                np.asarray(degree_differences)))
                n = n.astype('int')
                ax.set_xticks(bins)
                bin_centers = 0.5 * np.diff(bins) + bins[:-1]
                ax.yaxis.set_major_formatter(PercentFormatter())
                plt.xlabel('Angle (degrees)')
                plt.ylabel('Proportion')
                plt.title(f"Stimulation to R-peak, {trigger_name}, {subject_id}")
                # Give ourselves some more room at the bottom of the plot
                plt.subplots_adjust(bottom=0.15)
                plt.savefig(save_path_single + f"{subject_id}_{cond_name}.png")
                plt.savefig(save_path_single + f"{subject_id}_{cond_name}.pdf", bbox_inches='tight', format="pdf")
                plt.close()

            subject_averages.append(np.mean(degree_differences))

        # Now we have each subjects average, put on a polar plot
        # if plot_images:
            # r = 2
            # average_radians = np.deg2rad(np.asarray(subject_averages))
            # ax = plt.subplot(111, projection='polar')
            # for rad in average_radians:
            #     plt.polar(rad, r, 'g.', linewidth=4)
            # ax.set_theta_zero_location('N')
            # ax.set_theta_direction(-1)  # clockwise
            # ax.grid(False)
            # ax.set_rticks([])
            # ax.spines['polar'].set_visible(False)
            # ax.plot(np.linspace(0, 2*np.pi, 100), np.ones(100)*r, color='black', linewidth=0.5)
            # plt.ylim([0, r+0.25])
            # plt.title(f"All Subjects Average, {cond_name}")
            # plt.show()
            # Add p_values to df

        df[f"{cond_name}_uncorrpval"] = p_values
        hyp, corrected_p = fdrcorrection(df[f"{cond_name}_uncorrpval"].values)
        reject, p_bonf, _, _ = multipletests(df[f"{cond_name}_uncorrpval"].values, method='bonferroni')
        df[f"{cond_name}_pcorr_fdr"] = corrected_p
        df[f"{cond_name}_pcorr_bonf"] = p_bonf
        # df[f"{cond_name}_reject"] = hyp

        if plot_images:
            if cond_name == 'median':
                color = colors[0]
            else:
                color = colors[1]
            fig, ax = plt.subplots()
            n, bins, patches = plt.hist(np.asarray(all_degrees), range=tuple((0, 360)), bins=12, facecolor=color,
                                        edgecolor='#e0e0e0', linewidth=1.0, alpha=0.7,
                                        weights=np.ones_like(np.asarray(all_degrees))*100 / len(np.asarray(all_degrees)))
            n = n.astype('int')  # it MUST be integer

            # Set the ticks to be at the edges of the bins.
            ax.set_xticks(bins)
            # Label the raw counts and the percentages below the x-axis...
            bin_centers = 0.5 * np.diff(bins) + bins[:-1]
            ax.yaxis.set_major_formatter(PercentFormatter())
            plt.xlabel('Angle (degrees)')
            plt.ylabel('Proportion')
            plt.title(f"Stimulation to R-peak, {trigger_name}")
            # Give ourselves some more room at the bottom of the plot
            plt.subplots_adjust(bottom=0.15)
            plt.savefig(save_path+f"{cond_name}.png")
            plt.savefig(save_path+f"{cond_name}.pdf", bbox_inches='tight', format="pdf")
            plt.close()
            # plt.show()

        pval, z = rayleigh(np.deg2rad(np.asarray(all_degrees)))
        print(pval)
        # input('Just want a pause (hit Enter):')

    print(df)
    print(df_corrected)
    df.to_excel('/data/pt_02569/tmp_data/Rayleigh.xlsx')




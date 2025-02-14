# Read in the text files that save the number of components marked for removal by the ICA algorithm
# Get the mean, std, range


import numpy as np


if __name__ == '__main__':
    option = 'Separated-ICA'  # ICA, Anterior-ICA, Separated-ICA
    subjects = np.arange(1, 37)

    for cond_name in ['median', 'tibial']:
        no_comps_removed = []

        for subject in subjects:
            subject_id = f'sub-{str(subject).zfill(3)}'
            file_path = f'/data/pt_02569/tmp_data/baseline_ica_py/{subject_id}/'
            if option == 'ICA':
                file_name = f'ecg_indices_{cond_name}.txt'
            elif option == 'Anterior-ICA':
                file_name = f'anterior_ecg_indices_{cond_name}.txt'
            elif option == 'Separated-ICA':
                file_name = f'separated_ecg_indices_{cond_name}.txt'

            f = open(file_path+file_name, "r")
            count=0
            for line in f:
                count += 1
            no_comps_removed.append(count)

        print(f'\n{option}, {cond_name}')
        print(f'mean - {np.mean(no_comps_removed)}')
        print(f'std - {np.std(no_comps_removed)}')
        print(f'min - {np.min(no_comps_removed)}')
        print(f'max - {np.max(no_comps_removed)}')


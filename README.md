[![GitHub Release](https://img.shields.io/github/v/release/eippertlab/cardiac-artefact-removal)](https://github.com/eippertlab/cardiac-artefact-removal/releases/tag/v1.0)
[![DOI](https://zenodo.org/badge/446424355.svg)](https://zenodo.org/doi/10.5281/zenodo.13693032)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# cardiac-artefact-removal #

This repository is associated with the following [manuscript](https://direct.mit.edu/imag/article/doi/10.1162/IMAG.a.938/133390/Evaluating-cardiac-noise-correction-approaches-for) and the corresponding [dataset](https://openneuro.org/datasets/ds004388). If you have any questions related 
to this code, please feel free to contact bailey@cbs.mpg.de.

## Content ##
This repository contains the preprocessing and analysis code used to remove the cardiac artefact from spinal electrophysiology 
data as presented in the above-mentioned manuscript. 

### Main Processing ###
**main.py** is a wrapper script which can be used to detail the stages of analysis to run. These steps include:

* Data Import (incl. downsampling, r-peak event annotation, stimulus artefact removal, filtering, file concatenation)
* Cardiac artefact removal via:
  * PCA_OBS (Principal Component Analysis Optimal Basis Sets)
  * ICA (Independent Components Analysis, including variants involving anteriorly re-referenced data and applying ICA
  separately to the lumbar and cervical spinal patch of electrodes)
  * SSP (Signal Space Projection with varying numbers of projectors)
  * CCA (Canonical Correlation Analysis with varying numbers of components)
  * DSS (Denoising Separation of Sources with varying numbers of components)
* Signal enhancement analysis via:
  * CCA (Canonical Correlation Analysis)
  * DSS (Denoising Separation of Sources)

### Metrics ###
Scripts contained in **/Metrics** are used to compute the:
* SNR (Signal-to-Noise Ratio of the SEPs)
* RI (Residual Intensity of the cardiac artefact)
* INPSR (Improved Normalised Power Spectrum Ratio of the cardiac artefact)
* CoV (Coefficient of Variation)

### Statistics ###
Scripts contained in **/Statistics** are used to determine statistical significance via repeated measures ANOVA
and one-sample permutation t-tests for the SNR, RI, INPSR and CoV results.

### Plotting_Code_Publication ###
Scripts contained in **/Plotting_Code_Publication** are used to generate all manuscript and supplement figures, 
as seen in the above mentioned manuscript.

### QRS Timing ###
The R-peak latencies for each subject and condition has been provided for convenience in **/QRS_Timing**. To run the PCA_OBS method in particular, 
precise knowledge of the timing of each R-peak is required. These QRS timings should be copied into the relevant data folder 
and this file path should then be substituted for import_path_m within the following files:

* import_data.py
* rm_heart_artefact.py
* rm_heart_artefact_tukey.py

These timings are generated using the method employed by [Nierula et al., 2024](https://www.biorxiv.org/content/10.1101/2022.12.05.519148v2) 
and the accompanying code can be found in the [associated repository](https://github.com/eippertlab/spinal_sep1).

## Required Software ##
All scripts run with python 3.9 and MNE 1.0.3, for an extensive list of required packages see requirements.txt

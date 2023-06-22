# Localizing Syntactic Composition with Left-Corner Recurrent Neural Network Grammars

This repository provides the code for the paper 'Localizing Syntactic Composition with Left-Corner Recurrent Neural Network Grammars' Neurobiolgy of Language.

## Requirements
`python==3.9.12`
`R==4.2.1`

## Preparing for Regressors
The outputs from five gram, LSTM, and RNNGs are from the models that were implemented in RNNG-EyeTrack. Place the outputs (.tsv file) on `regressors/`. `BCCWJ-fMRI_regressors.ipynb` convolves the regressors with the HRF (hrf_model = "spm", TR =2.0) to estimate the BOLD signals. Correlations among predictors are also calculated. `Perplexity.ipynb` calculates the perplexities for five-gram, LSTM and RNNGs. `BCCWJ-fMRI_regressors.ipynb` and `Perplexity.ipynb` are located in the `regressors` directory.

## Concatenating regressors and fMRI data
After convolving the regressors with HRF, the csv file was manually separated into four blocks (A,B,C,D). For each block, remove the first 10TR (20 seconds) and place the files on `data/model-regressors/`. These were used for concatenating with the fMRI data via `nii2csv.ipynb`. Once the fMRI data become publicly available, add fMRI data to `data/nii/` and the head movement parameters to `data/rp/`. `nii2csv.ipynb` is located in the `regressors` directory.

## ROI analysis
The output of the concatenating regressors and fMRI data is used for ROI analysis. The script for ROI analysis is `BCCWJ-fMRI_ROI_analysis.R`, which is located in the `R_ROI_analysis` directory.

## Whole Brain Analysis
There are two steps for the whole brain analysis. For First-level analysis, use `GLM_first_level_analysis-BCCWJ-fMRI-all.ipynb` (Once the fMRI data become publicly available, add it to `data/nii_wb/`.). Based on the results from the first-level analysis, `GLM_second_level_analysis_BCCWJ-fMRI.ipynb` performs the second-level analysis. Both `GLM_first_level_analysis-BCCWJ-fMRI-all.ipynb` and `GLM_second_level_analysis_BCCWJ-fMRI.ipynb` are in the `whole_brain_analysis` directory.
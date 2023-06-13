
# Localizing Syntactic Composition with Left-corner Recurrent Neural Network Grammars in Naturalistic Reading

This repository provides the code for the paper 'Localizing Syntactic Composition with Left-corner Recurrent Neural Network Grammars in Naturalistic Reading' Neurobiolgy of Language.

## Requirements
    python==3.9.12 (jupyter notebook)
    R==4.2.1

## Prepare for Regressors
The file `BCCWJ-fMRI-regressors.ipynb` convlves the regressors to the estimated fMRI data (using the hrf function (hrf_model = "spm", TR =2.0)).

## Concatenating regressors and fMRI datapoints
After convolving the regressors, the csv files was manually separated into four blocks (A,B,C,D) and these were used for concatenating with fMRI data via `nii2csv.ipynb`.

## ROI analysis
The output of the concatenating regressors and fMRI data is used for ROI analysis. The script for ROI analysis is `BCCWJ-fMRI_ROI analysis.R`.

## Whole Brain Analysis
For the whole brain analysis, there are two steps. For First-level analysis, use `GLM first level analysis-BCCWJ-fMRI-all.ipynb`. Then execute `GLM second level analysis_BCCWJ-fMRI.ipynb`.

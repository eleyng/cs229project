# Predicting Metabolic Cost using data collected during Human-in-the-Loop Optimization
Autumn 2018 C229 Project \\
Eley Ng and Erez Krimsky

To run, simply load all .mat files and run the MATLAB scripts.

# File Information
- data_processing.m: processes raw data collected from the ongoing learning study
- k_fold_cv.m: runs k-fold cross-validation to tune the network architecture
- feature_selection.m: runs a cross-validated forward stepwise function
- model_selection.m: uses results from feature_selection to predict
- plots_and_tables.m: runs PCA and curvefitting, produces plots and table results
- data_scaling.m: function to scale any input data to zero mean and unit variance

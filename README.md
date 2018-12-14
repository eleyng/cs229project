# Predicting Metabolic Cost During Human-in-the-Loop Optimization
## Eley Ng and Erez Krimsky
Autumn 2018, CS 229 Project

## Description

Human-in-the-Loop Optimization (HILO) is a technique using assistive robotic
devices to augment human walking performance to overcome
this challenge. A common metric used to determine human
performance is metabolic cost, which is the amount of energy
used by the human to perform a certain task. We are interested in predicting metabolic
cost using human walking data collected during human-in-theloop
optimization experiments.

To run, simply load all .mat files and run the MATLAB scripts.

# File Information
- data_processing.m: processes raw data collected from the ongoing learning study
- k_fold_cv.m: runs k-fold cross-validation to tune the network architecture
- feature_selection.m: runs a cross-validated forward stepwise function
- model_selection.m: uses results from feature_selection to predict
- plots_and_tables.m: runs PCA and curvefitting, produces plots and table results
- data_scaling.m: function to scale any input data to zero mean and unit variance

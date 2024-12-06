# Projet-Big-Data

Title: Biomedical Signal Anomaly Detection

Description:
This project involves the analysis, preprocessing, and classification of electrocardiogram (ECG) signals from the ECG5000 dataset. The dataset consists of labeled ECG sequences used for anomaly detection, distinguishing normal signals from abnormal ones. This repository contains all the code necessary to explore the data, preprocess it, and prepare it for machine learning tasks.

About Dataset
The original dataset for "ECG5000" is a 20-hour long ECG downloaded from Physionet. The name is BIDMC Congestive Heart Failure Database(chfdb) and it is record "chf07". It was originally published in "Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23)". The data was pre-processed in two steps: (1) extract each heartbeat, (2) make each heartbeat equal length using interpolation. This dataset was originally used in paper "A general framework for never-ending learning from time series streams", DAMI 29(6). After that, 5,000 heartbeats were randomly selected. The patient has severe congestive heart failure and the class values were obtained by automated annotation.

Time series 140 features and 5000 sequences of ECG for classification, prediction, or detection anomalie.
length of the sequence ECG:140
the first column for classification value1 for normal sequence else abnormal

Key Features:

Exploratory Data Analysis (EDA):

Data overview with statistical insights.
Visualizations of class distributions, feature correlations, and signal histograms.
Preprocessing Pipeline:

Binarization of labels for anomaly detection.
Feature scaling using StandardScaler.
Splitting data into training and testing sets, stratified by class distribution.
Visualization:

Graphical representation of example ECG signals (normal and abnormal).
Heatmap of feature correlations.
Data Saving:

Preprocessed data is saved in a .npz file format for further use in downstream tasks like model training and evaluation.
How to Use:

Clone this repository.
Run the code to reproduce the preprocessing steps or adapt it for custom datasets.
The saved preprocessed dataset can directly be used for machine learning experiments.
Dataset:
The ECG5000 dataset was sourced from the UCR Time Series Classification Repository. It consists of ECG signal segments with 5 classes, reduced to a binary classification task for this project.

Dependencies:

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn

Applications:
This project can be extended to develop real-time ECG anomaly detection systems, aiding in medical diagnosis and monitoring

# Modbus-Anomaly-Detection
This is part of my master thesis on anomaly detection on maritime OT, with especially focus on modbus
Three diferent models are beeing trained and testet on CIC modbus 2023 dataset
The models implemented are OCSVM, LSTM autoencoder and a hybrid model

CSV files, and the pretrained models used are included in this repository

The repository consists of jupyter notebook, seperate for training and testing of each model totaling to six jupyter notebooks
There is one .py file named functions including the common functions used across the different jupyter notebooks

File structure is the following

Project folder includes the following subfolders:
"models" --> contains the pretrained models
"CSV files" --> Contains CSV files, both benign, attack data for testing, and attack log for labeling

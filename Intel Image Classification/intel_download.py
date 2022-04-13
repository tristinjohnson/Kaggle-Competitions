"""
Download Intel Image Classification Data from Kaggle

In order to download the data from Kaggle, you will need your Kaggle username and Kaggle API
"""
import opendatasets as od
import os

# make data directory
#os.makedirs('../data')
#data_dir = '../data/'

# change to data directory
#os.chdir(data_dir)

# download Intel Image Classification
dataset_url = "https://www.kaggle.com/puneet6060/intel-image-classification"
od.download(dataset_url)

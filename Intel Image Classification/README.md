# Intel Image Classification

This repository contains the code in order to train state-of-the-art pre-trained models on the Intel Image Classification dataset provided by Kaggle. Some of the models used include ResNet50, GoogleNet, VGG-19 and Inception-V3. You will see 2 different Python scripts:

		1. intel_download.py
		2. train.py

The 'intel_download.py' allows you to download all of the images directly from Kaggle.com. You will need a Kaggle account in order to do this, and will need to login with your Kaggle Username and Kaggle API Token. The 'train.py' script will initally use ResNet50 for training, as this was the most accurate model for this dataset. In order to use the other models stated earlier, simply uncomment one of the lines between line 85 and 88. Feel free to explore even more pre-trained models and hope you enjoy training!

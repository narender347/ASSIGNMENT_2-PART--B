**PART-B**
partB has the code for using the pretrained Vision models from pytorch and fine-tuning them on the iNaturalist dataset.
This contains the code necessary to fine-tune pretrained Vision models from PyTorch on the iNaturalist dataset.

Main Files:
main.py
This script handles fetching and fine-tuning pretrained models. It allows customization such as adding extra DenseNet layers and freezing specific layers prior to training.

pretrained.csv
This CSV file lists the available pretrained models that can be used for fine-tuning. All models are sourced from the PyTorch library and are sorted in descending order based on their ImageNet accuracy.

train.py (Work in Progress)
This script is intended as a command-line interface for training models. However, it currently does not support all models, as different architectures have varying input requirements and output layer configurations. A more abstract and unified implementation might be possible in the future.

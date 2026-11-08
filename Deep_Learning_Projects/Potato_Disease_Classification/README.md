# Potato Disease Classification Using CNN

A TensorFlow/Keras image-classification project that trains a convolutional neural network (CNN) on a PlantVillage directory dataset.

## Project workflow

- Loads image data from class-specific folders.
- Uses 256 × 256 RGB images and batches of 32.
- Splits the dataset into training, validation, and test subsets (80/10/10).
- Applies image rescaling and data augmentation (random flips and rotations).
- Builds a sequential CNN with convolution, max-pooling, and dense layers.
- Trains for up to 50 epochs using Adam and sparse categorical cross-entropy.
- Evaluates the model, plots training/validation accuracy and loss, and demonstrates predictions.
- Saves the trained model in Keras and HDF5 formats.

## Files

- [Python implementation](./potato_disease_classification.py)
- [Jupyter Notebook](./potato-disease-classification-model.ipynb)

## Requirements

```bash
pip install tensorflow matplotlib numpy jupyter
```

## Dataset setup

The original code expects a PlantVillage dataset arranged in class-specific subfolders. Update the dataset directory in `potato_disease_classification.py` (and the notebook, if using a different location) to match your local dataset path.

## Notes

This repository contains the implementation, not the dataset. No test accuracy is reported here because a verified evaluation result was not included with the submitted source files. The prediction helper in the original code may need a small variable-reference correction before running the batch visualization cell.

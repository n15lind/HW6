# Homework 6: Analyzing SHRED Models
**Neil Lindgren**

For this assignment, I looked at the effect of various parameters on the performance of Shallow Recurrent Decoder (SHRED) models.

## Sec. I. Introduction and Overview

The original code for the SHRED models was developed by Jan P. Williams, Olivia Zahn, and J. Nathan Kutz for the paper "Sensing with shallow recurrent decoder networks". To quote the description from https://github.com/Jan-Williams/pyshred, "SHallow REcurrent Decoders (SHRED) are models that learn a mapping from trajectories of sensor measurements to a high-dimensional, spatio-temporal state." I will be modifying this code to see what happens when I change the number of lags used, the number of sensors used, and the amount of noise in the data. The models are built to work with a few different datasets, but the dataset I will be using is a collection of sea-surface temperatures.

## Sec. II. Theoretical Background

This model is a form of Long Short-Term Memory (LSTM) network, which is a type of recursive neural network. In other words, some of the layers of the neural net provide "feedback" to previous layers to improve the performance of the network. This particular network uses an input layer, an output layer, and two hidden layers.

## Sec. III. Algorithm Implementation and Development
## Sec. IV. Computational Results
## Sec. V. Summary and Conclusions

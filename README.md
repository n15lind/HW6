# Homework 6: Analyzing SHRED Models
**Neil Lindgren**

For this assignment, I looked at the effect of various parameters on the performance of Shallow Recurrent Decoder (SHRED) models.

## Sec. I. Introduction and Overview

The original code for the SHRED models was developed by Jan P. Williams, Olivia Zahn, and J. Nathan Kutz for the paper "Sensing with shallow recurrent decoder networks". To quote the description from https://github.com/Jan-Williams/pyshred, "SHallow REcurrent Decoders (SHRED) are models that learn a mapping from trajectories of sensor measurements to a high-dimensional, spatio-temporal state." I will be modifying this code to see what happens when I change the number of lags used, the number of sensors used, and the amount of noise in the data. The models are built to work with a few different datasets, but the dataset I will be using is a collection of sea-surface temperatures (SST).

## Sec. II. Theoretical Background

This model is a form of Long Short-Term Memory (LSTM) network, which is a type of recursive neural network. In other words, some of the layers of the neural net provide "feedback" to previous layers to improve the performance of the network. This particular network uses an input layer, an output layer, and two hidden layers.

## Sec. III. Algorithm Implementation and Development

The Algorithm was mostly the same for each of the tests. First, I defined the initial parameters and loaded the SST data.
```
num_sensors = 3 
lags = 52
load_X = load_data('SST')
n = load_X.shape[0]
m = load_X.shape[1]
sensor_locations = np.random.choice(m, size=num_sensors, replace=False)
```
Second, I divided the data into different sets for training, validation, and testing
```
train_indices = np.random.choice(n - lags, size=1000, replace=False)
mask = np.ones(n - lags)
mask[train_indices] = 0
valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]]
valid_indices = valid_test_indices[::2]
test_indices = valid_test_indices[1::2]
```


## Sec. IV. Computational Results



## Sec. V. Summary and Conclusions


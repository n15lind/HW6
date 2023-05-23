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
Second, I divided the data into different sets for training, validation, and testing.
```
train_indices = np.random.choice(n - lags, size=1000, replace=False)
mask = np.ones(n - lags)
mask[train_indices] = 0
valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]]
valid_indices = valid_test_indices[::2]
test_indices = valid_test_indices[1::2]
```
I then prepared the data and generated input/output pairs for the data sets.
```
sc = MinMaxScaler()
sc = sc.fit(load_X[train_indices])
transformed_X = sc.transform(load_X)

all_data_in = np.zeros((n - lags, lags, num_sensors))
for i in range(len(all_data_in)):
    all_data_in[i] = transformed_X[i:i+lags, sensor_locations]
```
I then converted the data sets to torch tensors and modified them to get them ready for training.
```
train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
```
After that, I trained the model over 1000 epochs
```
shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=1000, lr=1e-3, verbose=True, patience=5)
```
Finally, I tested the model and calculated error by comparing the MSE to the ground truth.
```
test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
print(np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth))
```
I first implemented the code as-is to see how it performed and plot the results. I then modified it to train with 5 different numbers of lags (26, 32, 39, 44, and 52) and compared their performances. I then went back to the default lag value of 52 and added varying levels of Gaussian noise (std = 0, 0.05, 0.1, 0.15, and 0.2) to see how that affected the performance of the model. Finally, I removed the noise and tried different numbers of sensors (1, 2, 3, 4, and 5).

## Sec. IV. Computational Results

The validation error for the baseline test is shown below in Figure 1. The overall error for this test was 0.02

## Sec. V. Summary and Conclusions


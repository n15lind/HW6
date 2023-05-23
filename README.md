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

The validation error over the training epochs for the baseline test is shown below in Figure 1. Note that the x-axis is from 0 to 50 because samples were taken every 20 epochs (1/50 of the entire set of 1000 epochs). The overall error compared to the ground truth was 0.020010263.

![Baseline](https://github.com/n15lind/HW6/assets/130141391/ee7f665f-7b76-413d-acd8-f5e4aaeffc4d)

_Figure 1: Validation error vs epochs for baseline test_

Figure 2 shows the validation error over the training epochs for various numbers of lags. Note that the x-axis is from 0 to 50 because samples were taken every 20 epochs (1/50 of the entire set of 1000 epochs). Figure 3 shows the errors for the different lag values compared to the ground truth.

![Lags Epochs](https://github.com/n15lind/HW6/assets/130141391/b6f1a309-3cf4-47cd-8f15-bb9e24a3c6d1)

_Figure 2: Validation error vs epochs for different numbers of lags_

![Lags Errors](https://github.com/n15lind/HW6/assets/130141391/42cceb29-0290-4ff6-8433-14f49ded5a4d)

_Figure 3: Error vs number of lags_

Figure 4 shows the validation error over the training epochs for various amounts of noise. The "noise level" is the standard deviation of the Gaussian noise. Note that the x-axis is from 0 to 50 because samples were taken every 20 epochs (1/50 of the entire set of 1000 epochs). Figure 5 shows the errors for the different lag values compared to the ground truth.

![Noise Epochs](https://github.com/n15lind/HW6/assets/130141391/798c839a-1ee1-4bc6-ad02-a08a1041e471)

_Figure 4: Validation error vs epochs for different standard deviations of Gaussian noise_

![Noise Error](https://github.com/n15lind/HW6/assets/130141391/65897c38-fc1a-4f56-b6ca-383f6a787ca5)

_Figure 5: Error vs standard deviation of Gaussian noise_

Figure 6 shows the validation error over the training epochs for various amounts of noise. The "noise level" is the standard deviation of the Gaussian noise. Note that the x-axis is from 0 to 50 because samples were taken every 20 epochs (1/50 of the entire set of 1000 epochs). Figure 7 shows the errors for the different lag values compared to the ground truth.

![Sensors Epochs](https://github.com/n15lind/HW6/assets/130141391/3f94266c-201c-4394-b580-88ec4894fd2a)

_Figure 6: Validation error vs epochs for different numbers of sensors_

![Sensors Error](https://github.com/n15lind/HW6/assets/130141391/64f9aec0-68cb-40a4-891f-76959b6b1ec1)

_Figure 7: Error vs number of sensors used_

## Sec. V. Summary and Conclusions


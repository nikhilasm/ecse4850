# Nikhilas Murthy
# ECSE 4850 Final Project
# Data Conversion and Splitting Code

import numpy as np
import pickle as pkl

###############################################################################
# Parameter to determine the ratio of training sample count to testing sample
# count, i.e. a VALID_SPLIT_RATIO of 0.75 means that 75% of the samples will be
# in the training set and 25% of the samples will be in the validation set.
VALID_SPLIT_RATIO = 0.75

###############################################################################

# Load the given data files and concatenate them together into one data set
f = open('youtube_action_train_data_part1.pkl', 'rb')
data1, labels1 = pkl.load(f)
f.close()
f = open('youtube_action_train_data_part2.pkl', 'rb')
data2, labels2 = pkl.load(f)
f.close()

data_all = np.concatenate((data1, data2))
labels_all = np.concatenate((labels1, labels2))
del data1, labels1, data2, labels2

# Use the indices for class boundaries in the data set to split the data into
# sets of each separate class. For example, the slice of the dataset [0:607]
# contains all samples of class=0.
class_cutoffs = [607, 1282, 2004, 2653, 3584, 4313, 4946, 5696, 6241, 6707]

data_cw = np.split(data_all, class_cutoffs)
labels_cw = np.split(labels_all, class_cutoffs)

# Shuffle the order of all samples in each class to create more diverse
# training and validation sets
rng = rng = np.random.default_rng()
for i in range(11):
    order = rng.choice(len(data_cw[i]), len(data_cw[i]), replace=False)
    data_cw[i] = data_cw[i][order]
    labels_cw[i] = labels_cw[i][order]

del data_all, labels_all

# Initialize arrays to store the training and validation data sets
train_data = np.array([], dtype=np.int16).reshape(0, 30, 64, 64, 3)
train_labels = np.array([], dtype=np.int16)
valid_data = np.array([], dtype=np.int16).reshape(0, 30, 64, 64, 3)
valid_labels = np.array([], dtype=np.int16)

# Iterate through each class and split according to the given VALID_SPLIT_RATIO.
# This ensures that both the training set and the validation set will have
# examples of every class.
for k in range(11):
    train_data_split, valid_data_split = np.split(data_cw[k], [int(len(data_cw[k])*VALID_SPLIT_RATIO)])
    train_labels_split, valid_labels_split = np.split(labels_cw[k], [int(len(labels_cw[k])*VALID_SPLIT_RATIO)])
    
    train_data = np.concatenate((train_data, train_data_split))
    train_labels = np.append(train_labels, train_labels_split)
    valid_data = np.concatenate((valid_data, valid_data_split))
    valid_labels = np.append(valid_labels, valid_labels_split)

# Save the results
np.save("yt_action_data_train.npy", train_data)
np.save("yt_action_labels_train.npy", train_labels)
np.save("yt_action_data_valid.npy", valid_data)
np.save("yt_action_labels_valid.npy", valid_labels)

print("Saved 4 files with training/validation data & labels.")
print("With split ratio = " + str(VALID_SPLIT_RATIO) + ", resulting data size is:")
print("    Training Data: " + str(train_data.shape))
print("    Validation Data: " + str(valid_data.shape))

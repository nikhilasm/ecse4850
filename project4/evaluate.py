# Nikhilas Murthy
# ECSE 4850 Programming Assignment 4
# Model Loading & Evaluation Code

import numpy as np
import tensorflow as tf

###############################################################################
# Set these parameters and then run the program
TEST_DATA_PATH = "videoframes_clips_valid.npy"
TEST_LABEL_PATH = "joint_3d_clips_valid.npy"
MODEL_PATH = "DeepDynamicModel"
BATCH_SIZE = 3

###############################################################################
def calculateMPJPE(label, pred):
    batch_size = label.shape[0]
    value = 0
    for i in range(batch_size):
        for j in range(8):
            for k in range(17):
                value += tf.norm(label[i, j, k] - pred[i, j, k], ord="euclidean").numpy()
    
    return 1000*(value / (batch_size*8*17))

def dataGenerator():
    order = np.array(range(DATA_LENGTH))
    cur_idx = 0
    end_idx = 0 + BATCH_SIZE
    while cur_idx < DATA_LENGTH:
        videos = TEST_DATA[order[cur_idx:end_idx]] / 255.0
        labels = TEST_LABEL[order[cur_idx:end_idx]]
        yield (videos, labels)
        cur_idx = end_idx
        end_idx = min(DATA_LENGTH, cur_idx + BATCH_SIZE)

# Load data, rescale, build one-hot labels, then evaluate the model
# Prints the average loss, the test accuracy, the inaccuracy for each class, and
# the average inaccuracy

# Testing data is loaded to disk and read piecewise due to my machine's lack of
# memory space. Changing the "BATCH_SIZE" parameter can improve performance.
print("[*] Loading data...")
TEST_DATA = np.load(TEST_DATA_PATH)
TEST_LABEL = np.load(TEST_LABEL_PATH)
DATA_LENGTH = TEST_DATA.shape[0]

print("[*] Loading model and building dataset...")
test_ds = tf.data.Dataset.from_generator(dataGenerator, output_types=(tf.float32, tf.float32))
model = tf.keras.models.load_model(MODEL_PATH)
mse = tf.keras.losses.MeanSquaredError()

print("[*] Evaluating model performance. This may take a while...")
test_loss = np.array([])
test_mpjpe = np.array([])
for videos, labels in test_ds:
    pred = model(videos, training=False)
    loss = mse(labels, pred)
    mpjpe = calculateMPJPE(labels, pred)
    
    test_loss = np.append(test_loss, loss)
    test_mpjpe = np.append(test_mpjpe, mpjpe)

test_loss_val = np.mean(test_loss)
test_mpjpe_val = np.mean(test_mpjpe)

print("=== EVALUATION RESULTS ===")
print("Average Loss: " + str(test_loss_val))
print("Average MPJPE: " + str(test_mpjpe_val))

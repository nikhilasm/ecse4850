# Nikhilas Murthy
# ECSE 4850 Final Project
# Model Loading & Evaluation Code

import numpy as np
import tensorflow as tf
import plotter

###############################################################################
# Set these parameters and then run the program
TEST_DATA_PATH = "yt_action_data_valid.npy"
TEST_LABEL_PATH = "yt_action_labels_valid.npy"
MODEL_PATH = "DeepClassifierModel"
BATCH_SIZE = 8

###############################################################################
# This data generator will iterate over the testing dataset, returning a tuple
# of (videos, labels), where videos is a 5D tensor and labels is a 1D list.
def dataGenerator():
    order = np.array(range(DATA_LENGTH))
    cur_idx = 0
    end_idx = 0 + BATCH_SIZE
    while cur_idx < DATA_LENGTH:
        videos = TEST_DATA[order[cur_idx:end_idx]] / 255.0
        labels = tf.one_hot(TEST_LABEL[order[cur_idx:end_idx]], 11)
        yield (videos, labels)
        cur_idx = end_idx
        end_idx = min(DATA_LENGTH, cur_idx + BATCH_SIZE)

# Load the data
print("[*] Loading data...")
TEST_DATA = np.load(TEST_DATA_PATH)
TEST_LABEL = np.load(TEST_LABEL_PATH)
DATA_LENGTH = TEST_DATA.shape[0]

# Build the Tensorflow dataset object and load the saved model
print("[*] Loading model and building dataset...")
test_ds = tf.data.Dataset.from_generator(dataGenerator, output_types=(tf.float32, tf.float32))
model = tf.keras.models.load_model(MODEL_PATH)

# Iterate through the testing data and evaluate model performance
print("[*] Evaluating model performance. This may take a while...")
test_loss = np.array([])
all_predictions = np.array([])
all_labels = np.array([])
test_correct = np.array([0,0,0,0,0,0,0,0,0,0,0])
test_appearances = np.array([0,0,0,0,0,0,0,0,0,0,0])

for videos, labels in test_ds:
    # Make a prediction and compute the loss
    pred = model(videos, training=False)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred)
    
    # Compute and save data for metrics
    test_loss = np.append(test_loss, loss)
    all_predictions = np.append(all_predictions, tf.argmax(pred, axis=1).numpy())
    all_labels = np.append(all_labels, tf.argmax(labels, axis=1).numpy())
    
    pred = tf.one_hot(tf.math.argmax(pred, 1), 11)
    for k in range(11):
        correct = tf.reduce_sum(tf.multiply(pred[:, k], labels[:, k]))
        count = tf.reduce_sum(labels[:, k])
        test_correct[k] += correct
        test_appearances[k] += count
        
# Compute final evaluation metrics
test_loss = np.mean(test_loss)
test_accuracy = np.sum(test_correct) / np.sum(test_appearances)
test_accuracy_cw = [(test_correct[i] / test_appearances[i]) for i in range(11)]

# Print all statistics
classes = ["b_shooting", "cycling", "diving", "g_swinging", "h_riding", "s_juggling",
               "swinging", "t_swinging", "t_jumping", "v_spiking", "d_walking"]
print("=== EVALUATION RESULTS ===")
print("Average Loss: " + str(round(test_loss, 4)))
print("Overall Accuracy %: " + str(round(test_accuracy * 100, 4)))
print("Class-wise Accuracy %: ")
for i, c in enumerate(classes):
    print("  " + c.rjust(10) + "\t" + str(round(test_accuracy_cw[i] * 100, 4)))

# Generate the confusion matrix
matrix = tf.math.confusion_matrix(all_labels, all_predictions)
plotter.makeConfusionMatrix(matrix, classes)

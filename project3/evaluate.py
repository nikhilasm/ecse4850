# Nikhilas Murthy
# ECSE 4850 Programming Assignment 3
# Model Loading & Evaluation Code

import numpy as np
import tensorflow as tf

# Set these parameters and then run the program
TEST_DATA_PATH = "testing_data.npy"
TEST_LABEL_PATH = "testing_label.npy"
MODEL_PATH = "results10/cnn_model"

# Load data, rescale, build one-hot labels, then evaluate the model
# Prints the average loss, the test accuracy, the inaccuracy for each class, and
# the average inaccuracy
testing_data = np.load(TEST_DATA_PATH)
testing_data = tf.convert_to_tensor(testing_data / 255.0)

testing_label = np.load(TEST_LABEL_PATH)
testing_label = tf.reshape(tf.one_hot(testing_label, 10), [5000, 10])

test_ds = tf.data.Dataset.from_tensor_slices((testing_data, testing_label)).batch(100)

model = tf.keras.models.load_model(MODEL_PATH)

test_correct = np.array([0,0,0,0,0,0,0,0,0,0])
test_appearances = np.array([0,0,0,0,0,0,0,0,0,0])
test_loss = np.array([])
for test_images, test_labels in test_ds:
    pred = model(test_images, training=False)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=test_labels, logits=pred)
    
    test_loss = np.append(test_loss, loss)
    
    pred = tf.one_hot(tf.math.argmax(pred, 1), 10)
    for k in range(10):
        correct = tf.reduce_sum(tf.multiply(pred[:, k], test_labels[:, k]))
        count = tf.reduce_sum(test_labels[:, k])
        test_correct[k] += correct
        test_appearances[k] += count

test_loss = np.mean(test_loss)
test_accuracy = np.sum(test_correct) / np.sum(test_appearances)
test_inaccuracy = [(1 - (test_correct[i] / test_appearances[i])) for i in range(10)]
avg_test_inaccuracy = np.mean(test_inaccuracy)

print("EVALUATION RESULTS")
print("Average Loss: " + str(test_loss))
print("Overall Accuracy %: " + str(test_accuracy * 100))
print("Class-wise Error %: ")
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
for i, class_name in enumerate(classes):
    print("  " + class_name.rjust(10) + "\t" + str(test_inaccuracy[i] * 100))
print("Average Error %: " + str(avg_test_inaccuracy * 100))
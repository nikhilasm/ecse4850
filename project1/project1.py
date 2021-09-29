# Nikhilas Murthy
# ECSE 4850 Project 1
# Multi-class Logistic Regressor

import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Import data and preprocess to match expected vector dimensions and formats
def preprocessData(mode='train'):
    
    # Import images, convert to vector form, and assemble X^T
    XT = np.zeros((25112, 785))
    path = mode + "_data"
    for row, filename in enumerate(os.listdir(path)):
        img = mpimg.imread(os.path.join(path, filename))
        img = img.reshape([1, 784])
        img = img / 255
        x = np.append(img, 1)
        
        XT[row] = x
    
    # Load labels and convert to 1-of-k vector form, assemble T
    labels = readLabels(mode + "_label.txt")
    T = np.zeros((25112, 5))
    for row, label in enumerate(labels):
        T[row] = [int(i==label-1) for i in range(5)]
    
    return (tf.convert_to_tensor(XT), tf.convert_to_tensor(T))

# Load the labels for the data set from a file and return a list of integers
def readLabels(filename):
    with open(os.path.join("labels", filename), 'r') as f:
        lines = f.readlines()
    
    labels = [int(s.strip()) for s in lines]
    return labels

# Initialize Theta to have random values chosen from a Gaussian distribution of
# N(0, 1)
def initialize():
    Theta = np.random.normal(0, 1, (785, 5))
    Theta[784] = [0,0,0,0,0]
    
    return tf.convert_to_tensor(Theta)

# Main training loop function. Calculates loss and loss gradient, updates weights
# via gradient descent method, and collects performance data for plotting
# afterwards
def train(X_data, Y_data, Theta, X_test_data, Y_test_data, epochs=10, learning_rate=0.01):
    data = {"train_loss": [],
            "test_loss": [],
            "train_error_1": [],
            "train_error_2": [],
            "train_error_3": [],
            "train_error_4": [],
            "train_error_5": [],
            "test_error_1": [],
            "test_error_2": [],
            "test_error_3": [],
            "test_error_4": [],
            "test_error_5": [],
            }

    # Begin main training loop
    for epoch in range(epochs):
        print("Epoch #{}:".format(epoch+1))
        
        # Compute training loss and accuracy
        digit_counts = [0,0,0,0,0]
        errors = [0,0,0,0,0]
        loss = 0
        for m in range(25112):
            k = np.where(Y_data[m].numpy() == 1)[0].item()
            X_m = tf.gather(X_data, [m], axis=0)
            loss += log_softmax(X_m, Theta, k).numpy()
            if not np.isfinite(loss):
                print(X_m)
                print(Theta)
                print(k)
                print(softmax(X_m, Theta, k))
                raise ValueError
            
            classification = tf.math.argmax(tf.linalg.matmul(X_m, Theta), 1)
            if classification.numpy().item() != k:
                errors[k] += 1
            digit_counts[k] += 1
        
        loss = -1*loss
        errors = [errors[i]/digit_counts[i] for i in range(5)]
        print("    Train Loss = {}".format(loss.item()))
        print("    Train Inaccuracy = {}".format([round(e, 4) for e in errors]))
        
        # Compute test loss and accuracy
        test_loss, test_errors = validate(X_test_data, Y_test_data, Theta)
        print("    Test Loss = {}".format(test_loss.item()))
        print("    Test Inaccuracy = {}".format([round(e, 4) for e in test_errors]))
        
        # Compute the loss function gradient
        gradients = []
        for k in range(5):
            gradient = tf.zeros([785, 1], dtype=tf.float64)
            for m in range(25112):
                X_m = tf.gather(X_data, [m], axis=0)
                scalar = Y_data[m][k] - softmax(X_m, Theta, k)
                gradient += tf.multiply(scalar, tf.transpose(X_m))
            gradients.append(tf.multiply(-1, gradient))
       
        gradient_matrix = tf.concat(gradients, axis=1)
       
        # Update weights via gradient descent method
        Theta = Theta - tf.cast(tf.multiply(learning_rate, (gradient_matrix)), tf.float64)
        
        # Collect data values for plotting later
        data["train_loss"].append(loss)
        data["test_loss"].append(test_loss)
        for i in range(5):
            data["train_error_" + str(i+1)].append(errors[i])
            data["test_error_" + str(i+1)].append(test_errors[i])

    return (Theta, data)

# Compute the log of the softmax function, with numerical stability considerations
# as detailed in the report. The input X is equal to X^T[m] in the equations.
def log_softmax(X, Theta, k):
    x = tf.cast(tf.linalg.matmul(X, Theta), tf.float64)
    x = x - tf.math.reduce_max(x)
    log_term = tf.math.log(tf.reduce_sum(tf.exp(x)))
    result = x - log_term
    
    return result[0][k]

# Compute the softmax function, with numerical stability considerations as
# detailed in the report. The input X is equal to X^T[m] in the equations.
def softmax(X, Theta, k):
    x = tf.cast(tf.linalg.matmul(X, Theta), tf.float64)
    x = x - tf.math.reduce_max(x)
    num = tf.cast(tf.exp(x), tf.float64)
    den = tf.cast(tf.math.reduce_sum(num), tf.float64)
    result = num / den
    
    return result[0][k]

# Compute the loss and accuracy of the model when classifying the given test data
def validate(XT, T, W):    
    # Iterate through training data, classify, count occurrences of digits
    errors = [0,0,0,0,0]
    counts = [0,0,0,0,0]
    loss = 0
    for m in range(4982):
        k = np.where(T[m].numpy() == 1)[0].item()
        X_m = tf.gather(XT, [m], axis=0)
        loss += log_softmax(X_m, W, k).numpy()
        
        k = np.where(T[m].numpy() == 1)[0].item()
        classification = tf.math.argmax(tf.linalg.matmul(X_m, W), 1)
        if classification.numpy().item() != k:
            errors[k] += 1
        counts[k] += 1
    
    errors = [errors[i]/counts[i] for i in range(5)]
    return (-1*loss, errors)

# MAIN FUNCTION
if __name__ == "__main__":
    # Import data, preprocess, and initialize Theta
    print("[*] Importing and preprocessing input data and labels...")
    XT, T = preprocessData('train')
    XT_test, T_test = preprocessData('test')
    Theta = initialize()
    
    # Plot initial weights as images
    for k in range(5):
        W_k = tf.gather(Theta, [k], axis=1)
        W_k = tf.slice(W_k, [0, 0], [784, 1]).numpy()
    
        img = W_k.reshape(28, 28)
        plt.imshow(img)
        plt.colorbar()
        plt.show()
    
    # Train model and save the returned weights
    print("[*] Training model with learning_rate=0.01 ...")
    W, data = train(XT, T, Theta, XT_test, T_test, epochs=25)
    filehandler = open("multiclass_parameters.txt", "wb")
    pickle.dump(W, filehandler)
    filehandler.close()
    
    # Plot the final weights as images
    for k in range(5):
        W_k = tf.gather(W, [k], axis=1)
        W_k = tf.slice(W_k, [0, 0], [784, 1]).numpy()
    
        img = W_k.reshape(28, 28)
        plt.imshow(img)
        plt.colorbar()
        plt.show()
    
    # Plot the training and testing loss over time
    loss_plot = plt.figure(1)
    plt.plot(range(25), data["train_loss"], ".-b", range(25), data["test_loss"], ".-r")
    plt.legend(["Training Loss", "Testing Loss"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
    
    # Plot the training and testing inaccuracy for each class over time
    for k in range(5):
        plot = plt.figure(k+2)
        plt.plot(range(25), data["train_error_" + str(k+1)], ".-b", range(25), data["test_error_" + str(k+1)], ".-r")
        plt.legend(["Training Error", "Testing Error"])
        plt.title("k=" + str(k+1))
        plt.xlabel("epoch")
        plt.ylabel("classification inaccuracy")
        plt.show()
    
    
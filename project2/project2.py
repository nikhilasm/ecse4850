# Nikhilas Murthy
# ECSE 4850 Project 2
# Multi-class Neural Network Classifier (2 hidden layers)

import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random as rand
import math

LEARNING_RATE = 0.008

# Import data and preprocess to match expected vector dimensions and formats
def preprocessData(mode='train'):
    
    if mode=='train':
        size=50000
    else:
        size=5000
    
    # Import images, convert to vector form, and assemble X^T
    X = np.zeros((784, size))
    path = mode + "_data"
    for col, filename in enumerate(os.listdir(path)):
        img = mpimg.imread(os.path.join(path, filename))
        img = img.reshape([1, 784])
        img = img / 255
        
        X[:, col] = img
    
    # Load labels and convert to 1-of-k vector form, assemble T
    labels = readLabels(mode + "_label.txt")
    Y = np.zeros((10, size))
    for col, label in enumerate(labels):
        Y[:, col] = [int(i==label) for i in range(10)]
    
    return (tf.cast(tf.convert_to_tensor(X), tf.float32), tf.cast(tf.convert_to_tensor(Y), tf.float32))

# Load the labels for the data set from a file and return a list of integers
def readLabels(filename):
    with open(os.path.join("labels", filename), 'r') as f:
        lines = f.readlines()
    
    labels = [int(s.strip()) for s in lines]
    return labels

# Returns a list of 1000 lists of 50 image IDs (1000 batches)
def getBatches(epoch):
    ids = list(range(50000))
    rand.seed(epoch)
    rand.shuffle(ids)
    return np.split(np.array(ids), 1000, axis=0)

# Main training loop. Continues to train the model until error rate <= 0.7 or
# the maximum epoch threshold is hit.
def train(X_train, Y_train, X_test, Y_test):
    global LEARNING_RATE
    X_test = tf.reshape(tf.transpose(X_test), [5000, 1, 784])
    Y_test = tf.reshape(tf.transpose(Y_test), [5000, 10, 1])
    
    # DEFINE LAYERS AND INITIALIZATION
    #   H1 has 100 nodes and will be fed a set of 50 1x784 vectors
    #       H1 has 784x100 weight matrix and a 100x1 bias vector
    #   H2 has 100 nodes and will be fed a set of 50 1x100 vectors
    #       H2 has 100x100 weight matrix and a 100x1 bias vector
    #   Y_hat has 10 nodes and will be fed a set of 50 1x100 vectors
    #       Y_hat has 100x10 weight matrix and a 10x1 bias vector
    # All weights are initialized to ~N(0, 0.1) and biases are initialized to 0
    H1 = tf.keras.layers.Dense(
        units=100,
        activation='relu',
        use_bias=True,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
        bias_initializer=tf.keras.initializers.Zeros()
    )
    
    H2 = tf.keras.layers.Dense(
        units=100,
        activation='relu',
        use_bias=True,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
        bias_initializer=tf.keras.initializers.Zeros()
    )
    
    Y_hat = tf.keras.layers.Dense(
        units=10,
        activation='softmax',
        use_bias=True,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
        bias_initializer=tf.keras.initializers.Zeros()
    )
    
    data = {"train_loss": [], "test_loss": [],
            "train_error_0": [], "test_error_0": [],
            "train_error_1": [], "test_error_1": [],
            "train_error_2": [], "test_error_2": [],
            "train_error_3": [], "test_error_3": [],
            "train_error_4": [], "test_error_4": [],
            "train_error_5": [], "test_error_5": [],
            "train_error_6": [], "test_error_6": [],
            "train_error_7": [], "test_error_7": [],
            "train_error_8": [], "test_error_8": [],
            "train_error_9": [], "test_error_9": [],
            "avg_train_error": [], "avg_test_error": []
            }

    epoch = 0
    while True:
        for batch_num, batch in enumerate(getBatches(epoch)):
            X = tf.reshape(tf.transpose(tf.gather(X_train, batch, axis=1)), [50, 1, 784])
            Y = tf.reshape(tf.transpose(tf.gather(Y_train, batch, axis=1)), [50, 10, 1])
        
            # FORWARD PROPAGATION =====================================================
            # Run NN on training data
            H1_out = H1(X)
            H2_out = H2(H1_out)
            Y_out = Y_hat(H2_out)
            
            H1_out = tf.transpose(H1_out, [0, 2, 1])
            H2_out = tf.transpose(H2_out, [0, 2, 1])
            Y_out = tf.transpose(Y_out, [0, 2, 1])
            
            # Run NN on testing data
            H1_out_test = H1(X_test)
            H2_out_test = H2(H1_out_test)
            Y_out_test = Y_hat(H2_out_test)
            
            H1_out_test = tf.transpose(H1_out_test, [0, 2, 1])
            H2_out_test = tf.transpose(H2_out_test, [0, 2, 1])
            Y_out_test = tf.transpose(Y_out_test, [0, 2, 1])
            
            X = tf.transpose(X, [0, 2, 1])
            
            # COMPUTE LOSS AND TRAIN/TEST INACCURACY ==================================
            # Training loss and inaccuracy
            digit_appearances = [0,0,0,0,0,0,0,0,0,0]
            digit_errors = [0,0,0,0,0,0,0,0,0,0]
            train_loss = 0
            for m in range(50):
                k = np.where(Y[m].numpy() == 1)[0].item()
                train_loss += tf.math.log(Y_out[m, k, 0]).numpy()
                
                digit_appearances[k] += 1
                classification = tf.math.argmax(Y_out[m])
                if classification.numpy().item() != k:
                    digit_errors[k] += 1
            
            train_loss = -1*train_loss
            train_inaccuracy = [digit_errors[i]/digit_appearances[i] if digit_appearances[i] != 0 else 0 for i in range(10)]
            avg_train_inaccuracy = sum(train_inaccuracy)/10
            
            # Testing loss and inaccuracy
            digit_appearances = [0,0,0,0,0,0,0,0,0,0]
            digit_errors = [0,0,0,0,0,0,0,0,0,0]
            test_loss = 0
            for m in range(5000):
                k = np.where(Y_test[m].numpy() == 1)[0].item()
                test_loss += tf.math.log(Y_out_test[m, k, 0]).numpy()
                
                digit_appearances[k] += 1
                classification = tf.math.argmax(Y_out_test[m])
                if classification.numpy().item() != k:
                    digit_errors[k] += 1
            
            test_loss = -1*test_loss
            test_inaccuracy = [digit_errors[i]/digit_appearances[i] for i in range(10)]
            avg_test_inaccuracy = sum(test_inaccuracy)/10
            
            data['train_loss'].append(train_loss)
            data['test_loss'].append(test_loss)
            data['avg_train_error'].append(avg_train_inaccuracy)
            data['avg_test_error'].append(avg_test_inaccuracy)
            for k in range(10):
                data['train_error_' + str(k)].append(train_inaccuracy[k])
                data['test_error_' + str(k)].append(test_inaccuracy[k])
            
            print("Epoch: {0}, Batch: {1}".format(epoch, batch_num))
            print("    Train Loss: {0}, Train Inaccuracy: {1}".format(round(train_loss, 2), [round(e, 2) for e in train_inaccuracy]))
            print("    Test Loss: {0}, Test Inaccuracy %: {1}".format(round(test_loss, 2), [round(e, 2) for e in test_inaccuracy]))
            
            # STOPPING CRITERION
            if avg_test_inaccuracy <= 0.07 or epoch == 8:
                W1, W1_0 = H1.get_weights()
                W2, W2_0 = H2.get_weights()
                W3, W3_0 = Y_hat.get_weights()
                return (data, [W1, W1_0, W2, W2_0, W3, W3_0])
            
            # BACKWARD PROPAGATION ====================================================
            
            # Compute gradient of Y_out, H2, W3, and W3_0
            W3, W3_0 = Y_hat.get_weights()
            Y_out_gradient = -1*tf.cast(tf.math.divide(Y, Y_out), tf.float32)
            H2_gradient, W3_gradient, W3_0_gradient = computeGradient_Yhat(Y_out_gradient, W3, H2_out, Y_out)
            
            # Compute gradient of H1, W2, and W2_0
            W2, W2_0 = H2.get_weights()
            H1_gradient, W2_gradient, W2_0_gradient = computeGradient_H2(H2_gradient, W2, H1_out, H2_out)
            
            # Compute gradient of W1 and W1_0
            W1, W1_0 = H1.get_weights()
            W1_gradient, W1_0_gradient = computeGradient_H1(H1_gradient, W1, X, H1_out)
            
            # Compute average gradients for weights over the 50 data points
            W1_gradient_avg = tf.reduce_mean(tf.cast(W1_gradient, tf.float32), 0)
            W1_0_gradient_avg = tf.reduce_mean(tf.cast(W1_0_gradient, tf.float32), 0)
            W2_gradient_avg = tf.reduce_mean(tf.cast(W2_gradient, tf.float32), 0)
            W2_0_gradient_avg = tf.reduce_mean(tf.cast(W2_0_gradient, tf.float32), 0)
            W3_gradient_avg = tf.reduce_mean(tf.cast(W3_gradient, tf.float32), 0)
            W3_0_gradient_avg = tf.reduce_mean(tf.cast(W3_0_gradient, tf.float32), 0)
            
            # Update weights and apply to the layers
            W1 = tf.subtract(W1, 0.5*LEARNING_RATE*W1_gradient_avg)
            W1_0 = tf.subtract(W1_0, tf.reshape(0.5*LEARNING_RATE*W1_0_gradient_avg, [100]))
            H1.set_weights([W1.numpy(), W1_0.numpy()])
            
            W2 = tf.subtract(W2, LEARNING_RATE*W2_gradient_avg)
            W2_0 = tf.subtract(W2_0, tf.reshape(LEARNING_RATE*W2_0_gradient_avg, [100]))
            H2.set_weights([W2.numpy(), W2_0.numpy()])
            
            W3 = tf.subtract(W3, 2*LEARNING_RATE*W3_gradient_avg)
            W3_0 = tf.subtract(W3_0, tf.reshape(2*LEARNING_RATE*W3_0_gradient_avg, [10]))
            Y_hat.set_weights([W3.numpy(), W3_0.numpy()])
        
        epoch += 1
        LEARNING_RATE = 0.008*(1/math.sqrt(epoch+1))
    

# Computes the gradients of the output layer Y_hat:
#   gradient of H2, gradient of W3, and gradient of W3_0
def computeGradient_Yhat(Y_out_gradient, W3, H2_out, Y_out):
    dSigmaZ_dZ = np.zeros([50, 10, 10])
    dZ_dW = np.zeros([50, 100, 10, 10])
    for m in range(50):
        SigmaZ = Y_out[m].numpy()
        
        for i in range(10):
            for j in range(10):
                if i == j:
                    dSigmaZ_dZ[m, i, j] = SigmaZ[i, 0]*(1 - SigmaZ[i, 0])
                    dZ_dW[m, :, i, j] = H2_out.numpy()[m, :, 0]
                else:
                    dSigmaZ_dZ[m, i, j] = -1*SigmaZ[j, 0]*SigmaZ[i, 0]    
    
    W3_tiled = tf.reshape(tf.tile(W3, [50, 1]), [50, 100, 10])
    H2_gradient = tf.matmul(tf.matmul(W3_tiled, dSigmaZ_dZ), Y_out_gradient)
    W3_0_gradient = tf.matmul(dSigmaZ_dZ, tf.cast(Y_out_gradient, tf.float64))
    
    z = tf.matmul(dSigmaZ_dZ, tf.cast(Y_out_gradient, tf.float64))
    W3_gradient = np.zeros([50, 100, 10])
    for m in range(50):
        W3_gradient[m] = tf.reshape(tf.matmul(dZ_dW[m], z[m]), [100, 10])

    return(H2_gradient, W3_gradient, W3_0_gradient)

# Computes the gradients of the hidden layer H2:
#   gradient of H1, gradient of W2, and gradient of W2_0
def computeGradient_H2(H2_gradient, W2, H1_out, H2_out):
    dPhiZ_dZ = np.zeros([50, 100, 100])
    dZ_dW = np.zeros([50, 100, 100, 100])
    for m in range(50):
        PhiZ = H2_out[m].numpy()
        
        #Only apply values on the 'diagonals'
        for i in range(100):
            if PhiZ[i, 0] > 0:
                dPhiZ_dZ[m, i, i] = 1
            dZ_dW[m, :, i, i] = H1_out.numpy()[m, :, 0]   
    
    W2_tiled = tf.reshape(tf.tile(W2, [50, 1]), [50, 100, 100])
    H1_gradient = tf.matmul(tf.matmul(W2_tiled, dPhiZ_dZ), H2_gradient)
    W2_0_gradient = tf.matmul(dPhiZ_dZ, tf.cast(H2_gradient, tf.float64))
    
    z = tf.matmul(dPhiZ_dZ, tf.cast(H2_gradient, tf.float64))
    W2_gradient = np.zeros([50, 100, 100])
    for m in range(50):
        W2_gradient[m] = tf.reshape(tf.matmul(dZ_dW[m], z[m]), [100, 100])

    return(H1_gradient, W2_gradient, W2_0_gradient)

# Computes the gradients of the hidden layer H1:
#   gradient of W1, and gradient of W1_0
def computeGradient_H1(H1_gradient, W1, X, H1_out):
    dPhiZ_dZ = np.zeros([50, 100, 100])
    dZ_dW = np.zeros([50, 784, 100, 100])
    for m in range(50):
        PhiZ = H1_out[m].numpy()
        
        #Only apply values on the 'diagonals'
        for i in range(100):
            if PhiZ[i, 0] > 0:
                dPhiZ_dZ[m, i, i] = 1
            dZ_dW[m, :, i, i] = X.numpy()[m, :, 0]   
    
    W1_0_gradient = tf.matmul(dPhiZ_dZ, tf.cast(H1_gradient, tf.float64))
    
    z = tf.matmul(dPhiZ_dZ, tf.cast(H1_gradient, tf.float64))
    W1_gradient = np.zeros([50, 784, 100])
    for m in range(50):
        W1_gradient[m] = tf.reshape(tf.matmul(dZ_dW[m], z[m]), [784, 100])

    return(W1_gradient, W1_0_gradient)


# MAIN FUNCTION: Imports data, trains model, saves parameters, and creates plots
if __name__ == "__main__":
    print("[*] Importing & preprocessing data...")
    X, Y = preprocessData('train')
    X_test, Y_test = preprocessData('test')
    
    print("[*] Training NN with adaptive learning rate starting at learning_rate={}".format(LEARNING_RATE))
    data, Theta = train(X, Y, X_test, Y_test)
    
    #Save parameters
    print("[*] Saving parameters...")
    f = open("nn_parameters.txt", "wb")
    pickle.dump(Theta, f, protocol=2)
    f.close()
    
    iterations = len(data['test_loss'])
    
    print("[*] Creating plots...")
    # Plot the training and testing loss over time
    fig, ax = plt.subplots()
    ax.plot(range(iterations), data["train_loss"], "-b")
    ax.set_xlabel("iteration")
    ax.set_ylabel("train loss", color="blue")
    ax2 = ax.twinx()
    ax2.plot(range(iterations), data["test_loss"], "-r")
    ax2.set_ylabel("test loss", color="red")
    plt.show()
    
    # Plot the training and testing inaccuracy for each class over time
    for k in range(10):
        plot = plt.figure(k+2)
        plt.plot(range(iterations), data["train_error_" + str(k)], "-b", range(iterations), data["test_error_" + str(k)], "-r")
        plt.legend(["Training Error", "Testing Error"])
        plt.title("k=" + str(k))
        plt.xlabel("iteration")
        plt.ylabel("classification inaccuracy")
        plt.show()
    
    # Plot the average training and testing classification inaccuracy over time
    loss_plot = plt.figure(12)
    plt.plot(range(iterations), data["avg_train_error"], "-b", range(iterations), data["avg_test_error"], "-r")
    plt.legend(["Average Training Error", "Average Testing Error"])
    plt.xlabel("iteration")
    plt.ylabel("classification inaccuracy")
    plt.show()
    
    
    

    
# Nikhilas Murthy
# ECSE 4850 Final Project
# Plot and Visualization Code

import seaborn as sn
import matplotlib.pyplot as plt

# The makePlots() function will create plots of model loss, overall model
# accuracy, and class-wise accuracy over all training epochs.
# Arguments: data - dictionary with various data sequences
#            classes - a list of all 11 class labels
#            epochs - an integer indicating the number of epochs the model
#                     was trained for
# Returns: None
def makePlots(data, classes, epochs):
    # Create the loss plot
    plt.plot(range(epochs), data["train_loss"], "-b", range(epochs), data["valid_loss"], "-r")
    plt.legend(["Avg Training Loss", "Avg Validation Loss"])
    plt.title("Average Model Loss vs. Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.show()
    
    # Create the overall accuracy plot
    plt.plot(range(epochs), data["train_accuracy"], "-b", range(epochs), data["valid_accuracy"], "-r")
    plt.legend(["Training Accuracy", "Validation Accuracy"])
    plt.title("Overall Model Accuracy vs. Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()
    
    # Create the class-wise accuracy plots
    for k in range(11):
        plot = plt.figure(k+2)
        plt.plot(range(epochs), data["train_accuracy_" + str(k)], "-b", range(epochs), data["valid_accuracy_" + str(k)], "-r")
        plt.legend(["Training Accuracy", "Validation Accuracy"])
        plt.title("Class '" + str(classes[k]) + "' Accuracy vs. Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.show()

# The makeConfusionMatrix() function will generate a visual heatmap of the
# confusion matrix for the model's predictions.
# Arguments: matrix - an 11x11 2D array containing the number of predictions
#                     for each class
#            classes - a list of all 11 class labels
# Returns: None
def makeConfusionMatrix(matrix, classes):
    plt.figure(figsize=(8, 6))
    sn.heatmap(matrix, annot=True,
               xticklabels=classes,
               yticklabels=classes,
               fmt='d')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

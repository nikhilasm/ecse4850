# Nikhilas Murthy
# ECSE 4850 Final Project
# Using a CNN-LSTM model for human action recognition

import numpy as np
import tensorflow as tf
import plotter

###############################################################################
# HYPER PARAMETERS
# These are manually tuned to improve the performance of the model.
EPOCHS = 25
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
IMG_FLIP_PROB = 0.5


###############################################################################
# Load all of the data into RAM for quick and simple access. Slices of these
# arrays will be copied to the GPU during training/testing since the GPU does
# not have enough memory to store all of the data.
TRAIN_DATA = np.load("yt_action_data_train.npy")
TRAIN_LABELS = np.load("yt_action_labels_train.npy")
TRAIN_LENGTH = TRAIN_LABELS.shape[0]
VALID_DATA = np.load("yt_action_data_valid.npy")
VALID_LABELS = np.load("yt_action_labels_valid.npy")
VALID_LENGTH = VALID_LABELS.shape[0]

# MAIN MODEL CLASS
# This is a subclass of the tf.keras.Model class and defines the architecture
# of the model:
#       LAYER       DETAILS         ACTIVATION
#   1.  Conv2D      32 @ 3x3        ReLU
#   2.  MaxPool2D   2x2
#   3.  Conv2D      32 @ 3x3        ReLU
#   4.  MaxPool2D   2x2
#   5.  Conv2D      64 @ 3x3        ReLU
#   6.  MaxPool2D   2x2
#   7.  Conv2D      128 @ 3x3       ReLU
#   8.  Conv2D      128 @ 3x3       ReLU
#   9.  MaxPool2D   2x2
#   10. Flatten
#   11. Dense       256             ReLU
#   12. LSTM        128
#   13. Dense       11              Linear
class DeepClassifier(tf.keras.Model):
    def __init__(self):
        super(DeepClassifier, self).__init__()
        self.conv1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(
            filters=32, input_shape=[64, 64, 3], kernel_size=3,
            kernel_initializer='random_normal', kernel_regularizer='l2',
            padding='same', activation='relu'))
        self.max1 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(2, 2))
        self.conv2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, kernel_initializer='random_normal',
            kernel_regularizer='l2', padding='same', activation='relu'))
        self.max2 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(2, 2))
        self.conv3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, kernel_initializer='random_normal',
            kernel_regularizer='l2', padding='same', activation='relu'))
        self.max3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(2, 2))
        self.conv4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(
            filters=128, kernel_size=3, kernel_initializer='random_normal',
            kernel_regularizer='l2', padding='same', activation='relu'))
        self.conv5 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(
            filters=128, kernel_size=3, kernel_initializer='random_normal',
            kernel_regularizer='l2', padding='same', activation='relu'))
        self.max4 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(2, 2))
        self.flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
        self.lin = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(
            units=256, kernel_initializer='random_normal', kernel_regularizer='l2', activation='relu'))
        self.lstm = tf.keras.layers.LSTM(128)
        self.out = tf.keras.layers.Dense(11, kernel_initializer='random_normal', kernel_regularizer='l2')
        
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.conv3(x)
        x = self.max3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.max4(x)
        x = self.flatten(x)
        x = self.lin(x)
        x = self.lstm(x)
        return self.out(x)
        
# trainDataGenerator() will return batches of training data in the form
# (videos, labels) where videos is a 5D tensor and labels is a 1D list. The
# data generator will create a list of random indices and will use those indices
# to access slices of the training data for return.
def trainDataGenerator():
    # Generate a random sequence for training samples to decouple epochs
    rng = np.random.default_rng()
    order = rng.choice(TRAIN_LENGTH, TRAIN_LENGTH, replace=False)
    cur_idx = 0
    end_idx = 0 + BATCH_SIZE
    while cur_idx < TRAIN_LENGTH:
        # Normalize image data
        videos = TRAIN_DATA[order[cur_idx:end_idx]] / 255.0
        # Randomly horizontally flip data samples
        for i in range(len(videos)):
            if np.random.uniform() < IMG_FLIP_PROB:
                videos[i] = np.flip(videos[i], 2)
        # Generate one-hot labels
        labels = tf.one_hot(TRAIN_LABELS[order[cur_idx:end_idx]], 11)
        yield (videos, labels)
        # Update indices
        cur_idx = end_idx
        end_idx = min(TRAIN_LENGTH, cur_idx + BATCH_SIZE)

# validDataGenerator() will return batches of validation data in the form
# (videos, labels) where videos is a 5D tensor and labels is a 1D list. The
# data generator will access slices of the validation data for return.
def validDataGenerator():
    order = np.array(range(VALID_LENGTH))
    cur_idx = 0
    end_idx = 0 + BATCH_SIZE
    while cur_idx < VALID_LENGTH:
        # Normalize image data
        videos = VALID_DATA[order[cur_idx:end_idx]] / 255.0
        # Generate one-hot labels
        labels = tf.one_hot(VALID_LABELS[order[cur_idx:end_idx]], 11)
        yield (videos, labels)
        # Update indices
        cur_idx = end_idx
        end_idx = min(VALID_LENGTH, cur_idx + BATCH_SIZE)


if __name__ == "__main__":
    # Enable GPU memory growth if it is not enabled already. This can fix some
    # memory limit warnings.
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    
    classes = ["b_shooting", "cycling", "diving", "g_swinging", "h_riding", "s_juggling",
               "swinging", "t_swinging", "t_jumping", "v_spiking", "d_walking"]
    
    # Create the two Tensorflow dataset objects and initialize the model
    train_ds = tf.data.Dataset.from_generator(trainDataGenerator, output_types=(tf.float32, tf.float32))
    valid_ds = tf.data.Dataset.from_generator(validDataGenerator, output_types=(tf.float32, tf.float32))
    model = DeepClassifier()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    # Initialize our data structure to store metrics for plotting
    plot_data = {"train_loss": [], "valid_loss": [],
                 "train_accuracy": [], "valid_accuracy": [],
                 "train_accuracy_0": [], "valid_accuracy_0": [],
                 "train_accuracy_1": [], "valid_accuracy_1": [],
                 "train_accuracy_2": [], "valid_accuracy_2": [],
                 "train_accuracy_3": [], "valid_accuracy_3": [],
                 "train_accuracy_4": [], "valid_accuracy_4": [],
                 "train_accuracy_5": [], "valid_accuracy_5": [],
                 "train_accuracy_6": [], "valid_accuracy_6": [],
                 "train_accuracy_7": [], "valid_accuracy_7": [],
                 "train_accuracy_8": [], "valid_accuracy_8": [],
                 "train_accuracy_9": [], "valid_accuracy_9": [],
                 "train_accuracy_10": [], "valid_accuracy_10": []
                 }
    
    # MAIN TRAINING LOOP
    for epoch in range(EPOCHS):
        train_loss = np.array([])
        valid_loss = np.array([])
        
        train_correct = np.array([0,0,0,0,0,0,0,0,0,0,0])
        train_appearances = np.array([0,0,0,0,0,0,0,0,0,0,0])
        valid_correct = np.array([0,0,0,0,0,0,0,0,0,0,0])
        valid_appearances = np.array([0,0,0,0,0,0,0,0,0,0,0])
        
        # TRAINING STEP
        # Iterate through the training data, compute loss, and update weights
        for videos, labels in train_ds:
            labels = tf.stop_gradient(labels)
            # Compute loss and gradient, apply gradient to model weights
            with tf.GradientTape() as T:
                pred = model(videos, training=True)
                loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred)
            model_gradient = T.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(model_gradient, model.trainable_variables))
            
            # Save loss and other data for metrics calculations, such as class-
            # wise label appearances and correct predictions
            train_loss = np.append(train_loss, loss)
            pred = tf.one_hot(tf.math.argmax(pred, 1), 11)
            for k in range(11):
                correct = tf.reduce_sum(tf.multiply(pred[:, k], labels[:, k]))
                count = tf.reduce_sum(labels[:, k])
                train_correct[k] += correct
                train_appearances[k] += count
        
        
        # If this is the final epoch, save all predictions and label appearances
        # during validation for the creation of a confusion matrix
        if epoch+1 == EPOCHS:
            all_predictions = np.array([])
            all_labels = np.array([])
        
        # VALIDATION STEP
        # Iterate through the validation data, compute loss and accuracy
        for videos, labels in valid_ds:
            # Compute loss
            pred = model(videos, training=False)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred)
            
            # If this is the final epoch, save all predictions and label appearances
            if epoch+1 == EPOCHS:
                all_predictions = np.append(all_predictions, tf.argmax(pred, axis=1).numpy())
                all_labels = np.append(all_labels, tf.argmax(labels, axis=1).numpy())
            
            # Save loss and other data for metrics calculations, such as class-
            # wise label appearances and correct predictions
            valid_loss = np.append(valid_loss, loss)
            pred = tf.one_hot(tf.math.argmax(pred, 1), 11)
            for k in range(11):
                correct = tf.reduce_sum(tf.multiply(pred[:, k], labels[:, k]))
                count = tf.reduce_sum(labels[:, k])
                valid_correct[k] += correct
                valid_appearances[k] += count
                
        # METRIC COMPUTATIONS
        # Compute various metrics for plotting, including average training/validation
        # loss for this epoch, overall model accuracy for this epoch, and class-wise
        # accuracy for this epoch.
        train_loss = np.mean(train_loss)
        valid_loss = np.mean(valid_loss)
        train_accuracy = np.sum(train_correct) / np.sum(train_appearances)
        valid_accuracy = np.sum(valid_correct) / np.sum(valid_appearances)
        train_accuracy_cw = [(train_correct[i] / train_appearances[i]) for i in range(11)]
        valid_accuracy_cw = [(valid_correct[i] / valid_appearances[i]) for i in range(11)]
        
        # STORE DATA FOR PLOTTING
        # Add the computed metrics to our data structure for plotting later
        plot_data['train_loss'].append(train_loss)
        plot_data['train_accuracy'].append(train_accuracy)
        plot_data['valid_loss'].append(valid_loss)
        plot_data['valid_accuracy'].append(valid_accuracy)
        for k in range(11):
            plot_data["train_accuracy_" + str(k)].append(train_accuracy_cw[k])
            plot_data["valid_accuracy_" + str(k)].append(valid_accuracy_cw[k])
        
        # If this is the final epoch, create the confusion matrix
        if epoch+1 == EPOCHS:
            matrix = tf.math.confusion_matrix(all_labels, all_predictions)
            plotter.makeConfusionMatrix(matrix, classes)
        
        # PRINT STATISTICS
        # At the end of the epoch, print all computed metrics so that model
        # training can be monitored
        print("EPOCH {0}".format(epoch+1))
        print("    Train Loss: {0}, Train Accuracy: {1}".format(round(train_loss, 2), round(train_accuracy, 2)))
        print("    Classwise: {0}".format([round(e, 2) for e in train_accuracy_cw]))
        print("    Valid Loss: {0}, Valid Accuracy: {1}".format(round(valid_loss, 2), round(valid_accuracy, 2)))
        print("    Classwise: {0}".format([round(e, 2) for e in valid_accuracy_cw]))
    
    # Once training is complete, make all plots and save the model
    plotter.makePlots(plot_data, classes, EPOCHS)
    model.save("DeepClassifierModel")
    
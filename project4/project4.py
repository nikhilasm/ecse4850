# Nikhilas Murthy
# ECSE 4850 Programming Assignment 4
# Using CNN, LSTM, and MLP Regressor to estimate human poses

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

LEARNING_RATE = 0.01
BATCH_SIZE = 3
TRAIN_SAMPLE_MAX = 5964 #5964
VALID_SAMPLE_MAX = 1368 #1368
EPOCHS = 20

TRAIN_DATA = np.load("videoframes_clips_train.npy", mmap_mode="r")
TRAIN_LABEL = np.load("joint_3d_clips_train.npy", mmap_mode="r")
VALID_DATA = np.load("videoframes_clips_valid.npy", mmap_mode="r")
VALID_LABEL = np.load("joint_3d_clips_valid.npy", mmap_mode="r")

class DeepDynamic(tf.keras.Model):
    def __init__(self):
        super(DeepDynamic, self).__init__()
        self.conv1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=64, input_shape=[224, 224, 3], kernel_size=3, padding='same', activation='relu'))
        self.conv2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
        self.max1 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(2, 2))
        self.norm1 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        self.conv3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
        self.conv4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
        self.max2 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(2, 2))
        self.norm2 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        self.conv5 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
        self.conv6 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
        self.max3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(2, 2))
        self.norm3 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        self.conv7 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
        self.conv8 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
        self.max4 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(2, 2))
        self.norm4 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        self.conv9 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
        self.conv10 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
        self.max5 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(2, 2))
        self.norm5 = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        self.flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
        self.lin1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024, activation='relu'))
        self.lin2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024, activation='relu'))
        self.lstm = tf.keras.layers.LSTM(512, return_sequences=True)
        self.mlp1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(512, activation='relu'))
        self.mlp2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(256, activation='relu'))
        self.mlp3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(51, activation='linear'))
        self.reshape = tf.keras.layers.TimeDistributed(tf.keras.layers.Reshape([17, 3]))
        
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max1(x)
        x = self.norm1(x, training)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max2(x)
        x = self.norm2(x, training)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.max3(x)
        x = self.norm3(x, training)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.max4(x)
        x = self.norm4(x, training)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.max5(x)
        x = self.norm5(x, training)
        x = self.flatten(x)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lstm(x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        return self.reshape(x)

def trainDataGenerator():
    rng = np.random.default_rng()
    order = rng.choice(5964, TRAIN_SAMPLE_MAX, replace=False)
    np.random.shuffle(order)
    cur_idx = 0
    end_idx = 0 + BATCH_SIZE
    while cur_idx < TRAIN_SAMPLE_MAX:
        videos = TRAIN_DATA[order[cur_idx:end_idx]] / 255.0
        labels = TRAIN_LABEL[order[cur_idx:end_idx]]
        yield (videos, labels)
        cur_idx = end_idx
        end_idx = min(TRAIN_SAMPLE_MAX, cur_idx + BATCH_SIZE)

def validDataGenerator():
    rng = np.random.default_rng()
    order = rng.choice(1368, VALID_SAMPLE_MAX, replace=False)
    np.random.shuffle(order)
    cur_idx = 0
    end_idx = 0 + BATCH_SIZE
    while cur_idx < VALID_SAMPLE_MAX:
        videos = VALID_DATA[order[cur_idx:end_idx]] / 255.0
        labels = VALID_LABEL[order[cur_idx:end_idx]]
        yield (videos, labels)
        cur_idx = end_idx
        end_idx = min(VALID_SAMPLE_MAX, cur_idx + BATCH_SIZE)

# Generate plots of loss and mpjpe over epochs
def plot(data):
    plt.plot(range(EPOCHS), data["train_loss"], "-b", range(EPOCHS), data["valid_loss"], "-r")
    plt.legend(["Avg Training Loss", "Avg Validation Loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.show()
    
    plt.plot(range(EPOCHS), data["train_mpjpe"], "-b", range(EPOCHS), data["valid_mpjpe"], "-r")
    plt.legend(["Training MPJPE", "Validation MPJPE"])
    plt.xlabel("Epoch")
    plt.ylabel("MPJPE (mm)")
    plt.show()

def calculateMPJPE(label, pred):
    batch_size = label.shape[0]
    value = 0
    for i in range(batch_size):
        for j in range(8):
            for k in range(17):
                value += tf.norm(label[i, j, k] - pred[i, j, k], ord="euclidean").numpy()
    
    return 1000*(value / (batch_size*8*17))
    
##############################################################################
if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    
    train_ds = tf.data.Dataset.from_generator(trainDataGenerator, output_types=(tf.float32, tf.float32))
    valid_ds = tf.data.Dataset.from_generator(validDataGenerator, output_types=(tf.float32, tf.float32))
    model = DeepDynamic()
    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    mse = tf.keras.losses.MeanSquaredError()
    
    plot_data = {"train_loss": [], "valid_loss": [],
                 "train_mpjpe": [], "valid_mpjpe": []}

    # TRAINING LOOP
    for epoch in range(EPOCHS):
        train_loss = np.array([])
        valid_loss = np.array([])
        train_mpjpe = np.array([])
        valid_mpjpe = np.array([])
        
        # TRAINING STEP
        for videos, labels in train_ds:
            labels = tf.stop_gradient(labels)
            with tf.GradientTape() as T:
                pred = model(videos, training=True)
                loss = mse(labels, pred)
            model_gradient = T.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(model_gradient, model.trainable_variables))
            mpjpe = calculateMPJPE(labels, pred)
            
            train_loss = np.append(train_loss, loss)
            train_mpjpe = np.append(train_mpjpe, mpjpe)
        
        # TESTING STEP
        for videos, labels in valid_ds:
            pred = model(videos, training=False)
            loss = mse(labels, pred)
            mpjpe = calculateMPJPE(labels, pred)
            
            valid_loss = np.append(valid_loss, loss)
            valid_mpjpe = np.append(valid_mpjpe, mpjpe)
                
        # SAVE DATA FOR PLOTTING
        plot_data["train_loss"].append(np.mean(train_loss))
        plot_data["train_mpjpe"].append(np.mean(train_mpjpe))
        plot_data["valid_loss"].append(np.mean(valid_loss))
        plot_data["valid_mpjpe"].append(np.mean(valid_mpjpe))
        
        # PRINT STATUS
        print("EPOCH {0}".format(epoch+1))
        print("    Train Loss: {0}, Train MPJPE: {1}".format(round(np.mean(train_loss), 5), round(np.mean(train_mpjpe), 5)))
        print("    Valid Loss: {0}, Valid MPJPE: {1}".format(round(np.mean(valid_loss), 5), round(np.mean(valid_mpjpe), 5)))
              
    # get filters from the first layer
    plot(plot_data)
    model.save("DeepDynamicModel")
    
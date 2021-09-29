# Nikhilas Murthy
# ECSE 4850 Programming Assignment 3
# Convolutional Neural Network in Tensorflow

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

LEARNING_RATE = 0.01
BATCH_SIZE = 200
EPOCHS = 250
BASE_DROPOUT_RATE = 0.1
INIT_MEAN = 0.0
INIT_STDDEV = 1.0
INIT_SCALE = 0.01

class SmallRandomNormal(tf.keras.initializers.Initializer):
    def __init__(self, mean, stddev, seed):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
    
    def __call__(self, shape, dtype=None, **kwargs):
        return tf.multiply(INIT_SCALE, tf.random.normal(shape, mean=self.mean, stddev=self.stddev, dtype=dtype, seed=self.seed))

class ConvNet(tf.keras.Model):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[5, 5],
            input_shape=[32, 32, 3],
            activation='relu',
            use_bias=True,
            kernel_initializer=SmallRandomNormal(mean=INIT_MEAN, stddev=INIT_STDDEV, seed=1),
            bias_initializer=tf.keras.initializers.Zeros())
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.max1 = tf.keras.layers.MaxPool2D(2, 2)
        self.dropout1 = tf.keras.layers.Dropout(BASE_DROPOUT_RATE)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[5, 5],
            input_shape=[14, 14, 32],
            activation='relu',
            use_bias=True,
            kernel_initializer=SmallRandomNormal(mean=INIT_MEAN, stddev=INIT_STDDEV, seed=2),
            bias_initializer=tf.keras.initializers.Zeros())
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.max2 = tf.keras.layers.MaxPool2D(2, 2)
        self.dropout2 = tf.keras.layers.Dropout(BASE_DROPOUT_RATE + 0.1)
        self.conv3 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            input_shape=[5, 5, 32],
            activation='relu',
            use_bias=True,
            kernel_initializer=SmallRandomNormal(mean=INIT_MEAN, stddev=INIT_STDDEV, seed=3),
            bias_initializer=tf.keras.initializers.Zeros())
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.dropout3 = tf.keras.layers.Dropout(BASE_DROPOUT_RATE + 0.2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(
            units=10,
            kernel_initializer=SmallRandomNormal(mean=INIT_MEAN, stddev=INIT_STDDEV, seed=4),
            bias_initializer=tf.keras.initializers.Zeros())
        
        
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.batch_norm1(x, training)
        x = self.max1(x)
        x = self.dropout1(x, training)
        x = self.conv2(x)
        x = self.batch_norm2(x, training)
        x = self.max2(x)
        x = self.dropout2(x, training)
        x = self.conv3(x)
        x = self.batch_norm3(x, training)
        x = self.dropout3(x, training)
        x = self.flatten(x)
        return self.dense(x)
    

def importData():
    training_data = np.load('training_data.npy')
    training_data = tf.convert_to_tensor(training_data / 255.0)
    print(training_data)
    
    training_label = np.load('training_label.npy')
    training_label = tf.reshape(tf.one_hot(training_label, 10), [50000, 10])
    
    testing_data = np.load('testing_data.npy')
    testing_data = tf.convert_to_tensor(testing_data / 255.0)
    
    testing_label = np.load('testing_label.npy')
    testing_label = tf.reshape(tf.one_hot(testing_label, 10), [5000, 10])
    
    test_ds = tf.data.Dataset.from_tensor_slices((testing_data, testing_label)).batch(BATCH_SIZE)
    
    return (training_data, training_label, test_ds)

# Generate plots of accuracy and loss over epochs, and plot filters as images
def plot(data, filters, classes):
    # CREATE PLOTS FOR LOSS, ACCURACY, and AVERAGE INACCURACY
    plt.plot(range(EPOCHS), data["train_loss"], "-b", range(EPOCHS), data["test_loss"], "-r")
    plt.legend(["Avg Training Loss", "Avg Testing Loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.show()
    
    plt.plot(range(EPOCHS), data["train_accuracy"], "-b", range(EPOCHS), data["test_accuracy"], "-r")
    plt.legend(["Training Accuracy", "Testing Accuracy"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()
    
    plt.plot(range(EPOCHS), data["avg_train_inaccuracy"], "-b", range(EPOCHS), data["avg_test_inaccuracy"], "-r")
    plt.legend(["Avg Training Inaccuracy", "Avg Testing Inaccuracy"])
    plt.xlabel("Epoch")
    plt.ylabel("Error Rate")
    plt.show()
    
    # CREATE INACCURACY PLOTS FOR EACH CLASS
    fig, axs = plt.subplots(5, 2, figsize=(11, 19))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    for k, ax in enumerate(axs.flat):
        ax.plot(range(EPOCHS), data['train_inaccuracy_' + str(k)], "-b", range(EPOCHS), data['test_inaccuracy_' + str(k)], "-r")
        ax.set_title("'{0}' class".format(classes[k]))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error Rate")
        ax.legend(["Training Inaccuracy", "Testing Inaccuracy"])
    plt.show()
    
    # CREATE IMAGE PLOTS FOR FILTERS
    fig, axs = plt.subplots(4, 8, figsize=(15, 8))
    for i, ax in enumerate(axs.flat):
        ax.imshow(filters[:,:,:,i])
        ax.set_title("Filter {0}".format(i+1))
        ax.axis('off')
    plt.show()
    
##############################################################################
if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    
    tf.keras.backend.set_floatx('float64')
    
    x_train, y_train, test_ds = importData()
    model = ConvNet()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                #width_shift_range=0.1,
                #height_shift_range=0.1,
                horizontal_flip=True
                )
    
    plot_data = {"train_loss": [], "test_loss": [],
                 "train_accuracy": [], "test_accuracy": [],
                 "train_inaccuracy_0": [], "test_inaccuracy_0": [],
                 "train_inaccuracy_1": [], "test_inaccuracy_1": [],
                 "train_inaccuracy_2": [], "test_inaccuracy_2": [],
                 "train_inaccuracy_3": [], "test_inaccuracy_3": [],
                 "train_inaccuracy_4": [], "test_inaccuracy_4": [],
                 "train_inaccuracy_5": [], "test_inaccuracy_5": [],
                 "train_inaccuracy_6": [], "test_inaccuracy_6": [],
                 "train_inaccuracy_7": [], "test_inaccuracy_7": [],
                 "train_inaccuracy_8": [], "test_inaccuracy_8": [],
                 "train_inaccuracy_9": [], "test_inaccuracy_9": [],
                 "avg_train_inaccuracy": [], "avg_test_inaccuracy": []
                 }
    
    # TRAINING LOOP
    for epoch in range(EPOCHS):
        train_loss = np.array([])
        test_loss = np.array([])
        
        train_correct = np.array([0,0,0,0,0,0,0,0,0,0])
        train_appearances = np.array([0,0,0,0,0,0,0,0,0,0])
        test_correct = np.array([0,0,0,0,0,0,0,0,0,0])
        test_appearances = np.array([0,0,0,0,0,0,0,0,0,0])
        
        # TRAINING STEP
        batches = 0
        for images, labels in datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, seed=epoch):
            labels = tf.stop_gradient(labels)
            with tf.GradientTape() as T:
                pred = model(images, training=True)
                loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred)
            model_gradient = T.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(model_gradient, model.trainable_variables))
            
            train_loss = np.append(train_loss, loss)
            
            pred = tf.one_hot(tf.math.argmax(pred, 1), 10)
            for k in range(10):
                correct = tf.reduce_sum(tf.multiply(pred[:, k], labels[:, k]))
                count = tf.reduce_sum(labels[:, k])
                train_correct[k] += correct
                train_appearances[k] += count
            
            batches += 1
            if batches >= len(x_train) / BATCH_SIZE:
                break
        
        # TESTING STEP
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
                
        # COMPUTE METRICS
        train_loss = np.mean(train_loss)
        test_loss = np.mean(test_loss)
        
        train_accuracy = np.sum(train_correct) / np.sum(train_appearances)
        test_accuracy = np.sum(test_correct) / np.sum(test_appearances)
        
        train_inaccuracy = [(1 - (train_correct[i] / train_appearances[i])) for i in range(10)]
        avg_train_inaccuracy = np.mean(train_inaccuracy)
        test_inaccuracy = [(1 - (test_correct[i] / test_appearances[i])) for i in range(10)]
        avg_test_inaccuracy = np.mean(test_inaccuracy)
        
        # STORE DATA FOR PLOTTING
        plot_data['train_loss'].append(train_loss)
        plot_data['train_accuracy'].append(train_accuracy)
        plot_data['avg_train_inaccuracy'].append(avg_train_inaccuracy)
        plot_data['test_loss'].append(test_loss)
        plot_data['test_accuracy'].append(test_accuracy)
        plot_data['avg_test_inaccuracy'].append(avg_test_inaccuracy)
        for k in range(10):
            plot_data["train_inaccuracy_" + str(k)].append(train_inaccuracy[k])
            plot_data["test_inaccuracy_" + str(k)].append(test_inaccuracy[k])
        
        # PRINT STATUS
        print("EPOCH {0}".format(epoch+1))
        print("    Train Loss: {0}, Train Accuracy: {1}".format(round(train_loss, 2), round(train_accuracy, 2)))
        print("    Train Inaccuracy: {0}".format([round(e, 2) for e in train_inaccuracy]))
        print("    Test Loss: {0}, Test Accuracy: {1}".format(round(test_loss, 2), round(test_accuracy, 2)))
        print("    Test Inaccuracy: {0}".format([round(e, 2) for e in test_inaccuracy]))
        
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        
    # get filters from the first layer
    filters = model.conv1.get_weights()[0]
    plot(plot_data, filters, classes)
    model.save("cnn_model")
    
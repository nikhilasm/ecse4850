## Notes for ECSE 4850 Project 1

### Objective
- Train **multi-class logistic regressor** using the **gradient descent method**
- Learn the **discriminant function regression parameters** theta = (W_k, W_k0)^T, k = 1,2,3,4,5
    - W_k is a vector for the kth discriminant function and W_k0 is the bias

### Data Formatting
- Input: x[m] is a 28x28 image (a 784x1) vector
    - Convert image to vector and normalize it to [0,1] by dividing intensity of each pixel by 255
    - Then create input vector X = [x[m]^T 1]
    - Matrix X will be 785x25112
- Output: y[m] is a 1-of-k encoded (a 5x1 vector)

- M = 25112, indices are [0, 25111]
- K = 5, indices are [0, 4]

### Learning Parameters theta
- Implement gradient descent method to solve for theta iteratively using all data in D.
- Initialize theta to small values and iteratively update theta with appropriate learning rate until convergence
- Save theta and plot (W_k) for each class using matplotlib.pyplot functions
- DO NOT use Tensorflow's existing gradient descent method
- Watch for overfitting. Keep an eye on the testing error

### DO NOT USE THESE FUNCTIONS
- `tf.keras.datasets`
- `tf.data.Dataset`

- `tf.keras.Sequential()`
- `tf.keras.layers`
- `tf.keras.models`
- `tf.keras.Model()`
- `tf.nn`

- `tf.keras.losses`
- `tf.keras.metrics`

- `tf.keras.optimizers`
- `tf.gradients()`
- `tf.GradientTape()`



### Things to Submit
- A report summarizing the theories for multi-class logistic regression
    - experimental settings
    - hyper parameters
    - plot the training and testing errors over epochs or iterations and the learnt weights
- **Things to include in the report**
    1. Pseudocode for the algorithm (Just recommended and no penalties will be applied if you miss this)
    2. Equations of the loss function (symbols clearly defined)
    3. Equations of the gradients of loss with respect to parameters (symbols clearly defined)
    4. Training error (the value of the loss function), testing error, 
        training accuracy, testing accuracy (the accuracy of all the testing data) curves.
        Plots including these four curves must be included in your report.
    5. Plots of parameters. Since the parameters (without the bias) have the same dimension as the 
        images. You just reshape them as 28x28 and plot them.
    6. Classification errors and the average classification error. See the definition of
        them in the assignment.
    7. A brief discussion (describe the settings (Tensorflow version and environment settings)
        and the hyper-parameters you use)
- Show classification performance in terms of classification error for each digit
    - for each digit, calculate the ratio of incorrect classifications to the total number of images for that digit
- Tensorflow code
- Saved weights W in the required format
    - W = [W1  W2  W3  W4  W5
           W10 W20 W30 W40 W50]
    - Use the following code to save
        `import pickle`
        `filehandler = open("multiclass_parameters.txt", "wb")`
        `pickle.dump(W, filehandler)`
        `filehandler.close()`
    - Use the following code to plot W_k as an image for each digit
        `import matplotlib.pyplot as plt`
        `img = W_k.reshape(28, 28)`
        `plt.imshow(img)`
        `plt.colorbar`
        `plt.show`


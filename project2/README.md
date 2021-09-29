## Project 2: Training a Neural Network with Two Hidden Layers

This project required the creation of a neural network model to classify images of handwritten digits 0 through 9. This task had the added constraint that **no Tensorflow gradient descent or back-propagation functions could be used**. All parameter updating for all layers had to be computed and applied manually.

The model developed by this code achieved about a 90% accuracy after being trained for 8 epochs (about 8,000 iterations of 50 images each).

My implementation of the gradient descent and back-propagation calculations were not very efficient, and therefore training the model takes a very long time. Replacing some of the ```for```-loops in my code with pre-defined Tensorflow tensor operations would be more performant.

For more information on the implementation and the theory applied, please see ```Deep_Learning_Project_2_Report.pdf```.

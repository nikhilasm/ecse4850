===============================================================================
ECSE 4850 Final Project Submission
Nikhilas Murthy
5/12/2021
===============================================================================

EVALUATING THE MODEL
-------------------------------------------------------------------------------
The file 'run_file.py' contains all of the code necessary to evaluate the
model. Here are the steps necessary to run the program:

1. In order to generate the validation data set from the given .pkl files, you
   will need to run the "data_builder.py" script. Place the two .pkl files:
   		youtube_action_train_data_part1.pkl
   		youtube_action_train_data_part2.pkl
   in the same directory as the data builder script and run the program. The
   program will output 4 files:
   		yt_action_data_train.npy
   		yt_action_labels_train.npy
   		yt_action_data_valid.npy
   		yt_action_labels_valid.npy
   The validation data and labels will be used when running the 'run_file.py'
   script.

2. Open 'run_file.py' in a code/text editor and edit the following parameters
   at the top of the file.
   		TEST_DATA_PATH	-	This variable should store a string containing the
   							path to the testing data set. The data file must
   							be a .npy file that stores a 5-D tensor of video
   							data:
   								num_samples X frames X height X width X channels
   							This data should be non-normalized, i.e. all values
   							should be integers in the range [0, 255].

   		TEST_LABEL_PATH	-	This variable should store a string containing the
   							path to the testing label set. The label file must
   							be a .npy file that stores a 1-D list of class
   							labels.	This should be a list of integers in the
   							range [0, 10]. (the one-hot vectors will be built
   							by the program)

   		MODEL_PATH		-	This variable should store a string containing the
   							path to the saved model. This will likely not need
   							to change.

   		BATCH_SIZE		-	This variable should be an integer storing the size
   							of each batch during evaluation. A larger batch
   							size will improve performance but increase memory
   							usage.

3. Save the file and run the program using your preferred method.
4. The program will:
	a) Print the average loss and accuracy values
	b) Print the class-wise accuracy values
	c) Create a confusion matrix heatmap and show the figure
-------------------------------------------------------------------------------

OTHER INFORMATION
-------------------------------------------------------------------------------
Here I will detail what each file in my submission is:
	
	DeepClassifierModel/	This directory contains the saved Keras model.

	data_builder.py			This file is a helper script used to separate the
							given data files into training and validation sets.
							It then saves those data sets as .npy files to be
							loaded during training.

	plotter.py				This file contains helper functions for the
							creation of the many plots shown in my final report.

	final_project.py		This file contains all of the main code for the
							project. It loads the data, creates the model
							architecture, runs the training loop, and creates
							plots of model performance.

	run_file.py				This file contains the code necessary to load and
							evaluate the saved model. See above for more
							information.

	Deep_Learning_Final_Project_Report.pdf	This file contains my written
											report for this project.
-------------------------------------------------------------------------------

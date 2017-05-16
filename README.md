Repository used to hold implementations of perceptron models for Ubuntu disentanglement project.  
Usage: <programName.py> <training.list> <testing.list> <finalBenchmark.list>  

The file lists contain the files names (in pairs consisting of data files followed directly by the corresponding annotation file) of the files used for their respective purposes.

Parameters than can be modified within program code: (all parameters are initialized at top of code for easy access)  
Perceptron, Averaged Perceptron, Structured Perceptron: **number of epochs**  
AdaGrad Model: **number of epochs, learning rate, regularization constant, delta constant**

The programs will output the performance metrics for both testing and training after each epoch.
The programs will then run on the benchmark set using the weights that generated the greatest training fscore.
A file containing the predicted annotations will be generated for every file in the benchmark set. 
The files generated will be named according to the following format: <initial_filename>.annotated.<perceptron_type>

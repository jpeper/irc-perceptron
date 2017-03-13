# irc-perceptron
Repository to hold code for implementation of perceptron model for CLAIR IRC project.

Command line arguments consists of (data_file, annotation_file) pairs. 
The final pair is the training set the perceptron upon which the perceptron will make predictions. 
Predictions on test set are output to 'outputfile.txt'.

Example command line arguments:
./perceptron busy.txt busy.annotations samatman_excerpt_0_49.txt samatman_excerpt_0_49.annotations

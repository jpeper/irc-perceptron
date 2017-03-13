# implementation of perceptron model for ubuntu disentanglement project

import re
import random
import sys
import string
import numpy
numpy.set_printoptions(threshold=numpy.nan)

class Post(object):    
	# constructor
	def __init__(self, username, message_words, message, postid, raw_line):		
		# stores name of user
		self.username = username
		# stores list of strings which contains every 'word' in the message
		self.message_words = self.strip_words(message_words)
		# string of message body
		self.message = message
		# stores id number assigned to post
		self.postid = postid		
		# stores actual line of raw input corresponding to current post
		self.raw_line = raw_line

	def __str__(self):
		return 'Username: ' + self.username + ', Post ID: ' + str(self.postid) + '\nMessage: ' + self.message 

	def strip_words(self, message_words_list):
		new_set = set()
		for word in message_words_list:
			new_set.add(word.strip(string.punctuation))
		return new_set

class TrainingExample:
	# constructor
	def __init__(self, response_line, prev_line, features, correct_prediction = 0, prediction = -1):
		
		self.response_line = response_line
		self.prev_line = prev_line		
		self.features = features
		self.correct_prediction = correct_prediction
		self.prediction = prediction

	
def file_processing(entries, file_in):
	
	''' 
	Example of typical line:
	[08:11] <Peppernrino> Hello everyone!
	The parser will extract from this message the username (Peppernrino) and the message (Hello everyone!)
	'''
	post_num = 0

	for line in file_in:			

		# if line contains a normal post, store information in Post object 
		current_line = re.search(r'\[.+?\]\s<(.+?)>', line)
		if current_line:
			username = current_line.group(1)		
			# tokenize post body and store strings in list
			current_entry = re.findall(r'\[.+?\]\s<.+?>\s(.*)', line)	
			split = current_entry[0].split()
			# add object storing current comment's information to data list
			new_post = Post(username, split, current_entry[0], post_num, line)
			entries.append(new_post)

		# otherwise, create default object
		else:
			new_post = Post("", [], "", post_num, "")
			entries.append(new_post)			
		post_num += 1	

def add_to_dictionary(d_file, master_dictionary):
	entries = []
	data_file = open(d_file, 'r')
	# read posts from data file and store post objects in entries
	file_processing(entries, data_file)
	for post in entries:
		for word in post.message_words:
			master_dictionary.add(word)

def annotation_processing(features_grid, annotation_file_in):
	
	# go through all possible message pairs and set link bool
	# to true if link exists in the annotation file
	for line in annotation_file_in:
		links = re.findall(r'\d+', line)
		for link in range(1, len(links)):	
			features_grid[int(links[0])][int(links[link])].correct_prediction = 1

def generate_features(current_post, prev_post, compiled_dictionary):

	features = numpy.zeros(13 + len(compiled_dictionary))
	

	
	# correct classification held in element zero
	# features begin at element one
	
	# bias
	features[0] = 1
	# 'distance' categories
	distance = current_post.postid - prev_post.postid
	if distance == 1:
		features[1] = 1
	elif distance < 6:
		features[2] = 1
	elif distance < 21:
		features[3] = 1
	elif distance < 51:
		features[4] = 1
	else: 
		features[5] = 1


	# compare usernames
	if prev_post.username == current_post.username:
		features[6] = 1	
	
	# if current message contains username of previous poster
	if prev_post.username in current_post.message_words:
		features[7] = 1

	# if previous message contains username of current poster
	if current_post.username in prev_post.message_words:
		features[8] = 1
	

	size_intersection = len(set(prev_post.message_words).intersection(set(current_post.message_words)))
	
	if size_intersection == 0:
		features[9] = 1
	elif size_intersection == 1:
		features[10] = 1
	elif size_intersection < 6:
		features[11] = 1
	else:
		features[12] = 1

	index = 13
	for i in compiled_dictionary:
		if i in prev_post.message_words and i in current_post.message_words:
			features[index] = 1			
		index += 1

	return features
	
def create_training_set(d_file, a_file, compiled_training_set, compiled_dictionary):
	
	entries = []
	data_file = open(d_file, 'r')
	annotation_file = open(a_file, 'r')
	# read posts from data file and store post objects in entries
	file_processing(entries, data_file)	

	dim = len(entries)
	# generate 2-dim list to hold the possible training examples
	features_grid = [[None for x in range(dim)] for y in range(dim)]
	
	
	# generate features for every prediction possibility and store in feature grid
	for i in range(100, dim):		
		for j in range(i):
			# generate training example features	
			features_grid[i][j] = TrainingExample(i, j, generate_features(entries[i], entries[j], compiled_dictionary))	

	# add correct classification to zeroth element of features vector
	annotation_processing(features_grid, annotation_file)

	# add all valid training examples from current file to compiled training examples list
	for i in range(100, dim):
		for j in range(i): 				
			compiled_training_set.append(features_grid[i][j])	

def make_prediction(weights, features):

	summation = 0
		
	summation = numpy.dot(weights, features)

	activation = 1

	if summation < 0:
		activation = 0

	return activation

def train_perceptron(weights, compiled_training_set):

	random.shuffle(compiled_training_set)
	
	for i in range(len(compiled_training_set)):

		prediction = make_prediction(weights, compiled_training_set[i].features)
		
		diff = compiled_training_set[i].correct_prediction - prediction
		
		weights += numpy.multiply(diff,compiled_training_set[i].features)
		


def calculate_predictions(weights, test_set_file, correct_predictions, compiled_dictionary):

	print ("\nPerceptron weights:")
	print (weights)

	# generate test set from file
	test_set = []
	create_training_set(test_set_file, correct_predictions, test_set, compiled_dictionary)	

	correct_matches = 0
	true_pos = 0
	false_pos = 0
	false_neg = 0
	
	# determine diff in prediction vs actual but don't update weights
	for i in range(len(test_set)):

		prediction = make_prediction(weights, test_set[i].features)		
		diff = test_set[i].correct_prediction - prediction

		# if prediction was correct
		if diff == 0:
			correct_matches += 1
			if prediction == 1:
				true_pos += 1

		else:
			if prediction == 1:
				false_pos += 1
			else:
				false_neg += 1
	accuracy = correct_matches/len(test_set)
	precision = true_pos / (true_pos + false_pos)
	recall = true_pos / (true_pos + false_neg)
	fscore = 2 * precision * recall / (precision + recall)


	print ("Accuracy: " + str(accuracy) + "\nPrecision: " + str(precision) + "\nRecall: " + str(recall) + "\nFscore:" + str(fscore))

def generate_annotation_file(weights, d_file, a_file, compiled_dictionary):

	
	# generate test set from file
	test_set = []
	create_training_set(d_file, a_file, test_set, compiled_dictionary)
	output_dict	= {}

	count = 0
	
	# determine diff in prediction vs actual but don't update weights
	for i in range(len(test_set)):

		test_set[i].prediction = make_prediction(weights, test_set[i].features)		
		if test_set[i].prediction == 1:
			line_key = test_set[i].response_line
			line_val = test_set[i].prev_line
			output_dict.setdefault(line_key, []).append(line_val)
			
	f = open('outputfile.txt', 'w')

	for line in output_dict:
		output = str(line) + ' -'
		for val in output_dict[line]:
			output += ' ' + str(val)
		output += '\n'
		f.write(output)


if __name__ == "__main__":			

	compiled_training_set = []
	compiled_dictionary = set()



	# index of every data file from argv
	# command line arguments are in <data_file, annotation_file>
	# pairs with the last pair of files being the test set

	if len(sys.argv)//2 -1 < 1:
		print("ERROR: Please specify training and testing files")
		exit()

	for k in range(len(sys.argv)//2 - 1):
		print ("adding to dictionary")
		add_to_dictionary(sys.argv[2*k+1], compiled_dictionary)

	compiled_dictionary = list(compiled_dictionary)

	for k in range(len(sys.argv)//2 - 1):
		print("creating training set")
		create_training_set(sys.argv[2*k+1], sys.argv[2*k+2], compiled_training_set, compiled_dictionary)

	# create zero-initialized list of same length as feature set
	weights = numpy.zeros(len(compiled_training_set[0].features))
	
	epochs = 1
	for i in range(epochs):
		print ("epoch " + str(i) + "of perceptron training")
		train_perceptron(weights, compiled_training_set)
		if(i // 100000 == 0):
			print(weights)

	print("PRINTING THE WEIGHTS IN MAIN")
	print (weights)
	print("testing perceptron and calculating predictions")
	calculate_predictions(weights, sys.argv[len(sys.argv)-2], sys.argv[len(sys.argv)-1], compiled_dictionary)
	print("generating output file")
	generate_annotation_file(weights, sys.argv[len(sys.argv)-2], sys.argv[len(sys.argv)-1], compiled_dictionary)
	

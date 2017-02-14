import re
import numpy
import random
import sys

class Post(object):    
	# constructor
	def __init__(self, username, message_words, message, postid, raw_line):
		
		# member variables
		# stores name of user
		self.username = username
		# stores list of strings which contains every 'word' in the message
		self.message_words = message_words
		# string of message body
		self.message = message
		# stores id number assigned to post
		self.postid = postid		
		# stores actual line of raw input corresponding to current post
		self.raw_line = raw_line

	def __str__(self):
		return 'Username: ' + self.username + ', Post ID: ' + str(self.postid) + '\nMessage: ' + self.message 
	
	def get_username(self):
		return self.username	

	def get_message_words(self):
		return self.message_words

	def get_message(self):
		return self.message      

	def get_postid(self):
		return self.postid

	def get_raw_line(self):
		return self.raw_line
	
def file_processing(entries, file_in):
	
	post_num = 0

	for line in file_in:			

		# if line contains a normal post, add username to dictionary		 
		current_line = re.search(r'\[.+?\]\s<(.+?)>', line)
		if current_line:
			username = current_line.group(1)		
			# tokenize post body and store strings in list
			current_entry = re.findall(r'\[.+?\]\s<.+?>\s(.*)', line)	
			split = current_entry[0].split()
			# add object storing current comment's information to data list
			new_post = Post(username, split, current_entry[0], post_num, line)
			entries.append(new_post)
		else:
			entries.append(0)
		post_num += 1	

def annotation_processing(features_grid, annotation_file_in):
	
	# go through all possible message pairs and set link bool
	# to true of link exists in the annotation file
	for line in annotation_file_in:
		bob = re.findall(r'\d+', line)
		for link in range(1, len(bob)):					
			features_grid[int(bob[0])][int(bob[link])][0] = 1

def generate_features(current, input):

	features = numpy.zeros(5)
	# correct classification held in element zero
	# features begin at element one

	# bias
	features[1] = 1 

	# 'distance' categories
	distance = current.get_postid() - input.get_postid() 
	if (distance == 1):
		features[2] = 1
	elif (distance < 6):
		features[3] = 1
	else:
		features[4] = 1

	return features
	
def create_training_set(d_file, a_file, compiled_training_set):
	
	entries = []
	data_file = open(d_file, 'r')
	annotation_file = open(a_file, 'r')
	# read posts from data file and store post objects in entries
	file_processing(entries, data_file)	

	dim = len(entries)
	# generate 2-dim list to hold the possible training examples
	features_grid = [[None for x in range(dim)] for y in range(dim)]
	
	# generate features for every prediction possibility and store in feature grid
	for i in range(dim):
		# if the current entry contains a valid message
		if(entries[i] != 0):
			for j in range(i):
				# if previous entry is a valid message
				if (entries[j] != 0):	
					# generate training example features				
					features_grid[i][j] = generate_features(entries[i], entries[j])	

	# add correct classification to zeroth element of features vector
	annotation_processing(features_grid, annotation_file)

	# add all valid training examples from current file to compiled training examples list
	for i in range(len(features_grid)):
		for j in range(i): 				
			if ((features_grid[i][j] is not None)):
				compiled_training_set.append(features_grid[i][j])	

def train_perceptron(weights, compiled_training_set):

	random.shuffle(compiled_training_set)
	
	for i in range(len(compiled_training_set)):

		sum = 0
		activation = 1

		# ignore correct classfication held in elt zero of the feature vector
		for feature in range(1, len(compiled_training_set[i])):			
			sum += weights[feature] * compiled_training_set[i][feature]

		if (sum < 0):
			activation = 0

		diff = compiled_training_set[i][0] - activation

		# update weights if diff is non-zero
		for feature in range(1, len(compiled_training_set[i])):
			weights[feature] += diff*compiled_training_set[i][feature]		

def make_predictions(weights, test_set_file, correct_predictions):

	print ("Percepton weights:")
	print (weights)

	# generate test set from file
	test_set = []
	create_training_set(test_set_file, correct_predictions, test_set)	

	count = 0
	
	# determine diff in prediction vs actual but don't update weights
	for i in range(len(test_set)):

		sum = 0
		activation = 1

		for feature in range(1, len(compiled_training_set[i])):			
			sum += weights[feature] * compiled_training_set[i][feature]

		if (sum < 0):
			activation = 0

		diff = compiled_training_set[i][0] - activation

		if (diff == 0):
			count += 1

	print ("Agreement proportion:")
	print (count/len(test_set))

def make_output_predictions(weights, d_file, a_file):

	### still a work in progress ###

	'''  make_output_predictions() outputs predictions back to file in 
	the same format as the annotations.	unfortunately the make_predictions()
	function generates a training set whose examples are unaware of their origin, 
	so it is difficult to re-build an annotation file using it - which is why 
	this similar and rather redundant function exists  '''
	
	# output predictions in same format as annotation file
	entries = []
	data_file = open(d_file, 'r')
	annotation_file = open(a_file, 'r')
	file_processing(entries, data_file)	

	dim = len(entries)
	# keep track of all training examples 
	features_grid = [[None for x in range(dim)] for y in range(dim)]
	

	# generate all of the features for every prediction possibility and store in 
	for i in range(dim):
		if(entries[i] != 0):
			for j in range(i):
				if (entries[j] != 0):					
					features_grid[i][j] = generate_features(entries[i], entries[j])	

	annotation_processing(features_grid, annotation_file)

	f = open('outputfile.txt', 'w')
	
	for i in range(len(features_grid)):		
		line = ""
		line += str(i) + ' - '
		count = 0
		for j in range(i): 				
			if ((features_grid[i][j] is not None)):

				sum = 0
				activation = 1

				for feature in range(1, len(features_grid[i][j])):			
					sum += weights[feature] * features_grid[i][j][feature]

				if (sum < 0):
					activation = 0

				if (activation == 1):
					count += 1
					line = line + " " + str(j)

		# line that will be output if at least one 
		# activation takes place for a given message
		if (count > 0):
			line += " \n"
			f.write(line)

	f.close()

compiled_training_set = [];

# index of every data file from argv
# command line arguments are in (data_file, annotation_file) 
# pairs with the last pair being the test set
for k in range(len(sys.argv)//2 - 1):
	create_training_set(sys.argv[2*k+1], sys.argv[2*k+2], compiled_training_set);

weights = numpy.zeros(5)
epochs = 15
for i in range(epochs):
	train_perceptron(weights, compiled_training_set)


make_predictions(weights, sys.argv[len(sys.argv)-2], sys.argv[len(sys.argv)-1])
make_output_predictions(weights, sys.argv[len(sys.argv)-2], sys.argv[len(sys.argv)-1])


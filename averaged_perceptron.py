# implementation of averaged perceptron model for ubuntu disentanglement project
EPOCHS = 10

import re
import random
import sys
import string
import numpy
import collections
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
        # set containing the line number of every message this post is linked to
        self.links = set()

    def __str__(self):
        return 'Username: ' + self.username + ', Post ID: ' + str(self.postid) + '\nMessage: ' + self.message 

    def strip_words(self, message_words_list):
        new_set = set()
        for word in message_words_list:
            new_set.add(word.strip(string.punctuation))
            new_set.add(word)
        return new_set

class TrainingExample:
    # constructor
    def __init__(self, response_line, prev_line, features, correct_prediction = 0, prediction = -1):
        
        self.features = features
        self.response_line = response_line
        self.prev_line = prev_line
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
    global dictionary_index
    entries = []
    data_file = open(d_file, 'r')
    # read posts from data file and store post objects in entries
    file_processing(entries, data_file)
    for post in entries:
        for word in post.message_words:
            if word not in master_dictionary:
                master_dictionary[word] = dictionary_index
                dictionary_index += 1

def annotation_processing(entries, annotation_file_in):
    
    # go through all possible message pairs and set link bool
    # to true if link exists in the annotation file
    for line in annotation_file_in:
        links = re.findall(r'\d+', line)
        for link in range(1, len(links)):   
            (entries[int(links[0])].links).add(int(links[link]))

def generate_features(current_post, prev_post, post_file, compiled_dictionary, linked_pairs):

    #features = numpy.zeros(13, bool)
    features = numpy.zeros(18 + 3 * len(linked_pairs), bool)
    
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

    size_intersection = len(prev_post.message_words.intersection(current_post.message_words))
    
    if size_intersection == 0:
        features[9] = 1
    elif size_intersection == 1:
        features[10] = 1
    elif size_intersection < 6:
        features[11] = 1
    elif size_intersection < 15:
        features[12] = 1
    else:
        features[13] = 1
    # of posts by previous user between current and previous post
    posts_between = 0
    for i in range(1, current_post.postid - prev_post.postid):
        if(post_file[prev_post.postid + i].username == prev_post.username):
            posts_between += 1
    if posts_between == 0:
        features[14] = 1
    elif posts_between == 1:
        features[15] = 1
    elif posts_between < 5:
        features[16] = 1
    else:
        features[17] = 1

    index = 18
    
    hash_range = len(linked_pairs)
    for i in prev_post.message_words:
        for j in current_post.message_words:
            if (i, j) in linked_pairs:
                features[index + linked_pairs[(i,j)]] = 1
            else:
                features[index + len(linked_pairs) + (hash(i+j) % (2*len(linked_pairs)))] = 1
    
    '''
    for i in compiled_dictionary:
        if i in prev_post.message_words and i in current_post.message_words:
            features[index + compiled_dictionary[i]] = 1      
    '''
    
    return features
    
def create_message_file(d_file, a_file):
    
    entries = []
    data_file = open(d_file, 'r')
    annotation_file = open(a_file, 'r')
    # read posts from data file and store post objects in entries
    file_processing(entries, data_file) 

    dim = len(entries)
    # generate 2-dim list to hold the possible training examples
        
    # add correct classification to zeroth element of features vector
    annotation_processing(entries, annotation_file)
    return entries

def make_prediction(pred_weights, features):

    summation = 0
        
    summation = numpy.dot(pred_weights, features)

    activation = 0

    if summation > 0:
        activation = 1

    return activation

def train_perceptron(weights, running_weights, running_count, enumerated_examples, files, compiled_dictionary, linked_pairs):

    correct_matches = 0
    num_examples = 0
    true_pos = 0
    false_pos = 0
    false_neg = 0 

    random.shuffle(enumerated_examples)

    for i in range(len(enumerated_examples)):
        if i % ((len(enumerated_examples) - (len(enumerated_examples) % 1000)) // 10)  == 0:
            print("training on example " + str(i))
        running_count += 1
        
        file_num = enumerated_examples[i][0]
        msg_line = enumerated_examples[i][1]
        prev_msg_line = enumerated_examples[i][2]


        features = generate_features(files[file_num][msg_line], files[file_num][prev_msg_line], files[file_num], compiled_dictionary, linked_pairs)

        prediction = make_prediction(weights, features)
        
        correct_prediction = 0
        if prev_msg_line in files[file_num][msg_line].links:
            correct_prediction = 1

        num_examples += 1

        diff = correct_prediction - prediction
        if diff != 0:
            weights += numpy.multiply(diff,features)
            running_weights += numpy.multiply(1/running_count, weights - running_weights)
            if prediction == 1:
                false_pos += 1
            else:
                false_neg += 1

        else:
            correct_matches += 1
            if prediction == 1:
                true_pos += 1

    
    accuracy = correct_matches/num_examples
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    fscore = 2 * precision * recall / (precision + recall)

    print("Training accuracy: " + str(accuracy) + "\nTraining precision:" + str(precision) + "\nTraining recall:" + str(recall) + "\nTraining fscore: " + str(fscore))

    return running_count

def calculate_predictions(calc_weights, testing_list, compiled_dictionary, linked_pairs):

    # generate test set from file
    
    correct_matches = 0
    num_examples = 0
    true_pos = 0
    false_pos = 0
    false_neg = 0    

    for k in range(0, len(testing_list), 2):
        print("Working on test file " + str(k//2 + 1) + " of " + str(len(testing_list)//2))
        test_set = create_message_file(testing_list[k], testing_list[k+1])  
        
        
        # determine diff in prediction vs actual but don't update weights
        for message in range(100, len(test_set)):
            for prev_message in range(message):       
                num_examples += 1
                features = generate_features(test_set[message], test_set[prev_message], test_set, compiled_dictionary, linked_pairs)

                prediction = make_prediction(calc_weights, features)     
                
                correct_prediction = 0
                if prev_message in test_set[message].links:
                    correct_prediction = 1

                diff = correct_prediction - prediction

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
    

    accuracy = correct_matches/num_examples
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    fscore = 2 * precision * recall / (precision + recall)


    print ("Accuracy: " + str(accuracy) + "\nPrecision: " + str(precision) + "\nRecall: " + str(recall) + "\nFscore:" + str(fscore))
    return fscore

def generate_annotation_file(weights, benchmark_list, compiled_dictionary, linked_pairs):

    for k in range(0, len(benchmark_list), 2):
        test_set = create_message_file(benchmark_list[k], benchmark_list[k+1])  
        output_dict = {}

        # determine diff in prediction vs actual but don't update weights
        for message in range(100, len(test_set)):
            for prev_message in range(message):       
                
                features = generate_features(test_set[message], test_set[prev_message], test_set, compiled_dictionary, linked_pairs)
                prediction = make_prediction(weights, features)  
                if prediction == 1:
                    output_dict.setdefault(message, []).append(prev_message) 

        output_file_name = benchmark_list[k] + ".annotated.averaged"
        f = open(output_file_name, 'w')

        for line in output_dict:
            output = str(line) + ' -'
            for val in output_dict[line]:
                output += ' ' + str(val)
            output += '\n'
            f.write(output)

def create_pairs(enumerated_examples, linked_pairs, files):

    location = 0
    for i in range(len(enumerated_examples)):
        file_num = enumerated_examples[i][0]
        msg_line = enumerated_examples[i][1]
        prev_msg_line = enumerated_examples[i][2]

        if prev_msg_line in files[file_num][msg_line].links:
            for wordA in files[file_num][prev_msg_line].message_words:
                for wordB in files[file_num][msg_line].message_words:
                    if (wordA != files[file_num][msg_line].username and wordB != files[file_num][prev_msg_line].username):
                        pair_tuple = (wordA, wordB)
                        if(pair_tuple not in linked_pairs):
                            linked_pairs[pair_tuple] = [location, 1]
                            location += 1
                        else:
                            linked_pairs[pair_tuple][1] += 1

def remove_overfitted_pairs(linked_pairs):
    new_pairs = collections.OrderedDict()
    pair_index = 0
    for word_pair in linked_pairs:
        if linked_pairs[word_pair][1] > 10:
            new_pairs[word_pair] = pair_index
            pair_index += 1

    return new_pairs

       
if __name__ == "__main__":          

    training_files = []
    compiled_dictionary = collections.OrderedDict()
    dictionary_index = 0
    linked_pairs = collections.OrderedDict()


    training_list_file = open(sys.argv[1], 'r')
    testing_list_file = open(sys.argv[2], 'r')
    benchmark_list_file = open(sys.argv[3], 'r')

    training_list = []
    testing_list = []
    benchmark_list = []

    for line in training_list_file:
        lines = line.strip() 
        training_list.append(lines)
    for line in testing_list_file:
        lines = line.strip() 
        testing_list.append(lines)
    for line in benchmark_list_file:
        lines = line.strip() 
        benchmark_list.append(lines)


    # index of every data file from argv
    # command line arguments are in <data_file, annotation_file>
    # pairs with the last pair of files being the test set
    print ("creating dictionary")
    for k in range(0, len(training_list), 2):        
        add_to_dictionary(training_list[k], compiled_dictionary)

    
    print ("size of compiled dictionary:")
    print(len(compiled_dictionary))
    

    print("creating training files")
    for k in range(0, len(training_list), 2):
        training_files.append(create_message_file(training_list[k], training_list[k+1]))

    enumerated_examples = []
    for file in range(len(training_files)):
        for message in range(100, len(training_files[file])):
            for prev_message in range(message):
                enumerated_examples.append((file, message, prev_message))

    create_pairs(enumerated_examples, linked_pairs, training_files)
    print("number of linked pairs is " + str(len(linked_pairs)))
    linked_pairs = remove_overfitted_pairs(linked_pairs)    
    print("number of training examples is " + str(len(enumerated_examples)))
    print("number of reduced pairs is " + str(len(linked_pairs)))
    #for line in linked_pairs:
    #    print(line[0] + "\t" + line[1])

    dummy_feature_vector = generate_features(training_files[0][100], training_files[0][3], training_files[0], compiled_dictionary, linked_pairs)

    weights = numpy.zeros(len(dummy_feature_vector), numpy.int16)
    running_weights = numpy.zeros(len(dummy_feature_vector))
    running_count = 0
    best_weights = 0
    best_fscore = -1

    
    for i in range(EPOCHS):
        print ("\nepoch " + str(i + 1) + " of perceptron training")
        running_count = train_perceptron(weights, running_weights, running_count, enumerated_examples, training_files, compiled_dictionary, linked_pairs)
        print ("running count: " + str(running_count))
        print("\ntesting perceptron and calculating predictions for epoch " + str(i + 1))
        
        fscore = calculate_predictions(running_weights, testing_list, compiled_dictionary, linked_pairs)
        if fscore > best_fscore:
            best_weights = numpy.copy(running_weights)
            best_fscore = fscore
   
        
   
    #print("testing perceptron and calculating predictions for epoch " + str(i) + "\n")
    #calculate_predictions(weights, sys.argv[len(sys.argv)-2], sys.argv[len(sys.argv)-1], compiled_dictionary)
    print("\nrunning perceptron on benchmark files using best weights:")
    calculate_predictions(best_weights, benchmark_list, compiled_dictionary, linked_pairs)
    print(best_weights)
    print("\ngenerating annotation files")
    generate_annotation_file(best_weights, benchmark_list, compiled_dictionary, linked_pairs)

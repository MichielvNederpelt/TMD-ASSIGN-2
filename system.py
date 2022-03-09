from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import sys
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
import csv
import random
import os
from sklearn.metrics import make_scorer
from collections import defaultdict

features = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats','head', 'deprel', 'sense', 'children', 'NE', 'rb-arg']

#features = ['id', 'form', 'lemma', 'xpos', 'head', 'deprel', 'NE','children']
feature_to_idx = {'id': 0,'form': 1, 'lemma':2, 'upos': 3, 'xpos':4, 'feats':5, 'head':6, 'deprel': 7, 'sense':8, 'children':9, 'NE':10, 'rb-predicate':11, 'rb-arg':12}

def read_in_conll_file(conll_file, delimiter='\t'):
    '''
    THIS CODE WAS ADAPTED FROM THE ML4NLP COURSE
    Read in conll file and return structured object
    :param conll_file: path to conll_file
    :param delimiter: specifies how columns are separated. Tabs are standard in conll
    :returns structured representation of information included in conll file
    '''
    my_conll = open(conll_file, 'r', encoding='utf-8')
    conll_as_csvreader = csv.reader(my_conll, delimiter=delimiter, quoting=csv.QUOTE_NONE)
    return conll_as_csvreader

def extract_feature_value_pair(predicaterow, argumentrow, selected_features):
    '''
    Function that extracts feature value pairs from two rows and puts them in a dict

    :param predicaterow: row from conll file (list)
    :param argumentrow: other row from conll file (list)
    :param selected_features: list of selected features

    :returns: dictionary of feature value pairs
    '''
    feature_value = {}
    for feature in selected_features:
        r_index = feature_to_idx.get(feature)
        feature_value['pred_' + feature] = predicaterow[r_index]
        feature_value['arg_'+ feature] = argumentrow[r_index]
    return feature_value

def create_classifier(train_features, train_targets):
    '''
     Creates a SVM classifier and fits features and targets
        param: train_features: list of dicts {'token' : TOKEN}
        param: train_targets: list of targets
        returns: model: the fitted logistic regression model
        returns: vec: the dict vectorizer object used to transform the train_features

    '''
    classifier = LinearSVC()
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(train_features)
    model = classifier.fit(features_vectorized, train_targets)
    return model, vec


def read_sentences(conll_object):
    '''
    Reads the conll object sentence by sentence and collect a structure containing the sentences and the
    (rule-based identified) predicate (unique for each sentence copy in the conll file).

    param: conll_object: an input file containing samples on each row, where feature token is the first word on each row
    returns: data: a dict with the list of predicate indexes and a list of sentences (as a list of the conll rows)
        '''
    data = defaultdict(list)
    current_sent = []

    next(conll_object)
    collected_predicate = False # keeps track if a predicate was found in a sentence (since there are sentences which don't have one)
    for row in conll_object:
        if len(row) > 0:
            if row[11] == 'RB_PRED': # if the current row is a predicate, append it to the list
                data['predicate'].append(row[0])
                collected_predicate = True
            
            current_sent.append(row) # append row to sentence in any case
        else:
            if not collected_predicate: # if we have not collected a predicate append a dummy variable x 
                data['predicate'].append('X')
            collected_predicate = False
            data['sentences'].append(current_sent) # we append the sentence as a whole
            current_sent = [] # empty the sentence
    return data

# def classify_data_rb(model, vec, selected_features, inputdata, outputfile, model_type):
#     conll_object = read_in_conll_file(inputdata)
#     #return the next item for the itterator
#     next(conll_object)
#     sentences = read_sentences(conll_object)
#     classified = []
#     i = 0
#     l = len(sentences)
#     for i, sentence in enumerate(sentences['sentences']):
#         currently_selected = sentences['predicate'][i]
#         if i%l == 50:
#             print(i/l * 100, '%')
#         for row in sentence:
#             if row[11] == 'RB_PRED': # if this is the predicate we are currently looking at, just copy the value directly from rules
#                 extended_row = '\t'.join(row + ['PREDICATE'])
#                 classified.append(extended_row)
#             # if currently_selected == 'X':
#             #     extended_row = '\t'.join(row + ['O'])
#             #     classified.append(extended_row)
#             # #     continue
#             # if 'RB_ARG' in row[12]: # if arg
#                 # predicate_idxs = row[12].strip('RB_ARG:').split('-') # the idxs of  predicates the arguments should be connexted to
#             if currently_selected == 'X': # if the argument is not connected to the predicate currently running on, skip
#                 extended_row = '\t'.join(row + ['O'])
#                 classified.append(extended_row)
#                 continue
#                 #     continue
#             predicate_row = sentence[int(currently_selected)-1] # the row the predicate where the arg should be connected to
#             pred_arg_pair = extract_feature_value_pair(row,predicate_row, features) # take this as a feature pair
#             feat = vec.transform(pred_arg_pair)
#             label = model.predict(feat) # predict label
#             extended_row = '\t'.join(row + list(label))
#             classified.append(extended_row)

#         classified.append('')

#     outfile = open(outputfile, 'w', encoding='utf-8')
#     header = ['id', 'form', 'lemma', 'xpos', 'head', 'deprel', 'NE','children', 'rb-predicate', 'rb-arg', 'golflabel', 'label\n\n']
#     outfile.write('\t'.join(header))

#     for line in classified:
#         outfile.write(line + '\n') # add prediction to file
#     outfile.close()

def classify_data_rb(model, vec, inputdata, outputfile):
    ''' Classify the dataset using a combination of the rule-based identification and SRL argument labeling model. 
        Writes features and labels to a new file in conll format.
        param: model: a trained SVM Model 
        patam: vec: a fitted DictVectorizer 
        param: inputdata: the file containing the development/testset to classify 
        param: outputfile: the path to the output file'''
    conll_object = read_in_conll_file(inputdata)
    next(conll_object) # skip the header
    sentences = read_sentences(conll_object) # the sentence, predicate dict
    classified = [] # store the classified rows here 
    for i, sentence in enumerate(sentences['sentences']):
        currently_selected = sentences['predicate'][i]
        for row in sentence:
            if 'RB_ARG' in row[12]: # if current row is argument according to rule-based implementation 
                predicate_idxs = row[12].strip('RB_ARG:').split('-') # get the idxs of  predicates the arguments should be connected to
                if currently_selected not in predicate_idxs: # if the argument is not connected to the predicate currently running on, skip
                    extended_row = '\t'.join(row + ['O']) # write label (outside-)
                    classified.append(extended_row)
                    continue
                predicate_row = sentence[int(currently_selected)-1] # get the row ofthe predicate where the arg should be connected to
                pred_arg_pair = extract_feature_value_pair(row,predicate_row, features) # take this row + the arg as a feature pair
                feat = vec.transform(pred_arg_pair)
                label = model.predict(feat) # predict label for predicate, arg pair
                extended_row = '\t'.join(row + list(label)) # write the label 
                classified.append(extended_row)

            elif row[11] == 'RB_PRED': # if this is the predicate we are currently looking at, just copy the value directly from rules
                extended_row = '\t'.join(row + ['PREDICATE'])
                classified.append(extended_row)
            else:
                extended_row = '\t'.join(row + ['O']) # if the current row is not an argument, just write the O label
                classified.append(extended_row)
        classified.append('')

    
    write_modelfile(classified, outputfile)



def write_modelfile(classified_data, outputfilepath):
    outfile = open(outputfilepath, 'w', encoding='utf-8')
    header = ['id', 'form', 'lemma', 'xpos', 'head', 'deprel', 'NE','children', 'rb-predicate', 'rb-arg', 'goldlabel', 'label\n\n']
    outfile.write('\t'.join(header))

    for line in classified_data:
        outfile.write(line + '\n') # add prediction to file
    outfile.close()

def extract_features_and_gold_label_pairs(conll_file, selected_features):
    '''Function that extracts features and gold label from preprocessed conll as predicate, argument pairs.
    
    :param conllfile: path to the (preprocessed) conll file
    
    :return features: a list of dictionaries, with key-value pair providing the value for the feature `token' for individual instances
    :return labels: a list of gold labels of individual instances
    '''
    features = []
    labels = []
    conll_object = read_in_conll_file(conll_file)
    next(conll_object)

    pred_arg_structures = dict() # we create a datastructure to store the predicate, argument pairs
    i=0
    for row in conll_object:
        if len(row) > 0:
            if row[-1] == 'PREDICATE': # if we have a predicate
                if i not in pred_arg_structures.keys(): # if our sentence entry does not exist yet, create it
                    pred_arg_structures[i] = {'P':row}
                else: 
                    pred_arg_structures[i].update({'P':row}) # else update it with the predicate row
            elif row[-1] != 'O': # if it is not O, it should then be an argument
                if i not in pred_arg_structures.keys(): # if our sentence entry does not exist yet, create it 
                    pred_arg_structures[i] = {'A':[row]}
                else: 
                    if 'A' in pred_arg_structures[i].keys(): # else either append the argument to the list of arguments belonging to this predicate
                        pred_arg_structures[i]['A'].append(row)
                    else: 
                        pred_arg_structures[i].update({'A':[row]}) # or create the arguments list if it does not exist yet
                        
        else:
            i+=1
    for i in pred_arg_structures.keys():
        try:
            predicate = pred_arg_structures[i]['P'] # the predicate of the sentence
            if 'A' in pred_arg_structures[i].keys(): # if there are arguments:
                for argument in pred_arg_structures[i]['A']: # go through all arguments
                    feature_values = extract_feature_value_pair(argument, predicate, selected_features)
                    features.append(feature_values)
                    labels.append(argument[-1]) #The last column of the argument row provides the gold label (= the correct answer). 
            
        except KeyError:
            continue
    return features, labels

def main(argv=None):
    '''Run in commandline as:
    [arg1] = path tor training file 
    [arg2] = path to development/testset 
    [arg3] = path to modelfile to store output in '''
    if argv is None:
        argv = sys.argv

    trainingfile = argv[1]
    inputfile = argv[2]
    output_basepath = argv[3]

    print('Starting training')

    # These are all feature combinations that were tested in the feature ablation

    # If we don't want to run the ablation, the standard system is run with all the features
    model = 'SVM'
    print('Features selected in this:', features)
    #Load the features

    training_features, gold_labels =  extract_features_and_gold_label_pairs(trainingfile, features)#extract_features_and_labels(trainingfile, selected_features=features)
    print('finished feature extraction')
    # classify
    ml_model, vec = create_classifier(training_features, gold_labels)
    classify_data_rb(ml_model, vec, inputfile, output_basepath)
    print('finished training the ', model, 'model on', features )

#args = ['python','../../data/conll2003_ret.train-preprocessed_with_feats.conll', '../../data/conll2003_ret.test-preprocessed_chunks.conll', '../../models/1612_cl_fa_non_scaled_', r'C:\Users\Tessel Wisman\Documents\TextMining\GoogleNews-vectors-negative300\GoogleNews-vectors-negative300.bin', False, True]
#main(args)
if __name__ == '__main__':
    main()
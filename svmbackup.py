

features = ['id', 'form', 'lemma', 'xpos', 'head', 'deprel', 'NE','children', 'rb-predicate', 'rb-arg']
feature_to_idx = {'id': 0,'form': 1, 'lemma':2, 'xpos':3, 'head':4, 'deprel':5, 'NE':6, 'children':7, 'rb-predicate':8, 'rb-arg':9}

def extract_feature_values(row, selected_features):
    '''
    Function that extracts feature value pairs from row and puts them in a dict

    :param row: row from conll file (list)
    :param selected_features: list of selected features

    :returns: dictionary of feature value pairs
    '''
    feature_values = {}
    for feature in selected_features:
        try: 
            r_index = feature_to_idx.get(feature)
            feature_values[feature] = row[r_index]
        except IndexError:
            print(row)

    return feature_values

def extract_features_and_labels(trainingfile, selected_features):
    '''
    Extracts the features and labels from the gold datafile.
        param: trainingfile: the .conll file with samples on each line. First element in the line is the token,
                final element is the label
        param: selected_features: list of selected features
        returns: data: list of dicts {'token': TOKEN}
        returns: targets: list of target names for the tokens
            '''
    data = []
    targets = []
    with open(trainingfile, 'r', encoding='utf8') as infile:
        next(infile)
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                feature_dict = extract_feature_values(components, selected_features)

                data.append(feature_dict)
                #gold is in the last column
                targets.append(components[-1])
    return data, targets

def extract_features(inputfile, selected_features):
    '''
    Similar to extract_'features_and_labels, but only extracts data
        params: inputfile: an input file containing samples on each row, where feature token is the first word on each row
        param: selected_features: list of selected features
        returns: data: a list of dicts ('token': TOKEN)
            '''
    data = []
    with open(inputfile, 'r', encoding='utf8') as infile:
        next(infile)
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                feature_dict = extract_feature_values(components, selected_features)
                data.append(feature_dict)
    return data

def create_classifier(train_features, train_targets, classifier_type, grid_search=False):
    '''
     Creates a Logistic Regression classifier and fits features and targets
        param: train_features: list of dicts {'token' : TOKEN}
        param: train_targets: list of targets
        param:classifier_type: classifier used (CRF or SVM)
        param: grid_search: to do grid search or not (Boolean value)
        returns: model: the fitted logistic regression model
        returns: vec: the dict vectorizer object used to transform the train_features

    '''
    if classifier_type == 'SVM':
        classifier = LinearSVC()
        parameters = dict(C = (0.01, 0.1, 1.0), loss= ('hinge', 'squared_hinge'), tol=(1e-4, 1e-3, 1e-2, 1e-1)) # parameters for grid search

        vec = DictVectorizer()
        features_vectorized = vec.fit_transform(train_features)
    elif classifier_type == 'CRF':
        classifier = CRF(algorithm='l2sgd', all_possible_transitions=True) # set standard mparams
        parameters = dict(algorithm= ('lbfgs', 'l2sgd'), all_possible_transitions=(True, False))
        features_vectorized = train_features
        vec = None

    if grid_search:
        f1_scorer = make_scorer(metrics.flat_f1_score, average='macro')
        grid = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='f1_scorer')
        model = grid.fit(features_vectorized, train_targets)
        print('Parameters', grid.best_params_)
        classifier = grid.best_estimator_
        return model, vec
    else:
        # if we don't do grid search, just fit the model
        model = classifier.fit(features_vectorized, train_targets)
        return model, vec

def classify_data(model, vec, selected_features,inputdata, outputfile, model_type):
    '''
    Function that performs the named entity recognition and writes an output file that is the same as the input file
        except classification result is added at the end of each sample row.
        param: model: a fitted LogisticRegression model
        param: vec: a fitted DictVectorizer
        param: inputdata: input data file to be classified
        param: outputfile: file to write output
        param: model_type: which classifier is used (CRF or SVM)

    '''
    if model_type == 'CRF':
        features = extract_CRF_features(inputdata,selected_features) # we need a different way to extract features when using CRF
    else:
        features = extract_features(inputdata,selected_features)
        features = vec.transform(features)

    predictions = model.predict(features) # predict classifications

    if model_type == 'CRF': # if CRF is used, we need to align results with our input and put it in lists of sentences
        predictions = [prediction for sent in predictions for prediction in sent]

    outfile = open(outputfile, 'w', encoding='utf-8')
    header = ['id', 'form', 'lemma', 'xpos', 'head', 'deprel', 'NE','children', 'rb-predicate', 'rb-arg', 'label', 'pr_label\n']
    outfile.write('\t'.join(header))

    counter = 0

    with open(inputdata, 'r', encoding='utf-8') as infile:
        next(infile) # skip header
        for line in infile:
            stripped_line = line.rstrip('\n')
            if len(stripped_line.split()) > 0: # check if not empty
                outfile.write(stripped_line + '\t' + predictions[counter] + '\n') # add prediction to file
                counter += 1
            else:
                outfile.write('\n')
    outfile.close()

def main(argv=None):

    if argv is None:
        argv = sys.argv

    trainingfile = argv[1]#r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\SEM-2012-SharedTask-CD-SCO-simple.v2\SEM-2012-SharedTask-CD-SCO-training-simple.v2-preprocessed.txt'
    inputfile = argv[2]#r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\SEM-2012-SharedTask-CD-SCO-simple.v2\SEM-2012-SharedTask-CD-SCO-test-preprocessed.txt'
    output_basepath = argv[3]#r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\Group3_TMWork\models\testrun010222'

    print('Starting training')

    # These are all feature combinations that were tested in the feature ablation

    # If we don't want to run the ablation, the standard system is run with all the features
    model = 'SVM'
    print('Features selected in this:', features)
    #Load the features

    training_features, gold_labels = extract_features_and_labels(trainingfile, selected_features=features)
    print('finished feature extraction')
    # classify
    ml_model, vec = create_classifier(training_features, gold_labels, model)
    classify_data(ml_model, vec, features, inputfile, output_basepath, model)
    print('finished training the ', model, 'model on', features )

#args = ['python','../../data/conll2003_ret.train-preprocessed_with_feats.conll', '../../data/conll2003_ret.test-preprocessed_chunks.conll', '../../models/1612_cl_fa_non_scaled_', r'C:\Users\Tessel Wisman\Documents\TextMining\GoogleNews-vectors-negative300\GoogleNews-vectors-negative300.bin', False, True]
#main(args)
if __name__ == '__main__':
    main()

#########################
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

features = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats','head', 'deprel', 'sense', 'children', 'NE']

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
def extract_feature_values(row, selected_features):
    '''
    Function that extracts feature value pairs from row and puts them in a dict

    :param row: row from conll file (list)
    :param selected_features: list of selected features

    :returns: dictionary of feature value pairs
    '''
    feature_values = {}
    for feature in selected_features:
        r_index = feature_to_idx.get(feature)
        feature_values[feature] = row[r_index]
    return feature_values

def extract_feature_value_pair(row1, row2, selected_features):
    '''
    Function that extracts feature value pairs from row and puts them in a dict

    :param row: row from conll file (list)
    :param selected_features: list of selected features

    :returns: dictionary of feature value pairs
    '''
    feature_value = {}
    for feature in selected_features:
        r_index = feature_to_idx.get(feature)
        feature_value['pred_' + feature] = row1[r_index]
        feature_value['arg_'+ feature] = row2[r_index]
    return feature_value

def extract_features_and_labels(trainingfile, selected_features):
    '''
    Extracts the features and labels from the gold datafile.
        param: trainingfile: the .conll file with samples on each line. First element in the line is the token,
                final element is the label
        param: selected_features: list of selected features
        returns: data: list of dicts {'token': TOKEN}
        returns: targets: list of target names for the tokens
            '''
    data = []
    targets = []
    i = 0
    with open(trainingfile, 'r', encoding='utf8') as infile:
        next(infile)
        for line in infile:
            i+=1
            if i>500000:
                break
            components = line.rstrip('\n').split()
            if len(components) > 0:
                feature_dict = extract_feature_values(components, selected_features)

                data.append(feature_dict)
                #gold is in the last column
                targets.append(components[-1])
    return data, targets



def extract_features(inputfile, selected_features):
    '''
    Similar to extract_'features_and_labels, but only extracts data
        params: inputfile: an input file containing samples on each row, where feature token is the first word on each row
        param: selected_features: list of selected features
        returns: data: a list of dicts ('token': TOKEN)
            '''
    data = []
    with open(inputfile, 'r', encoding='utf8') as infile:
        next(infile)
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                feature_dict = extract_feature_values(components, selected_features)
                data.append(feature_dict)
    return data

def create_classifier(train_features, train_targets, classifier_type, grid_search=False):
    '''
     Creates a Logistic Regression classifier and fits features and targets
        param: train_features: list of dicts {'token' : TOKEN}
        param: train_targets: list of targets
        param:classifier_type: classifier used (CRF or SVM)
        param: grid_search: to do grid search or not (Boolean value)
        returns: model: the fitted logistic regression model
        returns: vec: the dict vectorizer object used to transform the train_features

    '''
    classifier = LinearSVC()
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(train_features)
    # if we don't do grid search, just fit the model
    model = classifier.fit(features_vectorized, train_targets)
    return model, vec


def read_sentences(conll_object):
    '''
    Similar to extract_features_and_labels, but only extracts data (catered to CRF architecture
    by putting the data in lists of lists(sentences) of dicts)
        param: inputfile: an input file containing samples on each row, where feature token is the first word on each row
        param: selected_features: the list of selected features in this model
        returns: data: a list of lists dicts ('token': TOKEN)
            '''
    data = defaultdict(list)
    current_sent = []
    next(conll_object)
    for row in conll_object:
        if len(row) > 0:
            if row[11] == 'RB_PRED':
                data['predicates'].append(row[0])
            current_sent.append(row)
        else:
            data['sentences'].append(current_sent)
            current_sent = []

    return data

# def is_exact_match(pred, arg):
#     n_predicates = len(pred) - 7
#     for n in range(n_predicates):
#         if pred[7+n] == 'PREDICATE':
#             return arg[7+n]
#     return 'O'

def classify_data_rb(model, vec, selected_features, inputdata, outputfile, model_type):
    conll_object = read_in_conll_file(inputdata)
    #return the next item for the itterator
    next(conll_object)
    sentences = read_sentences(conll_object)
    classified = []
    i = 0
    l = len(sentences)
    for sentence in sentences['sentences']:
        i+=1
        if i % l == 500:
            print(i/l)
        for row in sentence:

            if 'RB_ARG' in row[12]: # if arg
                pred = row[12].strip('RB_ARG:')
                predicate_row = sentence[int(pred)-1]
                pred_arg_pair = extract_feature_value_pair(row,predicate_row, features)
                feat = vec.transform(pred_arg_pair)
                label = model.predict(feat)
                extended_row = '\t'.join(row + list(label))
                classified.append(extended_row)

            elif row[11] == 'RB_PRED':
                extended_row = '\t'.join(row + ['PREDICATE'])
                classified.append(extended_row)
            else:
                extended_row = '\t'.join(row + ['O'])
                classified.append(extended_row)
        classified.append('')

    outfile = open(outputfile, 'w', encoding='utf-8')
    header = ['id', 'form', 'lemma', 'xpos', 'head', 'deprel', 'NE','children', 'rb-predicate', 'rb-arg', 'golflabel', 'label\n\n']
    outfile.write('\t'.join(header))

    for line in classified:
        outfile.write(line + '\n') # add prediction to file
    outfile.close()






def classify_data(model, vec, selected_features,inputdata, outputfile, model_type):
    '''
    Function that performs the named entity recognition and writes an output file that is the same as the input file
        except classification result is added at the end of each sample row.
        param: model: a fitted LogisticRegression model
        param: vec: a fitted DictVectorizer
        param: inputdata: input data file to be classified
        param: outputfile: file to write output
        param: model_type: which classifier is used (CRF or SVM)

    '''
    if model_type == 'CRF':
        features = extract_CRF_features(inputdata,selected_features) # we need a different way to extract features when using CRF
    else:
        features = extract_features(inputdata,selected_features)
        print(features[:10])
        features = vec.transform(features)

    predictions = model.predict(features) # predict classifications

    if model_type == 'CRF': # if CRF is used, we need to align results with our input and put it in lists of sentences
        predictions = [prediction for sent in predictions for prediction in sent]

    outfile = open(outputfile, 'w', encoding='utf-8')
    header = ['id', 'form', 'lemma', 'xpos', 'head', 'deprel', 'NE','children', 'rb-predicate', 'rb-arg', 'label', 'pr_label\n']
    outfile.write('\t'.join(header))

    counter = 0

    with open(inputdata, 'r', encoding='utf-8') as infile:
        next(infile) # skip header
        for line in infile:
            stripped_line = line.rstrip('\n')
            if len(stripped_line.split()) > 0: # check if not empty
                outfile.write(stripped_line + '\t' + predictions[counter] + '\n') # add prediction to file
                counter += 1
            else:
                outfile.write('\n')
    outfile.close()

def extract_features_and_gold_labels(conllfile, selected_features):
    '''Function that extracts features and gold label from preprocessed conll (here: tokens only).
    
    :param conllfile: path to the (preprocessed) conll file
    :type conllfile: string
    
    
    :return features: a list of dictionaries, with key-value pair providing the value for the feature `token' for individual instances
    :return labels: a list of gold labels of individual instances
    '''
    feature_to_index = {'index': 0, 'form': 1, 'lemma':2, 'xpos':3, 'head':4, 'deprel':5, 'NE':6, 'children':7}
    features = []
    labels = []
    conllinput = open(conllfile, 'r', encoding="utf8")
    #delimiter indicates we are working with a tab separated value (default is comma)
    #quotechar has as default value '"', which is used to indicate the borders of a cell containing longer pieces of text
    #in this file, we have only one token as text, but this token can be '"', which then messes up the format. We set quotechar to a character that does not occur in our file
    csvreader = csv.reader(conllinput, delimiter='\t', quotechar='|')
    next(csvreader, None)
    pred_arg_structures = dict()
    i=0
    for row in csvreader:
        #I preprocessed the file so that all rows with instances should contain 6 values, the others are empty lines indicating the beginning of a sentence
        if len(row) > 0:
           # print(row[-1])
            if row[-1] == 'PREDICATE':
                if i not in pred_arg_structures.keys():
                    pred_arg_structures[i] = {'P':row}
                else: 
                    pred_arg_structures[i].update({'P':row})
            elif row[-1] != 'O':
                if i not in pred_arg_structures.keys():
                    pred_arg_structures[i] = {'A':[row]}
                else: 
                    if 'A' in pred_arg_structures[i].keys():
                        pred_arg_structures[i]['A'].append(row)
                    else: 
                        pred_arg_structures[i].update({'A':[row]})
                pred_arg_structures[i]['L'] = row[-1]
                        
        else:
            i+=1
    for i in pred_arg_structures.keys():
        #print('I',i)
       # print(pred_arg_structures[i])
        try:
            predicate = pred_arg_structures[i]['P']
            if 'A' in pred_arg_structures[i].keys():
                for argument in pred_arg_structures[i]['A']:
                    #print('arg',argument)
                    feature_values = extract_feature_value_pair(argument, predicate, selected_features)
                    features.append(feature_values)
                    #The last column provides the gold label (= the correct answer). 
                    labels.append(pred_arg_structures[i]['L'])
            
        except KeyError:
            continue
    return features, labels

def main(argv=None):

    if argv is None:
        argv = sys.argv

    trainingfile = argv[1]#r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\SEM-2012-SharedTask-CD-SCO-simple.v2\SEM-2012-SharedTask-CD-SCO-training-simple.v2-preprocessed.txt'
    inputfile = argv[2]#r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\SEM-2012-SharedTask-CD-SCO-simple.v2\SEM-2012-SharedTask-CD-SCO-test-preprocessed.txt'
    output_basepath = argv[3]#r'C:\Users\Tessel Wisman\Documents\TextMining\AppliedTMMethods\Group3_TMWork\models\testrun010222'

    print('Starting training')

    # These are all feature combinations that were tested in the feature ablation

    # If we don't want to run the ablation, the standard system is run with all the features
    model = 'SVM'
    print('Features selected in this:', features)
    #Load the features

    training_features, gold_labels =  extract_features_and_gold_labels(trainingfile, features)#extract_features_and_labels(trainingfile, selected_features=features)
    print('finished feature extraction')
    # classify
    ml_model, vec = create_classifier(training_features, gold_labels, model)
    classify_data_rb(ml_model, vec, features, inputfile, output_basepath, model)
    print('finished training the ', model, 'model on', features )

#args = ['python','../../data/conll2003_ret.train-preprocessed_with_feats.conll', '../../data/conll2003_ret.test-preprocessed_chunks.conll', '../../models/1612_cl_fa_non_scaled_', r'C:\Users\Tessel Wisman\Documents\TextMining\GoogleNews-vectors-negative300\GoogleNews-vectors-negative300.bin', False, True]
#main(args)
if __name__ == '__main__':
    main()

def classify_data_rb(model, vec, selected_features, inputdata, outputfile, model_type):
    conll_object = read_in_conll_file(inputdata)
    #return the next item for the itterator
    next(conll_object)
    sentences = read_sentences(conll_object)
    classified = []
    i = 0
    l = len(sentences)
    for i, sentence in enumerate(sentences['sentences']):
        currently_selected = sentences['predicate'][i]
        for row in sentence:
            # if currently_selected == 'X':
            #     extended_row = '\t'.join(row + ['O'])
            #     classified.append(extended_row)
            #     continue
            if 'RB_ARG' in row[12]: # if arg
                predicate_idxs = row[12].strip('RB_ARG:').split('-') # the idxs of  predicates the arguments should be connexted to
                if currently_selected not in predicate_idxs: # if the argument is not connected to the predicate currently running on, skip
                    extended_row = '\t'.join(row + ['O'])
                    classified.append(extended_row)
                    continue
                predicate_row = sentence[int(currently_selected)-1] # the row the predicate where the arg should be connected to
                pred_arg_pair = extract_feature_value_pair(row,predicate_row, features) # take this as a feature pair
                feat = vec.transform(pred_arg_pair)
                label = model.predict(feat) # predict label
                extended_row = '\t'.join(row + list(label))
                classified.append(extended_row)

            elif row[11] == 'RB_PRED': # if this is the predicate we are currently looking at, just copy the value directly from rules
                extended_row = '\t'.join(row + ['PREDICATE'])
                classified.append(extended_row)
            else:
                extended_row = '\t'.join(row + ['O'])
                classified.append(extended_row)
        classified.append('')

    outfile = open(outputfile, 'w', encoding='utf-8')
    header = ['id', 'form', 'lemma', 'xpos', 'head', 'deprel', 'NE','children', 'rb-predicate', 'rb-arg', 'golflabel', 'label\n\n']
    outfile.write('\t'.join(header))

    for line in classified:
        outfile.write(line + '\n') # add prediction to file
    outfile.close()
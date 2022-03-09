from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import sys
from sklearn.svm import LinearSVC
import csv
import random
import os
from sklearn.metrics import make_scorer

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
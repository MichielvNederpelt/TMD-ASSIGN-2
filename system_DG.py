import pandas as pd
import numpy as np
#import nltk
#nltk.download('averaged_perceptron_tagger')
#from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import csv

from sklearn import metrics

feature_to_idx = {'form': 1, 'lemma':2, 'xpos':3, 'head':4, 'deprel':5, 'NE':6, 'children':7}
trainfile = r"C:/Users/desir/desktop/text_mining/nlp_technology/UP_English-EWT/en_ewt-up-train-parta-squeeze_try_combined.conllu"
testfile = r"C:/Users/desir/desktop/text_mining/nlp_technology/UP_English-EWT/en_ewt-up-dev-squeeze_try.conllu"

# def extract_feature_values(row, selected_features):
#     '''
#     Function that extracts feature value pairs from row and puts them in a dict

#     :param row: row from conll file (list)
#     :param selected_features: list of selected features

#     :returns: dictionary of feature value pairs
#     '''
#     feature_values = {}
#     for feature in selected_features:
#         r_index = feature_to_idx.get(feature)
#         feature_values[feature] = row[r_index]

#     return feature_values

def extract_features_and_labels(trainingfile, selected_features):
    '''
    Extracts the features and labels from the gold datafile.
        param: trainingfile: the .conll file with samples on each line. First element in the line is the token,
                final element is the label
        param: selected_features: list of selected features
        returns: data: list of dicts {'token': TOKEN}
        returns: targets: list of target names for the tokens
    '''
    feature_to_idx = {'form': 1, 'lemma':2, 'xpos':3, 'head':4, 'deprel':5, 'NE':6, 'children':7}        
    data = []
    targets = []
    with open(trainingfile, 'r', encoding='utf8') as infile:
        next(infile)
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                feature_dict = extract_feature_values(components, selected_features)

                features.append(feature_dict)
                #gold is in the last column
                labels.append(components[-1])
    return features, labels

# def extract_features(inputfile, selected_features):
#     '''
#     Similar to extract_'features_and_labels, but only extracts data
#         params: inputfile: an input file containing samples on each row, where feature token is the first word on each row
#         param: selected_features: list of selected features
#         returns: data: a list of dicts ('token': TOKEN)
#             '''
#     data = []
#     with open(inputfile, 'r', encoding='utf8') as infile:
#         next(infile)
#         for line in infile:
#             components = line.rstrip('\n').split()
#             if len(components) > 0:
#                 feature_dict = extract_feature_values(components, selected_features)
#                 data.append(feature_dict)
#     return data

def create_vectorizer_and_classifier(features, labels):
    '''
    Function that takes feature-value pairs and gold labels as input and trains a logistic regression classifier
    
    :param features: feature-value pairs
    :param labels: gold labels
    :type features: a list of dictionaries
    :type labels: a list of strings
    
    :return lr_classifier: a trained LogisticRegression classifier
    :return vec: a DictVectorizer to which the feature values are fitted. 
    '''
    
    vec = DictVectorizer()
    #fit creates a mapping between observed feature values and dimensions in a one-hot vector, transform represents the current values as a vector 
    tokens_vectorized = vec.fit_transform(features)
    svm_classifier = LinearSVC()   
    svm_classifier.fit(tokens_vectorized, labels)
    
    return svm_classifier, vec

def get_predicted_and_gold_labels(testfile, vectorizer, classifier, selected_features, outputfile):
    '''
    Function that extracts features and runs classifier on a test file returning predicted and gold labels
    
    :param testfile: path to the (preprocessed) test file
    :param vectorizer: vectorizer in which the mapping between feature values and dimensions is stored
    :param classifier: the trained classifier
    :type testfile: string
    :type vectorizer: DictVectorizer
    :type classifier: LogisticRegression()
    
    
    
    :return predictions: list of output labels provided by the classifier on the test file
    :return goldlabels: list of gold labels as included in the test file
    '''
    
    #we use the same function as above (guarantees features have the same name and form)
    features, goldlabels = extract_features_and_gold_labels(testfile, selected_features)
    #we need to use the same fitting as before, so now we only transform the current features according to this mapping (using only transform)
    test_features_vectorized = vectorizer.transform(features)
    predictions = classifier.predict(test_features_vectorized)
    
    return predictions, goldlabels

def print_confusion_matrix(predictions, goldlabels):
    '''
    Function that prints out a confusion matrix
    
    :param predictions: predicted labels
    :param goldlabels: gold standard labels
    :type predictions, goldlabels: list of strings
    ''' 
    
    #based on example from https://datatofish.com/confusion-matrix-python/ 
    data = {'Gold':    goldlabels[1:], 'Predicted': predictions[1:]    }
    df = pd.DataFrame(data, columns=['Gold','Predicted'])

    confusion_matrix = pd.crosstab(df['Gold'], df['Predicted'], rownames=['Gold'], colnames=['Predicted'])
    print (confusion_matrix)


def print_precision_recall_fscore(predictions, goldlabels):
    '''
    Function that prints out precision, recall and f-score
    
    :param predictions: predicted output by classifier
    :param goldlabels: original gold labels
    :type predictions, goldlabels: list of strings
    '''
    
    precision = metrics.precision_score(y_true=goldlabels,
                        y_pred=predictions,
                        average='macro')

    recall = metrics.recall_score(y_true=goldlabels,
                     y_pred=predictions,
                     average='macro')


    fscore = metrics.f1_score(y_true=goldlabels,
                 y_pred=predictions,
                 average='macro')

    print('P:', precision, 'R:', recall, 'F1:', fscore)


    
selected_features = ['form', 'lemma', 'xpos', 'head', 'deprel', 'NE', 'children']
    
feature_values, labels = extract_features_and_labels(trainfile, selected_features)
svm_classifier, vectorizer = create_vectorizer_and_classifier(features, labels)
predictions, goldlabels = get_predicted_and_gold_labels(testfile, vectorizer, svm_classifier, selected_features, 'ewt_test_predictions')
print_confusion_matrix(predictions, goldlabels)
print_precision_recall_fscore(predictions, goldlabels)
report = classification_report(goldlabels,predictions,digits = 7, zero_division=0)
print(report)
import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import csv

from sklearn import metrics

trainfile = r"C:\Users\Tessel Wisman\Documents\TextMining\NLPTech\UP_English-EWT\en_ewt-up-train-squeeze-feats.conllu"
testfile = r"C:\Users\Tessel Wisman\Documents\TextMining\NLPTech\UP_English-EWT\en_ewt-up-dev-prep-feats-preds-args.conllu"

# training_path_opt ='SEM-2012-SharedTask-CD-SCO-training-simple.v2.features.conll'
# dev_path_opt = 'SEM-2012-SharedTask-CD-SCO-test-circle.features.conll'

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

def is_predicate(row, predicates_in_sentence):
    #print(row[6], row[7])
    if row[4] == '0':
        predicates_in_sentence.append(row[0])
        return True, predicates_in_sentence
    elif predicates_in_sentence and (row[5] in ['xcomp', 'ccomp', 'parataxis', 'advcl'] or (row[5] == 'conj' and row[3] == 'VERB')) and row[4] in predicates_in_sentence:
        predicates_in_sentence.append(row[0])
        return True, predicates_in_sentence
    return False, predicates_in_sentence 

def is_argument(predicate, row):
    if ('nsubj' in row[5] or 'obj' in row[5] or 'obl' in row[5]) and row[4] == predicate:
        return True
    return False

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
    
#vectorizer and lr_classifier are the vectorizer and classifiers created in the previous cell.
#it is important that the same vectorizer is used for both training and testing: they should use the same mapping from values to dimensions
# predictions, goldlabels = get_predicted_and_gold_labels_token_only(testfile, vectorizer, lr_classifier)
# print_confusion_matrix(predictions, goldlabels)

# the functions with multiple features and analysis

#defines the column in which each feature is located (note: you can also define headers and use csv.DictReader)
#feature_to_index = {'TOKEN': 0, 'POS': 1, 'LEMMA': 2, 'PUNCTUATION': 3, 'STARTSWITH_CAPITAL_LETTER': 4, 'IS_STOPWORD': 5}

def implement_features(pred_arg_structures, feature_to_index):
    features = []
    labels = []
    for i in sorted(pred_arg_structures.keys()):
        try:
            for predicate_idx, pred_arg_pairs in sorted(pred_arg_structures[i].items()):
                predicate_row = pred_arg_pairs['P']
                if 'A' in pred_arg_pairs.keys():
                    for argument in pred_arg_pairs['A']:
                        feature_value = {}
                        for feature_name in selected_features:
                            row_index = feature_to_index.get(feature_name)
                            try:
                                feature_value[feature_name] = (argument[row_index], predicate_row[row_index])
                            except IndexError:
                                print(row_index, argument, predicate_row)

                        features.append(feature_value)
                        #The last column provides the gold label (= the correct answer). 
                        label = is_exact_match(predicate_row, argument)
                        labels.append(label)
        except KeyError:
            continue
    return features, labels
        
def extract_features_rule_based(conllfile, selected_features):
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
    sentence_idx=0
    for row in csvreader:
        #I preprocessed the file so that all rows with instances should contain 6 values, the others are empty lines indicating the beginning of a sentence
        if len(row) > 0:
           # print(row[-1])
            #predicate, predicates_in_sentence = is_predicate(row, predicates_in_sentence)
            # if the row is classified as a predicate, add to sentence dict as predicate_idx: P: row
            if row[8] == 'RB_PRED':
                pred_ix = row[0]
                if sentence_idx not in pred_arg_structures.keys():
                    pred_arg_structures[sentence_idx] = {pred_ix: {'P':row}}
                else: 
                    if pred_ix in pred_arg_structures[sentence_idx].keys():
                        pred_arg_structures[sentence_idx][pred_ix]['P'] = row 
                    else:
                        pred_arg_structures[sentence_idx][pred_ix] = {'P':row}
                #pred_arg_structures[sentence_idx]['L'] = row[-1]

            elif row[9].startswith('RB_ARG:'):
                predicate_attatchments = row[9].strip('RB_ARG:').split('-')
                for predicate in predicate_attatchments:
                    if sentence_idx not in pred_arg_structures.keys(): # if this is the first entry for the sentence, add complete structure
                        pred_arg_structures[sentence_idx] = {predicate : {'A':[row]}}
                    else: 

                        if predicate not in pred_arg_structures[sentence_idx].keys(): # if the sentence is already set up but we have an unseen predicate
                            pred_arg_structures[sentence_idx][predicate] = {'A': [row]}
                        else: 
                            if 'A' in pred_arg_structures[sentence_idx][predicate].keys(): # if we know the sentence, pred and already have args
                                pred_arg_structures[sentence_idx][predicate]['A'].append(row)
                            else: # if we know sentence and predicate but this is the first arg
                                pred_arg_structures[sentence_idx][predicate]['A'] = [row]
                # else:
                #     if sentence_idx in pred_arg_structures.keys():
                #         pred_arg_structures[sentence_idx][row[0]] = row
                #     else:
                #         pred_arg_structures[sentence_idx] = {row[0]:row}


                        
        else:
            sentence_idx +=1
    features, labels = implement_features(pred_arg_structures, feature_to_index)
    return features, labels

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
                pred_arg_structures[i]['L'] = row[-1]
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
                    feature_value = {}
                    for feature_name in selected_features:
                        row_index = feature_to_index.get(feature_name)
                        try:
                            feature_value[feature_name] = (argument[row_index], predicate[row_index])
                        except IndexError:
                            print(row_index, predicate, argument)

                    features.append(feature_value)
                    #The last column provides the gold label (= the correct answer). 
                    labels.append(pred_arg_structures[i]['L'])
            
        except KeyError:
            continue
    return features, labels

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



def is_exact_match(pred, arg):
    n_predicates = len(pred) - 7
    for n in range(n_predicates):
        if pred[7+n] == 'PREDICATE':
            return arg[7+n]
    return 'O'



def find_best_parameters(trainfile, selected_features):
    """
    Function to find the best parameters for the classifier.
    """
    training_opt = pd.read_csv(training_path_opt, encoding ='utf-8', sep='\t')
    
    x_train_opt, _ = extract_features_and_gold_labels(training_path_opt, selected_features)
    
    # Transforming the features to vectors:
    vec = DictVectorizer()
    x_train_opt_vec = vec.fit_transform(x_train_opt)
    # Adding labels to a seperate list:
    y_train_opt = training_opt.Label.to_list()
    
    classifier = LinearSVC()
    parameters = dict(
        C = (0.01,0.1,1.0),
        loss = ('hinge','squared_hinge'),
        tol = (1e-4,1e-3,1e-2,1e-1))

    grid = GridSearchCV(estimator = classifier, param_grid=parameters, cv=5, scoring='f1_macro')
    grid.fit(x_train_opt_vec, y_train_opt)
    classifier = grid.best_estimator_
    print('Best parameters:',grid.best_params_)
    
# def error_analysis(testfile):
    
    
#     error_analysis_df = pd.read_csv(testfile, sep="\t", header=None, names=["Book", "Sent nr", "Token nr", "Token", "Label"])
#     tokenlist = error_analysis_df['Token'].tolist()
#     # d = {'token': tokenlist, 'true': goldlabels, 'pred': predictions}
#     # error_df = pd.DataFrame(d)

#     # get all matches together in lists
#     matches = [[],[],[]]
#     errors = [[],[],[]]
#     i = 0
#     for token in tokenlist:
#         if goldlabels[i] == predictions[i]:
#             matches[0].append(token)
#             matches[1].append(goldlabels[i])
#             matches[2].append(predictions[i])
#         if goldlabels[i] != predictions[i]:
#             errors[0].append(token)
#             errors[1].append(goldlabels[i])
#             errors[2].append(predictions[i]) 
#         i +=1

#     d = {'token': matches[0], 'true': matches[1], 'pred': matches[2]}
#     matches_df = pd.DataFrame(d)
#     matches_df

#     d = {'token': errors[0], 'true': errors[1], 'pred': errors[2]}
#     errors_df = pd.DataFrame(d)
#     errors_df


#define which from the available features will be used (names must match key names of dictionary feature_to_index)
# all_features = ['Token', 'Pre-token', 'Next-token', 'Lemma', 'Pre-lemma', 'Next-lemma', 'POS', 'Pre-POS', 'Next-POS', 'POS_classified', 'Punctuation_python', 'MatchesNegExp', 'HasNegAffix', 'Negated event', 'NegAffix']

# print('confusion matrix and classification report for all features:')

# sparse_feature_reps, labels = extract_features_and_gold_labels(trainfile, all_features)
# #we can use the same function as before for creating the classifier and vectorizer
# svm_classifier, vectorizer = create_vectorizer_and_classifier(sparse_feature_reps, labels)
# #when applying our model to new data, we need to use the same features
# predictions, goldlabels = get_predicted_and_gold_labels(testfile, vectorizer, svm_classifier, all_features)
# print_confusion_matrix(predictions, goldlabels)
# print_precision_recall_fscore(predictions, goldlabels)
# report = classification_report(goldlabels,predictions,digits = 7, zero_division=0)
# print(report)

def display_output(predications, labels):
    for i in range(len(predictions)):
        print(predications[i], labels[i])

print('confusion matrix and classification report for selected features:')

#define which from the available features will be used (names must match key names of dictionary feature_to_index)
selected_features = ['index', 'form', 'lemma', 'xpos', 'head', 'deprel', 'NE', 'children']

feature_values, labels = extract_features_and_gold_labels(trainfile, selected_features)
print('features extracted')
#we can use the same function as before for creating the classifier and vectorizer
svm_classifier, vectorizer = create_vectorizer_and_classifier(feature_values, labels)
print('classifier is fit')
#when applying our model to new data, we need to use the same features
predictions, goldlabels = get_predicted_and_gold_labels(testfile, vectorizer, svm_classifier, selected_features, 'ewt_test_predictions')
display_output(predictions, goldlabels)
print_confusion_matrix(predictions, goldlabels)
print_precision_recall_fscore(predictions, goldlabels)
report = classification_report(goldlabels,predictions,digits = 7, zero_division=0)
print(report)

#find_best_parameters(trainfile, selected_features)


# error_analysis(testfile)
import sys
import pandas as pd
import os
from collections import defaultdict, Counter
import glob
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn import metrics
import csv

## THIS SCRIPT WAS ADAPTED FROM THE ML4NLP COURSE

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

def extract_annotations(inputfile, delimiter='\t'):
    '''
    This function extracts annotations represented in the conll format from a file

    :param inputfile: the path to the conll file (str)
    :param delimiter: optional parameter to overwrite the default delimiter (tab) (str)
    :returns: the annotations as a list
    '''
    conll_input = pd.read_csv(inputfile, sep=delimiter, encoding='utf-8')
    annotations = conll_input['label'].tolist()
    return annotations

def evaluate_correct_predicates(conll_file):
    conll_object = list(read_in_conll_file(conll_file))
    total = len(conll_object)
    identified = 0
    missed = 0
    wrong=0
    for row in conll_object:
        if len(row) > 0:
            if row[-1] == 'PREDICATE' and row[11] != 'O': # true positive
                identified +=1
            elif row[-1] == 'PREDICATE': # false negative
                missed +=1
            elif row[-1] != 'PREDICATE' and row[11] != 'O': # false positive
                wrong +=1
    total_n_predicates = identified + missed
    #true_negatives = total-total_n_predicates
    precision = identified/(total_n_predicates)
    recall = identified/(missed+identified)
    f = (2*precision*recall) / (precision+recall)
    print('PREDICATE DETECTION SCORES')
    print(f'Precision: {precision} - Recall: {recall} - F-score: {f}')

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


def evaluate_correct_arguments(conll_file):
    conll_object = read_in_conll_file(conll_file)
    sentences = read_sentences(conll_object)
    identified = 0
    missed = 0
    wrong=0
    for i, sentence in enumerate(sentences['sentences']):
        current_predicate = sentences['predicate'][i]
        for row in sentence:
            if len(row) > 0:
                if 'ARG' in row[-1] and row[12] != 'O':
                    pred_references = row[12].strip('RB_ARG:').split('-')
                    #print(current_predicate, pred_references)
                    if current_predicate in pred_references:
                        identified +=1
                    else:
                        wrong +=1
                elif 'ARG' in row[-1]:
                    missed +=1
                elif row[12] != 'O' and 'ARG' not in row[-1]:
                    wrong +=1
    total_n_args = identified + missed
    #true_negatives = total-total_n_predicates
    precision = identified/(total_n_args)
    recall = identified/(missed+identified)
    f = (2*precision*recall) / (precision+recall)
    print('ARGUMENT DETECTION SCORES')
    print(f'Precision: {precision} - Recall: {recall} - F-score: {f}')

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

def provide_output_tables(gold_annotations, system_annotations):
    '''
    Create tables based on the evaluation of various systems

    :param evaluations: the outcome of evaluating one or more systems
    '''
    #https:stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
    # report = pd.DataFrame(classification_report(gold_annotations, system_annotations, output_dict = True)).transpose()
    # print('Precision, recall and f-score:')
    # #print(evaluations_pddf)
    # print('\n')
    # print(report.to_latex())
    # evaluation_counts = obtain_counts(gold_annotations, system_annotations)
    # provide_confusion_matrix(evaluation_counts)
    print_confusion_matrix(system_annotations, gold_annotations)
    print_precision_recall_fscore(system_annotations, gold_annotations)
    report = classification_report(system_annotations, gold_annotations,digits = 7, zero_division=0)
    print(report)

def main(my_args=None):
    '''
    Runs the evaluation of a single model (see README for instructions)
    '''
    if my_args is None:
        my_args = sys.argv
    gold = my_args[1]
    modelfile = my_args[2]

    print('FILE:    ', '-'.join(modelfile.split('\\')[-1:][0].split("_"))[:-4])
    gold_annotations = extract_annotations(gold)
    print(set(gold_annotations))
    system_annotations = extract_annotations(modelfile)
    print(set(system_annotations))
    provide_output_tables(gold_annotations, system_annotations)
    evaluate_correct_predicates(gold)
    evaluate_correct_arguments(gold)
#my_args =['python', '../../SEM-2012-SharedTask-CD-SCO-simple.v2/SEM-2012-SharedTask-CD-SCO-test-preprocessed.txt', '../../models/testset//']
if __name__ == '__main__':
    print('EVAL')
    main()

#main(my_args)

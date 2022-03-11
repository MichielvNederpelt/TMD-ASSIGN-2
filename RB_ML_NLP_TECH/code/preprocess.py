import csv
from numpy import number
import spacy
from spacy.tokens import Doc
import sys

n_features = 9

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

def parse_doc(texts):
    '''
    Parse list of input text
    param texts: list of texts
    returns: texts processed in spacy pipeline
    '''
    nlp = spacy.load("en_core_web_sm")
    return nlp.pipe(texts)

def get_spacy_repres_of_conll(conll_file):
    '''
    parse conll file as sentences.
    param conll_file: path to conll file
    returns: a list of strings containing sentences
    '''
    conll_object = read_in_conll_file(conll_file)
    text = []
    #return the next item for the itterator
    next(conll_object)
    sentence = []
    while conll_object != None:
        try:
            row = next(conll_object)
            if len(row) > 0: # if not empty
                if row[0].startswith('#'): # skip these headers
                    continue
                else:
                    sentence.append(row[1]) # add row to sentence

            else:
                sentence.append('') # else append an empty newline
                text.append(' '.join(sentence)) # append the sentence as text
                sentence = [] # reset sentence container
        except StopIteration:
            break
    return text


def add_features(conll_file):
    '''
    Add Named Entity Labels and Children of current token features to the conll file
    param conll_file: path to conll file
    returns: conll file with added features
    '''
    conll_object = read_in_conll_file(conll_file)
    texts = get_spacy_repres_of_conll(conll_file)
    print('parsing the doc')
    docs = list(parse_doc(texts)) # a list of the parsed sentences
    print('done')
    with open(conll_file[:-7] + '-feats.conllu', 'w', newline='', encoding='utf-8') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        header = ['id', 'form', 'lemma', 'xpos', 'head', 'deprel', 'NE_label', 'children', 'label']
        csvwriter.writerow(header) # write header
        next(conll_object) # skip initial header
        sentence_idx = 0 # keeps track of current sentence to align with our spacy representation
        while conll_object != None:
            sentence_rows = []
            while True:
                try:
                    next_row = next(conll_object)
                except StopIteration: # abort if conll is empty
                    return
                if len(next_row) > 0:
                    if next_row[0].startswith('#'): # skip sentence headers (shouldn't be there but in case)
                        continue
                    sentence_rows.append(next_row) # append row to sentence structure
                if len(next_row) <=0:
                    csvwriter.writerow('') # if newline, write this down immediately
                    break
            doc = docs[sentence_idx] # get aligning doc
            entities = doc.ents # entitites
            entities_text = [e.text for e in entities] # entity names
            for i, row in enumerate(sentence_rows[:-1]): # skip final empty row
                children = '|'.join([tok.text for tok in list(doc[i].children)]) # store list of children as string separated by |

                if children:
                    row.insert(9, children) # insert list of children
                else:
                    row.insert(9,'O')
                if doc[i].text in entities_text:
                    idx = entities_text.index(doc[i].text)
                    row.insert(10, entities[idx].label_) # insert entity label
                else:
                    row.insert(10, 'O')
                csvwriter.writerow(row) # write to file
            sentence_idx +=1





def is_predicate(row, predicates_in_sentence):
    '''
    Check if current row is a predicate based on set of rules
    param row: current row
    param predicates_in_sentence: list of predicate idx that were detected in the sentence
    '''
    if row[6] == '0':
        predicates_in_sentence.append(row[0])
        return True, predicates_in_sentence
    elif (predicates_in_sentence and (row[7] in ['xcomp', 'ccomp', 'parataxis', 'advcl'] or (row[7] == 'conj' and row[3] == 'VERB') \
        or (row[7]) == 'conj' and row[6] ) and row[6] in predicates_in_sentence) or (row[3] == 'AUX' and row[4] in ['VBD', 'VBZ', 'VBN', 'VBP', 'VB', 'VBG'])\
            or (('acl' in row[7] or 'advcl' in row[7]) and row[3] == 'VERB'):
        predicates_in_sentence.append(row[0])
        return True, predicates_in_sentence
    return False, predicates_in_sentence

def is_argument(predicates, row):
    '''
    Check if current row is an argument
    param predicates: list of predicates in sentence (list of conll row lists)
    param row: current row (list)
    returns: list_of_connected_predicate_idx: a list containing the token indexes in the sentence of the predicate the argument is related to
    '''
    list_of_connected_predicate_idxs = []
    for predicate in predicates:
        predicate_idx = predicate[0]
        if ('nsubj' in row[7] or 'obj' in row[7] or 'obl' in row[7]) and row[6] == predicate_idx\
            or (predicate[7] == 'xcomp' and predicate[6] in list_of_connected_predicate_idxs and ('nsubj' in row[7] and row[6] == predicate[6])):
            # predicate is an open clausal complement to another predicate and the current row is subject to this predicate
            list_of_connected_predicate_idxs.append(predicate_idx)
    return list_of_connected_predicate_idxs

def squeeze_gold(conll_file):
    """
    Preprocesses the files. Adjust labels and, for every predicate in a sentence, copy the sentence so that each copy only contains
    one 'active' predicate.
    param conll_file: path to conll file
    returns: an extended, preprocessed conll file with sentence predicates redistributed per predicate
    """
    conll_object = read_in_conll_file(conll_file)
    with open(conll_file[:-7] + '-sq.conllu', 'w', newline='', encoding='utf-8') as outputcsv:
        # write header
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        header = ['id', 'form', 'lemma', 'xpos', 'head', 'deprel', 'pr_label']
        csvwriter.writerow(header)
        next(conll_object)
        while conll_object != None:
            list_of_sentence_rows = []
            while True: # we collect the sentence
                try:
                    next_row = next(conll_object)
                    if len(next_row) > 0: # only if not empty
                        if next_row[0].startswith('#'): # skip non-tokens
                            continue
                        else: # delete unwanted features
                            del next_row[9] # extra features
                            del next_row[8] #dependencies (head + relation are already separately represented)
                    list_of_sentence_rows.append(next_row)
                    if len(next_row) <=0: # the sentence ends if next row is newline
                        break
                except StopIteration:
                    break
            try:
                number_of_predicates = len(list_of_sentence_rows[0])-n_features # the number of predicates equals the number of rows added to the number of features
            except IndexError: # safety measure for when we try to parse an empty line
                break

            if number_of_predicates == 0: # if there are no predicates in the sentence, we just copy the sentence.
                for row in list_of_sentence_rows:
                    csvwriter.writerow(row[:n_features-1] + ['O'])

            for i in range(1, number_of_predicates+1): # go through all predicates
                for row in list_of_sentence_rows:
                    if len(row) <= 0: # write down empty lines unchanged
                        csvwriter.writerow(row)
                        continue
                    try:
                        target_col = row[n_features-1+i] # we want to select labels from one column at the time
                    except IndexError: # the conll file contains rows that are labeled as a copy of another row and then contain no predicate info
                        if 'CopyOf=' in row[n_features-1]:
                            idx = int(row[n_features-1].strip('CopyOf=')) # in this case, we check the row it was copied from
                            target_col = list_of_sentence_rows[idx][n_features-1+i] # and collect the label from this row


                    target_col = preprocess_labels(target_col)
                    extended_row = row[:n_features] + [target_col]
                    csvwriter.writerow(extended_row)

def preprocess_labels(target_col):
    '''Preprocess/simplify the label
    param: target_col: a label (str)
    returns: target_col: the updated label'''
    if 'C-' in target_col:
        target_col = target_col.strip('C-')
    if '-CXN' in target_col:
        target_col = target_col.strip('-CXN')
    if 'R-' in target_col:
        target_col = target_col.strip('R-')
    if '-DSP' in target_col:
        target_col = target_col.strip('-DSP')
    if 'ARGM' in target_col: # we simplify the labels by treating all modifiers as one class
        target_col = 'ARGM'
    if target_col == 'V': # replace the V label with the more straightforward PREDICATE
        target_col = 'PREDICATE'

    if target_col == '_': # instead of _ we do O for non-labeled instances
        target_col = 'O'
    return target_col

def add_rulebased_args(conll_file):
    '''
    Identify arguments rule-based and write to new conll file
    param conll_file: path to conll file
    returns: new conll file with predicates with their relation added
     '''
    print('ADDING ARGUMENTS')
    conll_object = read_in_conll_file(conll_file)
    with open(conll_file[:-17] + '-complete.conllu', 'w', newline='', encoding='utf-8') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        header = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats','head', 'deprel', 'sense', 'children', 'NE', 'rb-predicate', 'rb-arg', 'label']
        csvwriter.writerow(header)
        next(conll_object)

        while conll_object != None:
            list_of_sentence_rows = []
            list_of_sentence_predicates = []
            while True: # while we are in a sentence
                try:
                    next_row = next(conll_object)
                    list_of_sentence_rows.append(next_row) # store sentence rows
                    if len(next_row) > 0:
                        if next_row[11] == 'RB_PRED' or next_row[11] == 'INC_PRED':
                            list_of_sentence_predicates.append(next_row) # if this row is a predicate, save it too
                    else:
                        break
                except StopIteration:
                    return
            for i, row in enumerate(list_of_sentence_rows): # go through rows in the sentence
                if len(row) <= 0:
                    csvwriter.writerow(row) # if we have new line, write down
                    continue
                predicate_relations = is_argument(list_of_sentence_predicates, row) # get the predicates the argument is related to

                if predicate_relations: # if the argument is related to any predicates, write it down as RB_ARG:[list of predicate indices]
                    row.insert(12, 'RB_ARG:'+ '-'.join(predicate_relations))
                else:
                    row.insert(12, 'O')
                #print(row)
                csvwriter.writerow(row)

def read_sentences_as_list_of_lists(conll_file):
    sentences = [] # where we collect all the sentences
    current_sent = [] # initalize current sentence
    conll_object = read_in_conll_file(conll_file)
    next(conll_object)
    for row in conll_object:
        if len(row)>0 :
            current_sent.append(row)
        else:
            sentences.append(current_sent)
            current_sent = []
    return sentences

def add_rulebased_predicates(conll_file):
    '''
    Identifies the predicates in the sentence rule based and write to new conll file
    param conll_file: path to conll file
    returns: new conll file with collumn of token being a predicate or not
     '''
    print('ADDING PREDICATES')
    with open(conll_file[:-7] + '-p.conllu', 'w', newline='', encoding='utf-8') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        header = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats','head', 'deprel', 'sense', 'children', 'NE', 'rb-predicate', 'rb-arg', 'label']
        csvwriter.writerow(header)
        sentences = read_sentences_as_list_of_lists(conll_file)

        for sentence in sentences: # we go through the sentence two times: front to back and back to front in order to capture all predicates that may depend on one another
            labeled_sentence = label_predicates(sentence)
            reverse_labeled_sentence = reversed(label_predicates(labeled_sentence, reverse=True))

            for row in reverse_labeled_sentence:
                csvwriter.writerow(row)
            csvwriter.writerow('')


def label_predicates(sentence, reverse=False):
    '''Labels the predicates using rules
    param: sentence: the sentence to label
    param: reverse: if reverse is True, the sentence is processed in reverse'''
    predicates_in_sentence = []
    if reverse:
        sentence = list(reversed(sentence))
        for row in sentence:
            val_pred, predicates_in_sentence = is_predicate(row, predicates_in_sentence)
            gold_label = row[-1]
            if val_pred == True and gold_label == 'PREDICATE':
                row[11] = 'RB_PRED' # if our rule-based label is predicate, we write this label if it aligns with the gold data (to ensure alignment in further ML applications)
            elif val_pred == True:
                row[11] =  'INC_PRED' # else, we still store it as predicate, but as predicate that is not applicable in the sentence
        else:
            predicates_in_sentence = []

    else:
        for row in sentence:
            val_pred, predicates_in_sentence = is_predicate(row, predicates_in_sentence)

            gold_label = row[-1]
            if val_pred == True and gold_label == 'PREDICATE':
                row.insert(11, 'RB_PRED')
            elif val_pred == True:
                row.insert(11, 'INC_PRED')
            else:
                row.insert(11, 'O')
        else:
            predicates_in_sentence = []
    return sentence

#Adapt paths to relative paths for now hard coded paths are fine

def main(argv=None):
    '''
    Runs the evaluation of a single model (see README for instructions)
    '''
    if argv is None:
        argv = sys.argv
    trainfile = argv[1]
    devfile = argv[2]
    testfile = argv[3]
    for file in [trainfile, devfile, testfile]:
        squeezed_file = file[:-7] + '-sq.conllu'
        feature_file = squeezed_file[:-7] + '-feats.conllu'
        predicate_file = feature_file[:-7] + '-p.conllu'

        squeeze_gold(file)
        add_features(squeezed_file)
        add_rulebased_predicates(feature_file)
        add_rulebased_args(predicate_file)

main()
# squeeze_gold(path)
# add_features(sq_path)
# add_rulebased_predicates(feature_path)
# add_rulebased_args(pred_path)

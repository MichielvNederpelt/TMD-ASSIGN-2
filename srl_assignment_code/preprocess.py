import csv
import spacy
from spacy.tokens import Doc

def custom_tokenizer(text):
    """
    this should be removed right?
    """
    return Doc(nlp.vocab, tokens_dict[text])

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
    print('parsing')
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
                children = list(doc[i].children)

                if children:
                    row.insert(6, children) # insert list of children
                else:
                    row.insert(6,'O')
                if doc[i].text in entities_text:
                    idx = entities_text.index(doc[i].text)
                    row.insert(6, entities[idx].label_) # insert entity label
                else:
                    row.insert(6, 'O')
                csvwriter.writerow(row) # write to file
            sentence_idx +=1


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


def is_predicate(row, predicates_in_sentence):
    '''
    Check if current row is a predicate based on set of rules
    param row: current row
    param predicates_in_sentence: list of predicate idx that were detected in the sentence
    '''
    if row[4] == '0':
        predicates_in_sentence.append(row[0])
        return True, predicates_in_sentence
    elif predicates_in_sentence and (row[5] in ['xcomp', 'ccomp', 'parataxis', 'advcl'] or (row[5] == 'conj' and row[3] == 'VERB')) and row[4] in predicates_in_sentence:
        predicates_in_sentence.append(row[0])
        return True, predicates_in_sentence
    return False, predicates_in_sentence

def is_argument(predicate, row):
    '''
    Check if current row is an argument
    param predicate: predicate position (idx (int)) of referent predicate
    param row: current row (list)
    '''
    if ('nsubj' in row[5] or 'obj' in row[5] or 'obl' in row[5]) and row[4] == predicate:
        return True
    return False

def preprocess_gold(conll_file):
    """
    preprocess the gold labels in the conll file and overwrite these to a new conll file
    param conll_file: path to conll file
    returns: conll file with preprocessed labels added
    """
    conll_object = read_in_conll_file(conll_file)
    with open(conll_file[:-7] + '-prep.conllu', 'w', newline='', encoding='utf-8') as outputcsv:
        # write header
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        header = ['id', 'form', 'lemma', 'xpos', 'head', 'deprel', 'label']
        csvwriter.writerow(header)
        next(conll_object)
        while conll_object != None:
            list_of_sentence_rows = []
            try:
                next_row = next(conll_object)
                if len(next_row) <= 0:
                    csvwriter.writerow(next_row)
                    continue
                if len(next_row) > 0: # only if not empty
                    if next_row[0].startswith('#'): # skip non-tokens
                        continue
                    else: # delete unwanted features
                        del next_row[10]
                        del next_row[9]
                        del next_row[5]
                        del next_row[3]
            except StopIteration:
                print('empty')
                break

            number_of_predicates = len(next_row)-7
            converted_cols = []
            for i in range(1, number_of_predicates+1):
                try:
                    target_col = next_row[6+i]
                    if target_col == '_':
                        target_col = 'O'
                    elif target_col == 'V':
                        target_col = 'PREDICATE'
                    if 'C-' in target_col:
                        target_col = target_col.strip('C-')
                    if 'ARGM' in target_col:
                        target_col = 'ARGM'

                except IndexError:
                    if 'CopyOf=' in next_row[6]:
                        idx = int(next_row[6].strip('CopyOf='))
                        target_col = list_of_sentence_rows[idx][6+i]
                converted_cols.append(target_col)
            extended_row = next_row[:6] + converted_cols
            csvwriter.writerow(extended_row)

def squeeze_gold(conll_file):
    """
    for every predicate in a sentence make new line with predicate and its arguments
    param conll_file: path to conll file
    returns: an extended conll file with sentence predicates redistributed per predicate line
    """
    conll_object = read_in_conll_file(conll_file)
    with open(conll_file[:-7] + '-squeeze.conllu', 'w', newline='', encoding='utf-8') as outputcsv:
        # write header
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        header = ['id', 'form', 'lemma', 'xpos', 'head', 'deprel', 'label']
        csvwriter.writerow(header)
        next(conll_object)
        while conll_object != None:
            list_of_sentence_rows = []
            while True:
                try:
                    next_row = next(conll_object)
                    if len(next_row) > 0: # only if not empty
                        if next_row[0].startswith('#'): # skip non-tokens
                            continue
                        else: # delete unwanted features
                            del next_row[10]
                            del next_row[9]
                            del next_row[5]
                            del next_row[3]
                    list_of_sentence_rows.append(next_row)
                    if len(next_row) <=0:
                        break
                except StopIteration:
                    print('empty')
                    break

            number_of_predicates = len(list_of_sentence_rows[0])-7
            for i in range(1, number_of_predicates+1):
                #print('predicate', i)
                for row in list_of_sentence_rows:
                    #print(row)
                    if len(row) <= 0:
                        csvwriter.writerow(row)
                        continue
                    try:
                        target_col = row[6+i]
                        if target_col == '_':
                            target_col = 'O'
                        elif target_col == 'V':
                            target_col = 'PREDICATE'
                        if 'C-' in target_col:
                            target_col = target_col.strip('C-')
                        if 'ARGM' in target_col:
                            target_col = 'ARGM'

                    except IndexError:
                        if 'CopyOf=' in row[6]:
                            idx = int(row[6].strip('CopyOf='))
                            target_col = list_of_sentence_rows[idx][6+i]

                    extended_row = row[:6] + [target_col]
                    csvwriter.writerow(extended_row)


def preprocess_args(conll_file):
    '''
    preprocess arguments and write to new conll file
    param conll_file: path to conll file
    returns: new conll file with predicates with their relation added
     '''
    conll_object = read_in_conll_file(conll_file)
    with open(conll_file[:-7] + '-args.conllu', 'w', newline='', encoding='utf-8') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        header = ['id', 'form', 'lemma', 'xpos', 'head', 'deprel', 'NE_label', 'children', 'rb-predicate', 'rb-arg', 'label']
        csvwriter.writerow(header)
        next(conll_object)

        while conll_object != None:
           # print('new sentence')
            list_of_sentence_rows = []
            list_of_sentence_predicates = []
            while True: # while we are in a sentence
                try:
                    next_row = next(conll_object)
                    list_of_sentence_rows.append(next_row)
                    if len(next_row) > 0:
                        if next_row[8] == 'RB_PRED':
                            list_of_sentence_predicates.append(next_row[0])
                    else:
                        break
                except StopIteration:
                    return
          #  print('predicates in this sentence detected:', list_of_sentence_predicates)
            for i, row in enumerate(list_of_sentence_rows):
                if len(row) <= 0:
                    csvwriter.writerow(row)
                    continue
                predicate_relations = []
                for pred in list_of_sentence_predicates:
                    val = is_argument(pred, row)
                    #print(pred, row)
                    if val == True:
                        #print('Predicate', pred, 'has an argument', row[1])
                    #print('true')
                        predicate_relations.append(pred)
                if predicate_relations:
                    row.insert(9, 'RB_ARG:' + '-'.join(predicate_relations))
                else:
                    row.insert(9, 'O')
                #print(row)
                csvwriter.writerow(row)

def preprocess_predicates(conll_file):
    '''
    preprocess predicates, one per sentence, and write to new conll file
    param conll_file: path to conll file
    returns: new conll file with collumn of token being a predicate or not
     '''
    conll_object = read_in_conll_file(conll_file)
    with open(conll_file[:-7] + '-preds.conllu', 'w', newline='', encoding='utf-8') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        #header = ['id', 'form', 'lemma', 'xpos', 'head', 'deprel', 'label']
        #csvwriter.writerow(header)
        predicates_in_sentence = []
        for row in conll_object:
            if len(row)>0 :
                #if row[0].startswith('#'):
                #    continue
                val_pred, predicates_in_sentence = is_predicate(row, predicates_in_sentence)
                if val_pred == True:
                    row.insert(8, 'RB_PRED')
                else:
                    row.insert(8, 'O')
            else:
                predicates_in_sentence = []
            csvwriter.writerow(row)


#Adapt paths to relative paths for now hard coded paths are fine
path = r"C:\Users\Tessel Wisman\Documents\TextMining\NLPTech\UP_English-EWT\en_ewt-up-dev.conllu"
sq_path = r"C:\Users\Tessel Wisman\Documents\TextMining\NLPTech\UP_English-EWT\en_ewt-up-dev-prep.conllu"
feature_path = r"C:\Users\Tessel Wisman\Documents\TextMining\NLPTech\UP_English-EWT\en_ewt-up-dev-prep-feats.conllu"
pred_path =r"C:\Users\Tessel Wisman\Documents\TextMining\NLPTech\UP_English-EWT\en_ewt-up-dev-prep-feats-preds.conllu"

preprocess_gold(path)
add_features(sq_path)
preprocess_predicates(feature_path)
preprocess_args(pred_path)

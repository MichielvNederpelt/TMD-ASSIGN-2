import csv

from numpy import number 

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

def extract_sentences(conll_file):
    '''extracts the sentences (# text =) from conll files and saves'''
    conll_object = read_in_conll_file(conll_file)
    with open(conll_file[:-7] + '_sents.conllu', 'w', newline='', encoding='utf-8') as output:
        for row in conll_object:
            if len(row) > 0 and row[0].startswith('# text'):
                output.write(row[0][9:] + '\n')

def get_predicates(sentence):
    root = get_root(sentence)
    predicates = [root]
    for word in sentence.words:
        if word.deprel == 'xcomp'  or word.deprel == 'ccomp' or \
           (word.deprel == 'conj' and word.pos == 'VERB') or word.deprel == 'parataxis' or word.deprel == 'advcl':
            # if it is relation to another predicate
            for preds in predicates:
                if preds.id == word.head:
                    predicates.append(word)
    return predicates

def is_predicate(row, predicates_in_sentence):
    #print(row[6], row[7])
    if row[6] == '0':
        predicates_in_sentence.append(row[0])
        return True, predicates_in_sentence
    elif predicates_in_sentence and (row[7] in ['xcomp', 'ccomp', 'parataxis', 'advcl'] or (row[7] == 'conj' and row[3] == 'VERB')) and row[6] in predicates_in_sentence:
        predicates_in_sentence.append(row[0])
        return True, predicates_in_sentence
    return False, predicates_in_sentence

def is_argument(predicate, row):
    if ('nsubj' in row[7] or 'obj' in row[7] or 'obl' in row[7]) and row[6] == predicate:
        return True
    return False

def squeeze_gold(conll_file):
    ''' Checks if the file contains any blank lines or lines starting with # (comments) and removes them'''
    conll_object = read_in_conll_file(conll_file)
    with open(conll_file[:-7] + '-squeeze.conllu', 'w', newline='', encoding='utf-8') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        header = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'predicate', 'pred_sense','arg_structs']
        csvwriter.writerow(header)
        next(conll_object)
        #length = len(list(conll_object))
        while conll_object != None:
            list_of_sentence_rows = []
            while True:
                try:
                    next_row = next(conll_object)
                    if len(next_row) > 0 and next_row[0].startswith('#'):
                        continue
                    list_of_sentence_rows.append(next_row)
                    if len(next_row) <=0:
                        break
                except StopIteration:
                    return
            
            #print([row[1] for row in list_of_sentence_rows], '\n')
            number_of_predicates = len(list_of_sentence_rows[0])-11
            for i in range(1, number_of_predicates+1):
                #print('predicate', i)
                for row in list_of_sentence_rows:
                    #print(row)
                    if len(row) <= 0:
                        csvwriter.writerow(row)
                        continue
                    try:
                        target_col = row[10+i]
                    except IndexError:
                        if 'CopyOf=' in row[10]:
                            idx = int(row[10].strip('CopyOf='))
                            target_col = list_of_sentence_rows[idx][10+i]
                        #print(i, len(row), number_of_predicates)
                        #print(row, '\n')
                        #for r in list_of_sentence_rows:
                        #    print(r)
                        #print('\n')
                        
                    #print(row[:10] + [target_col])
                    extended_row = row[:10] + [target_col]
                    csvwriter.writerow(extended_row)
                #csvwriter.writerow('\n')

def preprocess_args(conll_file):
    ''' Checks if the file contains any blank lines or lines starting with # (comments) and removes them'''
    conll_object = read_in_conll_file(conll_file)
    with open(conll_file[:-7] + '-args.conllu', 'w', newline='', encoding='utf-8') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        header = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc', 'predicate', 'argument']
        csvwriter.writerow(header)
        next(conll_object)
        #length = len(list(conll_object))
        c = 0
        while conll_object != None:
            list_of_sentence_rows = []
            list_of_sentence_predicates = []
            while True:
                try:
                    next_row = next(conll_object)
                    list_of_sentence_rows.append(next_row)
                    if len(next_row) <=0:
                        break
                    if next_row[10] == 'PRED':
                        list_of_sentence_predicates.append(next_row[0])
                except StopIteration:
                    return
            #print([row[1] for row in list_of_sentence_rows], '\n')
            for pred in list_of_sentence_predicates:
                for i, row in enumerate(list_of_sentence_rows):
                    if len(row) <= 0:
                        csvwriter.writerow(row)
                        continue
                    if i+1 != int(pred):
                        row[10] = '_'
                    val = is_argument(pred, row)
                    if val == True:
                        #print('true')
                        row.insert(11, 'ARG')
                    else:
                        row.insert(11, '_')
                    #print(row)
                    csvwriter.writerow(row)
                #csvwriter.writerow('\n')

                
def is_predicate(row, predicates_in_sentence):
    #print(row[6], row[7])
    if row[6] == '0':
        predicates_in_sentence.append(row[0])
        return True, predicates_in_sentence
    elif predicates_in_sentence and (row[7] in ['xcomp', 'ccomp', 'parataxis', 'advcl'] or (row[7] == 'conj' and row[3] == 'VERB')) and row[6] in predicates_in_sentence:
        predicates_in_sentence.append(row[0])
        return True, predicates_in_sentence
    return False, predicates_in_sentence

def preprocess_predicates(conll_file):
    ''' Checks if the file contains any blank lines or lines starting with # (comments) and removes them'''
    conll_object = read_in_conll_file(conll_file)
    with open(conll_file[:-7] + '-preds.conllu', 'w', newline='', encoding='utf-8') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        header = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc','predicate']
        csvwriter.writerow(header)
        predicates_in_sentence = []
        for row in conll_object:
            if len(row)>0 :
                if row[0].startswith('#'):
                    continue
                #if len(row)>10:
                #    row.insert(10, 'GOLD_PRED')
                #else:
                #    row.insert(10, 'X')
                val_pred, predicates_in_sentence = is_predicate(row, predicates_in_sentence)
                if val_pred == True:
                    row.insert(10, 'PRED')
                else:
                    row.insert(10, '_')
            else:
                predicates_in_sentence = []
            csvwriter.writerow(row[:11])

path = r"C:\Users\Tessel Wisman\Documents\TextMining\NLPTech\UP_English-EWT\en_ewt-up-train.conllu"

pred_path = r"C:\Users\Tessel Wisman\Documents\TextMining\NLPTech\UP_English-EWT\en_ewt-up-train-preds.conllu"
squeeze_gold(path)
#preprocess_predicates(path)
#preprocess_args(pred_path)
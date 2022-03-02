import csv
import spacy
from spacy.tokens import Doc



def parse_doc(text):
    nlp = spacy.load("en_core_web_sm")
    #nlp.max_length = 5300000

    text = text.lower()
    doc = nlp(text)
    # from the Stanford doc: https://stanfordnlp.github.io/stanza/depparse.html
    #print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')
    return doc

def get_spacy_repres_of_conll(conll_file):
    conll_object = read_in_conll_file(conll_file)
    text = []
    c=0
    next(conll_object)
    for row in conll_object:
        if len(row) > 0:
            text.append(row[1])
        else:
            text.append('')
    doc = parse_doc(' '.join(text))
    return doc

def add_features2(conll_file):
    conll_object = read_in_conll_file(conll_file)
    #doc = get_spacy_repres_of_conll(conll_file)

    with open(conll_file[:-7] + '_try.conllu', 'w', newline='', encoding='utf-8') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        header = ['id', 'form', 'lemma', 'xpos', 'head', 'deprel', 'NE_label', 'children', 'label']
        csvwriter.writerow(header)
        next(conll_object)
        c = 0
        l = 1078502
        i=0
        while conll_object != None:
            c+=1
            if c % 100 ==0:
                print(c/l)        
            next_row = next(conll_object)
            c+=1
            if c % 1000 ==0:
                print(c/l)
            if len(next_row) > 0:
                if next_row[0].startswith('# text ='):
                    sentence = next_row[0].strip('# text =')
                    doc = parse_doc(sentence)
                    entities = doc.ents
                    entities_text = [e.text for e in entities]
                    continue
            if len(next_row) <=0:
                i=0
                continue 
            print(doc[i].text, next_row[1], i)
            children = list(doc[i].children)
            if children: 
                next_row.insert(6, children)
            else:
                next_row.insert(6,'O')
                #print(type(entities[0]))
            if doc[i].text in entities_text:
                idx = entities_text.index(doc[i].text)
                next_row.insert(6, entities[idx].label_)
            else:
                next_row.insert(6, 'O')
            csvwriter.writerow(next_row)


            i+=1




def add_features(conll_file):
    conll_object = read_in_conll_file(conll_file)
    #doc = get_spacy_repres_of_conll(conll_file)

    with open(conll_file[:-7] + '_try.conllu', 'w', newline='', encoding='utf-8') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        header = ['id', 'form', 'lemma', 'xpos', 'head', 'deprel', 'NE_label', 'children', 'label']
        csvwriter.writerow(header)
        next(conll_object)
        c = 0
        l = 1078502
        while conll_object != None:
            sentence_rows = []   
            c+=1
            if c % 10000 ==0:
                print(c/l) 
            x = 0       
            while True:
                next_row = next(conll_object)
                c+=1
                x+=1
                if c % 1000 ==0:
                    print(c/l)
                if len(next_row) > 0:
                    if next_row[0].startswith('#'):
                        continue
                    sentence_rows.append(next_row)
                if len(next_row) <=0 and x > 100000:
                    x=0
                    print('bye')
                    break
            sentence = ' '.join([row[1] for row in sentence_rows])
            print('boiii')
            doc = parse_doc(sentence)
            entities = doc.ents
            entities_text = [e.text for e in entities]
            for i, row in enumerate(sentence_rows):
                print('parsinggg')    
                children = list(doc[i].children)
                if children: 
                    row.insert(6, children)
                else:
                    row.insert(6,'O')
                    #print(type(entities[0]))
                if doc[i].text in entities_text:
                    idx = entities_text.index(doc[i].text)
                    row.insert(6, entities[idx].label_)
                else:
                    row.insert(6, 'O')
                csvwriter.writerow(row)

def add_features1(conll_file):
    conll_object = read_in_conll_file(conll_file)
    doc = get_spacy_repres_of_conll(conll_file)
    entities = doc.ents
    entities_text = [e.text for e in entities]
    with open(conll_file[:-7] + '_feats.conllu', 'w', newline='', encoding='utf-8') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        header = ['id', 'form', 'lemma', 'xpos', 'head', 'deprel', 'NE', 'children', 'label']
        csvwriter.writerow(header)
        next(conll_object)
        for i, row in enumerate(conll_object):
            if len(row)>0:
                children = list(doc[i].children)
                if children: 
                    row.insert(6, children)
                else:
                    row.insert(6,'O')
                if doc[i].text in entities_text:
                    idx = entities_text.index(doc[i].text)
                    row.insert(6, entities[idx].label_)
                else:
                    row.insert(6, 'O')
            csvwriter.writerow(row)
        
def add_features3(conll_file):
    conll_object = read_in_conll_file(conll_file)
    doc = get_spacy_repres_of_conll(conll_file)
    entities = doc.ents
    entities_text = [e.text for e in entities]
    with open(conll_file[:-7] + '_feats.conllu', 'w', newline='', encoding='utf-8') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        header = ['id', 'form', 'lemma', 'xpos', 'head', 'deprel', 'NE', 'children', 'label']
        csvwriter.writerow(header)
        next(conll_object)
        for i, row in enumerate(conll_object):
            if len(row)>0:
                children = list(doc[i].children)
                if children: 
                    row.insert(6, children)
                else:
                    row.insert(6,'O')
                if doc[i].text in entities_text:
                    idx = entities_text.index(doc[i].text)
                    row.insert(6, entities[idx].label_)
                else:
                    row.insert(6, 'O')
            csvwriter.writerow(row)


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

def squeeze_gold(conll_file):
    ''' Checks if the file contains any blank lines or lines starting with # (comments) and removes them'''
    conll_object = read_in_conll_file(conll_file)
    with open(conll_file[:-7] + '-squeeze.conllu', 'w', newline='', encoding='utf-8') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        header = ['id', 'form', 'lemma', 'xpos', 'head', 'deprel', 'label']
        csvwriter.writerow(header)
        next(conll_object)
        #length = len(list(conll_object))
        while conll_object != None:
            list_of_sentence_rows = []
            while True:
                try:
                    next_row = next(conll_object)
                    if len(next_row) > 0:
                        if next_row[0].startswith('#'):
                            continue
                        else:
                            del next_row[10]
                            del next_row[9]
                            del next_row[5]
                            del next_row[3]
                    list_of_sentence_rows.append(next_row)
                    if len(next_row) <=0:
                        break
                except StopIteration:
                    return
            #print([row[1] for row in list_of_sentence_rows], '\n')
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
                    except IndexError:
                        if 'CopyOf=' in row[6]:
                            idx = int(row[6].strip('CopyOf='))
                            target_col = list_of_sentence_rows[idx][6+i]
                    extended_row = row[:6] + [target_col]
                    csvwriter.writerow(extended_row)




path = r"C:\Users\Tessel Wisman\Documents\TextMining\NLPTech\UP_English-EWT\en_ewt-up-train1.conllu"
sq_path = r"C:\Users\Tessel Wisman\Documents\TextMining\NLPTech\UP_English-EWT\en_ewt-up-train1-squeeze.conllu"
pred_path = r"C:\Users\Tessel Wisman\Documents\TextMining\NLPTech\UP_English-EWT\en_ewt-up-train-squeeze-preds.conllu"
squeeze_gold(path)
#preprocess_predicates(sq_path)
#preprocess_args(pred_path)

add_features(sq_path)
import stanza

def read_file(path):
    with open(path, 'r', encoding='utf-8') as infile:
        text = infile.readlines()
    # print(''.join(text))
    return text

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
                    if next_row[6] == 'RB_PRED':
                        list_of_sentence_predicates.append(next_row[0])
                except StopIteration:
                    return
            #print([row[1] for row in list_of_sentence_rows], '\n')
            for i, row in enumerate(list_of_sentence_rows):
                if len(row) <= 0:
                    csvwriter.writerow(row)
                    continue
                # if i+1 != int(pred):
                #     row[6] = '_'
                for pred in list_of_sentence_predicates:
                    val = is_argument(pred, row)
                    if val == True:
                    #print('true')
                        row[6]= 'ARG'
                #print(row)
                csvwriter.writerow(row)
                
def preprocess_predicates(conll_file):
    ''' Checks if the file contains any blank lines or lines starting with # (comments) and removes them'''
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
                    row.insert(6, 'RB_PRED')
                else:
                    row.insert(6, 'O')
            else:
                predicates_in_sentence = []
            csvwriter.writerow(row)
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
            while True:
                next_row = next(conll_object)
                c+=1
                if c % 100 ==0:
                    print(c/l)
                if len(next_row) > 0:
                    if next_row[0].startswith('#'):
                        continue
                    sentence_rows.append(next_row)
                if len(next_row) <=0:
                    break
            sentence = ' '.join([row[1] for row in sentence_rows])
            doc = parse_doc(sentence)
            entities = doc.ents
            entities_text = [e.text for e in entities]
            for i, row in enumerate(sentence_rows):    
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


def add_features(conll_file):
    conll_object = read_in_conll_file(conll_file)
    doc = get_spacy_repres_of_conll(conll_file)
    entities = doc.ents
    entities_text = [e.text for e in entities]
    with open(conll_file[:-7] + '_try.conllu', 'w', newline='', encoding='utf-8') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        header = ['id', 'form', 'lemma', 'xpos', 'head', 'deprel', 'children', 'label']
        csvwriter.writerow(header)
        next(conll_object)
        for i, row in enumerate(conll_object):
            if len(row)>0:
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
        


def parse_doc(text):
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse, constituency')
    text = text.lower()
    doc = nlp(text)
    # from the Stanford doc: https://stanfordnlp.github.io/stanza/depparse.html
    #print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')
    return doc

def get_dependents(word, sentence):
    dependents = []
    for depword in sentence.words:
        if depword.head == word.id:
            dependents.append(depword)
    #print([(w.text, w.deprel) for w in dependents])
    return dependents 


def get_root(sentence):
    for word in sentence.words:
        if word.head <=0:
            return word



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

def get_subject(predicate, sentence):
    for word in sentence.words:
        if 'nsubj' in word.deprel and word.head == predicate.id:
            subject = word
            incl_dependents = get_all_dependents(subject, sentence)
            return incl_dependents
    return 
        
def get_object(predicate, sentence):
    for word in sentence.words:
        if 'obj' in word.deprel and word.head == predicate.id:    
            object = word
            entire_predicate = get_all_dependents(object, sentence)
            return entire_predicate
    return

def get_direct_dependents(head, sentence):
    deps = []
    for word in sentence.words:
        if word.head == head.id:
            deps.append(word)
    return deps

def get_substructure(head_of_struct, dep_struct):
    subargs = []
    for word in dep_struct.words:
        if word.id == head_of_struct.id:
            subarg = get_all_dependents(word, dep_struct)
            #print([s.text for s in subarg])
            subargs.append(subarg)
    return subargs


def get_obl(predicate, sentence):
    for word in sentence.words:
        if 'obl' in word.deprel and word.head == predicate.id:    
            obl = word
            entire_arg = get_substructure(obl,sentence)
            return entire_arg
    return


#print(wordtest.text, [w.text for w in sentencetest.words])
#get_dependents(wordtest, sentencetest)
def get_all_dependents(word, sentence):
    dependents = [word]
    stack = [word]
    while True:
        #print(stack)
        #print(dependents)
        current_node = stack.pop(0)
        cn_dependents = get_dependents(current_node, sentence)
        dependents += cn_dependents
        stack += cn_dependents
        if not stack:
            break 
    dependents = sorted(dependents, key= lambda word: word.id)
    #print([w.text for w in dependents])
    return dependents


def role_labeling(sentence):
    sent_text = [' '.join([s.text for s in sentence.words])]
    predicates = get_predicates(sentence)
    sentence_parse = {'sent': sent_text}
    for predicate in predicates:
        #print('PREDICATE:', predicate.text)
        subject = get_subject(predicate, sent)
        subject_txt = [s.text for s in subject] if subject else subject
        #if subject:
            #print('SUBJECT:', [s.text for s in subject])
        object = get_object(predicate, sent)
        object_txt = [o.text for o in object] if object else object
        #if object:
            #print([w.text for w in object])
        obl = get_obl(predicate, sentence)
        obl_txt = []
        if obl:
            for sstr in obl:
                obl_txt.append([o.text for o in sstr])
        sentence_parse[predicate.text] = {'ARG0': subject_txt, 'ARG1': object_txt, 'ARG2': obl_txt}
    return sentence_parse



testfile = read_file(r'..\UP_English-EWT\en_ewt-up-dev_sents.conllu')[10:15]
#print(''.join(testfile))
doc = parse_doc(''.join(testfile))


for sent in doc.sentences:
    print('----------------------------')
    #print(' '.join([w.text for w in sent.words]))
    
    parse = role_labeling(sent)
    print(parse)
    print([(w.text, w.deprel, sent.words[w.head-1].text) for w in sent.words])

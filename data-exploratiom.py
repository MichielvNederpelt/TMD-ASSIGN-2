import csv

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

def copy_predicates(conll_file):
    '''extracts the sentences (# text =) from conll files and saves'''
    conll_object = read_in_conll_file(conll_file)
    with open(conll_file[:-7] + '_sents.conllu', 'w', newline='', encoding='utf-8') as output:
        for row in conll_object:
            if len(row) <= 0:
                print('\n')
            if len(row) > 0 and not row[0].startswith('#'):
                print(row[11:])

path = r"C:\Users\Tessel Wisman\Documents\TextMining\NLPTech\UP_English-EWT\en_ewt-up-train.conllu"

with open(path, 'r', encoding='utf-8') as infile:
    conll=infile.readlines()

first = conll[:127935]
second = conll[127935:]

with open(r"C:\Users\Tessel Wisman\Documents\TextMining\NLPTech\UP_English-EWT\en_ewt-up-train1.conllu", 'w', encoding='utf-8') as outfile:
    for line in first:
        outfile.write(line)

with open(r"C:\Users\Tessel Wisman\Documents\TextMining\NLPTech\UP_English-EWT\en_ewt-up-train2.conllu", 'w', encoding='utf-8') as outfile:
    for line in second:
        outfile.write(line)
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

def extract_sentences(conll_file):
    '''extracts the sentences (# text =) from conll files and saves'''
    conll_object = read_in_conll_file(conll_file)
    with open(conll_file[:-7] + '_sents.conllu', 'w', newline='', encoding='utf-8') as output:
        for row in conll_object:
            if len(row) > 0 and row[0].startswith('# text'):
                output.write(row[0][9:] + '\n')

def preprocess(conll_file):
    ''' Checks if the file contains any blank lines or lines starting with # (comments) and removes them'''
    conll_object = read_in_conll_file(conll_file)
    with open(conll_file[:-7] + '_prep.conllu', 'w', newline='', encoding='utf-8') as outputcsv:
        csvwriter = csv.writer(outputcsv, delimiter='\t')
        for row in conll_object:
            if len(row) <= 0 or row[0].startswith('#'):
                continue
            csvwriter.writerow(row)

path = r"C:\Users\Tessel Wisman\Documents\TextMining\NLPTech\UP_English-EWT\en_ewt-up-dev.conllu"
preprocess(path)
extract_sentences(path)
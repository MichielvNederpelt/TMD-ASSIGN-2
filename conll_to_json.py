import json
import collections
import csv




def read_in_conll_file(conll_file, delimiter='\t'):
    '''
    Read in conll file and return structured object
    :param conll_file: path to conll_file
    :param delimiter: specifies how columns are separated. Tabs are standard in conll
    :returns structured representation of information included in conll file
    '''
    my_conll = open(conll_file, 'r')
    conll_as_csvreader = csv.reader(my_conll, delimiter=delimiter, quoting=csv.QUOTE_NONE)
    return conll_as_csvreader

def make_dictionary(conll_file):
    conll_as_csv = read_in_conll_file(conll_file)
    #partly from https://www.adamsmith.haus/python/answers/how-to-append-an-element-to-a-key-in-a-dictionary-with-python
    predicate_dict = collections.defaultdict(list)
    predicate_dict2 = collections.defaultdict(list)
    predicate_dict3 = collections.defaultdict(list)
    predicate_dict4 = collections.defaultdict(list)
    predicate_dict5 = collections.defaultdict(list)
    #return the next item for the itterator
    next(conll_as_csv)
    while conll_as_csv != None:
        try:
            row = next(conll_as_csv)
            if len(row) == 0:
                with open(conll_file.replace(".conll", ".jsonl"), 'a') as fp:
                    json.dump(predicate_dict, fp, indent = 4)
                    if len(predicate_dict2.values()) >2:
                        json.dump(predicate_dict2, fp, indent = 4)
                    if len(predicate_dict3.values()) >3:
                        json.dump(predicate_dict3, fp, indent = 4)
                    if len(predicate_dict4.values()) >3:
                        json.dump(predicate_dict4, fp, indent = 4)
                    if len(predicate_dict5.values()) >3:
                        json.dump(predicate_dict5, fp, indent = 4)
                    print("json dumped")
                predicate_dict.clear()
                predicate_dict2.clear()
                predicate_dict3.clear()
                predicate_dict4.clear()
                predicate_dict5.clear()
            if len(row) > 0: # if not empty
                if row[0].startswith('#'): # skip these headers
                    continue
                else:
                    predicate_dict["seq_words"].append(row[1]) # add token to seq_words key in dict
                    if row[11] != "_":
                        predicate_dict["BIO"].append(row[11])
                    else:
                        predicate_dict["BIO"].append("O")
                    if row[10] != "_" and row[11] == "V":
                        predicate_dict["pred_sense"].append(int(row[0])-1)
                        predicate_dict["pred_sense"].append(row[10])
                        predicate_dict["pred_sense"].append(row[11])
                        predicate_dict["pred_sense"].append(row[4])
                    else:
                        predicate_dict["pred_sense"].append("O")
                        predicate_dict["pred_sense"].append("O"))
                        predicate_dict["pred_sense"].append("O"))
                        predicate_dict["pred_sense"].append("O"))
            try:
                predicate_dict2["seq_words"] = predicate_dict["seq_words"]

                if row[12] != "_":
                    predicate_dict2["BIO"].append(row[12])
                else:
                    predicate_dict2["BIO"].append("O")
                if row[12] == "V":
                    predicate_dict2["pred_sense"].append(int(row[0])-1)
                    predicate_dict2["pred_sense"].append(row[10])
                    predicate_dict2["pred_sense"].append( row[12])
                    predicate_dict2["pred_sense"].append( row[4])
            except IndexError:
                continue
            try:
                predicate_dict3["seq_words"] = predicate_dict["seq_words"]

                if row[13] != "_":
                    predicate_dict3["BIO"].append(row[13])
                else:
                    predicate_dict3["BIO"].append("O")
                if row[13] == "V":
                    predicate_dict3["pred_sense"].append(int(row[0])-1)
                    predicate_dict3["pred_sense"].append(row[10])
                    predicate_dict3["pred_sense"].append( row[13])
                    predicate_dict3["pred_sense"].append( row[4])
            except IndexError:
                continue

            try:
                predicate_dict4["seq_words"] = predicate_dict["seq_words"]

                if row[14] != "_":
                    predicate_dict4["BIO"].append(row[14])
                else:
                    predicate_dict4["BIO"].append("O")
                if row[14] == "V":
                    predicate_dict3["pred_sense"].append(int(row[0])-1)
                    predicate_dict3["pred_sense"].append(row[10])
                    predicate_dict3["pred_sense"].append( row[14])
                    predicate_dict3["pred_sense"].append( row[4])
            except IndexError:
                continue
            try:
                predicate_dict5["seq_words"] = predicate_dict["seq_words"]

                if row[15] != "_":
                    predicate_dict5["BIO"].append(row[15])
                else:
                    predicate_dict5["BIO"].append("O")
                if row[15] == "V":
                    predicate_dict3["pred_sense"].append(int(row[0])-1)
                    predicate_dict3["pred_sense"].append(row[10])
                    predicate_dict3["pred_sense"].append( row[15])
                    predicate_dict3["pred_sense"].append( row[4])
            except IndexError:
                continue

        except StopIteration:
            break
    return predicate_dict


files_conll = ["../data/srl_univprop_en.train.conll", "../data/srl_univprop_en.dev.conll"]
for conll_file in files_conll:
    trying = make_dictionary(conll_file)

# inspiration from:https://github.com/Amberpichel/CoNLL-to-JSON-and-JSON-to-CoNLL/blob/main/conll_json2.py

import json

def process_conll_file(conll_path):
    """
    given a list of txt_paths
    -process each
    :param paths: list of content volume
    :return: list of dicts
    """
    
    
    with open(conll_path, 'rt', encoding='utf8') as infile:
        content = infile.readlines()
        
    list_dicts = []
    #strip newline, split on space char and make components for the dictionary
    
    # Label should be included!!
    for line in content:
        components = line.rstrip('\n').split()
        if len(components) > 0:
            sent_id = components[0]
            form = components[1]
            lemma = components[2]
            xpos = components[3]
            head = components[4]
            deprel = components[5]
            NE = components[6]
            children = components[7]

           
            
            feature_dict = {'sent_id': sent_id,
                            'form': form,
                            'lemma': lemma,
                            'xpos': xpos,
                            'head': head,
                            'deprel': deprel,
                            'NE': NE,
                            'children': children}
            list_dicts.append(feature_dict)

    return list_dicts

def write_file(list_dicts):
    """
    write volumes to new directory
    :param list_dicts: list_of dicts
    :param input_folder: folder with CONLL files
    :param text: pathname of CONLL file
    """
    outputfile = 'UP_English-EWT/en_ewt-up-combined.json'
    #write file to json format
    jsondumps = json.dumps(list_dicts)
    jsonfile = open(outputfile, "w")
    jsonfile.write(jsondumps)
    jsonfile.close()
    
dict_list = process_conll_file("C:/Users/desir/desktop/text_mining/nlp_technology/UP_English-EWT/en_ewt-up-train-parta-squeeze_try_combined.conllu")

write_file(dict_list)
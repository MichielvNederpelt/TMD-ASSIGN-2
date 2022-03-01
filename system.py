features = ['form', 'lemma', 'xpos', 'head', 'deprel', 'NE','children']
feature_to_idx = {'form': 1, 'lemma':2, 'xpos':3, 'head':4, 'deprel':5, 'NE':6, 'children':7}

def extract_feature_values(row, selected_features):
    '''
    Function that extracts feature value pairs from row and puts them in a dict

    :param row: row from conll file (list)
    :param selected_features: list of selected features

    :returns: dictionary of feature value pairs
    '''
    feature_values = {}
    for feature in selected_features:
        r_index = feature_to_idx.get(feature_name)
        feature_values[feature] = row[r_index]

    return feature_values

def extract_features_and_labels(trainingfile, selected_features):
    '''
    Extracts the features and labels from the gold datafile.
        param: trainingfile: the .conll file with samples on each line. First element in the line is the token,
                final element is the label
        param: selected_features: list of selected features
        returns: data: list of dicts {'token': TOKEN}
        returns: targets: list of target names for the tokens
            '''
    data = []
    targets = []
    with open(trainingfile, 'r', encoding='utf8') as infile:
        next(infile)
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                feature_dict = extract_feature_values(components, selected_features)

                data.append(feature_dict)
                #gold is in the last column
                targets.append(components[-1])
    return data, targets

def extract_features(inputfile, selected_features):
    '''
    Similar to extract_'features_and_labels, but only extracts data
        params: inputfile: an input file containing samples on each row, where feature token is the first word on each row
        param: selected_features: list of selected features
        returns: data: a list of dicts ('token': TOKEN)
            '''
    data = []
    with open(inputfile, 'r', encoding='utf8') as infile:
        next(infile)
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                feature_dict = extract_feature_values(components, selected_features)
                data.append(feature_dict)
    return data
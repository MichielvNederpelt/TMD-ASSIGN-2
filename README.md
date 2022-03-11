# NLP technology-ASSIGN-2

- Desiree Gerritsen
- Tessel Wisman
- Michiel van Nederpelt

The Github is structured as follows:

- ReadMe
- NN_code_NLP_TECH
  - Data
    -   xsrl_univprop_en.dev.conll
    -  srl_univprop_en.dev.jsonl
    - srl_univprop_en.train.conll
    - srl_univprop_en.train.jsonl
  - Tools
    - conll_to_json.py
    - srl_model_bert.py
    - srl_model_lstm.py
    - srl_predictor.py
    - srl_reader.py
  - srl_main.py

- RB_ML_NLP_TECH
  - code
    -   x
    -   x
    -   x
    -   x
  - models?
    -   x   
    -   x
    -   x
    -   x 

Requirements to run the scripts:

- Preprocess.py:
This scripts needs three modules to run. SpaCy (en_core_web_sm); the csv module and NumPy

- System.py:
In order to run this script, one needs the following modules and packages: sklearn, pandas, NumPy, csv, collections, os, random and sys.

- Evaluation.py:
Requirements for this script are: sys, pandas, os, collections, glob, sklearn and csv.

- Conll_to_json.py:
Necessary packages and modules are: json, collections and csv.

- srl scripts (BERT, lstm, predictor and reader):
Necessary packages and modules are: warnings, typing, torch, transformers, allennlp-models, SpaCy, NumPy, logging and json.
In order to run this scripts, one needs to have access to a Mac computer. Allennlp does not work on windows.


Overview of how to run the scripts:
- Step 1: Run preprocess.py. 
This script preprocesses the original file into several files, adding features and extracting predicates and arguments via a rule-based approach.

- Step 2: Run system.py
This script will train the SVM classifier and test it on the test data.

- Step 3: Run evaluation.py
This script will compare the predictions made by the classifier with the gold data.
Also, it shows the an evaluation on the extraction of predicates and arguments (compared to the gold data) to detect if extracts these properly.

- step 4: Run conll_to_json
This script will split the sentences into a number of different dictionary representations in a to be created conll file depending on the number of predicates in the sentence. The sentence, per predicate, will show a dictionary with seqeunce of words, BIO-representation of tokens and predicate_sense as keys.
To run on different files (than develop and train) adjust list at the bottom of the script and run.

-step 5: run srl_main.py
to run this script the AllenNLP model should be downloaded: follow steps described here to download: https://github.com/allenai/allennlp/blob/main/README.md#installing-via-pip
Unfortunatly in our project installing AllenNLP did not go as planned (mac was unable to install certain wheels) thus the code was not properly run on the data and the LSTM was not trained. However all functions are described and relative paths are in place so when the installment has succeeded the script should be ready to run. 

## Description of the classification task for argument classification:
The argument classification task is a task which concernes the characterization of events. In order to do so, predicate(s) of a sentence need to be identified. A predicate is the main token that establishes 'what' took place. Other information, such as 
'who', 'where', 'when' to 'whom' provide more information about the event, and therefore it's important to be able to identify them. So the primary task is to identify the predicates and the associated arguments (either participants or properties). After this is done, a system will be trained on the identified labels which can be tested on new test data.

## Description of predicate and argument extraction:

First, the dataset (Universal proposition banks)  was preprocessed in such a way that all predicates and arguments were found. This was done by counting all the extra columns in our dataset, which denote an extra predicate. Then, the sentence was duplicated the same amount of times as the previous count. Each duplicated sentence focused on a seperate predicate and it's arguments. 

In order to extract predicates, a rule-based approach was used by looking at dependency labels. We identified predicates when having the following dependency label: 'root', 'xcomp', 'ccomp', 'parataxis', 'advcl' or 'conj' with VERB as part-of-speech tag. Also, predicates were extracted if they had an UPOS of AUX, in combination with an XPOS of VBD, VBZ, VBN, VBP, VB or VBG.

Based on the predicates, arguments need to be identified. For the rule based approach, three classes of arguments were categorized. ARG0 is assigned to the subject, ARG1 is assigned to the object and ARG2 to other arguments such as time or date. We used the dependency label to identify these arguments. Using the dependency label, arguments consisting of nsubj, obj or obl were identified as arguments. 

This approach is a simple way of extracting predicates and arguments. Unfortunately, it isn't flawless since rules are hard coded which does not provide space for exceptions. We found that some predicates had a different dependency label, which results in overlooking that predicate. Most of the time, predicates are verbs, and therefore we accounted for verbs to be roots. But some predicates were nouns, which means that these were not extracted. Another problem was labeling incorrect predicates or arguments based on the rules. for instance, we coded the dependency label 'obl' to be an argument indicator for ARG2. But not all 'obl' instances are an argument, or they are a different argument type. Replicating this approach can be done by running predicates_from_dependency.py. These flaws will also flow through the experiments. 

## List of features: 
Some features are already present in our dataset, such as dependencies, lemma, univeral pos-tag, and specific POS tags (XPOS). 

Based on previous research, we propose to add the following features (Marquez et al, 2008):

Lexical features:
- Children of token:
In order to identify which possible arguments a token and possible predicate takes, we want to add children as a lexical feature. By using SpaCy, we labeled all possible children of a token, and added this to the dataset. 

- distance to predicate (in our dataset called head):
In order to identify the distance from the predicate a token has. The system could pick up possible patterns showing that certain arguments are closer to the predicate compared to others.

- Named entity labels:
Labeling named entities can help to identify a role of a token such as person or organization. This information can help to identify easily is something is a specific argument, since arguments can denote an agent, or patient or location (Gübür, 2021).

Syntactic features:
- Dependency labels:
This feature is already provided in the dataset, but it is a very useful feature that identifies which syntactic relationships between tokens and the root of a sentence. 

- Dependency relation to head:
This feature is already provided in the dataset. It is a useful feature which indicates how a token is related to the head. This way, the system can possibly identify patterns that are common and uncommon. 

## Choice of Machine learning algorithm:

### SVM

The system which is used is the Support Vector Machine (SVM), which is a supervised learning model that can deal with a large number of data and features. It is a system can be used for classification tasks where the system tries to find the best seperation line (also called hyperplane) between datapoints of classes. Other research (Hacioglu, 2004; Pradhan et al, 2005) has shown that SVM was a good system to use, and therefore we want to use it as well in combination with our extracted features. 

### LSTM

For a second system, we will make use of a LSTM neural network based on AllenNLP. This system makes use of three main concepts, the DatasetReader, Model and Trainer, and the Predictor.  

The DatasetReader component basically does what it says, its function is in its name. It is the component of the pipeline that allows the raw (input) data format to be in a readable, or processable, format for later steps in the pipeline. In its most basic form, the component separates the (Instance) object into a minimum of two fields. The first is the token/text snippet, sentence etc. which needs to be classified (TextField) from the labels that it should classify (gold label) (LabelField). The Fields into which the data is separated can either be input or output. The fields will be converted to Tensors which will then be fed to the model. 
For training one would like to have instances separated in input fields and the labels used for prediction (the output fields). For the Semantic role labeling task, the input will be sequence of words, BIO labels and the predicate senses with the position they take and XPOS tag. The expected output will be a list of classification instances represented as tensors.

The Model and Trainer component will take as input a batch of Instances. First, the model converts the tokens into a vector. This will create a large tensor since each token is vectorized. In order to create smaller vectors, a sequence of vectors for each token is squased into a single vector.

The model will combine word-level features into a document level feature vector. Then it classifies that vector into one of the labels, which will be the expected output. Lastly, each single feature vector is classified as a label which provides some information about the probability distribution of the labels. 

The predictor takes as input the vector for each instance in a batch and predicts a label for it. The output is expected to be a score for each possible label and the computed loss. 

# Evaluation:
After running evaluation.py, one can see precision, recall and F-scores for each argument class and predicates. The macro-average weight shows a performance of 0.401 (F-score). Recall is 0.35, which means that 35% of relevant items were retrieved. Precision, however, is 0.55 which means that of the retrieved elements 55% is classified correctly. 

A part can be explained by our rule-based approach. The rules were sufficient for extracting predicates, which has an F-score of 0.838. Unfortunately, the rules were not sufficient enough to detect and extract all arguments, which makes it of course also more difficult to predict them when information is missing for the system.

### Error analysis test set:
In the table below, some examples of correct and incorrect predicted labels are displayed. Please note that this is only a fraction. 
Example 1 and 2 show two cases in which the argument was correctly predicted as an ARG0. In example 3 and 4, the tokens were labeled as ARG0 arguments, whereas they were in fact different. Compared to row 1 and 2, they have in common that the UPOS and XPOS are also a pronoun and PRP. This indicated that the system might think that such a combination is an indication of an ARG0. Taking into account example 5 and 6, the system also sometimes does not recognize an ARG0. In case of example 5, it might be due to the fact that the combination of UPOS and XPOS is more indicative of an ARGM instead of an ARG0. Example 7 shows a correctly predicted ARGM with similar features to example 5. Example 6 has both UPOS and XPOS in common with the correct examples, but the dependency is different. 
This indicates that the system is very keen on specific feature combinations as patterns. However, there are a lot of exceptions which cannot be detected due to the current system. Especially in cases where there are few examples (such as ARG3 and ARG4 and ARG5). 

|  | TOKEN | UPOS | XPOS | Dependency | Gold | Predicted |
|:---:| :---:|:---:|:---:|:---:|:---:|:---:|
| 1| they | PRON | PRP| nsubj | ARG0| ARG0 |
| 2| Syria | PROPN | NNP| nsubj | ARG0| ARG0 |
| 3| I | PRON | PRP| nsubj | O | ARG0 |
| 4| Them | PRON | PRP| obl | ARG2| ARG0 |
| 5| Message | NOUN | NN| obl | ARG0| ARGM |
| 6| Muhsin | PROPN | NNP| obl | ARG0| ARGM|
| 7| Attempts | NOUN | NNS| obl | ARGM| ARGM |

Although not displayed, the same patterns can be seen for the other arguments. Patterns that consist of specific pos tag combinations together with the dependency label that pushes the system to a certain prediction. Exceptions are not strong enough to push the system to another possible choice. Therefore, we would recommend to add some more contextual features, such as word embeddings or syntactic n-grams.  

### References:
- Màrquez, L., Carreras, X., Litkowski, K. C., & Stevenson, S. (2008). Semantic role labeling: an introduction to the special issue. Computational linguistics, 34(2), 145-159.
- Hacioglu, K. (2004). Semantic role labeling using dependency trees. In COLING 2004: Proceedings of the 20th International Conference on Computational Linguistics (pp. 1273-1276).
- Pradhan, S., Ward, W., Hacioglu, K., Martin, J. H., & Jurafsky, D. (2005, June). Semantic role labeling using different syntactic views. In Proceedings of the 43rd Annual Meeting of the Association for Computational Linguistics (ACL’05) (pp. 581-588).
- Support vector machines: The linearly separable case. (z.d.). nlp stanford. Geraadpleegd op 28 februari 2022, van https://nlp.stanford.edu/IR-book/html/htmledition/support-vector-machines-the-linearly-separable-case-1.html
- Gübür, K. T. (2021, 19 juli). Named Entity Recognition: Definition, Examples, and Guide. Holistic SEO. Geraadpleegd op 28 februari 2022, van https://www.holisticseo.digital/theoretical-seo/named-entity-recognition/
- Training and prediction ·. (z.d.). A Guide to Natural Language Processing With AllenNLP. Geraadpleegd op 9 maart 2022, van https://guide.allennlp.org/training-and-prediction/

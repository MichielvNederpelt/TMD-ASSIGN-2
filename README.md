# NLP technology-ASSIGN-2

- Desiree Gerritsen (s2700295)
- Tessel 
- Michiel
- Nihed

## Description of predicate and argument extraction:

[ Tessel ] ?


## Description of the classification task for argument classification:
The argument classification task is a task which concernes the characterization of events. In order to do so, predicate(s) of a sentence need to be identified. A predicate is the main token that establishes 'what' took place. Other information, such as 
'who', 'where', 'when' to 'whom' provide more information about the event, and therefore it's important to be able to identify them. So the primary task is to identify the predicates and the associated arguments (either participants or properties). After this is done, a system will be trained on the identified labels which kcan be tested on new test data. 

## List of features: 
Some features are already present in our dataset, such as dependencies, lemma, POS tag, XPOS tags. 

Based on previous research, we propose to add the following features (references):

Lexical features:
- Phrase type
- Parent / direct headword
- Children (if any) of each token
- Constituent
- Named entity labels
- Word embeddings

Syntactic features:
- Dependency label
- Syntactic N-grams

## Choice of Machine learning algorithm:

The system which is used is the Support Vector Machine (SVM), which is a supervised learning model. This model will 

## Describe the generation of training and test instances:


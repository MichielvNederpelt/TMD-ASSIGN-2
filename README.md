# NLP technology-ASSIGN-2

- Desiree Gerritsen
- Tessel Wisman
- Michiel van Nederpelt
- Nihed Harrak

## Description of predicate and argument extraction:

First, the dataset was preprocessed in such a way that all predicates and arguments were found. This was done by counting all the extra columns in our dataset, which denote an extra predicate. Then, the sentence was duplicated the same amount of times as the previous count. Each duplicated sentence focused on a seperate predicate and it's arguments. 

In order to extract predicates, a rule-based approach was created. We identified predicates when having the following dependency label: 'root', 'xcomp', 'ccomp', 'parataxis', 'advcl' or 'conj' with VERB as part-of-speech tag.

Based on the predicates, arguments need to be identified. For the rule based approach, three classes of arguments were categorized. ARG0 is assigned to the subject, ARG1 is assigned to the object and ARG2 to other arguments such as time or date. We used the dependency label to identify these arguments. If the dependency label was 'nsubj', we assigned it ARG0. If the dependency label was 'obj', ARG1 was assigned. Lastly, when 'obl' was seen ARG2 is assigned. 

Since the rule-based approach is not the best, given that a lot of predicates were differently identified compared to the gold data, we choose to train the system on the gold labels provided in the dataset. 

## Description of the classification task for argument classification:
The argument classification task is a task which concernes the characterization of events. In order to do so, predicate(s) of a sentence need to be identified. A predicate is the main token that establishes 'what' took place. Other information, such as 
'who', 'where', 'when' to 'whom' provide more information about the event, and therefore it's important to be able to identify them. So the primary task is to identify the predicates and the associated arguments (either participants or properties). After this is done, a system will be trained on the identified labels which kcan be tested on new test data. 

## List of features: 
Some features are already present in our dataset, such as dependencies, lemma, POS tag, XPOS tags. 

Based on previous research, we propose to add the following features (Marquez et al, 2008):

Lexical features:
- Children of tokens:
In order to identify which possible arguments a token and possible predicate takes, we want to add children as a lexical feature. By using SpaCy, we labeled all possible children of a token, and added this to the dataset. 

- Named entity labels:
Labeling named entities can help to identify a role of a token such as person or organization. This information can help to identify easily is something is a specific argument, since arguments can denote an agent, or patient or location (Gübür, 2021).

- Word embeddings: ?


Syntactic features:
- Dependency labels:
This feature is already provided in the dataset, but it is a very useful feature that identifies which syntactic relationships between tokens and the root of a sentence. 

- Path to root:
The path to the root is a feature that can show the relationship between tokens in a sentence and also shows a more deeper structure underlying sentences. A system can learn patterns that show possible and impossible paths which can be useful for predicting arguments. 

- Syntactic N-grams:
This feature will provide the system with context and possible and impossible combinations of words sequences. 

## Choice of Machine learning algorithm:

The system which is used is the Support Vector Machine (SVM), which is a supervised learning model that can deal with a large number of data and features. It is a system can be used for classification tasks where the system tries to find the best seperation line (also called hyperplane) between datapoints of classes. Other research (Hacioglu, 2004; Pradhan et al, 2005) has shown that SVM was a good system to use, and therefore we want to use it as well in combination with our extracted features. 


### References:
- Màrquez, L., Carreras, X., Litkowski, K. C., & Stevenson, S. (2008). Semantic role labeling: an introduction to the special issue. Computational linguistics, 34(2), 145-159.
- Hacioglu, K. (2004). Semantic role labeling using dependency trees. In COLING 2004: Proceedings of the 20th International Conference on Computational Linguistics (pp. 1273-1276).
- Pradhan, S., Ward, W., Hacioglu, K., Martin, J. H., & Jurafsky, D. (2005, June). Semantic role labeling using different syntactic views. In Proceedings of the 43rd Annual Meeting of the Association for Computational Linguistics (ACL’05) (pp. 581-588).
- Support vector machines: The linearly separable case. (z.d.). nlp stanford. Geraadpleegd op 28 februari 2022, van https://nlp.stanford.edu/IR-book/html/htmledition/support-vector-machines-the-linearly-separable-case-1.html
- Gübür, K. T. (2021, 19 juli). Named Entity Recognition: Definition, Examples, and Guide. Holistic SEO. Geraadpleegd op 28 februari 2022, van https://www.holisticseo.digital/theoretical-seo/named-entity-recognition/

# NLP technology-ASSIGN-2

- Desiree Gerritsen
- Tessel Wisman
- Michiel van Nederpelt

## Description of predicate and argument extraction:

First, the dataset (Universal proposition banks)  was preprocessed in such a way that all predicates and arguments were found. This was done by counting all the extra columns in our dataset, which denote an extra predicate. Then, the sentence was duplicated the same amount of times as the previous count. Each duplicated sentence focused on a seperate predicate and it's arguments. 

In order to extract predicates, a rule-based approach was used by looking at dependency labels. We identified predicates when having the following dependency label: 'root', 'xcomp', 'ccomp', 'parataxis', 'advcl' or 'conj' with VERB as part-of-speech tag.

Based on the predicates, arguments need to be identified. For the rule based approach, three classes of arguments were categorized. ARG0 is assigned to the subject, ARG1 is assigned to the object and ARG2 to other arguments such as time or date. We used the dependency label to identify these arguments. If the dependency label was 'nsubj', we assigned it ARG0. If the dependency label was 'obj', ARG1 was assigned. Lastly, for the dependency label 'obl' ARG2 was assigned. 

This approach is a simple way of extracting predicates and arguments. Unfortunately, it isn't flawless since rules are hard coded which does not provide space for exceptions. We found that some predicates had a different dependency label, which results in overlooking that predicate. Another problem was labeling incorrect predicates or arguments based on the rules. Replicating this approach can be done by running predicates_from_dependency.py.

[ insert matching of rule based to gold labels ]

Due to these flaws, we choose to train the system on the gold labels provided in the dataset.

## Description of the classification task for argument classification:
The argument classification task is a task which concernes the characterization of events. In order to do so, predicate(s) of a sentence need to be identified. A predicate is the main token that establishes 'what' took place. Other information, such as 
'who', 'where', 'when' to 'whom' provide more information about the event, and therefore it's important to be able to identify them. So the primary task is to identify the predicates and the associated arguments (either participants or properties). After this is done, a system will be trained on the identified labels which kcan be tested on new test data. 

## List of features: 
Some features are already present in our dataset, such as dependencies, lemma, POS tag, XPOS tags. 

Based on previous research, we propose to add the following features (Marquez et al, 2008):

Lexical features:
- Children of token:
In order to identify which possible arguments a token and possible predicate takes, we want to add children as a lexical feature. By using SpaCy, we labeled all possible children of a token, and added this to the dataset. 

- Head of token:
In order to identify which possible predicate a token has, we want to add the tokens head as a lexical feature. By using SpaCy, we labeled all possible heads of a token, and added this to the dataset. 

- Named entity labels:
Labeling named entities can help to identify a role of a token such as person or organization. This information can help to identify easily is something is a specific argument, since arguments can denote an agent, or patient or location (Gübür, 2021).

Syntactic features:
- Dependency labels:
This feature is already provided in the dataset, but it is a very useful feature that identifies which syntactic relationships between tokens and the root of a sentence. 

- Dependency relation to head:
This feature is already provided in the dataset. It is a useful feature which indicates how a token is related to the head. This way, the system can possibly identify patterns that are common and uncommon. 

## Choice of Machine learning algorithm:

The system which is used is the Support Vector Machine (SVM), which is a supervised learning model that can deal with a large number of data and features. It is a system can be used for classification tasks where the system tries to find the best seperation line (also called hyperplane) between datapoints of classes. Other research (Hacioglu, 2004; Pradhan et al, 2005) has shown that SVM was a good system to use, and therefore we want to use it as well in combination with our extracted features. 

For a second system, we will make use of a LSTM neural network....


### References:
- Màrquez, L., Carreras, X., Litkowski, K. C., & Stevenson, S. (2008). Semantic role labeling: an introduction to the special issue. Computational linguistics, 34(2), 145-159.
- Hacioglu, K. (2004). Semantic role labeling using dependency trees. In COLING 2004: Proceedings of the 20th International Conference on Computational Linguistics (pp. 1273-1276).
- Pradhan, S., Ward, W., Hacioglu, K., Martin, J. H., & Jurafsky, D. (2005, June). Semantic role labeling using different syntactic views. In Proceedings of the 43rd Annual Meeting of the Association for Computational Linguistics (ACL’05) (pp. 581-588).
- Support vector machines: The linearly separable case. (z.d.). nlp stanford. Geraadpleegd op 28 februari 2022, van https://nlp.stanford.edu/IR-book/html/htmledition/support-vector-machines-the-linearly-separable-case-1.html
- Gübür, K. T. (2021, 19 juli). Named Entity Recognition: Definition, Examples, and Guide. Holistic SEO. Geraadpleegd op 28 februari 2022, van https://www.holisticseo.digital/theoretical-seo/named-entity-recognition/

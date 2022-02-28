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

Based on previous research, we propose to add the following features (Marquez et al, 2008):

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

The system which is used is the Support Vector Machine (SVM), which is a supervised learning model that can deal with a large number of data and features. It is a system can be used for classification tasks where the system tries to find the best seperation line (also called hyperplane) between datapoints of classes. Other research (Hacioglu, 2004; Pradhan et al, 2005) has shown that SVM was a good system to use, and therefore we want to use it as well in combination with our extracted features. 


### References:
Màrquez, L., Carreras, X., Litkowski, K. C., & Stevenson, S. (2008). Semantic role labeling: an introduction to the special issue. Computational linguistics, 34(2), 145-159.
Hacioglu, K. (2004). Semantic role labeling using dependency trees. In COLING 2004: Proceedings of the 20th International Conference on Computational Linguistics (pp. 1273-1276).
Pradhan, S., Ward, W., Hacioglu, K., Martin, J. H., & Jurafsky, D. (2005, June). Semantic role labeling using different syntactic views. In Proceedings of the 43rd Annual Meeting of the Association for Computational Linguistics (ACL’05) (pp. 581-588).

